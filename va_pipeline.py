#!/usr/bin/env python3
"""
ElevenLabs Voice Bot Call Processing Pipeline

Flow per conversation:
  1. Fetch list (paginated) → deduplicate
  2. Fetch full detail from ElevenLabs
  3. Store to calls_va_answered  (Category = null always)
  4. Skip if call duration < 40 s
  5. AI extraction (GPT-4.1-mini)
  6. BigQuery institution lookup
  7. Skip if no institution found
  8. Upsert call_notes_latest  (key: phone)
  9. Upsert call_notes_2       (key: doc_id)
 10. Send email alert if category is Complaint / Negotiation / PTP / Paid / Callback Request / Request to Speak to Agent

Run manually or via cron at 1 AM:
    0 1 * * * /path/to/venv/bin/python /path/to/pipeline.py >> /var/log/pipeline.log 2>&1
"""

import base64
import json
import logging
import os
import smtplib
import sys
import time
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

import requests
from dateutil import parser as dateutil_parser
from dotenv import load_dotenv
from google.cloud import bigquery, firestore
from google.oauth2 import service_account
from openai import OpenAI

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


# ── Environment ────────────────────────────────────────────────────────────────

ELEVENLABS_API_KEY  = os.environ["ELEVENLABS_API_KEY"]
OPENAI_API_KEY      = os.environ["OPENAI_API_KEY"]
GOOGLE_SA_FILE      = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE", "service_account.json")
GMAIL_SENDER        = os.environ["GMAIL_SENDER"]
GMAIL_APP_PASSWORD  = os.environ["GMAIL_APP_PASSWORD"]
ALERT_RECIPIENT     = os.getenv("ALERT_RECIPIENT", "payments@fsldigital.com")

AGENT_ID          = "agent_3401kkqz2vryek4rxgh1x3zef4wk"
BQ_PROJECT        = "fssspark"
FS_PROJECT        = "fssspark"
MIN_DURATION_SECS = 40

ALERT_CATEGORIES = {
    "Complaint",
    "Negotiation",
    "Request to Speak to Agent",
    "PTP",
    "Callback Request",
    "Paid",
}


# ── Prompts / schema ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a call analysis engine for a loan recovery company.

You will be given a JSON object from a completed voice bot call. Your job is to:
1. Extract structured information from the transcript
2. Classify the conversation into exactly one category

CLASSIFICATION CATEGORIES:
- Paid — Customer confirms they have already made a payment or settlement
- PTP — Customer makes a clear, specific promise to pay on a stated date or within a stated timeframe. A vague "I'll pay soon" is NOT a PTP.
- Negotiation — Customer is discussing repayment terms, requesting a reduced amount, an extension, or proposing an alternative arrangement
- Complaint — Customer is expressing dissatisfaction, disputing the loan, the charges, or the recovery process
- System Error — Call dropped unexpectedly, transcript is empty or garbled, duration is abnormally short with no meaningful exchange, or termination reason indicates a technical failure
- Non-Engagement — Customer was reached but refused to engage, gave no meaningful response, or repeatedly deflected without committing to anything
- Unresolved — Conversation ended without a clear outcome — no payment, no PTP, no complaint, no refusal. Customer may have been unreachable or call ended ambiguously.

CLASSIFICATION RULES:
- If transcript is empty, missing, or under 2 meaningful exchanges → System Error
- If customer was reached but said nothing useful → Non-Engagement
- A specific date or amount mentioned with intent to pay → PTP, not Negotiation
- Customer saying "I already paid" → Paid, not Complaint, even if they sound frustrated
- If two categories seem to apply, pick the dominant intent of the conversation

---

EXTRACTION FIELDS:

Amount_Promised: Exact amount customer committed to paying (blank if none)
Payment_Method: Only if customer claims prior payment not yet reflected. One of: Branch payment, Zenith bank, Credit officer, Paystack, Union Bank, Leader, Providus Bank, Sales Rep, Unknown
Promise_to_Pay: Yes/No/Unknown
Right_Party_Contact: Yes/No/Unknown
Willingness: Score the customer's willingness to repay based on their behaviour in the call. Choose exactly one:
  - "0 Unwilling (Not Speaking)" — Customer picked up but said nothing at all, completely silent
  - "1 Unwilling" — Customer spoke but explicitly refused to pay or showed no willingness
  - "2 Medium" — Customer was reluctant but did not outright refuse, OR agreed without a confirmed PTP (a confirmed PTP requires both a ptp_date AND an Amount_Promised)
  - "3 Willing" — Customer willingly agreed to pay, most often with a confirmed PTP (date + amount)
  - "Unknown" — Cannot be determined (e.g. call dropped, debt denied, transcript unclear)
agent: Always set to "Voice Bot"
alternative_phone: Alternate number provided by customer (blank if none)
callback_requested: Yes/No/Unknown
contact_method: Always set to "Voice Call"
denied_debt: Yes/No/Unknown
employment_status: Business Owner/Salary Earner/Both/Unknown
federal_deduction: Yes/No/Unknown
name_of_officer: Name of officer customer claims to have paid (blank if none)
offset_loan_with_savings: Yes/No/Unknown
other_information: Concise summary of the conversation written from the agent's perspective. Do not attribute agent observations to the customer.
ptp_date: Specific payment date in YYYY-MM-DD format. Convert relative dates (e.g. "tomorrow", "next Friday") to exact dates based on call date. Blank if none.
reason_for_default: Brief summary of why customer defaulted (blank if not provided)
language: Language customer spoke to the bot. One of: Yoruba, English, Igbo, Hausa, Sheng, Swahili, Pidgin
discount_accepted: true/false/null
Category: The classification category from the list above

EXTRACTION RULES:
- other_information must reflect what the CUSTOMER said or did, not what the bot observed
- "Not reachable", "no response", "switched off" = agent observation, NOT a customer complaint
- If the customer was unreachable, leave complaint-related fields blank and classify as Unresolved or System Error
- Never infer a ptp_date that was not stated or clearly implied

---

STRICT OUTPUT RULES:
1. Return a single valid JSON object — no markdown fences, no extra text
2. discount_accepted must be true, false, or null — never a string
3. All dates must be YYYY-MM-DD strings
4. If a field cannot be determined, use "" for text fields and null for booleans
5. Category is required — never leave it blank\
"""


# ── Google clients ─────────────────────────────────────────────────────────────

def _creds():
    _scopes = ["https://www.googleapis.com/auth/cloud-platform"]
    raw = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    if raw:
        info = json.loads(base64.b64decode(raw))
        return service_account.Credentials.from_service_account_info(info, scopes=_scopes)
    return service_account.Credentials.from_service_account_file(GOOGLE_SA_FILE, scopes=_scopes)


def _bq():
    return bigquery.Client(project=BQ_PROJECT, credentials=_creds())


def _fs():
    return firestore.Client(project=FS_PROJECT, credentials=_creds())


# ── ElevenLabs ─────────────────────────────────────────────────────────────────

_EL_HEADERS = {"xi-api-key": ELEVENLABS_API_KEY}


def _time_window() -> tuple[int, int]:
    wat   = timezone(timedelta(hours=1))
    now   = datetime.now(wat)
    after = int((now - timedelta(days=6)).replace(hour=0, minute=0, second=0, microsecond=0).timestamp())
    before = int(now.timestamp())
    return before, after


def fetch_all_conversations() -> list[dict]:
    before, after = _time_window()
    base_params = {
        "agent_id":               AGENT_ID,
        "page_size":              100,
        "call_start_before_unix": before,
        "call_start_after_unix":  after,
    }

    all_convs: list[dict] = []
    cursor: Optional[str] = None

    while True:
        params = {**base_params, **({"cursor": cursor} if cursor else {})}
        resp = requests.get(
            "https://api.elevenlabs.io/v1/convai/conversations",
            headers=_EL_HEADERS,
            params=params,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        all_convs.extend(data.get("conversations", []))
        cursor = data.get("next_cursor")
        if not cursor:
            break

    seen: dict[str, dict] = {}
    for c in all_convs:
        cid = c.get("conversation_id")
        if cid and cid not in seen:
            seen[cid] = c

    log.info("Fetched %d unique conversations", len(seen))
    return list(seen.values())


def fetch_conversation_detail(conversation_id: str) -> Optional[dict]:
    try:
        resp = requests.get(
            f"https://api.elevenlabs.io/v1/convai/conversations/{conversation_id}",
            headers=_EL_HEADERS,
            params={"agent_id": AGENT_ID},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        log.warning("Detail fetch failed for %s: %s", conversation_id, exc)
        return None


# ── OpenAI ─────────────────────────────────────────────────────────────────────

_oai_client: Optional[OpenAI] = None


def _oai() -> OpenAI:
    global _oai_client
    if _oai_client is None:
        _oai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _oai_client


def extract_call_data(detail: dict) -> dict:
    transcript  = detail.get("transcript") or []
    first_msg   = transcript[0].get("message", "") if transcript else ""
    analysis    = detail.get("analysis") or {}
    metadata    = detail.get("metadata") or {}
    dyn_vars    = ((detail.get("conversation_initiation_client_data") or {})
                   .get("dynamic_variables") or {})

    user_content = (
        f"Transcript: {first_msg}\n"
        f"Summary: {analysis.get('transcript_summary', '')}\n"
        f"Termination Reason: {metadata.get('termination_reason', '')}\n"
        f"Call Duration: {metadata.get('call_duration_secs', 0)} seconds\n"
        f"Full Analysis: {json.dumps(analysis)}\n"
        f"System Conversation History: {dyn_vars.get('system__conversation_history', '')}"
    )

    resp = _oai().chat.completions.create(
        model="gpt-4.1-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ],
    )
    return json.loads(resp.choices[0].message.content)


# ── BigQuery ───────────────────────────────────────────────────────────────────

def query_institution(user_id: str, bq_client: bigquery.Client) -> Optional[str]:
    phone = ("0" + user_id) if not user_id.startswith("0") else user_id
    sql = """
        SELECT ANY_VALUE(institution) AS institution
        FROM `fssspark.original_cohorts.all_leads`
        WHERE phone = @phone
    """
    cfg = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("phone", "STRING", phone)]
    )
    try:
        for row in bq_client.query(sql, job_config=cfg).result():
            return row.institution
    except Exception as exc:
        log.warning("BigQuery error for %s: %s", user_id, exc)
    return None


# ── Firestore ──────────────────────────────────────────────────────────────────

def fs_upsert(db: firestore.Client, collection: str, key_field: str, data: dict):
    key_value = data.get(key_field)
    if key_value is None:
        log.warning("Skipping upsert into '%s' — key field '%s' is missing", collection, key_field)
        return
    docs = list(
        db.collection(collection)
          .where(key_field, "==", key_value)
          .limit(1)
          .stream()
    )
    if docs:
        docs[0].reference.set(data, merge=True)
    else:
        db.collection(collection).add(data)


# ── Gmail ──────────────────────────────────────────────────────────────────────

def send_alert(
    phone: str,
    category: str,
    institution: str,
    call_dt: str,
    summary: str,
    other_info: str,
):
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"Voice Bot Follow-up Required - {category} | {phone}"
    msg["From"]    = f"voicebot <{GMAIL_SENDER}>"
    msg["To"]      = ALERT_RECIPIENT

    html = (
        "<h2>Voice Bot Follow-up Required</h2>"
        f"<p><strong>Phone:</strong> {phone}</p>"
        f"<p><strong>Category:</strong> {category}</p>"
        f"<p><strong>Institution:</strong> {institution}</p>"
        f"<p><strong>Date of Call:</strong> {call_dt}</p>"
        "<p><strong>Transcription:</strong></p>"
        f"<p>{summary}</p>"
        "<p><strong>Other Information:</strong></p>"
        f"<p>{other_info}</p>"
    )
    msg.attach(MIMEText(html, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(GMAIL_SENDER, GMAIL_APP_PASSWORD)
        smtp.send_message(msg)


# ── Helpers ────────────────────────────────────────────────────────────────────

def normalise_phone(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("+234"):
        raw = "0" + raw[4:]
    elif raw.startswith("234") and len(raw) >= 13:
        raw = "0" + raw[3:]
    return ("0" + raw)[-11:]


def parse_system_time(system_time: str) -> datetime:
    """Parse any date string ElevenLabs sends (ISO or human-readable) into a UTC datetime."""
    if not system_time:
        return datetime.now(timezone.utc)
    try:
        dt = dateutil_parser.parse(system_time)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return datetime.now(timezone.utc)


def to_iso(dt: datetime) -> str:
    """Format datetime as 2026-05-20T17:19:00.000Z"""
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def parse_call_dt(system_time: str) -> str:
    return parse_system_time(system_time).strftime("%Y-%m-%dT%H:%M:%S")


def epoch_ms(system_time: str) -> int:
    try:
        return int(parse_system_time(system_time).timestamp() * 1000)
    except Exception:
        return int(time.time() * 1000)


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run():
    log.info("=== Pipeline started ===")
    bq = _bq()
    db = _fs()

    conversations = fetch_all_conversations()
    log.info("Total to process: %d", len(conversations))

    for conv in conversations:
        conversation_id = conv.get("conversation_id")
        if not conversation_id:
            continue

        log.info("▶ %s", conversation_id)

        # ── 1. Fetch full conversation detail ────────────────────────────────
        detail = fetch_conversation_detail(conversation_id)
        if not detail:
            continue

        time.sleep(0.5)  # Gentle rate-limit guard

        metadata   = detail.get("metadata") or {}
        analysis   = detail.get("analysis") or {}
        phone_call = metadata.get("phone_call") or {}
        feat_usage = metadata.get("features_usage") or {}
        dyn_vars   = ((detail.get("conversation_initiation_client_data") or {})
                      .get("dynamic_variables") or {})

        user_id     = conv.get("user_id", "")
        duration    = metadata.get("call_duration_secs", 0) or 0
        system_time = dyn_vars.get("system__time", "")
        call_dt     = parse_call_dt(system_time)

        # ── 2. Store to calls_va_answered (Category always null here) ────────
        va_doc = {
            "phone_number":        user_id,
            "duration":            str(duration),
            "category":            None,
            "status":              analysis.get("call_successful", ""),
            "datetime":            call_dt,
            "timestamp":           datetime.now(timezone.utc).isoformat(),
            "call_type":           phone_call.get("direction", ""),
            "agent_id":            conv.get("agent_id", ""),
            "agent_name":          conv.get("agent_name", ""),
            "voicemail_detection": str((feat_usage.get("voicemail_detection") or {}).get("used", "")),
            "transcript_summary":  analysis.get("transcript_summary", ""),
            "conversation_id":     conversation_id,
        }
        try:
            db.collection("calls_va_answered").document(conversation_id).set(va_doc, merge=True)
            log.info("  ✓ calls_va_answered")
        except Exception as exc:
            log.error("  ✗ calls_va_answered: %s", exc)

        # ── 3. Duration gate — skip calls shorter than 40 seconds ────────────
        if duration < MIN_DURATION_SECS:
            log.info("  skip — %ds < %ds minimum", duration, MIN_DURATION_SECS)
            continue

        # ── 4. AI extraction ─────────────────────────────────────────────────
        try:
            extracted = extract_call_data(detail)
            log.info("  ✓ AI — Category: %s", extracted.get("Category"))
        except Exception as exc:
            log.error("  ✗ AI extraction: %s", exc)
            continue

        category = extracted.get("Category", "")

        # ── 5. Normalise phone (needed for both BQ lookup and doc fields) ────────
        ext_number = phone_call.get("external_number", "")
        phone_norm = normalise_phone(ext_number) if ext_number else normalise_phone(user_id)

        # ── 6. BigQuery — institution lookup ─────────────────────────────────
        institution = query_institution(phone_norm, bq)
        if not institution:
            log.info("  skip — no institution found for %s", phone_norm)
            continue

        # ── 7. Build shared document ──────────────────────────────────────────
        doc_id  = f"{phone_norm}_{epoch_ms(system_time)}"
        call_ts = to_iso(parse_system_time(system_time))
        now_dt  = to_iso(datetime.now(timezone(timedelta(hours=1))))

        call_notes = {
            "Amount_Promised":          extracted.get("Amount_Promised", ""),
            "Payment_Method":           extracted.get("Payment_Method", ""),
            "Promise_to_Pay":           extracted.get("Promise_to_Pay", ""),
            "Right_Party_Contact":      extracted.get("Right_Party_Contact", ""),
            "Willingness":              extracted.get("Willingness", ""),
            "agent":                    extracted.get("agent", ""),
            "alternative_phone":        extracted.get("alternative_phone", ""),
            "callback_requested":       extracted.get("callback_requested", ""),
            "contact_method":           extracted.get("contact_method", ""),
            "denied_debt":              extracted.get("denied_debt", ""),
            "employment_status":        extracted.get("employment_status", ""),
            "federal_deduction":        extracted.get("federal_deduction", ""),
            "institution":              institution,
            "name_of_officer":          extracted.get("name_of_officer", ""),
            "offset_loan_with_savings": extracted.get("offset_loan_with_savings", ""),
            "other_information":        extracted.get("other_information", ""),
            "phone":                    phone_norm,
            "ptp_date":                 extracted.get("ptp_date", ""),
            "reason_for_default":       extracted.get("reason_for_default", ""),
            "timestamp":                call_ts,
            "updated_at":               now_dt,
            "discount_accepted":        extracted.get("discount_accepted"),
            "language":                 extracted.get("language", ""),
        }

        # ── 7. Firestore — call_notes_latest (doc ID = phone) ───────────────
        try:
            db.collection("call_notes_latest").document(phone_norm).set(call_notes, merge=True)
            log.info("  ✓ call_notes_latest")
        except Exception as exc:
            log.error("  ✗ call_notes_latest: %s", exc)

        # ── 8. Firestore — call_notes_2 (doc ID = phone_timestamp) ───────────
        try:
            db.collection("call_notes_2").document(doc_id).set(call_notes, merge=True)
            log.info("  ✓ call_notes_2")
        except Exception as exc:
            log.error("  ✗ call_notes_2: %s", exc)

        # ── 9. Email alert for high-priority categories ───────────────────────
        if category in ALERT_CATEGORIES:
            try:
                send_alert(
                    phone=phone_norm,
                    category=category,
                    institution=institution,
                    call_dt=call_dt,
                    summary=analysis.get("transcript_summary", ""),
                    other_info=extracted.get("other_information", ""),
                )
                log.info("  ✓ email sent — %s", category)
            except Exception as exc:
                log.error("  ✗ email: %s", exc)

    log.info("=== Pipeline complete ===")


if __name__ == "__main__":
    run()
