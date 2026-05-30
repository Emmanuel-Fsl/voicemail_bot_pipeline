#!/usr/bin/env python3
"""
ElevenLabs WhatsApp / Chat Bot Conversation Processing Pipeline

Flow per conversation:
  1. Fetch list (paginated) → deduplicate
  2. Fetch full detail from ElevenLabs
  3. Extract data_collection_results + eval criteria early (used across all later steps)
  4. Store to wa_bot_conversations  (Category = null; tooling audit + outcome signals)
  5. Upload attached images to GCS → customer_uploads_bot  (runs before duration gate)
  6. Skip if message_count < 4  (no real exchange — bot greeting + 1 reply max)
  7. AI extraction (GPT-4.1-mini) for Category / Willingness / discount_accepted only
  8. Institution from data_collection_results; fallback to BigQuery lookup
  9. Skip if no institution found
 10. Upsert call_notes_latest  (key: phone)
 11. Upsert call_notes_2       (key: doc_id)
 12. Send email alert if category is Complaint / Negotiation / PTP / Paid / Callback Request / Request to Speak to Agent / Payment Receipt Received
     → Payment Receipt Received also attaches images; all emails routed by phone mod 5, CC payments@fsldigital.com

Run manually or via cron at 1 AM:
    0 1 * * * /path/to/venv/bin/python /path/to/wa_pipeline.py >> /var/log/wa_pipeline.log 2>&1
"""

import base64
import json
import logging
import os
import random
import re
import smtplib
import string
import sys
import time
from datetime import datetime, timedelta, timezone
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional
from urllib.parse import quote as url_quote

import requests
from dateutil import parser as dateutil_parser
from dotenv import load_dotenv
from google.cloud import bigquery, firestore, storage as gcs_storage
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

AGENT_ID           = "agent_6901kk137kxnfyctkhj2jsxv3vra"
BQ_PROJECT         = "fssspark"
FS_PROJECT         = "fssspark"
MIN_MESSAGE_COUNT  = 4   # fewer than 4 messages = no real exchange (bot greeting + 1 reply max)

ALERT_CATEGORIES = {
    "Complaint",
    "Negotiation",
    "Request to Speak to Agent",
    "PTP",
    "Callback Request",
    "Paid",
    "Payment Receipt Received",
}

RECIPIENT_MAP = {
    0: "gbolade@fsldigital.com",
    1: "dami.adeyanju@fsldigital.com",
    2: "jedidah@fsldigital.com",
    3: "kesta@fsldigital.com",
    4: "emmanuella@fsldigital.com",
}


# ── Prompts / schema ───────────────────────────────────────────────────────────
# WhatsApp bot already extracts most fields via ElevenLabs data_collection_results.
# GPT is only needed for Category, Willingness, and discount_accepted.

SYSTEM_PROMPT = """\
You are a chat conversation analysis engine for a loan recovery company.

You will be given structured data from a completed WhatsApp bot conversation — including pre-extracted fields and the conversation summary. Your job is to determine three values:
1. Category — classify the conversation outcome
2. Willingness — score the customer's willingness to repay
3. discount_accepted — whether the customer agreed to a reduced/discounted settlement

CLASSIFICATION CATEGORIES:
- Payment Receipt Received — Customer has sent a photo or screenshot of a payment receipt, bank transfer confirmation, or any visual proof of payment. Only use this category when has_images is true.
- Paid — Customer verbally confirms they have already made a payment or settlement but has NOT sent an image receipt
- PTP — Customer makes a clear, specific promise to pay on a stated date or within a stated timeframe. A vague "I'll pay soon" is NOT a PTP.
- Negotiation — Customer is discussing repayment terms, requesting a reduced amount, an extension, or proposing an alternative arrangement
- Complaint — Customer is expressing dissatisfaction, disputing the loan, the charges, or the recovery process
- System Error — Conversation dropped unexpectedly, transcript is empty or garbled, or termination reason indicates a technical failure
- Non-Engagement — Customer was reached but refused to engage, gave no meaningful response, or repeatedly deflected without committing to anything
- Callback Request — Customer explicitly requested to be called back or contacted at a later time
- Unresolved — Conversation ended without a clear outcome — no payment, no PTP, no complaint, no refusal

CLASSIFICATION RULES:
- If has_images is true AND customer is sharing payment proof → Payment Receipt Received (takes priority over Paid)
- If has_images is false → never use Payment Receipt Received
- If transcript is empty or under 2 meaningful exchanges → System Error
- If customer was reached but said nothing useful → Non-Engagement
- A specific date or amount mentioned with intent to pay → PTP, not Negotiation
- Customer saying "I already paid" without an image → Paid, not Complaint, even if they sound frustrated
- If two categories seem to apply, pick the dominant intent of the conversation
- Use the pre-extracted fields (Promise_to_Pay, ptp_date, denied_debt, callback_requested) as strong signals

---

Willingness SCALE — score the customer's willingness to repay based on their behaviour:
  - "0 Unwilling (Not Speaking)" — Customer connected but sent no messages at all, completely silent
  - "1 Unwilling" — Customer messaged but explicitly refused to pay or showed no willingness
  - "2 Medium" — Customer was reluctant but did not outright refuse, OR agreed without a confirmed PTP (a confirmed PTP requires both a ptp_date AND an Amount_Promised)
  - "3 Willing" — Customer willingly agreed to pay, most often with a confirmed PTP (date + amount)
  - "Unknown" — Cannot be determined (e.g. conversation dropped, debt denied, transcript unclear)

---

discount_accepted:
  - true — Customer explicitly agreed to a discounted/reduced settlement amount
  - false — Customer did not accept a discount or no discount was offered
  - null — Cannot be determined

---

STRICT OUTPUT RULES:
1. Return a single valid JSON object — no markdown fences, no extra text
2. discount_accepted must be true, false, or null — never a string
3. Return exactly: {"Category": "...", "Willingness": "...", "discount_accepted": true/false/null}\
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


GCS_BUCKET = "fssspark.firebasestorage.app"


def _gcs():
    return gcs_storage.Client(project=BQ_PROJECT, credentials=_creds())


# ── ElevenLabs ─────────────────────────────────────────────────────────────────

_EL_HEADERS = {"xi-api-key": ELEVENLABS_API_KEY}


def _time_window(hours: int = 12) -> tuple[int, int]:
    now    = datetime.now(timezone.utc)
    after  = int((now - timedelta(hours=hours)).timestamp())
    before = int(now.timestamp())
    return before, after


def fetch_all_conversations(hours: int = 12) -> list[dict]:
    before, after = _time_window(hours)
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


def extract_category_and_willingness(detail: dict, dc: dict, has_images: bool = False) -> dict:
    """Call GPT for Category, Willingness, and discount_accepted only.
    All other fields are already extracted by ElevenLabs data_collection_results.
    """
    analysis  = detail.get("analysis") or {}
    metadata  = detail.get("metadata") or {}
    dyn_vars  = ((detail.get("conversation_initiation_client_data") or {})
                 .get("dynamic_variables") or {})

    def dcv(key: str) -> str:
        return ((dc.get(key) or {}).get("value") or "") or ""

    user_content = (
        f"has_images: {str(has_images).lower()}\n"
        f"Transcript Summary: {analysis.get('transcript_summary', '')}\n"
        f"Call Summary Title: {analysis.get('call_summary_title', '')}\n"
        f"Termination Reason: {metadata.get('termination_reason', '')}\n"
        f"Message Count: {metadata.get('call_duration_secs', 0)}\n"
        f"Promise_to_Pay: {dcv('Promise_to_Pay')}\n"
        f"ptp_date: {dcv('ptp_date')}\n"
        f"Amount_Promised: {dcv('Amount_Promised')}\n"
        f"denied_debt: {dcv('denied_debt')}\n"
        f"callback_requested: {dcv('callback_requested')}\n"
        f"Right_Party_Contact: {dcv('Right_Party_Contact')}\n"
        f"Conversation History: {dyn_vars.get('system__conversation_history', '')}"
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


# ── Gmail ──────────────────────────────────────────────────────────────────────

def route_recipient(phone: str) -> str:
    digits = ''.join(c for c in phone if c.isdigit())
    last_two = int(digits[-2:]) if len(digits) >= 2 else 0
    return RECIPIENT_MAP[last_two % 5]


def send_alert(
    phone: str,
    category: str,
    institution: str,
    call_dt: str,
    summary: str,
    other_info: str,
    recipient: str,
    attachments: Optional[list] = None,
):
    msg = MIMEMultipart("mixed")
    msg["Subject"] = f"WhatsApp Bot Follow-up Required - {category} | {phone}"
    msg["From"]    = f"whatsappbot <{GMAIL_SENDER}>"
    msg["To"]      = recipient
    msg["Cc"]      = ALERT_RECIPIENT

    html = (
        "<h2>WhatsApp Bot Follow-up Required</h2>"
        f"<p><strong>Phone:</strong> {phone}</p>"
        f"<p><strong>Category:</strong> {category}</p>"
        f"<p><strong>Institution:</strong> {institution}</p>"
        f"<p><strong>Date of Conversation:</strong> {call_dt}</p>"
        "<p><strong>Summary:</strong></p>"
        f"<p>{summary}</p>"
        "<p><strong>Other Information:</strong></p>"
        f"<p>{other_info}</p>"
    )
    msg.attach(MIMEText(html, "html"))

    for att in (attachments or []):
        img = MIMEImage(att["data"], _subtype=att["ctype"].split("/")[-1])
        img.add_header("Content-Disposition", "attachment", filename=att["fname"])
        msg.attach(img)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(GMAIL_SENDER, GMAIL_APP_PASSWORD)
        smtp.sendmail(GMAIL_SENDER, [recipient, ALERT_RECIPIENT], msg.as_string())


# ── Helpers ────────────────────────────────────────────────────────────────────

_LANG_NAMES = {
    "en": "English", "fr": "French", "de": "German", "es": "Spanish",
    "pt": "Portuguese", "it": "Italian", "nl": "Dutch", "pl": "Polish",
    "hi": "Hindi", "ar": "Arabic", "yo": "Yoruba", "ig": "Igbo",
    "ha": "Hausa", "sw": "Swahili", "pcm": "Nigerian Pidgin",
}


def expand_language(code: str) -> str:
    return _LANG_NAMES.get((code or "").lower(), code)


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
    if not system_time:
        return ""
    return parse_system_time(system_time).strftime("%Y-%m-%dT%H:%M:%S")


def epoch_ms(system_time: str) -> int:
    try:
        return int(parse_system_time(system_time).timestamp() * 1000)
    except Exception:
        return int(time.time() * 1000)


def normalise_ptp_date(raw: str) -> str:
    """Convert DD/MM/YYYY or YYYY-MM-DD or any parseable date to YYYY-MM-DD."""
    if not raw:
        return ""
    # Handle DD/MM/YYYY explicitly (ElevenLabs returns this format for WA)
    match = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{4})", raw.strip())
    if match:
        d, m, y = match.groups()
        return f"{y}-{int(m):02d}-{int(d):02d}"
    # Already YYYY-MM-DD
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw.strip()):
        return raw.strip()
    # Fallback: try dateutil
    try:
        return dateutil_parser.parse(raw).strftime("%Y-%m-%d")
    except Exception:
        return raw


# ── Transcript helpers ─────────────────────────────────────────────────────────

def clean_transcript(transcript: list) -> list:
    """Return role + message text only, dropping tool calls, metrics, and LLM usage.
    Entries with no message text (pure tool-call turns) are omitted.
    Adds file_input: True when the user attached a file on that turn.
    """
    result = []
    for t in transcript:
        msg = t.get("message")
        if not msg:
            continue
        entry = {
            "role":    t["role"],
            "message": msg,
            "t":       t.get("time_in_call_secs", 0),
        }
        if t.get("file_input"):
            entry["file_input"] = True
        result.append(entry)
    return result


# ── File upload helpers ────────────────────────────────────────────────────────

def _upload_doc_id(now: datetime) -> str:
    """Matches JS pattern: {randomId}_{YYYYMMDD}_{HHmmss}"""
    rand = "".join(random.choices(string.ascii_lowercase + string.digits, k=9))
    return f"{rand}_{now.strftime('%Y%m%d')}_{now.strftime('%H%M%S')}"


def _bot_filename(name: str) -> str:
    """Append _bot before the file extension: proof.jpg → proof_bot.jpg"""
    if "." in name:
        base, ext = name.rsplit(".", 1)
        return f"{base}_bot.{ext}"
    return f"{name}_bot"


def _file_bytes(file_obj: dict) -> tuple[bytes, str, str]:
    """Extract (raw_bytes, content_type, filename) from a file_input entry.

    ElevenLabs stores files as signed GCS URLs that expire 15 minutes after the
    conversation detail is fetched.  The pipeline must download the image in the
    same processing pass — do not defer.

    Confirmed field names from live API (2026-05):
      file_url          — signed https://storage.googleapis.com/xi-backend/... URL
      original_filename — e.g. "IMG-20260416-WA0008.jpg"
      mime_type         — e.g. "image/jpeg"
      file_id           — ElevenLabs UUID
    """
    name  = file_obj.get("original_filename") or file_obj.get("file_id") or "attachment"
    ctype = file_obj.get("mime_type") or "application/octet-stream"
    url   = file_obj.get("file_url") or ""

    if not url:
        raise ValueError(f"file_input has no 'file_url': {list(file_obj.keys())}")

    resp = requests.get(url, timeout=30)
    if resp.status_code == 403:
        raise ValueError(
            f"Signed URL expired for {name} (file_id={file_obj.get('file_id')}). "
            "This happens if the pipeline runs long after fetch_conversation_detail()."
        )
    resp.raise_for_status()
    return resp.content, ctype, name


def upload_conversation_files(
    conversation_id: str,
    detail: dict,
    phone_norm: str,
    institution: str,
    other_info: str,
    db: firestore.Client,
    bucket,
) -> list[dict]:
    """Upload file attachments to GCS and write metadata to customer_uploads_bot.
    Returns a list of uploaded file dicts: {fname, data, ctype} — kept in memory
    so the email step can attach them without re-fetching.
    Runs before the message gate so images are captured from any conversation.
    """
    metadata = detail.get("metadata") or {}
    feat = (metadata.get("features_usage") or {}).get("file_input") or {}
    if not feat.get("used"):
        return []

    transcript  = detail.get("transcript") or []
    file_entries = [
        t["file_input"] for t in transcript
        if t.get("role") == "user" and t.get("file_input")
    ]
    if not file_entries:
        log.info("  file_input.used=True but no entries in transcript — skipping")
        return []

    uploaded = []
    for entry in file_entries:
        files = entry if isinstance(entry, list) else [entry]

        for file_obj in files:
            now    = datetime.now(timezone.utc)
            doc_id = _upload_doc_id(now)

            try:
                data, ctype, fname = _file_bytes(file_obj)
            except Exception as exc:
                log.warning("  ✗ could not read file_input: %s | keys=%s",
                            exc, list(file_obj.keys()) if isinstance(file_obj, dict) else "?")
                continue

            bot_fname = _bot_filename(fname)
            gcs_path  = f"customer_uploads/{doc_id}_{bot_fname}"

            try:
                blob = bucket.blob(gcs_path)
                blob.upload_from_string(data, content_type=ctype)
                download_url = (
                    f"https://firebasestorage.googleapis.com/v0/b/"
                    f"{GCS_BUCKET}/o/{url_quote(gcs_path, safe='')}?alt=media"
                )
            except Exception as exc:
                log.error("  ✗ GCS upload '%s': %s", fname, exc)
                continue

            try:
                db.collection("customer_uploads_bot").document(doc_id).set({
                    "created_at":      now.isoformat(),
                    "download_link":   download_url,
                    "phone":           phone_norm,
                    "message":         other_info,
                    "institution":     institution,
                    "client_id":       "",
                    "type":            "payment proof",
                    "file_name":       fname,
                    "file_id":         file_obj.get("file_id", ""),
                    "conversation_id": conversation_id,
                })
                log.info("  ✓ uploaded '%s' → doc %s", fname, doc_id)
                uploaded.append({"fname": fname, "data": data, "ctype": ctype})
            except Exception as exc:
                log.error("  ✗ customer_uploads_bot: %s", exc)

    return uploaded


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run():
    import argparse
    parser = argparse.ArgumentParser(description="WA Bot Conversation Pipeline")
    parser.add_argument(
        "--days", type=float, default=0,
        help="Look-back window in days (overrides default 12-hour window). E.g. --days 21"
    )
    args = parser.parse_args()

    hours = int(args.days * 24) if args.days else 12

    log.info("=== WA Pipeline started (window: %d hours) ===", hours)
    bq     = _bq()
    db     = _fs()
    bucket = _gcs().bucket(GCS_BUCKET)

    conversations = fetch_all_conversations(hours=hours)
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

        metadata  = detail.get("metadata") or {}
        analysis  = detail.get("analysis") or {}
        dyn_vars  = ((detail.get("conversation_initiation_client_data") or {})
                     .get("dynamic_variables") or {})

        # ── Phone: WhatsApp passes number as a dynamic variable, not phone_call ──
        raw_phone  = (
            dyn_vars.get("phone_number") or
            dyn_vars.get("p") or
            str(dyn_vars.get("system__caller_id") or "") or
            ""
        )
        phone_norm = normalise_phone(raw_phone) if raw_phone else conv.get("user_id", "")
        if not phone_norm:
            log.warning("  no phone number found for %s — call_notes writes will be skipped", conversation_id)

        duration        = metadata.get("call_duration_secs", 0) or 0
        message_count   = conv.get("message_count", 0) or 0
        system_time     = dyn_vars.get("system__time_utc") or dyn_vars.get("system__time", "")
        call_dt         = parse_call_dt(system_time)
        if not call_dt:
            start_unix = metadata.get("start_time_unix_secs")
            if start_unix:
                call_dt = datetime.fromtimestamp(start_unix, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

        # ── 2. Extract data_collection_results early (used in both wa_doc and call_notes) ──
        dc = analysis.get("data_collection_results") or {}

        def dcv(key: str) -> str:
            v = (dc.get(key) or {}).get("value")
            return v if v is not None else ""

        eval_crit   = analysis.get("evaluation_criteria_results") or {}
        ptp_result  = (eval_crit.get("ptp") or {}).get("result", "")
        tool_names  = conv.get("tool_names") or []

        # ── 3. Store to wa_bot_conversations ────────────────────────────────────
        # message_count == 0 covers all cases: widget opened but nothing typed,
        # including 10-min zombie timeouts and zero-duration sessions.
        if message_count == 0:
            log.info("  skip wa_bot_conversations — no messages")
        else:
            wa_doc = {
                # ── Identity ────────────────────────────────────────────────
                "phone_number":                  phone_norm,
                "conversation_id":               conversation_id,
                "agent_id":                      conv.get("agent_id", ""),
                "agent_name":                    conv.get("agent_name", ""),
                # ── Timing ──────────────────────────────────────────────────
                "datetime":                      call_dt,
                "timestamp":                     datetime.now(timezone.utc).isoformat(),
                "timezone":                      metadata.get("timezone", ""),
                # ── Conversation stats ───────────────────────────────────────
                "duration":                      str(duration),
                "message_count":                 message_count,
                "language":                      metadata.get("main_language", ""),
                # ── Outcome signals ──────────────────────────────────────────
                "status":                        analysis.get("call_successful", ""),
                "category":                      None,
                "termination_reason":            metadata.get("termination_reason", ""),
                "error":                         metadata.get("error"),
                "ptp_criteria_result":           ptp_result,
                # ── Tooling audit ────────────────────────────────────────────
                "tool_names":                    tool_names,
                "update_call_notes_called":      "update_call_notes" in tool_names,
                # ── Summaries ────────────────────────────────────────────────
                "call_summary_title":            analysis.get("call_summary_title", ""),
                "transcript_summary":            analysis.get("transcript_summary", ""),
                # ── Cost / source ────────────────────────────────────────────
                "cost_credits":                  metadata.get("cost", 0),
                "conversation_initiation_source": metadata.get("conversation_initiation_source", ""),
                # ── Full chat transcript ─────────────────────────────────────
                "transcript":                    clean_transcript(detail.get("transcript") or []),
            }
            try:
                db.collection("wa_bot_conversations").document(conversation_id).set(wa_doc, merge=True)
                log.info("  ✓ wa_bot_conversations")
            except Exception as exc:
                log.error("  ✗ wa_bot_conversations: %s", exc)

        # ── 4. File uploads (before duration gate — capture images from any conversation) ──
        uploaded_files = []
        try:
            uploaded_files = upload_conversation_files(
                conversation_id, detail, phone_norm,
                dcv("institution"), dcv("other_information"),
                db, bucket,
            )
            if uploaded_files:
                log.info("  ✓ %d file(s) uploaded to customer_uploads_bot", len(uploaded_files))
        except Exception as exc:
            log.error("  ✗ file uploads: %s", exc)

        # ── 5. Message count gate ────────────────────────────────────────────
        # Images bypass the gate — a payment receipt is actionable regardless of message count.
        if message_count < MIN_MESSAGE_COUNT and not uploaded_files:
            log.info("  skip — %d messages < %d minimum", message_count, MIN_MESSAGE_COUNT)
            continue

        # ── 6. AI extraction — Category, Willingness, discount_accepted ───────
        try:
            extracted = extract_category_and_willingness(detail, dc, has_images=bool(uploaded_files))
            log.info("  ✓ AI — Category: %s", extracted.get("Category"))
        except Exception as exc:
            log.error("  ✗ AI extraction: %s", exc)
            continue

        category = extracted.get("Category", "")

        # ── 7. Institution: from DC results, fallback to BigQuery ──────────────
        institution = dcv("institution") or None
        if not institution:
            log.info("  institution not in DC results — trying BigQuery")
            institution = query_institution(phone_norm, bq)
        if not institution:
            if uploaded_files:
                # Never skip a payment receipt just because institution is unknown —
                # the team still needs to see the image.
                log.info("  institution unknown but images present — continuing as 'Unknown'")
                institution = "Unknown"
            else:
                log.info("  skip — no institution found for %s", phone_norm)
                continue

        # ── 8. Build shared document ──────────────────────────────────────────
        doc_id  = f"{phone_norm}_{epoch_ms(system_time)}"
        call_ts = to_iso(parse_system_time(system_time))
        now_dt  = to_iso(datetime.now(timezone(timedelta(hours=1))))

        call_notes = {
            "Amount_Promised":          dcv("Amount_Promised"),
            "Payment_Method":           dcv("Payment_Method"),
            "Promise_to_Pay":           dcv("Promise_to_Pay"),
            "Right_Party_Contact":      dcv("Right_Party_Contact"),
            "Willingness":              extracted.get("Willingness", ""),
            "agent":                    "Whatsapp Bot",
            "alternative_phone":        dcv("alternative_phone"),
            "callback_requested":       dcv("callback_requested"),
            "contact_method":           "Whatsapp Bot",
            "denied_debt":              dcv("denied_debt"),
            "employment_status":        dcv("employment_status"),
            "federal_deduction":        dcv("federal_deduction"),
            "institution":              institution,
            "name_of_officer":          dcv("name_of_officer"),
            "offset_loan_with_savings": dcv("offset_loan_with_savings"),
            "other_information":        dcv("other_information"),
            "phone":                    phone_norm,
            "ptp_date":                 normalise_ptp_date(dcv("ptp_date")),
            "reason_for_default":       dcv("reason_for_default"),
            "timestamp":                call_ts,
            "updated_at":               now_dt,
            "discount_accepted":        extracted.get("discount_accepted"),
            "language":                 expand_language(metadata.get("main_language", "")),
        }

        # ── 9. Firestore — call_notes_latest (doc ID = phone) ───────────────
        if phone_norm:
            try:
                db.collection("call_notes_latest").document(phone_norm).set(call_notes, merge=True)
                log.info("  ✓ call_notes_latest")
            except Exception as exc:
                log.error("  ✗ call_notes_latest: %s", exc)
        else:
            log.warning("  skip call_notes_latest — no phone number")

        # ── 10. Firestore — call_notes_2 (doc ID = phone_timestamp) ───────────
        if phone_norm:
            try:
                db.collection("call_notes_2").document(doc_id).set(call_notes, merge=True)
                log.info("  ✓ call_notes_2")
            except Exception as exc:
                log.error("  ✗ call_notes_2: %s", exc)
        else:
            log.warning("  skip call_notes_2 — no phone number")

        # ── 11. Email alert for high-priority categories ──────────────────────
        if category in ALERT_CATEGORIES:
            try:
                is_receipt = category == "Payment Receipt Received"
                send_alert(
                    phone=phone_norm,
                    category=category,
                    institution=institution,
                    call_dt=call_dt,
                    summary=analysis.get("transcript_summary", ""),
                    other_info=dcv("other_information"),
                    recipient=route_recipient(phone_norm),
                    attachments=uploaded_files if is_receipt else None,
                )
                log.info("  ✓ email sent — %s", category)
            except Exception as exc:
                log.error("  ✗ email: %s", exc)

    log.info("=== WA Pipeline complete ===")


if __name__ == "__main__":
    run()
