#!/usr/bin/env python3
"""
Voicemail Pipeline

Flow per email:
  1. Fetch emails from Gmail SPAM folder (sender: JustGoCloud) — last 12 hours
  2. Extract phone from subject, duration + voicemail type from body
  3. Skip if no attachment or duration <= 5s
  4. Transcribe audio with OpenAI Whisper
  5. Meaningfulness check — skip if not loan-recovery relevant
  6. Extract call data with GPT-4.1-mini
  7. BigQuery institution lookup — skip if not found
  8. Upload audio to Firebase Storage → get download URL
  9. Store to voicemail_upload  (Firestore create, doc_id = phone_ms)
 10. Upsert call_notes_latest  (key: phone)
 11. Upsert call_notes_2       (key: doc_id)
 12. Send email alert if category matches

Cron (1 AM and 1 PM daily):
    0 1,13 * * * /Users/emmanuelokorie/Documents/Workflows/.venv/bin/python /Users/emmanuelokorie/Documents/Workflows/voicemail_pipeline.py >> /tmp/voicemail_pipeline.log 2>&1
"""

import base64
import email
import email.policy
import imaplib
import io
import json
import logging
import os
import sys
import re
import smtplib
import time
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional
from urllib.parse import quote

import requests
from dateutil import parser as dateutil_parser
from dotenv import load_dotenv
from google.auth.transport.requests import Request as GoogleAuthRequest
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

OPENAI_API_KEY        = os.environ["OPENAI_API_KEY"]
GOOGLE_SA_FILE        = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE", "service_account.json")
GMAIL_READER          = os.environ["GMAIL_READER"]            # account to READ voicemails from
GMAIL_READER_PASSWORD = os.environ["GMAIL_READER_PASSWORD"]   # app password for reader account
GMAIL_SENDER          = os.environ["GMAIL_SENDER"]            # account to SEND alerts from
GMAIL_APP_PASSWORD    = os.environ["GMAIL_APP_PASSWORD"]
ALERT_RECIPIENT       = os.getenv("ALERT_RECIPIENT", "payments@fsldigital.com")

VOICEMAIL_SENDER = "JustGoCloud@fsldigital.justgocloud.com"
FIREBASE_BUCKET  = "fssspark.firebasestorage.app"
BQ_PROJECT       = "fssspark"
FS_PROJECT       = "fssspark"

ALERT_CATEGORIES = {
    "Complaint",
    "Negotiation",
    "Request to Speak to Agent",
    "PTP (Promise to Pay)",
    "Callback Request",
    "Paid",
}

SYSTEM_PROMPT = """\
You are an AI assistant for a loan recovery company analyzing transcribed voicemail calls.

Your task is to extract structured information from the call transcript and classify it into one of these categories:
- Complaint
- Negotiation
- Request to Speak to Agent
- Dropped Call
- PTP (Promise to Pay)
- Callback Request
- Paid

Extract the following fields:
- Amount_Promised: Exact amount customer committed to paying (leave blank if none)
- Payment_Method: Only if customer claims prior payment not yet reflected. Options: Branch payment, Zenith bank, Credit officer, Paystack, Union Bank, Leader, Providus Bank, Sales Rep, Unknown
- Promise_to_Pay: Yes/No/Unknown
- Right_Party_Contact: Yes/No/Unknown
- Willingness: Always set to "2 Medium"
- agent: Always set to "Voicemail"
- alternative_phone: Alternate number provided (blank if none)
- callback_requested: Yes/No/Unknown
- contact_method: Always set to "Voice Call"
- denied_debt: Yes/No/Unknown
- employment_status: Business Owner/Salary Earner/Both/Unknown
- federal_deduction: Yes/No/Unknown
- name_of_officer: Name of officer customer claims to have paid (blank if none)
- offset_loan_with_savings: Yes/No/Unknown
- other_information: Concise summary of conversation
- ptp_date: Specific payment date (convert relative dates to exact dates, blank if none)
- reason_for_default: Brief summary of default reason (blank if not provided)
- discount_accepted: Yes if customer agreed to a settlement/discount offer, No if declined, Unknown if not mentioned
- language: Language the customer spoke. One of: English, Yoruba, Igbo, Hausa, Pidgin. Default to English if unclear.
- Category: One of the 7 categories listed above

Return a single valid JSON object — no markdown fences, no extra text.\
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


# ── OpenAI ─────────────────────────────────────────────────────────────────────

_oai_client: Optional[OpenAI] = None


def _oai() -> OpenAI:
    global _oai_client
    if _oai_client is None:
        _oai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _oai_client


def transcribe_audio(audio_bytes: bytes, filename: str) -> str:
    buf = io.BytesIO(audio_bytes)
    buf.name = filename
    result = _oai().audio.transcriptions.create(model="whisper-1", file=buf)
    return result.text


def is_meaningful(text: str) -> bool:
    resp = _oai().chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{
            "role": "user",
            "content": (
                "Is this transcription meaningful and contains actual conversation "
                "in the context of loan recovery (Promise to Pay, Complaints, Negotiation, "
                "Payment Made, Request to speak with Someone etc)? "
                f"Answer only with YES or NO: {text}"
            ),
        }],
    )
    return resp.choices[0].message.content.strip().upper().startswith("YES")


def extract_call_data(text: str) -> dict:
    resp = _oai().chat.completions.create(
        model="gpt-4.1-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": text},
        ],
    )
    return json.loads(resp.choices[0].message.content)


# ── Gmail IMAP ─────────────────────────────────────────────────────────────────

def fetch_voicemail_emails() -> list[dict]:
    cutoff   = datetime.now(timezone.utc) - timedelta(hours=12)
    since    = cutoff.strftime("%d-%b-%Y")

    mail = imaplib.IMAP4_SSL("imap.gmail.com")
    mail.login(GMAIL_READER, GMAIL_READER_PASSWORD)
    mail.select("[Gmail]/Spam")

    _, ids = mail.search(None, f'(FROM "{VOICEMAIL_SENDER}" SINCE {since})')
    message_ids = ids[0].split()
    log.info("Found %d candidate emails in SPAM", len(message_ids))

    results = []
    for msg_id in message_ids:
        _, data = mail.fetch(msg_id, "(RFC822)")
        raw_msg = data[0][1]
        msg = email.message_from_bytes(raw_msg, policy=email.policy.default)

        # Fine-grained 12-hour filter on the Date header
        try:
            msg_dt = dateutil_parser.parse(msg.get("Date", ""))
            if msg_dt.tzinfo is None:
                msg_dt = msg_dt.replace(tzinfo=timezone.utc)
            if msg_dt < cutoff:
                continue
        except Exception:
            pass  # if we can't parse the date, include it anyway

        subject          = msg.get("Subject", "")
        body             = ""
        attachment_bytes = None
        attachment_name  = "voicemail.mp3"
        attachment_mime  = "audio/mpeg"

        for part in msg.walk():
            ct = part.get_content_type()
            cd = part.get("Content-Disposition", "")
            if ct == "text/plain" and "attachment" not in cd:
                try:
                    body = part.get_content() or ""
                except Exception:
                    body = part.get_payload(decode=True).decode("utf-8", errors="ignore")
            elif "attachment" in cd or ct.startswith("audio/"):
                attachment_bytes = part.get_payload(decode=True)
                attachment_name  = part.get_filename() or attachment_name
                attachment_mime  = ct

        if not attachment_bytes:
            continue  # no audio, skip

        results.append({
            "subject":          subject,
            "body":             body,
            "date":             msg.get("Date", ""),
            "attachment_bytes": attachment_bytes,
            "attachment_name":  attachment_name,
            "attachment_mime":  attachment_mime,
        })

    mail.logout()
    log.info("Fetched %d voicemail emails with attachments", len(results))
    return results


# ── Email field parsers ────────────────────────────────────────────────────────

def extract_phone(subject: str) -> str:
    match = re.search(r"\d{10,}", subject)
    return normalise_phone(match.group()) if match else ""


def extract_duration(body: str) -> Optional[int]:
    match = re.search(r"(\d+):(\d+)(?::(\d+))?\s+long\s+message", body, re.IGNORECASE)
    if not match:
        return None
    if match.group(3):
        return int(match.group(1)) * 3600 + int(match.group(2)) * 60 + int(match.group(3))
    return int(match.group(1)) * 60 + int(match.group(2))


def extract_voicemail_type(body: str) -> str:
    match = re.search(r"Dear\s+(.+?)\s+voicemail:", body, re.IGNORECASE)
    raw = match.group(1).strip() if match else "Unknown"
    if raw == "Press 3":
        return "Complaint"
    if raw == "Press 4":
        return "Negotiation"
    return raw


# ── Firebase Storage ───────────────────────────────────────────────────────────

def upload_to_firebase(audio_bytes: bytes, filename: str, mime_type: str) -> str:
    creds = _creds()
    creds.refresh(GoogleAuthRequest())

    encoded = quote(filename, safe="")
    upload_url = f"https://firebasestorage.googleapis.com/v0/b/{FIREBASE_BUCKET}/o/{encoded}?alt=media"

    resp = requests.post(
        upload_url,
        headers={
            "Authorization": f"Bearer {creds.token}",
            "Content-Type":  mime_type,
        },
        data=audio_bytes,
        timeout=60,
    )
    resp.raise_for_status()
    token = resp.json().get("downloadTokens", "")
    return f"https://firebasestorage.googleapis.com/v0/b/{FIREBASE_BUCKET}/o/{encoded}?alt=media&token={token}"


# ── BigQuery ───────────────────────────────────────────────────────────────────

def query_institution(phone: str, bq_client: bigquery.Client) -> Optional[str]:
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
        log.warning("BigQuery error for %s: %s", phone, exc)
    return None


# ── Firestore ──────────────────────────────────────────────────────────────────

def fs_upsert(db: firestore.Client, collection: str, key_field: str, data: dict):
    key_value = data.get(key_field)
    if key_value is None:
        log.warning("Skipping upsert into '%s' — key '%s' missing", collection, key_field)
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


def fs_create(db: firestore.Client, collection: str, doc_id: str, data: dict):
    db.collection(collection).document(doc_id).set(data)


# ── Gmail alert ────────────────────────────────────────────────────────────────

def send_alert(phone: str, category: str, institution: str, voicemail_type: str, transcript: str, other_info: str):
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"Voicemail Follow-up Required - {category} | {phone}"
    msg["From"]    = f"voicemail <{GMAIL_SENDER}>"
    msg["To"]      = ALERT_RECIPIENT

    html = (
        "<h2>Voicemail Follow-up Required</h2>"
        f"<p><strong>Phone:</strong> {phone}</p>"
        f"<p><strong>Category:</strong> {category}</p>"
        f"<p><strong>Institution:</strong> {institution}</p>"
        f"<p><strong>Voicemail Type:</strong> {voicemail_type}</p>"
        "<p><strong>Transcription:</strong></p>"
        f"<p>{transcript}</p>"
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


def to_iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run():
    log.info("=== Voicemail pipeline started ===")
    bq = _bq()
    db = _fs()

    emails = fetch_voicemail_emails()

    for item in emails:
        subject      = item["subject"]
        body         = item["body"]
        date_str     = item["date"]
        audio_bytes  = item["attachment_bytes"]
        audio_name   = item["attachment_name"]
        audio_mime   = item["attachment_mime"]

        # ── 1. Parse email metadata ───────────────────────────────────────────
        phone          = extract_phone(subject)
        duration       = extract_duration(body)
        voicemail_type = extract_voicemail_type(body)

        log.info("▶ phone=%s  type=%s  duration=%s", phone or "?", voicemail_type, duration)

        if not phone:
            log.info("  skip — no phone found in subject")
            continue

        # ── 2. Duration gate (> 5 seconds) ────────────────────────────────────
        if duration is None or duration <= 5:
            log.info("  skip — duration %s <= 5s", duration)
            continue

        # ── 3. Transcribe audio ───────────────────────────────────────────────
        try:
            transcript = transcribe_audio(audio_bytes, audio_name)
            log.info("  ✓ transcribed (%d chars)", len(transcript))
        except Exception as exc:
            log.error("  ✗ transcription: %s", exc)
            continue

        # ── 4. Meaningfulness check ───────────────────────────────────────────
        try:
            if not is_meaningful(transcript):
                log.info("  skip — not meaningful")
                continue
            log.info("  ✓ meaningful")
        except Exception as exc:
            log.error("  ✗ meaningfulness check: %s", exc)
            continue

        # ── 5. AI extraction ──────────────────────────────────────────────────
        try:
            extracted = extract_call_data(transcript)
            log.info("  ✓ AI — Category: %s", extracted.get("Category"))
        except Exception as exc:
            log.error("  ✗ AI extraction: %s", exc)
            continue

        category = extracted.get("Category", "")

        # ── 6. BigQuery institution lookup ────────────────────────────────────
        institution = query_institution(phone, bq)
        if not institution:
            log.info("  skip — no institution for %s", phone)
            continue

        # ── 7. Upload audio to Firebase Storage ───────────────────────────────
        ext      = audio_name.rsplit(".", 1)[-1] if "." in audio_name else "mp3"
        now_ms   = int(time.time() * 1000)
        filename = f"voicemail_uploads/voicemail_{phone}_{now_ms}.{ext}"
        doc_id   = f"{phone}_{now_ms}"
        now_iso  = to_iso(datetime.now(timezone.utc))

        try:
            email_dt = dateutil_parser.parse(date_str)
            if email_dt.tzinfo is None:
                email_dt = email_dt.replace(tzinfo=timezone.utc)
            email_ts = to_iso(email_dt)
        except Exception:
            email_ts = now_iso

        try:
            download_url = upload_to_firebase(audio_bytes, filename, audio_mime)
            log.info("  ✓ Firebase Storage")
        except Exception as exc:
            log.error("  ✗ Firebase Storage: %s", exc)
            download_url = ""

        # ── 8. Store to voicemail_upload ──────────────────────────────────────
        try:
            fs_create(db, "voicemail_upload", doc_id, {
                "transcribed_text":   transcript,
                "voicemail_type":     voicemail_type,
                "phone":              phone,
                "download_link":      download_url,
                "voicemail_duration": duration,
                "timestamp":          email_ts,
                "updated_at":         now_iso,
            })
            log.info("  ✓ voicemail_upload")
        except Exception as exc:
            log.error("  ✗ voicemail_upload: %s", exc)

        # ── 9. Upsert call_notes_latest + call_notes_2 ────────────────────────
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
            "phone":                    phone,
            "ptp_date":                 extracted.get("ptp_date", ""),
            "reason_for_default":       extracted.get("reason_for_default", ""),
            "timestamp":                email_ts,
            "updated_at":               now_iso,
            "discount_accepted":        extracted.get("discount_accepted", ""),
            "language":                 extracted.get("language", ""),
            "doc_id":                   doc_id,
        }

        try:
            fs_upsert(db, "call_notes_latest", "phone", call_notes)
            log.info("  ✓ call_notes_latest")
        except Exception as exc:
            log.error("  ✗ call_notes_latest: %s", exc)

        try:
            fs_upsert(db, "call_notes_2", "doc_id", call_notes)
            log.info("  ✓ call_notes_2")
        except Exception as exc:
            log.error("  ✗ call_notes_2: %s", exc)

        # ── 10. Email alert ───────────────────────────────────────────────────
        if category in ALERT_CATEGORIES:
            try:
                send_alert(
                    phone=phone,
                    category=category,
                    institution=institution,
                    voicemail_type=voicemail_type,
                    transcript=transcript,
                    other_info=extracted.get("other_information", ""),
                )
                log.info("  ✓ email sent — %s", category)
            except Exception as exc:
                log.error("  ✗ email: %s", exc)

    log.info("=== Voicemail pipeline complete ===")


if __name__ == "__main__":
    run()
