#!/usr/bin/env python3
"""
classify_receipts.py
Replicates the n8n receipt classification workflow.
OCR.space is replaced with GPT-4o Vision — one API call reads and classifies the image.

Pipeline:
  BigQuery → filter → GPT Vision classify → filter Claims → GCS upload
  → Cloud Run trigger → fetch Excel → send email
"""

import base64
import json
import logging
import os
import smtplib
import time
from datetime import date, datetime, timezone
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from urllib.parse import unquote, urlparse

import fitz  # pymupdf
import requests
from dotenv import load_dotenv
from google.cloud import bigquery, storage
from google.oauth2 import service_account
import google.auth.transport.requests
from openai import OpenAI

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────
OPENAI_API_KEY      = os.environ["OPENAI_API_KEY"]
GMAIL_SENDER        = os.environ["GMAIL_SENDER"]
GMAIL_APP_PASSWORD  = os.environ["GMAIL_APP_PASSWORD"]
GOOGLE_SA_FILE      = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE", "service_account.json")

BQ_PROJECT          = "fssspark"
INSTITUTION         = "BAOBAB"
BQ_TABLE            = f"`{BQ_PROJECT}.original_cohorts.customer_uploads_with_details`"

GCS_UPLOAD_BUCKET   = "customer_upload"
GCS_UPLOAD_OBJECT   = "classified_output"
GCS_EXPORT_BUCKET   = "fss_firestore_export_storage"

CLOUD_RUN_URL       = "https://classify-runner-270783804340.us-central1.run.app"
CLOUD_RUN_SA_EMAIL  = "emmanuel@fssspark.iam.gserviceaccount.com"

EMAIL_TO            = "emmanuel@fsldigital.com"
EMAIL_CC            = ["tech@fsldigital.com", "joyce@fsldigital.com"]

GPT_MODEL           = "gpt-4o"

CLASSIFY_PROMPT = """\
You are a financial data classifier.

You will be given an image of a payment receipt or document.

Customer metadata:
- client_id: {client_id}
- full_name: {full_name}

Analyse the image and determine:
- Which bank the customer paid into
- The amount paid (if present)
- The date of payment (if present)

------------------------------
BANK IDENTIFICATION RULES
------------------------------
Look for both full names and common abbreviations:
  Wema → "Wema", "WMA"
  Providus → "Providus", "PVD", "PVS"
  Access → "Access", "ACS"
  (detect others as needed)

If multiple banks appear, choose the one most strongly implied by context
(nearest to the transaction line, amount, reference, or merchant info).
If no bank can be confidently identified, use "Unknown".

------------------------------
UPLOAD TYPE RULE
------------------------------
- Receipt  → ALL three fields detected (bank ≠ Unknown, amount present, date present)
- Claims   → ANY field missing or unclear

------------------------------
OUTPUT FORMAT (STRICT)
------------------------------
Return ONLY valid JSON — no explanation text:

{{
  "client_id": "...",
  "full_name": "...",
  "detected_bank": "...",
  "detected_amount": "...",
  "detected_date": "...",
  "confidence": "High | Medium | Low",
  "raw_text_snippet": "5–15 word snippet supporting your classification",
  "upload_type": "Receipt | Claims"
}}
"""

# ── Helpers ─────────────────────────────────────────────────────────────────

def _json_default(obj):
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _load_credentials() -> service_account.Credentials:
    raw = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    if raw:
        info = json.loads(raw)
        return service_account.Credentials.from_service_account_info(
            info, scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
    return service_account.Credentials.from_service_account_file(
        GOOGLE_SA_FILE,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )


def _download_bytes(url: str, creds: service_account.Credentials) -> tuple[bytes, str]:
    """Return (raw_bytes, guessed_mime). Uses GCS client for Firebase Storage URLs
    (which reject unauthenticated HTTP), plain requests for everything else."""
    if "firebasestorage.googleapis.com" in url:
        parsed  = urlparse(url)
        # path is  /v0/b/<bucket>/o/<encoded-object>
        parts   = parsed.path.split("/o/", 1)
        bucket_name = parts[0].replace("/v0/b/", "")
        object_path = unquote(parts[1])
        clean_path  = object_path.split("?")[0]  # strip any query string that crept in
        gcs  = storage.Client(credentials=creds)
        blob = gcs.bucket(bucket_name).blob(clean_path)
        raw  = blob.download_as_bytes()
        # guess mime from filename
        lower = clean_path.lower()
        mime  = "application/pdf" if lower.endswith(".pdf") else "image/jpeg"
        return raw, mime

    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            break
        except Exception as exc:
            if attempt == 2:
                raise
            log.warning("  download attempt %d failed: %s — retrying...", attempt + 1, exc)
            time.sleep(2 ** attempt)

    mime = resp.headers.get("Content-Type", "image/jpeg").split(";")[0].strip().lower()
    return resp.content, mime


def _to_vision_data_url(url: str, creds: service_account.Credentials) -> str:
    """Download a file and return a base64 data URL GPT-4o Vision can consume.
    PDFs are rendered to a PNG of page 1 only. Images are encoded as-is."""
    raw, mime = _download_bytes(url, creds)

    if mime == "application/pdf" or url.lower().split("?")[0].endswith(".pdf"):
        doc       = fitz.open(stream=raw, filetype="pdf")
        pix       = doc[0].get_pixmap(dpi=150)
        img_bytes = pix.tobytes("png")
        doc.close()
        mime = "image/png"
    else:
        img_bytes = raw
        if not mime.startswith("image/"):
            mime = "image/jpeg"

    b64 = base64.b64encode(img_bytes).decode()
    return f"data:{mime};base64,{b64}"


# ── Pipeline steps ───────────────────────────────────────────────────────────

def query_bigquery(creds: service_account.Credentials) -> list[dict]:
    bq = bigquery.Client(project=BQ_PROJECT, credentials=creds)
    sql = f"SELECT * FROM {BQ_TABLE} WHERE institution = '{INSTITUTION}'"
    log.info("BigQuery: querying %s ...", BQ_TABLE)
    rows = [dict(row) for row in bq.query(sql).result()]
    log.info("BigQuery: %d rows fetched", len(rows))
    return rows


def classify_with_vision(openai_client: OpenAI, row: dict, creds: service_account.Credentials) -> dict | None:
    client_id     = row.get("client_id")
    full_name     = row.get("full_name", "")
    download_link = row.get("download_link")

    if not client_id or not download_link:
        return None

    prompt = CLASSIFY_PROMPT.format(client_id=client_id, full_name=full_name)

    try:
        data_url = _to_vision_data_url(download_link, creds)
        resp = openai_client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            response_format={"type": "json_object"},
            max_tokens=512,
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as exc:
        log.error("  ✗ GPT Vision failed for client_id=%s: %s", client_id, exc)
        return None


def upload_to_gcs(creds: service_account.Credentials, data: list[dict]):
    gcs    = storage.Client(credentials=creds, project=BQ_PROJECT)
    bucket = gcs.bucket(GCS_UPLOAD_BUCKET)
    blob   = bucket.blob(GCS_UPLOAD_OBJECT)
    blob.upload_from_string(
        json.dumps(data, indent=2, default=_json_default),
        content_type="application/json",
    )
    log.info("✓ GCS: uploaded %d records to gs://%s/%s", len(data), GCS_UPLOAD_BUCKET, GCS_UPLOAD_OBJECT)


def get_id_token(creds: service_account.Credentials) -> str:
    creds.refresh(google.auth.transport.requests.Request())
    url  = (
        f"https://iamcredentials.googleapis.com/v1/projects/-"
        f"/serviceAccounts/{CLOUD_RUN_SA_EMAIL}:generateIdToken"
    )
    resp = requests.post(
        url,
        json={"audience": CLOUD_RUN_URL, "includeEmail": True},
        headers={"Authorization": f"Bearer {creds.token}"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["token"]


def trigger_cloud_run(token: str):
    resp = requests.post(
        CLOUD_RUN_URL,
        headers={"Authorization": f"Bearer {token}"},
        timeout=120,
    )
    resp.raise_for_status()
    log.info("✓ Cloud Run triggered: HTTP %s", resp.status_code)


def fetch_today_excel(creds: service_account.Credentials) -> bytes:
    today    = datetime.now(timezone.utc)
    date_str = today.strftime("%Y%m%d")
    obj_path = f"classified/{date_str}/classified_output_{date_str}.xlsx"
    gcs      = storage.Client(credentials=creds, project=BQ_PROJECT)
    blob     = gcs.bucket(GCS_EXPORT_BUCKET).blob(obj_path)
    log.info("GCS: fetching gs://%s/%s", GCS_EXPORT_BUCKET, obj_path)
    return blob.download_as_bytes()


def send_email(excel_bytes: bytes):
    today_str = datetime.now().strftime("%d %b %Y")
    date_str  = datetime.now().strftime("%Y%m%d")

    html = f"""\
<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"/>
<style>
  body{{font-family:Arial,sans-serif;font-size:14px;color:#333;background:#fff;margin:0;padding:0}}
  .c{{max-width:680px;margin:0 auto;padding:32px 24px}}
  .hd{{border-bottom:2px solid #7CB342;padding-bottom:12px;margin-bottom:24px}}
  .hd h2{{margin:0;font-size:18px;color:#222}}
  .hd p{{margin:4px 0 0;font-size:12px;color:#888}}
  .sec{{margin-bottom:28px}}
  .st{{font-size:13px;font-weight:bold;text-transform:uppercase;letter-spacing:.5px;color:#7CB342;margin-bottom:10px;border-bottom:1px solid #f0f0f0;padding-bottom:6px}}
  table{{width:100%;border-collapse:collapse;font-size:13px}}
  th{{background:#f5f5f5;text-align:left;padding:8px 10px;font-weight:bold;color:#444;border-bottom:1px solid #e0e0e0}}
  td{{padding:8px 10px;border-bottom:1px solid #f0f0f0;color:#333}}
  tr:last-child td{{border-bottom:none}}
  .ft{{margin-top:32px;font-size:12px;color:#aaa;border-top:1px solid #f0f0f0;padding-top:16px}}
</style></head><body><div class="c">
  <div class="hd"><h2>FSS — Payment Receipts</h2>
    <p>Prepared for: Joyce &nbsp;|&nbsp; Generated: {today_str}</p></div>
  <div class="sec"><p>Hey Joyce,</p>
    <p>Please find attached the payment receipts received by FSS for {INSTITUTION} classified for today.</p></div>
  <div class="sec"><div class="st">Classification Results Guide</div>
    <table><thead><tr><th>Field</th><th>Description</th></tr></thead><tbody>
      <tr><td><strong>Detected Bank</strong></td><td>The bank identified as the destination of the transaction.</td></tr>
      <tr><td><strong>Detected Amount</strong></td><td>The transaction amount extracted from the receipt.</td></tr>
      <tr><td><strong>Detected Date</strong></td><td>The date detected from the receipt.</td></tr>
      <tr><td><strong>Confidence</strong></td><td>The model's confidence level in the prediction.</td></tr>
      <tr><td><strong>Raw Text Snippet</strong></td><td>The part of the image text used for classification.</td></tr>
    </tbody></table></div>
  <div class="sec"><p>Do not hesitate to reach out if you have any questions.</p>
    <p>Best regards,<br/><strong>FSS Data Team</strong></p></div>
  <div class="ft">This mail is confidential and intended solely for the REMEDIAL team.</div>
</div></body></html>"""

    msg              = MIMEMultipart()
    msg["From"]      = f"FSS Data Team <{GMAIL_SENDER}>"
    msg["To"]        = EMAIL_TO
    msg["Cc"]        = ", ".join(EMAIL_CC)
    msg["Subject"]   = "Payment Receipts"
    msg.attach(MIMEText(html, "html"))

    part = MIMEBase("application", "octet-stream")
    part.set_payload(excel_bytes)
    encoders.encode_base64(part)
    part.add_header(
        "Content-Disposition",
        f'attachment; filename="classified_output_{date_str}.xlsx"',
    )
    msg.attach(part)

    all_recipients = [EMAIL_TO] + EMAIL_CC
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as server:
        server.login(GMAIL_SENDER, GMAIL_APP_PASSWORD)
        server.sendmail(GMAIL_SENDER, all_recipients, msg.as_string())
    log.info("✓ Email sent to %s (CC: %s)", EMAIL_TO, ", ".join(EMAIL_CC))


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    log.info("=== classify_receipts.py starting ===")

    creds         = _load_credentials()
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

    # 1. Fetch rows from BigQuery
    rows = query_bigquery(creds)

    # 2. Classify each row via GPT Vision (replaces OCR.space + separate GPT call)
    results = []
    valid   = [r for r in rows if r.get("client_id") and r.get("download_link")]
    log.info("Rows with client_id + download_link: %d / %d", len(valid), len(rows))

    for i, row in enumerate(valid):
        log.info("  [%d/%d] client_id=%s", i + 1, len(valid), row["client_id"])
        classification = classify_with_vision(openai_client, row, creds)
        if classification is None:
            continue
        results.append({**row, **classification})

    log.info("Classified: %d rows", len(results))

    # 3. Keep only Receipts (drop Claims)
    receipts = [r for r in results if r.get("upload_type") == "Receipt"]
    log.info("Receipts after filtering Claims: %d", len(receipts))

    # 4. Final filter — must still have client_id + download_link
    receipts = [r for r in receipts if r.get("client_id") and r.get("download_link")]

    if not receipts:
        log.info("No receipts to process. Exiting.")
        return

    # 5. Upload JSON to GCS
    upload_to_gcs(creds, receipts)

    # 6. Trigger Cloud Run (via short-lived ID token)
    token = get_id_token(creds)
    trigger_cloud_run(token)

    # 7. Fetch today's Excel produced by Cloud Run
    excel_bytes = fetch_today_excel(creds)

    # 8. Email it to Joyce
    send_email(excel_bytes)

    log.info("=== Done ===")


if __name__ == "__main__":
    main()
