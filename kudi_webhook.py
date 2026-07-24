#!/usr/bin/env python3
"""
kudi_webhook.py
Flask webhook receiver for Kudi email delivery status callbacks.

Pipeline:
  POST /kudi-email-status
    → BigQuery customer lookup (by recipient email)
    → classify status (Delivered / Expired / Undeliverable)
    → upsert to Firestore email_campaign_tracking
"""

import json
import logging
import os

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from google.cloud import bigquery, firestore
from google.oauth2 import service_account

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────────────
GOOGLE_SA_FILE  = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE", "service_account.json")
BQ_PROJECT      = "fssspark"
FS_COLLECTION   = "email_campaign_tracking"

# Status/code routing — mirrors the n8n Switch node (OR logic within each bucket)
DELIVERED_STATUSES     = {"DELIVRD", "DELIVERED"}
DELIVERED_CODES        = {"000"}
EXPIRED_STATUSES       = {"EXPIRED", "REJECTD", "UNKNOWN"}
EXPIRED_CODES          = {"101", "102", "150"}
UNDELIVERABLE_STATUSES = {"UNDELIVRD", "BOUNCED"}
UNDELIVERABLE_CODES    = {"99", "100"}

# ── Credentials (shared across requests) ────────────────────────────────────
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

creds     = _load_credentials()
bq_client = bigquery.Client(project=BQ_PROJECT, credentials=creds)
db        = firestore.Client(project=BQ_PROJECT, credentials=creds)

# ── Helpers ──────────────────────────────────────────────────────────────────

def _query_customer(email: str) -> dict:
    sql = """
        SELECT client_id, first_name, institution, email AS destination, phone
        FROM `fssspark.original_cohorts.all_leads`
        WHERE email = @email
        LIMIT 1
    """
    cfg  = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("email", "STRING", email)]
    )
    rows = list(bq_client.query(sql, job_config=cfg).result())
    return dict(rows[0]) if rows else {}


def _classify(status: str, code: str) -> str:
    if status in DELIVERED_STATUSES or code in DELIVERED_CODES:
        return "Delivered"
    if status in EXPIRED_STATUSES or code in EXPIRED_CODES:
        return "Expired"
    if status in UNDELIVERABLE_STATUSES or code in UNDELIVERABLE_CODES:
        return "Undeliverable"
    return "Unknown"


def _build_doc(payload: dict, customer: dict, status_label: str) -> dict:
    data      = payload.get("data", {})
    recipient = data.get("recipient", "")
    time_val  = data.get("time", "")
    api_ref   = payload.get("api_reference", "")

    if status_label == "Delivered":
        error       = "No Error (code 0)"
        error_group = "No Errors"
        seen_at     = time_val
    elif status_label == "Expired":
        error       = data.get("description", "")
        error_group = "User Errors"
        seen_at     = "--"
    else:  # Undeliverable / Unknown
        error       = "Recipient address is invalid (code 6006)"
        error_group = "User Errors"
        seen_at     = "--"

    return {
        "application_id":       "default",
        "client_id":            customer.get("client_id", ""),
        "destination":          recipient,
        "done_at":              time_val,
        "email":                recipient,
        "email_count":          1,
        "email_sent_date":      time_val.split(" ")[0] if time_val else "",
        "entity_id":            None,
        "error":                error,
        "error_group":          error_group,
        "external_message_id":  api_ref,
        "first_name":           customer.get("first_name", ""),
        "inbound":              None,
        "institution":          customer.get("institution", ""),
        "ivrResponses":         None,
        "ivr_mapped_responses": None,
        "outbound":             "Yes",
        "phone":                customer.get("phone", ""),
        "request_type":         "Email",
        "seen_at":              seen_at,
        "sender":               "payments@fsldigital.com",
        "source_file":          "Kudi_email_webhook",
        "status":               status_label,
        "text":                 None,
        "upload_batch_id":      None,
        "uploaded_at":          time_val,
    }


# ── Flask app ────────────────────────────────────────────────────────────────
app = Flask(__name__)


@app.route("/kudi-email-status", methods=["POST"])
def kudi_email_status():
    try:
        payload   = request.get_json(force=True) or {}
        log.info("Payload: %s", json.dumps(payload))

        data      = payload.get("data", {})
        recipient = data.get("recipient", "")
        status    = payload.get("status", "")
        code      = str(data.get("code", ""))   # code lives inside data, not at root
        api_ref   = payload.get("api_reference", "")

        if not recipient:
            log.warning("Missing recipient in payload")
            return jsonify({"error": "missing recipient"}), 400

        if not api_ref:
            log.warning("Missing api_reference in payload")
            return jsonify({"error": "missing api_reference"}), 400

        # Enrich with BigQuery customer data
        customer = _query_customer(recipient)
        if not customer:
            log.warning("No customer found for email=%s — continuing with empty fields", recipient)

        # Route status
        status_label = _classify(status, code)
        log.info("%-30s  status=%-15s code=%-5s → %s", recipient, status, code, status_label)

        # Build and upsert Firestore doc
        doc = _build_doc(payload, customer, status_label)
        db.collection(FS_COLLECTION).document(api_ref).set(doc, merge=True)
        log.info("✓ Firestore upserted doc=%s", api_ref)

        return jsonify({"ok": True, "status": status_label}), 200

    except Exception as exc:
        log.error("Webhook error: %s", exc, exc_info=True)
        return jsonify({"error": str(exc)}), 500


@app.route("/debug", methods=["POST", "GET"])
def debug():
    """Raw dump — logs everything Kudi sends so you can inspect the exact payload shape."""
    raw_body  = request.get_data(as_text=True)
    headers   = dict(request.headers)
    parsed    = request.get_json(force=True, silent=True)

    dump = {
        "method":  request.method,
        "headers": headers,
        "raw_body": raw_body,
        "parsed":  parsed,
    }
    log.info("=== /debug hit ===\n%s", json.dumps(dump, indent=2))
    return jsonify(dump), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True}), 200


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)
