"""
Which customer a phone number belongs to, so every call note carries the
customer's name, FSS ID and client ID (the FSS app lists and searches notes by
these). Matches the way the app tags notes (fss-notes-app/lib/noteCustomers.js):

  1. Active leads (phone, phone2, phone3) → fss_id, client_id, client_name, country.
     Several customers on one number: the one at the call's institution wins.
  2. Otherwise all leads (past customers) → client_id, client_name, country (no FSS ID).

Also who the customer's agent is (leads_active.assigned_agent, set in BigQuery), so
alerts go to that agent — see recipient_for().

Used by va_pipeline.py, wa_pipeline.py and livekit_pipeline.py.
"""

import logging
import re

from google.cloud import bigquery

log = logging.getLogger(__name__)

_cache: dict[str, dict] = {}
_agents: dict[str, str] = {}   # same keys as _cache → assigned_agent ("" when none)

# leads_active.assigned_agent → the agent's FSS email.
AGENT_EMAILS = {
    "kesta":      "kesta@fsldigital.com",
    "gbolade":    "gbolade@fsldigital.com",
    "dami":       "dami.adeyanju@fsldigital.com",
    "diana":      "diana@fsldigital.com",
    "emmanuella": "emmanuella@fsldigital.com",
    "itoro":      "itoro@fsldigital.com",
    "lynda":      "lynda@fsldigital.com",
    "samuel":     "samuel@fsldigital.com",
    "olatoye":    "olatoye@fsldigital.com",
}


def phone_variants(phone: str) -> list[str]:
    """Every stored form of a number: 0801…, 801…, 234801…, +234801… (and KE/UG)."""
    raw = str(phone or "").strip()
    digits = re.sub(r"\D", "", raw)
    if not digits:
        return []
    out = {raw, digits}
    m = re.match(r"^(234|254|256)(\d{9,10})$", digits)
    local = m.group(2) if m else digits.lstrip("0")
    if local:
        out.update({"0" + local, local})
        for cc in ("234", "254", "256"):
            if m and m.group(1) != cc:
                continue
            out.update({cc + local, "+" + cc + local})
    return [p for p in out if len(re.sub(r"\D", "", p)) >= 7]


def _same_institution(a: str, b: str) -> bool:
    x = re.sub(r"[^a-z0-9]", "", str(a or "").lower())
    y = re.sub(r"[^a-z0-9]", "", str(b or "").lower())
    return bool(x and y and (x in y or y in x))


def _pick(rows: list[dict], institution: str) -> dict:
    return next((r for r in rows if _same_institution(r.get("institution"), institution)), rows[0])


def find_customer(phone: str, institution: str, bq: bigquery.Client) -> dict:
    """Fields to add to the note — {} when the number matches no customer."""
    variants = phone_variants(phone)
    if not variants:
        return {}
    key = f"{variants[0]}|{institution}"
    if key in _cache:
        return _cache[key]
    cfg = bigquery.QueryJobConfig(query_parameters=[bigquery.ArrayQueryParameter("p", "STRING", variants)])
    found: dict = {}
    try:
        rows = [dict(r) for r in bq.query("""
            SELECT CAST(fss_id AS STRING) AS fss_id, CAST(client_id AS STRING) AS client_id,
                   client_name, country, institution, assigned_agent
            FROM `fssspark.recovery_methods_data.leads_active`
            WHERE phone IN UNNEST(@p) OR phone2 IN UNNEST(@p) OR phone3 IN UNNEST(@p)
        """, job_config=cfg).result()]
        if rows:
            r = _pick(rows, institution)
            found = {"fss_id": r["fss_id"] or "", "client_id": r["client_id"] or "",
                     "client_name": r["client_name"] or "", "country": r["country"] or ""}
            _agents[key] = str(r.get("assigned_agent") or "").strip().lower()
        else:
            rows = [dict(r) for r in bq.query("""
                SELECT CAST(client_id AS STRING) AS client_id, country, institution,
                       TRIM(CONCAT(IFNULL(first_name, ''), ' ', IFNULL(middle_name, ''), ' ', IFNULL(surname, ''))) AS client_name
                FROM `fssspark.original_cohorts.all_leads`
                WHERE phone IN UNNEST(@p) OR alt_phone IN UNNEST(@p) OR alt_phone_2 IN UNNEST(@p)
            """, job_config=cfg).result()]
            if rows:
                r = _pick(rows, institution)
                found = {"client_id": r["client_id"] or "", "client_name": re.sub(r"\s+", " ", r["client_name"] or "").strip(),
                         "country": r["country"] or ""}
    except Exception as exc:
        log.warning("Customer lookup failed for %s: %s", phone, exc)
        return {}   # not cached, so the next call retries
    found = {k: v for k, v in found.items() if v}
    _cache[key] = found
    return found


def recipient_for(phone: str, institution: str, bq) -> str:
    """The email of the customer's assigned agent in BigQuery — "" when there isn't one
    (not an active lead, an agent we don't know, or the lookup failed)."""
    if not bq:
        return ""
    find_customer(phone, institution, bq)   # cached; fills _agents
    variants = phone_variants(phone)
    key = f"{variants[0]}|{institution}" if variants else ""
    return AGENT_EMAILS.get(_agents.get(key, ""), "")
