#!/usr/bin/env python3
"""
One-off backfill for call_notes_2 and call_notes_latest — fixes notes the bot
pipelines wrote before they were updated:

  1. `timestamp` saved as text → native Firestore Timestamp (text sorts apart
     from Timestamps, which put notes out of date order). `updated_at` is left
     as it is: other jobs still write it as text, and nothing sorts on it.
  2. No customer on the note → fss_id, client_id, client_name, country, matched by
     phone the same way the pipelines now do (customer_lookup.py).

Only those fields are touched (update, never overwrite). Safe to re-run.

    python backfill_call_notes.py            # dry run: counts + samples, writes nothing
    python backfill_call_notes.py --apply    # write
"""

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from dateutil import parser as dateutil_parser
from dotenv import load_dotenv
from google.cloud import bigquery, firestore
from google.oauth2 import service_account

from customer_lookup import _pick, phone_variants

load_dotenv()
COLLECTIONS = ("call_notes_2", "call_notes_latest")
LAGOS = ZoneInfo("Africa/Lagos")
MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"], 1)}


def _creds():
    raw = (os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON") or "").strip()
    if raw:
        return service_account.Credentials.from_service_account_info(json.loads(raw))
    return service_account.Credentials.from_service_account_file(
        os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE", "service_account.json"))


def parse_text_date(raw: str):
    """Same formats as the app's parser (fss-notes-app/scripts/migrate-timestamps.js).
    No time zone given → Lagos time, which is how people and the app wrote them."""
    s = str(raw or "").strip()
    if not s:
        return None
    m = re.match(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}):(\d+)$", s)          # 2026-05-20 17:19:00:1234
    if m:
        s = f"{m.group(1)}.{m.group(2)[:3]}"
    m = re.match(r"^[A-Za-z]+,\s+(\d{1,2}):(\d{2})\s+(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})$", s)  # Friday, 17:19 20 May 2026
    if m and m.group(4).lower() in MONTHS:
        dt = datetime(int(m.group(5)), MONTHS[m.group(4).lower()], int(m.group(3)), int(m.group(1)), int(m.group(2)))
        return dt.replace(tzinfo=LAGOS)
    m = re.match(r"^(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})(?:\s+(\d{1,2}):(\d{2})(?::(\d{2}))?)?$", s)  # 20/05/2026 17:19
    if m:
        a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        y += 2000 if y < 100 else 0
        day, month = (b, a) if a <= 12 and b > 12 else (a, b)
        try:
            return datetime(y, month, day, int(m.group(4) or 0), int(m.group(5) or 0), int(m.group(6) or 0), tzinfo=LAGOS)
        except ValueError:
            return None
    try:
        dt = dateutil_parser.isoparse(s)
    except (ValueError, OverflowError):
        try:
            dt = dateutil_parser.parse(s)
        except (ValueError, OverflowError):
            return None
    return dt if dt.tzinfo else dt.replace(tzinfo=LAGOS)


def phone_key(p) -> str:
    d = re.sub(r"\D", "", str(p or ""))
    return d[-9:] if len(d) >= 9 else d


def lookup_customers(notes: list[dict], bq) -> dict:
    """{(phone_key, institution): fields} for every untagged note — two BigQuery
    queries in total instead of one per note."""
    keys = {phone_key(n["phone"]) for n in notes if phone_key(n["phone"])}
    variants = sorted({v for n in notes for v in phone_variants(str(n["phone"]))})
    by_key = defaultdict(list)
    past_by_key = defaultdict(list)
    for i in range(0, len(variants), 5000):
        cfg = bigquery.QueryJobConfig(query_parameters=[bigquery.ArrayQueryParameter("p", "STRING", variants[i:i + 5000])])
        for r in bq.query("""
            SELECT CAST(fss_id AS STRING) AS fss_id, CAST(client_id AS STRING) AS client_id,
                   client_name, country, institution, phone, phone2, phone3
            FROM `fssspark.recovery_methods_data.leads_active`
            WHERE phone IN UNNEST(@p) OR phone2 IN UNNEST(@p) OR phone3 IN UNNEST(@p)
        """, job_config=cfg).result():
            r = dict(r)
            for k in {phone_key(r[c]) for c in ("phone", "phone2", "phone3")} & keys:
                by_key[k].append(r)
        for r in bq.query("""
            SELECT CAST(client_id AS STRING) AS client_id, country, institution, phone, alt_phone, alt_phone_2,
                   TRIM(CONCAT(IFNULL(first_name, ''), ' ', IFNULL(middle_name, ''), ' ', IFNULL(surname, ''))) AS client_name
            FROM `fssspark.original_cohorts.all_leads`
            WHERE phone IN UNNEST(@p) OR alt_phone IN UNNEST(@p) OR alt_phone_2 IN UNNEST(@p)
        """, job_config=cfg).result():
            r = dict(r)
            for k in {phone_key(r[c]) for c in ("phone", "alt_phone", "alt_phone_2")} & keys:
                past_by_key[k].append(r)

    out = {}
    for n in notes:
        k, inst = phone_key(n["phone"]), n.get("institution") or ""
        if (k, inst) in out:
            continue
        if by_key.get(k):
            r = _pick(by_key[k], inst)
            f = {"fss_id": r["fss_id"], "client_id": r["client_id"], "client_name": r["client_name"], "country": r["country"]}
        elif past_by_key.get(k):
            r = _pick(past_by_key[k], inst)
            f = {"client_id": r["client_id"], "client_name": re.sub(r"\s+", " ", r["client_name"] or "").strip(), "country": r["country"]}
        else:
            f = {}
        out[(k, inst)] = {a: b for a, b in f.items() if b}
    return out


def run(apply: bool):
    creds = _creds()
    db = firestore.Client(project="fssspark", credentials=creds)
    bq = bigquery.Client(project="fssspark", credentials=creds)
    mode = "APPLY" if apply else "DRY RUN"

    for coll in COLLECTIONS:
        print(f"\n── {coll} ({mode}) ──")
        updates = defaultdict(dict)
        bad_dates = Counter()
        untagged = []
        samples = {}
        total = 0
        for d in db.collection(coll).select(["timestamp", "phone", "institution", "fss_id", "client_name"]).stream():
            x = d.to_dict()
            total += 1
            v = x.get("timestamp")
            if isinstance(v, str) and v.strip():
                dt = parse_text_date(v)
                if dt:
                    updates[d.reference]["timestamp"] = dt.astimezone(timezone.utc)
                    fmt = re.sub(r"\d", "9", v)
                    if fmt not in samples:
                        samples[fmt] = (v, dt.astimezone(timezone.utc).isoformat())
                else:
                    bad_dates[v[:40]] += 1
            if not x.get("fss_id") and not x.get("client_name") and phone_key(x.get("phone")):
                untagged.append({"ref": d.reference, "phone": str(x.get("phone")), "institution": x.get("institution") or ""})

        found = lookup_customers(untagged, bq) if untagged else {}
        tagged = Counter()
        for n in untagged:
            f = found.get((phone_key(n["phone"]), n["institution"]), {})
            tagged["active customer" if f.get("fss_id") else "past customer" if f else "no match"] += 1
            if f:
                updates[n["ref"]].update(f)

        ts_fixed = sum(1 for u in updates.values() if "timestamp" in u)
        print(f"docs: {total:,}")
        print(f"text timestamps → Timestamp: {ts_fixed:,}")
        for v, iso in list(samples.values())[:12]:
            print(f"    {v!r:42} → {iso}")
        print(f"notes without a customer: {len(untagged):,}  → {dict(tagged)}")
        if bad_dates:
            print(f"text dates that couldn't be read (left as they are): {sum(bad_dates.values()):,}", bad_dates.most_common(5))
        print(f"docs to update: {len(updates):,}")

        if apply:
            batch, n = db.batch(), 0
            for ref, fields in updates.items():
                batch.update(ref, fields)
                n += 1
                if n % 400 == 0:
                    batch.commit()
                    batch = db.batch()
                    print(f"  … {n:,}")
            batch.commit()
            print(f"updated {n:,} docs")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write the changes (default: dry run)")
    run(ap.parse_args().apply)
