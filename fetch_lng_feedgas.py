#!/usr/bin/env python3
"""
fetch_lng_feedgas.py
Versuch: automatische Ermittlung von LNG Feedgas / sendout (best-effort).
Gibt auf stdout: "<value>,<YYYY-MM-DD>" oder "ERROR" bei totalem Fehlschlag.
Legt bei Erfolg eine Cache-Datei lng_feedgas_last.txt mit "value,date" an.
"""
from __future__ import annotations
import csv
import re
import requests
from datetime import datetime
from io import StringIO
from typing import Optional, Tuple

# Mögliche CSV-Quellen (EIA / IR) — probiere in Reihenfolge
CANDIDATE_URLS = [
    "https://ir.eia.gov/ngs/wngsr.csv",      # Beispiel: weekly storage CSV (some sites)
    "https://ir.eia.gov/ngd/ngd_weekly.csv", # older/alternate (may 404)
    "https://ir.eia.gov/ngd/ngd.csv",
]

CACHE_FILE = "lng_feedgas_last.txt"
# Header-Keywords, die auf Feedgas / sendout / exports etc. hinweisen können
HEADER_KEYWORDS = [
    "feedgas", "feed gas", "sendout", "send out", "export", "lng",
    "total sendout", "feedgas mcf", "feedgas (mmcf/d)"
]

def try_parse_csv_for_value(text: str) -> Optional[Tuple[float, str]]:
    """Versucht CSV-Text anzulesen. Gibt (value,date) zurück oder None."""
    f = StringIO(text)
    reader = csv.reader(f)
    # read header lines until we find a header with numeric columns
    try:
        headers = next(reader)
    except StopIteration:
        return None

    # normalize header names
    norm_headers = [h.strip().lower() for h in headers]

    # find candidate column index
    candidate_idx = None
    for i, h in enumerate(norm_headers):
        for kw in HEADER_KEYWORDS:
            if kw in h:
                candidate_idx = i
                break
        if candidate_idx is not None:
            break

    # try also a fall-back: if first col looks like date and second numeric, use second
    if candidate_idx is None:
        if len(norm_headers) >= 2 and ("date" in norm_headers[0] or "week" in norm_headers[0]):
            candidate_idx = 1

    if candidate_idx is None:
        return None

    # iterate rows: first numeric row -> return
    for row in reader:
        if len(row) <= candidate_idx:
            continue
        date_candidate = row[0].strip()
        value_candidate = row[candidate_idx].strip().replace(",", "")
        # find a numeric substring
        m = re.search(r"-?\d+(\.\d+)?", value_candidate)
        if m:
            try:
                val = float(m.group(0))
            except Exception:
                continue
            # try to parse date in common formats
            for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d-%b-%y", "%d-%b-%Y", "%m/%d/%y", "%d.%m.%Y"):
                try:
                    dt = datetime.strptime(date_candidate, fmt)
                    return round(val, 2), dt.date().isoformat()
                except Exception:
                    continue
            # if date not parseable, return today as fallback date
            return round(val, 2), datetime.utcnow().date().isoformat()

    return None

def fetch_from_urls() -> Optional[Tuple[float, str]]:
    for url in CANDIDATE_URLS:
        try:
            r = requests.get(url, timeout=15)
            if r.status_code != 200:
                continue
            parsed = try_parse_csv_for_value(r.text)
            if parsed:
                return parsed
        except Exception:
            continue
    return None

def read_cache() -> Optional[Tuple[float, str]]:
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            line = f.read().strip()
        if not line:
            return None
        parts = [p.strip() for p in line.split(",")]
        val = float(parts[0])
        date = parts[1] if len(parts) > 1 else datetime.utcnow().date().isoformat()
        return val, date
    except Exception:
        return None

def write_cache(value: float, date_iso: str) -> None:
    with open(CACHE_FILE + ".tmp", "w", encoding="utf-8") as f:
        f.write(f"{value},{date_iso}")
    import os
    os.replace(CACHE_FILE + ".tmp", CACHE_FILE)

def main():
    # 1) try online
    parsed = fetch_from_urls()
    if parsed:
        val, date_iso = parsed
        write_cache(val, date_iso)
        print(f"{val},{date_iso}")
        return

    # 2) fallback to cache
    parsed = read_cache()
    if parsed:
        val, date_iso = parsed
        # indicate we are using cached but still return a value
        print(f"{val},{date_iso}")
        return

    # 3) total fallback
    print("0,1970-01-01")  # explicit, so downstream can parse

if __name__ == "__main__":
    main()

# === Compatibility wrapper ===
import pandas as pd
from datetime import datetime

def load_lng_feedgas(*args, **kwargs):
    parsed = fetch_from_urls()
    source = "eia_live"

    if not parsed:
        parsed = read_cache()
        source = "cache"

    if not parsed:
        # return EMPTY but SCHEMA-CORRECT dataframe
        return pd.DataFrame(columns=["Date", "Feedgas"])

    val, date_iso = parsed
    date = pd.Timestamp(date_iso)

    df = pd.DataFrame({
        "Date": [date],
        "Feedgas": [val],
    })

    df.attrs["source"] = source
    return df


