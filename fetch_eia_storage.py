#!/usr/bin/env python3
"""
fetch_eia_storage.py

Robuster EIA Working Gas in Storage Fetcher.

- Liest API-Key aus EIA_API_KEY (Environment / GitHub Secret).
- Versucht mehrere v2 backward-compatible Varianten:
    * /v2/seriesid/NG.WKST.S
    * /v2/seriesid/NG.WKST
- Falls beides fehlschlägt, versucht als Fallback die legacy v1 "series" endpoint.
- Debug-Ausgaben -> stderr. Nur der numeric value (oder "ERROR") -> stdout.
- Exit-Code 0 bei Erfolg, 1 bei Fehler.
"""

import os
import sys
import requests
from typing import Tuple, Optional

TIMEOUT = 15.0


def _print_debug(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def try_v2_seriesid(series_id: str, api_key: str) -> Tuple[Optional[float], Optional[str], Optional[str]]:
    """
    Try the v2 backward-compatible endpoint /v2/seriesid/{series_id}
    Returns (value, period, error_message)
    """
    url = f"https://api.eia.gov/v2/seriesid/{series_id}?api_key={api_key}"
    try:
        _print_debug("REQUEST_V2:", url.replace(api_key, "***"))
        r = requests.get(url, timeout=TIMEOUT)
    except Exception as e:
        return None, None, f"request-error: {e}"

    _print_debug("STATUS:", r.status_code)
    _print_debug("RAW_RESPONSE:", r.text[:1000])  # limit length in logs

    if r.status_code != 200:
        # return textual error (for logic in caller)
        try:
            j = r.json()
            msg = j.get("error") or j.get("message") or r.text
        except Exception:
            msg = r.text
        return None, None, f"http-{r.status_code}: {msg}"

    try:
        j = r.json()
        # v2 seriesid returns structure: response -> data -> [ { period, value }, ... ]
        if "response" in j and "data" in j["response"] and len(j["response"]["data"]) > 0:
            first = j["response"]["data"][0]
            # value might be in 'value' or in a named field (defensive)
            if "value" in first:
                value = float(first["value"])
            else:
                # try find first numeric field
                value = None
                for k, v in first.items():
                    if k == "period":
                        continue
                    try:
                        value = float(v)
                        break
                    except Exception:
                        continue
                if value is None:
                    return None, None, "parse-error: no numeric field in v2 response"
            period = first.get("period")
            return value, period, None
        # fallback: v2 may also return legacy 'series' block sometimes
        if "series" in j and isinstance(j["series"], list) and len(j["series"]) > 0:
            ser = j["series"][0]
            if "data" in ser and len(ser["data"]) > 0:
                row = ser["data"][0]
                # legacy v1 structure: [period, value]
                value = float(row[1])
                period = row[0]
                return value, period, None
        return None, None, "parse-error: unexpected v2 payload"
    except Exception as e:
        return None, None, f"parse-exception: {e}"


def try_v1_series(series_id: str, api_key: str) -> Tuple[Optional[float], Optional[str], Optional[str]]:
    """
    Try legacy v1 series endpoint (note: EIA deprecated v1, but their docs allow fallback)
    URL: https://api.eia.gov/series/?api_key=KEY&series_id=SERIES_ID
    """
    url = f"https://api.eia.gov/series/?api_key={api_key}&series_id={series_id}"
    try:
        _print_debug("REQUEST_V1:", url.replace(api_key, "***"))
        r = requests.get(url, timeout=TIMEOUT)
    except Exception as e:
        return None, None, f"request-error: {e}"

    _print_debug("STATUS_V1:", r.status_code)
    _print_debug("RAW_V1:", r.text[:1000])

    if r.status_code != 200:
        return None, None, f"http-{r.status_code}: {r.text}"

    try:
        j = r.json()
        # expected structure: series -> [ { data: [ [period, value], ... ] } ]
        if "series" in j and isinstance(j["series"], list) and len(j["series"]) > 0:
            ser = j["series"][0]
            data = ser.get("data")
            if data and len(data) > 0:
                period = data[0][0]
                value = float(data[0][1])
                return value, period, None
        return None, None, "parse-error-v1"
    except Exception as e:
        return None, None, f"parse-exception-v1: {e}"


def fetch_eia_storage() -> Tuple[Optional[float], Optional[str]]:
    api_key = os.getenv("EIA_API_KEY", "").strip()
    if not api_key:
        _print_debug("ERROR: EIA_API_KEY not set in environment")
        return None, None

    # Candidate series IDs (try most likely forms)
    candidates = ["NG.WKST.S", "NG.WKST", "NG.WST", "NG.WKST.SA"]  # included some variants defensively

    # Try v2 seriesid for each candidate
    for sid in candidates:
        val, period, err = try_v2_seriesid(sid, api_key)
        if err is None and val is not None:
            return val, period
        _print_debug(f"v2 attempt {sid} failed: {err}")

        # If error explicitly mentions invalid frequency 'S', try without .S
        if err and ("Invalid frequency" in err or "Invalid frequency" in str(err)):
            continue  # next candidate likely without dot suffix

    # As last resort, try legacy v1 series endpoint with the most common id
    fallback_ids = ["NG.WKST.S", "NG.WKST"]
    for fid in fallback_ids:
        val, period, err = try_v1_series(fid, api_key)
        if err is None and val is not None:
            return val, period
        _print_debug(f"v1 attempt {fid} failed: {err}")

    # nothing worked
    return None, None


if __name__ == "__main__":
    value, period = fetch_eia_storage()
    if value is None:
        # only "ERROR" on stdout for workflow detection; full details are on stderr
        print("ERROR")
        sys.exit(1)
    else:
        # print only numeric value to stdout (clean for GitHub Actions)
        # but also print debug to stderr
        _print_debug(f"SUCCESS: period={period}, value={value}")
        print(value)
        sys.exit(0)
