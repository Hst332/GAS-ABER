#!/usr/bin/env python3
"""
gas_price_forecast.py
Erzeugt forecast_output.txt aus den Input-Dateien:
 - eia_storage.txt  (format: "<value>,<YYYY-MM-DD>" oder eine einzige Zahl)
 - us_production.txt (analog)
 - lng_feedgas.txt   (aus fetch_lng_feedgas.py: "<value>,<YYYY-MM-DD>")

Falls Dateien fehlen, verwendet Platzhalter bzw. cache.
"""
from __future__ import annotations
from datetime import datetime
import os
import math

# Gewichtungen
FACTORS = {
    "EIA Storage": 0.25,
    "US Production": 0.20,
    "LNG Feedgas": 0.20,
    "Futures Curve": 0.20,
    "COT Managed Money": 0.15
}

# Default placeholders (0..10 scale). Für feel: 0 = kein Aufwärtsdruck, 10 = sehr stark.
DEFAULT_PLACEHOLDERS = {
    "EIA Storage": 0.0,
    "US Production": 0.0,
    "LNG Feedgas": 0.0,
    "Futures Curve": 0.0,
    "COT Managed Money": 0.0
}

# Pfade
EIA_FILE = "eia_storage.txt"
US_PROD_FILE = "us_production.txt"
LNG_FILE = "lng_feedgas.txt"
OUT_FILE = "forecast_output.txt"

def read_value_date(path: str):
    """Liess path, erwartet 'value' oder 'value,date'. Liefert (value:float, date_or_note:str, source_str)."""
    if not os.path.exists(path):
        return None
    try:
        s = open(path, "r", encoding="utf-8").read().strip()
        if not s:
            return None
        # falls "value,date"
        parts = [p.strip() for p in s.split(",")]
        val = float(parts[0])
        date = parts[1] if len(parts) > 1 else "unknown"
        return val, date, "auto"
    except Exception:
        # falls file contains non-numeric, return None
        return None

def normalize_to_0_10(name: str, raw_value: float) -> float:
    """
    Mappt raw_value auf 0..10. Für bekannte Größen werden plausiblere Ranges verwendet.
    - EIA Storage (Bcf): typ. Bereich ~ 500–4500 → 0..10 mappe linear
    - US Production (Bcf/d): typ. Bereich ~ 70–100 → mappe linear
    - LNG Feedgas (MMcf/d): typ. Bereich ~ 0–15k → mappe linear
    Für unknown: einfache scaling / clipping.
    """
    if raw_value is None:
        return 0.0
    try:
        if name == "EIA Storage":
            # smaller value -> higher price pressure -> invert mapping: low storage => high upward pressure.
            # But user previously wanted "höherer Wert = stärkerer Preisdruck nach oben" — falls you want normalised on that basis,
            # we will *invert* storage to reflect that *higher storage -> lower upward pressure*. To keep simple, compute percentile-like.
            # We'll map storage Bcf into 0..10 where higher storage -> higher numeric (user earlier used higher = more upward pressure,
            # but historically storage high -> downward pressure. Here we preserve user's "higher = stronger pressure" convention.)
            low = 0.0
            high = 4500.0
            v = max(min(raw_value, high), low)
            return round((v - low) / (high - low) * 10.0, 2)
        if name == "US Production":
            low, high = 60.0, 110.0  # Bcf/d
            v = max(min(raw_value, high), low)
            return round((v - low) / (high - low) * 10.0, 2)
        if name == "LNG Feedgas":
            # mmcf/d typical ~ 0..15000; we'll use 0..15000
            low, high = 0.0, 15000.0
            v = max(min(raw_value, high), low)
            return round((v - low) / (high - low) * 10.0, 2)
        # generic
        low, high = 0.0, 100.0
        v = max(min(raw_value, high), low)
        return round((v - low) / (high - low) * 10.0, 2)
    except Exception:
        return 0.0

def compute_weighted_probability(scaled_values: dict) -> (float, float):
    """Berechnet gewichteten Score (0..10) -> normiert auf 0..100%."""
    score = 0.0
    for k, w in FACTORS.items():
        score += scaled_values.get(k, 0.0) * w
    # max possible = 10 * sum(weights)
    max_score = 10.0 * sum(FACTORS.values())
    prob = (score / max_score) * 100.0
    # clip 0..100
    prob = max(0.0, min(100.0, prob))
    return round(score, 2), round(prob, 1)

def write_output_file(meta: dict, scaled_values: dict, weighted_score: float, prob: float):
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = []
    lines.append("="*35)
    lines.append("      NATURAL GAS PRICE FORECAST")
    lines.append("="*35)
    lines.append(f"Datum: {ts}")
    lines.append("")
    lines.append("Eingabewerte (0–10 Skala):")
    for k in FACTORS.keys():
        val = scaled_values.get(k, 0.0)
        status = meta.get(k, {}).get("status", "unknown")
        date = meta.get(k, {}).get("date", "unbekannt")
        lines.append(f"  {k:18s}: {val:5.2f}  [{status} vom {date}]")
    lines.append("")
    lines.append(f"Gewichteter Score: {weighted_score}")
    lines.append(f"Wahrscheinlichkeit, dass Gaspreis steigt: {prob:.1f}%")
    lines.append("="*35)
    out_text = "\n".join(lines) + "\n"

    # atomic write
    tmp = OUT_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(out_text)
    os.replace(tmp, OUT_FILE)
    print(out_text)  # print to CI logs as well

def main():
    # read inputs (try parsing value,date)
    meta = {}

    # EIA
    eia = read_value_date(EIA_FILE)
    if eia:
        raw_eia, date, src = eia
        meta["EIA Storage"] = {"status": "aktuell", "date": date, "source": src}
    else:
        raw_eia = None
        meta["EIA Storage"] = {"status": "Platzhalter", "date": "unbekannt", "source": "none"}

    # US Production
    us = read_value_date(US_PROD_FILE)
    if us:
        raw_us, date, src = us
        meta["US Production"] = {"status": "aktuell", "date": date, "source": src}
    else:
        raw_us = None
        meta["US Production"] = {"status": "Platzhalter", "date": "unbekannt", "source": "none"}

    # LNG
    lng = read_value_date(LNG_FILE)
    if lng:
        raw_lng, date, src = lng
        meta["LNG Feedgas"] = {"status": "aktuell", "date": date, "source": src}
    else:
        # try cached file from fetch_lng_feedgas.py (lng_feedgas_last.txt)
        if os.path.exists("lng_feedgas_last.txt"):
            try:
                s = open("lng_feedgas_last.txt", "r", encoding="utf-8").read().strip()
                parts = [p.strip() for p in s.split(",")]
                raw_lng = float(parts[0])
                date = parts[1] if len(parts) > 1 else "unbekannt"
                meta["LNG Feedgas"] = {"status": "cache", "date": date, "source": "cache"}
            except Exception:
                raw_lng = None
                meta["LNG Feedgas"] = {"status": "Platzhalter", "date": "unbekannt", "source": "none"}
        else:
            raw_lng = None
            meta["LNG Feedgas"] = {"status": "Platzhalter", "date": "unbekannt", "source": "none"}

    # Futures & COT placeholders (for now)
    raw_futures = None
    raw_cot = None
    meta["Futures Curve"] = {"status": "Platzhalter", "date": "unbekannt", "source": "none"}
    meta["COT Managed Money"] = {"status": "Platzhalter", "date": "unbekannt", "source": "none"}

    # normalize to 0..10
    scaled = {}
    scaled["EIA Storage"] = normalize_to_0_10("EIA Storage", raw_eia)
    scaled["US Production"] = normalize_to_0_10("US Production", raw_us)
    scaled["LNG Feedgas"] = normalize_to_0_10("LNG Feedgas", raw_lng)
    scaled["Futures Curve"] = normalize_to_0_10("Futures Curve", raw_futures)
    scaled["COT Managed Money"] = normalize_to_0_10("COT Managed Money", raw_cot)

    # compute
    weighted_score, prob = compute_weighted_probability(scaled)

    # write file
    write_output_file(meta, scaled, weighted_score, prob)

if __name__ == "__main__":
    main()
