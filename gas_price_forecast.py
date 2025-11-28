# gas_price_forecast.py
import argparse
from datetime import datetime

# ---------- Hilfsfunktionen ----------

def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def clamp(value, min_val=0.0, max_val=10.0):
    return max(min_val, min(max_val, value))

def normalize(value, max_realistic):
    """
    Skaliert reale Werte auf 0–10
    """
    if max_realistic <= 0:
        return 0.0
    return clamp((value / max_realistic) * 10.0)

# ---------- Hauptlogik ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eia-storage", required=True)
    parser.add_argument("--us-production", required=True)
    parser.add_argument("--lng-feedgas", required=True)
    parser.add_argument("--futures-curve", required=True)
    parser.add_argument("--cot-managed-money", required=True)

    args = parser.parse_args()

    # Rohwerte
    eia_storage_raw = safe_float(args.eia_storage)
    us_prod_raw = safe_float(args.us_production)
    lng_raw = safe_float(args.lng_feedgas)
    futures = safe_float(args.futures_curve)
    cot = safe_float(args.cot_managed_money)

    # Normalisierung (REALISTISCHE MAXIMA!)
    eia_storage = normalize(eia_storage_raw, 4000)   # Bcf
    us_production = normalize(us_prod_raw, 100)      # Bcf/d
    lng_feedgas = normalize(lng_raw, 14)              # Bcf/d

    futures = clamp(futures)
    cot = clamp(cot)

    # Gewichtung (Summe = 1.0)
    weights = {
        "eia": 0.30,
        "prod": 0.25,
        "lng": 0.20,
        "futures": 0.15,
        "cot": 0.10
    }

    score = (
        eia_storage * weights["eia"] +
        us_production * weights["prod"] +
        lng_feedgas * weights["lng"] +
        futures * weights["futures"] +
        cot * weights["cot"]
    )

    probability = clamp(score * 10, 0, 100)

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    output = f"""
===================================
      NATURAL GAS PRICE FORECAST
===================================
Datum: {now}

Eingabewerte (0–10 Skala):
  EIA Storage         : {eia_storage:.2f}  [vom Datenabruf]
  US Production       : {us_production:.2f}  [vom Datenabruf]
  LNG Feedgas         : {lng_feedgas:.2f}  [letzte verfügbare Schätzung]
  Futures Curve       : {futures:.2f}  [Platzhalter]
  COT Managed Money   : {cot:.2f}  [Platzhalter]

Gewichteter Score: {score:.2f}
Wahrscheinlichkeit, dass Gaspreis steigt: {probability:.1f}%
===================================
"""

    print(output.strip())

    with open("forecast_output.txt", "w", encoding="utf-8") as f:
        f.write(output.strip())

if __name__ == "__main__":
    main()
