#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from datetime import datetime


def clamp(value, min_value=0.0, max_value=100.0):
    return max(min_value, min(max_value, value))


def main():
    parser = argparse.ArgumentParser(description="Natural Gas Price Forecast")

    parser.add_argument("--eia-storage", type=float, required=True)
    parser.add_argument("--us-production", type=float, required=True)
    parser.add_argument("--lng-feedgas", type=float, required=True)
    parser.add_argument("--futures-curve", type=float, required=True)
    parser.add_argument("--cot-managed-money", type=float, required=True)

    args = parser.parse_args()

    # =========================
    # Eingabewerte
    # =========================
    eia = args.eia_storage
    us_prod = args.us_production
    lng = args.lng_feedgas
    futures = args.futures_curve
    cot = args.cot_managed_money

    # =========================
    # Status bestimmen
    # =========================
    eia_status = "aktuell"
    us_prod_status = "aktuell"

    if lng > 0:
        lng_status = "aktuell"
    else:
        lng_status = "aktuell (keine neue Meldung)"

    if futures > 0:
        futures_status = "aktuell"
    else:
        futures_status = "Platzhalter"

    if cot > 0:
        cot_status = "aktuell"
    else:
        cot_status = "Platzhalter"

    # =========================
    # Gewichtung
    # =========================
    weights = {
        "eia": 0.30,
        "us_prod": 0.25,
        "lng": 0.20,
        "futures": 0.15,
        "cot": 0.10,
    }

    weighted_score = (
        eia * weights["eia"] +
        us_prod * weights["us_prod"] +
        lng * weights["lng"] +
        futures * weights["futures"] +
        cot * weights["cot"]
    )

    probability = clamp(weighted_score * 10.0)

    now_utc = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    output = f"""
===================================
      NATURAL GAS PRICE FORECAST
===================================
Datum: {now_utc}

Eingabewerte (0–10 Skala):
  EIA Storage         : {eia:.2f}  [{eia_status}]
  US Production       : {us_prod:.2f}  [{us_prod_status}]
  LNG Feedgas         : {lng:.2f}  [{lng_status}]
  Futures Curve       : {futures:.2f}  [{futures_status}]
  COT Managed Money   : {cot:.2f}  [{cot_status}]

Gewichteter Score: {weighted_score:.2f}
Wahrscheinlichkeit, dass Gaspreis steigt: {probability:.1f}%
===================================
""".strip()

    # =========================
    # Ausgabe
    # =========================
    print(output)

    with open("forecast_output.txt", "w", encoding="utf-8") as f:
        f.write(output + "\n")


if __name__ == "__main__":
    main()
