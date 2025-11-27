#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from datetime import datetime

def load_input(file_path, default_value=0.0):
    """Liest einen Wert aus einer Textdatei oder liefert default."""
    try:
        with open(file_path, "r") as f:
            value = f.read().strip()
            return float(value)
    except Exception:
        return default_value

def normalize_score(value, min_val=0.0, max_val=100.0):
    """Normiert die Werte auf 0–10 Skala"""
    return max(0.0, min(10.0, (value - min_val) / (max_val - min_val) * 10))

def compute_weighted_score(eia, us_prod, lng, futures, cot):
    """Ein sehr einfaches Beispiel zur Berechnung eines Scores"""
    # Gewichte können angepasst werden
    w_eia = 0.3
    w_us = 0.3
    w_lng = 0.2
    w_fut = 0.1
    w_cot = 0.1
    score = (eia * w_eia + us_prod * w_us + lng * w_lng + futures * w_fut + cot * w_cot)
    return score

def compute_probability(score):
    """Rechnet Score in Wahrscheinlichkeit 0–100% um"""
    return min(100.0, max(0.0, score * 10))  # max 100%

def main():
    parser = argparse.ArgumentParser(description="Natural Gas Price Forecast")
    parser.add_argument("--eia-storage", type=float, required=True)
    parser.add_argument("--us-production", type=float, required=True)
    parser.add_argument("--lng-feedgas", type=float, required=True)
    parser.add_argument("--futures-curve", type=float, required=True)
    parser.add_argument("--cot-managed-money", type=float, required=True)
    args = parser.parse_args()

    # Normalisierung auf 0–10 Skala
    eia_score = normalize_score(args.eia_storage, min_val=0, max_val=4000)
    us_score = normalize_score(args.us_production, min_val=0, max_val=100)
    lng_score = normalize_score(args.lng_feedgas, min_val=0, max_val=10)
    futures_score = normalize_score(args.futures_curve, min_val=0, max_val=10)
    cot_score = normalize_score(args.cot_managed_money, min_val=0, max_val=10)

    # Weighted Score
    weighted_score = compute_weighted_score(eia_score, us_score, lng_score, futures_score, cot_score)
    prob_rise = compute_probability(weighted_score)

    # Statusanzeige für die Werte
    eia_status = "[aktuell vom Datum]"
    us_status = "[aktuell vom Datum]"
    lng_status = "[keine neue Meldung, letzte Woche]" if args.lng_feedgas == 0 else "[aktuell]"

    # Ausgabe
    output = f"""\
===================================
      NATURAL GAS PRICE FORECAST
===================================
Datum: {datetime.utcnow()} UTC

Eingabewerte (0–10 Skala):
  EIA Storage         : {eia_score:.2f}  {eia_status}
  US Production       : {us_score:.2f}  {us_status}
  LNG Feedgas         : {lng_score:.2f}  {lng_status}
  Futures Curve       : {futures_score:.2f}  [Platzhalter]
  COT Managed Money   : {cot_score:.2f}  [Platzhalter]

Gewichteter Score: {weighted_score:.2f}
Wahrscheinlichkeit, dass Gaspreis steigt: {prob_rise:.1f}%
===================================
"""
    print(output)

    # In forecast_output.txt speichern
    with open("forecast_output.txt", "w", encoding="utf-8") as f:
        f.write(output)

if __name__ == "__main__":
    main()
