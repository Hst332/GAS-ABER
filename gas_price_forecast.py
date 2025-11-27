import os
from datetime import datetime

def load_input(file_path):
    with open(file_path, "r") as f:
        return float(f.read().strip())

def run_forecast(eia_storage, eia_date,
                 us_prod, us_date,
                 lng_feedgas, lng_date,
                 futures_curve, cot_money):

    # Status für Anzeige
    storage_status = f"[Stand: {eia_date}]" if eia_date else "[Datum unbekannt]"
    us_status = f"[Stand: {us_date}]" if us_date else "[Datum unbekannt]"
    lng_status = f"[letzter bekannter Wert: {lng_date}]" if lng_date else "[keine Meldung]"

    # Score Berechnung (vereinfachtes Beispiel)
    score = eia_storage * 0.3 + us_prod * 0.3 + lng_feedgas * 0.2 + futures_curve * 0.1 + cot_money * 0.1
    prob_rise = min(max(score * 10, 0), 100)  # Prozent zwischen 0–100

    output = f"""
===================================
      NATURAL GAS PRICE FORECAST
===================================
Datum: {datetime.utcnow()} UTC

Eingabewerte (0–10 Skala):
  EIA Storage         : {eia_storage}  {storage_status}
  US Production       : {us_prod}  {us_status}
  LNG Feedgas         : {lng_feedgas}  {lng_status}
  Futures Curve       : {futures_curve}  [Platzhalter]
  COT Managed Money   : {cot_money}  [Platzhalter]

Gewichteter Score: {score:.2f}
Wahrscheinlichkeit, dass Gaspreis steigt: {prob_rise:.1f}%
===================================
"""
    # Ausgabe speichern
    with open("forecast_output.txt", "w", encoding="utf-8") as f:
        f.write(output)
    print(output)

def main():
    eia_storage = load_input("eia_storage.txt")
    us_prod = load_input("us_production.txt")
    lng_feedgas = load_input("lng_feedgas.txt")
    futures_curve = 0.0
    cot_money = 0.0

    # Für Demo: Datum aus den Dateien (falls separat gespeichert)
    # sonst None
    eia_date = os.getenv("EIA_DATE", None)
    us_date = os.getenv("US_PROD_DATE", None)
    lng_date = os.getenv("LNG_DATE", None)

    run_forecast(eia_storage, eia_date,
                 us_prod, us_date,
                 lng_feedgas, lng_date,
                 futures_curve, cot_money)

if __name__ == "__main__":
    main()
