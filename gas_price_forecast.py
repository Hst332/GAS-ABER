import os
from datetime import datetime

FORECAST_OUTPUT = "forecast_output.txt"

def load_input(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return float(f.read().strip())
    else:
        return 0.0

def run_forecast(eia_storage, eia_date,
                 us_prod, us_date,
                 lng_feedgas, lng_date,
                 futures_curve, cot_money):

    # Gewichteter Score (Beispielberechnung)
    weights = [1.5, 1.0, 1.0, 1.0, 1.0]
    inputs = [eia_storage, us_prod, lng_feedgas, futures_curve, cot_money]
    weighted_score = sum(w * v for w, v in zip(weights, inputs))

    # Wahrscheinlichkeit, dass Gaspreis steigt
    prob_rise = min(weighted_score, 100.0)  # Max 100%

    # Datum UTC
    now = datetime.utcnow()

    # Ausgabe
    output = [
        "===================================",
        "      NATURAL GAS PRICE FORECAST",
        "===================================",
        f"Datum: {now} UTC",
        "",
        "Eingabewerte (0–10 Skala):",
        f"  EIA Storage         : {eia_storage:.2f}  [aktuell vom {eia_date}]",
        f"  US Production       : {us_prod:.2f}  [aktuell vom {us_date}]",
        f"  LNG Feedgas         : {lng_feedgas:.2f}  [aktuell vom {lng_date}]" if lng_feedgas > 0 else
        f"  LNG Feedgas         : {lng_feedgas:.2f}  [keine neue Meldung, letzte Woche]",
        f"  Futures Curve       : {futures_curve:.2f}  [Platzhalter]",
        f"  COT Managed Money   : {cot_money:.2f}  [Platzhalter]",
        "",
        f"Gewichteter Score: {weighted_score:.2f}",
        f"Wahrscheinlichkeit, dass Gaspreis steigt: {prob_rise:.1f}%",
        "==================================="
    ]

    text = "\n".join(output)
    print(text)

    with open(FORECAST_OUTPUT, "w", encoding="utf-8") as f:
        f.write(text)

def main():
    eia_storage = load_input("eia_storage.txt")
    us_prod = load_input("us_production.txt")
    lng_feedgas = load_input("lng_feedgas.txt")
    futures_curve = 0.0
    cot_money = 0.0

    # Optional: Datum aus Umgebungsvariablen
    eia_date = os.getenv("EIA_DATE", "unbekannt")
    us_date = os.getenv("US_PROD_DATE", "unbekannt")
    lng_date = os.getenv("LNG_DATE", "unbekannt")

    run_forecast(eia_storage, eia_date,
                 us_prod, us_date,
                 lng_feedgas, lng_date,
                 futures_curve, cot_money)

if __name__ == "__main__":
    main()
