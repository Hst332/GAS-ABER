import argparse
from datetime import datetime


# Gewichtungen (fest, wie von dir definiert)
FACTORS = {
    "EIA Storage": 0.25,
    "US Production": 0.20,
    "LNG Feedgas": 0.20,
    "Futures Curve": 0.20,
    "COT Managed Money": 0.15
}


def calculate_probability(values: dict) -> float:
    """
    Berechnet die Wahrscheinlichkeit (0–100 %),
    dass der Gaspreis steigt.
    Erwartungsbereich der Werte: 0–10
    """
    weighted_score = sum(values[f] * w for f, w in FACTORS.items())

    max_value_per_factor = 10.0
    max_score = sum(max_value_per_factor * w for w in FACTORS.values())

    probability = (weighted_score / max_score) * 100
    return round(probability, 1)


def main():
    parser = argparse.ArgumentParser(description="Natural Gas Price Forecast")

    parser.add_argument("--eia-storage", type=float, required=True)
    parser.add_argument("--us-production", type=float, required=True)
    parser.add_argument("--lng-feedgas", type=float, required=True)
    parser.add_argument("--futures-curve", type=float, required=True)
    parser.add_argument("--cot-managed-money", type=float, required=True)

    args = parser.parse_args()

    # Eingabewerte sammeln
    values = {
        "EIA Storage": args.eia_storage,
        "US Production": args.us_production,
        "LNG Feedgas": args.lng_feedgas,
        "Futures Curve": args.futures_curve,
        "COT Managed Money": args.cot_managed_money
    }

    probability = calculate_probability(values)

    # Zeitstempel (UTC, stabil für GitHub Actions)
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    # Datenstand (bewusst textuell – nichts wird hier „erraten“)
    data_status = {
        "EIA Storage": "aktuell (Woche zum 26.11.)",
        "US Production": "aktuell (letzte verfügbare Woche)",
        "LNG Feedgas": "noch Platzhalter (nicht automatisch geladen)",
        "Futures Curve": "noch Platzhalter (nicht automatisch geladen)",
        "COT Managed Money": "noch Platzhalter (nicht automatisch geladen)"
    }

    output = f"""
NATURAL GAS PRICE FORECAST
===================================
Datum: {timestamp}

Eingabewerte (0–10 Skala):
  EIA Storage         : {values['EIA Storage']}  [{data_status['EIA Storage']}]
  US Production       : {values['US Production']}  [{data_status['US Production']}]
  LNG Feedgas         : {values['LNG Feedgas']}  [{data_status['LNG Feedgas']}]
  Futures Curve       : {values['Futures Curve']}  [{data_status['Futures Curve']}]
  COT Managed Money   : {values['COT Managed Money']}  [{data_status['COT Managed Money']}]

Wahrscheinlichkeit, dass Gaspreis steigt: {probability:.1f}%
===================================
""".strip()

    # Konsole
    print(output)

    # Datei (wird von deinem Workflow genutzt)
    with open("forecast_output.txt", "w", encoding="utf-8") as f:
        f.write(output + "\n")


if __name__ == "__main__":
    main()
