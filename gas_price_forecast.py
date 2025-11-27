import argparse
from datetime import datetime


WEIGHTS = {
    "eia_storage": 0.25,
    "us_production": 0.20,
    "lng_feedgas": 0.20,
    "futures_curve": 0.20,
    "cot_managed_money": 0.15,
}


def normalize(value, min_val=0.0, max_val=10.0):
    """
    Bringt einen Wert stabil auf 0–10
    """
    if value < min_val:
        return min_val
    if value > max_val:
        return max_val
    return value


def main():
    parser = argparse.ArgumentParser(description="Natural Gas Price Forecast")

    parser.add_argument("--eia-storage", type=float, required=True)
    parser.add_argument("--us-production", type=float, required=True)
    parser.add_argument("--lng-feedgas", type=float, required=True)
    parser.add_argument("--futures-curve", type=float, required=True)
    parser.add_argument("--cot-managed-money", type=float, required=True)

    args = parser.parse_args()

    # -----------------------------
    # NORMALISIERUNG (entscheidend!)
    # -----------------------------
    eia_score = normalize(args.eia_storage)
    prod_score = normalize(args.us_production)
    lng_score = normalize(args.lng_feedgas)
    curve_score = normalize(args.futures_curve)
    cot_score = normalize(args.cot_managed_money)

    weighted_score = (
        eia_score * WEIGHTS["eia_storage"]
        + prod_score * WEIGHTS["us_production"]
        + lng_score * WEIGHTS["lng_feedgas"]
        + curve_score * WEIGHTS["futures_curve"]
        + cot_score * WEIGHTS["cot_managed_money"]
    )

    probability = weighted_score * 10.0  # 0–100 %

    now_utc = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    output = (
        "\n===================================\n"
        "      NATURAL GAS PRICE FORECAST   \n"
        "===================================\n"
        f"Datum: {now_utc} UTC\n\n"
        "Eingabewerte (0–10 Skala):\n"
        f"  EIA Storage         : {eia_score:.2f}  [aktuell]\n"
        f"  US Production       : {prod_score:.2f}  [aktuell]\n"
        f"  LNG Feedgas         : {lng_score:.2f}  [aktuell]\n"
        f"  Futures Curve       : {curve_score:.2f}  [Platzhalter]\n"
        f"  COT Managed Money   : {cot_score:.2f}  [Platzhalter]\n\n"
        f"Gewichteter Score: {weighted_score:.2f}\n"
        f"Wahrscheinlichkeit, dass Gaspreis steigt: {probability:.1f}%\n"
        "===================================\n"
    )

    print(output)

    # -----------------------------
    # DATEI SCHREIBEN (immer!)
    # -----------------------------
    with open("forecast_output.txt", "w", encoding="utf-8") as f:
        f.write(output)


if __name__ == "__main__":
    main()
