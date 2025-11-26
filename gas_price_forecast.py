import argparse
from datetime import datetime


# Gewichtungen (bleiben UNVERÄNDERT)
FACTORS = {
    "EIA Storage": 0.25,
    "US Production": 0.20,
    "LNG Feedgas": 0.20,
    "Futures Curve": 0.20,
    "COT Managed Money": 0.15
}


def clamp(value: float, min_v=0.0, max_v=10.0) -> float:
    return max(min_v, min(value, max_v))


def normalize_eia_storage(bcf: float) -> float:
    # 3000 bullish (10), 4000 neutral (5), 4400 bearish (0)
    score = 10 - ((bcf - 3000) / (4400 - 3000)) * 10
    return clamp(score)


def normalize_us_production(bcfd: float) -> float:
    # 85 bullish (10), 92 neutral (5), 100 bearish (0)
    score = 10 - ((bcfd - 85) / (100 - 85)) * 10
    return clamp(score)


def calculate_probability(values: dict) -> float:
    weighted_score = sum(values[f] * w for f, w in FACTORS.items())
    max_score = sum(10 * w for w in FACTORS.values())
    return round((weighted_score / max_score) * 100, 1)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--eia-storage", type=float, required=True)
    parser.add_argument("--us-production", type=float, required=True)
    parser.add_argument("--lng-feedgas", type=float, required=True)
    parser.add_argument("--futures-curve", type=float, required=True)
    parser.add_argument("--cot-managed-money", type=float, required=True)

    args = parser.parse_args()

    # ✅ NORMALISIERUNG
    values = {
        "EIA Storage": normalize_eia_storage(args.eia_storage),
        "US Production": normalize_us_production(args.us_production),
        "LNG Feedgas": clamp(args.lng_feedgas),
        "Futures Curve": clamp(args.futures_curve),
        "COT Managed Money": clamp(args.cot_managed_money),
    }

    probability = calculate_probability(values)

    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    output = f"""
NATURAL GAS PRICE FORECAST
===================================
Datum: {timestamp}

Normalisierte Faktoren (0–10):
  EIA Storage         : {values['EIA Storage']:.2f}  (Lower 48, Bcf)
  US Production       : {values['US Production']:.2f}  (Bcf/d)
  LNG Feedgas         : {values['LNG Feedgas']:.2f}
  Futures Curve       : {values['Futures Curve']:.2f}
  COT Managed Money   : {values['COT Managed Money']:.2f}

Wahrscheinlichkeit, dass Gaspreis steigt: {probability:.1f}%
===================================
""".strip()

    print(output)

    with open("forecast_output.txt", "w", encoding="utf-8") as f:
        f.write(output + "\n")


if __name__ == "__main__":
    main()
