# gas_price_forecast.py
import argparse
from datetime import datetime

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eia-storage", type=float, required=True)
    parser.add_argument("--us-production", type=float, required=True)
    parser.add_argument("--lng-feedgas", type=float, required=True)
    parser.add_argument("--futures-curve", type=float, required=True)
    parser.add_argument("--cot-managed-money", type=float, required=True)
    args = parser.parse_args()

    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    # Gewichtung (bewusst konservativ)
    weights = {
        "storage": -0.30,
        "production": -0.25,
        "lng": 0.25,
        "futures": 0.15,
        "cot": 0.05,
    }

    score = (
        args.eia_storage * weights["storage"] +
        args.us_production * weights["production"] +
        args.lng_feedgas * weights["lng"] +
        args.futures_curve * weights["futures"] +
        args.cot_managed_money * weights["cot"]
    )

    probability = max(0.0, min(100.0, 50 + score))

    output = f"""===================================
      NATURAL GAS PRICE FORECAST
===================================
Datum: {timestamp}

Eingabewerte (0–10 Skala):
  EIA Storage         : {args.eia_storage:.2f}
  US Production       : {args.us_production:.2f}
  LNG Feedgas         : {args.lng_feedgas:.2f}
  Futures Curve       : {args.futures_curve:.2f}
  COT Managed Money   : {args.cot_managed_money:.2f}

Gewichteter Score: {score:.2f}
Wahrscheinlichkeit, dass Gaspreis steigt: {probability:.1f}%
===================================
"""

    # 🔒 HIER IST DER ENTSCHEIDENDE TEIL
    with open("forecast_output.txt", "w", encoding="utf-8") as f:
        f.write(output)

    # weiterhin Console Output
    print(output)

if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
