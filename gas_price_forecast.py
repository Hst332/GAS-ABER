import argparse

def main():
    parser = argparse.ArgumentParser(description="NatGas Forecast basierend auf gewichteten Fundamentaldaten")

    parser.add_argument("--eia-storage", type=float, required=True)
    parser.add_argument("--us-production", type=float, required=True)
    parser.add_argument("--lng-feedgas", type=float, required=True)
    parser.add_argument("--futures-curve", type=float, required=True)
    parser.add_argument("--cot-managed-money", type=float, required=True)

    args = parser.parse_args()

    # Gewichtungen
    factors = {
        "EIA Storage": 0.25,
        "US Production": 0.20,
        "LNG Feedgas": 0.20,
        "Futures Curve": 0.20,
        "COT Managed Money": 0.15
    }

    # Werte aus CLI
    values = {
        "EIA Storage": args.eia_storage,
        "US Production": args.us_production,
        "LNG Feedgas": args.lng_feedgas,
        "Futures Curve": args.futures_curve,
        "COT Managed Money": args.cot_managed_money
    }

    # Score berechnen
    weighted_score = sum(values[f] * w for f, w in factors.items())
    max_possible = 10  # Wir nehmen an: Faktorwerte 0–10

    max_score = sum(max_possible * w for w in factors.values())
    prob_rise = (weighted_score / max_score) * 100

    print("===================================")
    print("      NATURAL GAS PRICE FORECAST   ")
    print("===================================")
    print("Eingabewerte:")
    for k, v in values.items():
        print(f"  {k:20}: {v}")

    print("\nGewichteter Score:", round(weighted_score, 2))
    print(f"Wahrscheinlichkeit, dass Gaspreis steigt: {prob_rise:.1f}%")
    print("===================================")


if __name__ == "__main__":
    main()
