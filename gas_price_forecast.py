import argparse

def calculate_forecast(eia_storage, us_production, lng_feedgas, futures_curve, cot_managed_money):
    factors = {
        "EIA Storage": 0.25,
        "US Production": 0.20,
        "LNG Feedgas": 0.20,
        "Futures Curve": 0.20,
        "COT Managed Money": 0.15
    }

    values = {
        "EIA Storage": eia_storage,
        "US Production": us_production,
        "LNG Feedgas": lng_feedgas,
        "Futures Curve": futures_curve,
        "COT Managed Money": cot_managed_money
    }

    weighted_score = sum(values[f] * factors[f] for f in factors)

    max_possible = 10  # assumed scale of 0–10
    max_score = sum(max_possible * w for w in factors.values())
    prob_rise = (weighted_score / max_score) * 100

# --- SAFETY CLAMP ---
probability = float(probability)

# Falls Modell mit 0–1 arbeitet
if probability <= 1.0:
    probability *= 100.0

# Hard Clamp (absolute Sicherheit)
probability = max(0.0, min(probability, 100.0))

    return prob_rise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eia-storage", type=float, required=True)
    parser.add_argument("--us-production", type=float, required=True)
    parser.add_argument("--lng-feedgas", type=float, required=True)
    parser.add_argument("--futures-curve", type=float, required=True)
    parser.add_argument("--cot-managed-money", type=float, required=True)
    args = parser.parse_args()

    prob = calculate_forecast(
        args.eia_storage,
        args.us_production,
        args.lng_feedgas,
        args.futures_curve,
        args.cot_managed_money
    )

    print(f"Wahrscheinlichkeit, dass Gaspreis steigt: {prob:.1f}%")


if __name__ == "__main__":
    main()
