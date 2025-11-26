import argparse

def calculate_probability(
    eia_storage,
    us_production,
    lng_feedgas,
    futures_curve,
    cot_managed_money
):
    # --- einfache, stabile Heuristik ---
    score = (
        -0.3 * eia_storage +
         0.4 * us_production +
         0.3 * lng_feedgas +
         0.2 * futures_curve +
         0.1 * cot_managed_money
    )

    # Lineare Projektion
    probability = score

    # --- SAFETY NORMALIZATION ---
    probability = float(probability)

    if probability <= 1.0:
        probability *= 100.0

    probability = max(0.0, min(probability, 100.0))

    return probability


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eia-storage", type=float, required=True)
    parser.add_argument("--us-production", type=float, required=True)
    parser.add_argument("--lng-feedgas", type=float, required=True)
    parser.add_argument("--futures-curve", type=float, required=True)
    parser.add_argument("--cot-managed-money", type=float, required=True)

    args = parser.parse_args()

    prob_rise = calculate_probability(
        args.eia_storage,
        args.us_production,
        args.lng_feedgas,
        args.futures_curve,
        args.cot_managed_money
    )

    output_line = f"Wahrscheinlichkeit, dass Gaspreis steigt: {prob_rise:.1f}%"

    # ✅ 1. In Konsole ausgeben
    print(output_line)

    # ✅ 2. Zusätzlich als Datei speichern
    with open("forecast_output.txt", "w", encoding="utf-8") as f:
        f.write(output_line + "\n")
