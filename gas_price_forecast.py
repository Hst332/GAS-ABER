"""
gas_price_forecast.py
Einfaches Tool zur Einschätzung der Wahrscheinlichkeit, dass Gaspreise steigen.
Schritt 1: Nur EIA Storage berücksichtigt.
"""

def main():
    # Gewichtung
    weight_eia = 0.25  # 25 %

    # Maximalwert für Normierung
    max_value = 10  # Beispiel: Werte zwischen 0–10

    # Eingabe des aktuellen Faktors
    try:
        eia_value = float(input("EIA Storage (aktueller Wert, höher = stärkerer Preisdruck nach oben): "))
    except ValueError:
        print("Ungültige Eingabe! Bitte eine Zahl eingeben.")
        return

    # Gewichteten Score berechnen
    weighted_score = eia_value * weight_eia
    max_score = max_value * weight_eia
    prob_rise = (weighted_score / max_score) * 100

    # Ergebnis ausgeben
    print(f"\nWahrscheinlichkeit, dass Gaspreis steigt (nur EIA Storage berücksichtigt): {prob_rise:.1f}%")

if __name__ == "__main__":
    main()
