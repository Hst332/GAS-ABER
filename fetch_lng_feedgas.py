import csv
import requests


CSV_URL = "https://ir.eia.gov/ngd/ngd_weekly.csv"


def scale_lng_to_score(bcf_per_day: float) -> float:
    """
    Realistische Skalierung US LNG Feedgas
    """
    if bcf_per_day < 10:
        return 3.0
    elif bcf_per_day < 12:
        return 5.0
    elif bcf_per_day < 14:
        return 7.0
    else:
        return 9.0


def fetch_lng_feedgas():
    r = requests.get(CSV_URL, timeout=30)
    r.raise_for_status()

    lines = r.text.splitlines()
    reader = csv.DictReader(lines)

    for row in reader:
        try:
            value = float(
                row["U.S. Natural Gas Pipeline Imports to LNG Export Facilities (Bcf/d)"]
            )
            date = row["Date"]
            return value, date
        except (KeyError, ValueError):
            continue

    raise RuntimeError("Keine gültigen LNG Feedgas Daten gefunden")


def main():
    value, date = fetch_lng_feedgas()
    score = scale_lng_to_score(value)

    # NUR Score für Forecast
    print(f"{score:.2f}")

    # Debug-Datei (optional, zerstört nichts)
    with open("lng_feedgas_debug.txt", "w", encoding="utf-8") as f:
        f.write(f"Datum: {date}\n")
        f.write(f"LNG Feedgas (Bcf/d): {value}\n")
        f.write(f"Score (0–10): {score}\n")


if __name__ == "__main__":
    main()
