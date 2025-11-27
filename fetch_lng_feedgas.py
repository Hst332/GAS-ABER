import csv
import requests


URLS = [
    # falls EIA den wöchentlichen Feed wieder anbietet
    "https://ir.eia.gov/ngd/ngd_weekly.csv",
    # stabilste bekannte Fallback-Quelle
    "https://ir.eia.gov/ngd/ngd.csv",
]


TARGET_COLUMN = "U.S. Natural Gas Pipeline Imports to LNG Export Facilities (Bcf/d)"


def scale_lng_to_score(bcf_per_day: float) -> float:
    """
    Sehr konservative Normalisierung (0–10)
    """
    if bcf_per_day <= 0:
        return 0.0
    elif bcf_per_day < 8:
        return 3.0
    elif bcf_per_day < 11:
        return 5.0
    elif bcf_per_day < 13:
        return 7.0
    else:
        return 9.0


def try_fetch(url: str):
    r = requests.get(url, timeout=20)
    r.raise_for_status()

    reader = csv.DictReader(r.text.splitlines())

    for row in reader:
        try:
            value = float(row[TARGET_COLUMN])
            date = row.get("Date", "unbekannt")
            return value, date
        except Exception:
            continue

    raise ValueError("Keine verwertbaren LNG-Daten gefunden")


def main():
    for url in URLS:
        try:
            value, date = try_fetch(url)
            score = scale_lng_to_score(value)

            print(f"{score:.2f}")

            with open("lng_feedgas_debug.txt", "w", encoding="utf-8") as f:
                f.write(f"Quelle: {url}\n")
                f.write(f"Datum: {date}\n")
                f.write(f"LNG Feedgas (Bcf/d): {value}\n")
                f.write(f"Score: {score}\n")

            return

        except Exception:
            continue

    # ✅ stabiler Fallback — Workflow läuft weiter
    print("0.00")
    with open("lng_feedgas_debug.txt", "w", encoding="utf-8") as f:
        f.write("Keine LNG-Quelle erreichbar – Fallback 0.0 verwendet\n")


if __name__ == "__main__":
    main()
