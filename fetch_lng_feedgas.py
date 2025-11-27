import csv
import requests

CSV_URL = "https://ir.eia.gov/ngs/ng_lng_feedgas.csv"

def fetch_lng_feedgas():
    try:
        r = requests.get(CSV_URL, timeout=10)
        r.raise_for_status()

        lines = r.text.splitlines()
        reader = csv.DictReader(lines)

        rows = [row for row in reader if row.get("LNG Feedgas (Bcf/d)")]
        if not rows:
            raise ValueError("Keine LNG Feedgas Daten gefunden")

        latest = rows[-1]
        feedgas = float(latest["LNG Feedgas (Bcf/d)"])

        # ✅ Skalierung 0–10
        # Typischer Bereich ~10–15 Bcf/d
        scaled = max(0.0, min(10.0, (feedgas / 15.0) * 10.0))

        print(f"{scaled:.2f}")
        return

    except Exception as e:
        print("0.0")

if __name__ == "__main__":
    fetch_lng_feedgas()
