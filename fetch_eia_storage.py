import csv
import requests

CSV_URL = "https://ir.eia.gov/ngs/wngsr.csv"

def fetch_storage():
    try:
        r = requests.get(CSV_URL, timeout=10)
        r.raise_for_status()

        lines = r.text.splitlines()
        reader = csv.DictReader(lines)

        latest = next(reader)  # first row = latest week

        # Try both common column names
        for col in ["Total Working Gas", "Working Gas"]:
            if col in latest:
                value = latest[col].replace(",", "").strip()
                return float(value)

        # fallback: no column found
        return -1

    except Exception:
        return -1


if __name__ == "__main__":
    value = fetch_storage()
    print(value)
