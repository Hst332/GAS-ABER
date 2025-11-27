import csv
import requests

CSV_URL = "https://ir.eia.gov/ngd/ngd.csv"


def scale_lng_to_score(bcf_per_day: float) -> float:
    if bcf_per_day < 10:
        return 3.0
    elif bcf_per_day < 12:
        return 5.0
    elif bcf_per_day < 14:
        return 7.0
    else:
        return 9.0


def main():
    r = requests.get(CSV_URL, timeout=30)
    r.raise_for_status()

    lines = r.text.splitlines()
    reader = csv.reader(lines)

    header = next(reader)

    date_idx = header.index("Date")
    value_idx = header.index("U.S. Natural Gas Pipeline Imports to LNG Export Facilities (Bcf/d)")

    latest_value = None
    latest_date = None

    for row in reader:
        try:
            value = float(row[value_idx])
            date = row[date_idx]
            latest_value = value
            latest_date = date
            break
        except (ValueError, IndexError):
            continue

    if latest_value is None:
        raise RuntimeError("Keine LNG Feedgas Daten gefunden")

    score = scale_lng_to_score(latest_value)

    print(f"{score:.2f}")  # nur Score für Workflow
    print(f"LNG_FEEDGAS_RAW={latest_value}", file=open("lng_debug.txt", "w"))
    print(f"LNG_FEEDGAS_DATE={latest_date}", file=open("lng_debug.txt", "a"))


if __name__ == "__main__":
    main()
