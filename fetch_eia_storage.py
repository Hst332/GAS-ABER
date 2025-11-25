import csv
import requests
from datetime import datetime

CSV_URL = "https://ir.eia.gov/ngs/wngsr.csv"

def fetch_storage():
    r = requests.get(CSV_URL)
    r.raise_for_status()

    lines = r.text.splitlines()
    reader = csv.DictReader(lines)

    latest = next(reader)

    total_gas = latest["Total Working Gas"]

    date = latest["Week Ending"]
    date = datetime.strptime(date, "%m/%d/%Y").strftime("%Y-%m-%d")

    return float(total_gas), date


if __name__ == "__main__":
    try:
        value, date = fetch_storage()
        print(value)
    except Exception as e:
        print("ERROR")
        raise e
