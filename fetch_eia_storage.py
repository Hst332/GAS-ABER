# fetch_eia_storage.py
import csv
import requests

CSV_URL = "https://ir.eia.gov/ngs/wngsr.csv"

def fetch_storage():
    try:
        r = requests.get(CSV_URL)
        r.raise_for_status()
        lines = r.text.splitlines()
        reader = csv.reader(lines)

        # Überspringe die ersten paar Header-Zeilen bis zu den Spaltennamen
        headers = None
        for line in reader:
            if line and line[0] == "Region":
                headers = line
                break

        if not headers:
            raise ValueError("Regionen-Header nicht gefunden")

        # Suche nach der "Total"-Zeile
        for row in reader:
            if row and row[0].strip() == "Total":
                # Stocks in billion (Bcf) ist Spalte 1
                value = row[1].replace(',', '')
                return float(value)
        raise ValueError("Region 'Total Lower 48' nicht gefunden")
    except Exception as e:
        print("EIA fetch failed — debug:")
        raise e

if __name__ == "__main__":
    total = fetch_storage()
    print(total)

# === Compatibility wrapper for gas_price_forecast.py ===
import pandas as pd
from datetime import datetime

def load_storage_data(*args, **kwargs):
    value = fetch_storage()
    df = pd.DataFrame(
        {"Storage": [value]},
        index=[pd.Timestamp(datetime.utcnow().date())]
    )
    df.attrs["source"] = "eia_live"
    return df

