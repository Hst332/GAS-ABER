import csv
import requests

CSV_URL = "https://ir.eia.gov/ngs/wngsr.csv"

def fetch_storage():
    r = requests.get(CSV_URL)
    r.raise_for_status()

    # CSV einlesen
    lines = r.text.splitlines()
    reader = csv.reader(lines)

    # bis zur Zeile mit Region=="Total" suchen
    for row in reader:
        if len(row) > 1 and row[0].strip() == "Total":
            # aktueller Wert steht in Spalte 1: Beispiel: "3,946"
            raw_value = row[1].replace(",", "")
            return float(raw_value)

    raise ValueError("Total row not found")

if __name__ == "__main__":
    try:
        value = fetch_storage()
        print(value)
    except Exception as e:
        print("ERROR")
        print(str(e))
