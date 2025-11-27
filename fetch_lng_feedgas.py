import requests
import csv
from datetime import datetime

CSV_URL = "https://www.eia.gov/dnav/ng/hist/ng_feedgas_weekly.csv"  # Beispiel-URL, bitte prüfen
OUTPUT_FILE = "lng_feedgas.txt"

def fetch_lng_feedgas():
    try:
        r = requests.get(CSV_URL)
        r.raise_for_status()
        lines = r.text.splitlines()
        reader = csv.reader(lines)
        next(reader)  # Header überspringen

        latest_value = None
        latest_date = None
        for row in reader:
            if len(row) < 2:
                continue
            date_str, value_str = row[0], row[1]
            try:
                value = float(value_str.replace(",", ""))
                latest_value = value
                latest_date = datetime.strptime(date_str, "%m/%d/%Y").date()
                break  # wir nehmen die erste Zeile = neueste Woche
            except:
                continue

        if latest_value is None:
            raise ValueError("Keine LNG-Feedgas-Daten gefunden")

        with open(OUTPUT_FILE, "w") as f:
            f.write(str(latest_value))
        print(f"{latest_value}  ({latest_date})")
        return latest_value, latest_date

    except requests.exceptions.RequestException as e:
        print("Fehler beim Abrufen von LNG-Feedgas:", e)
        return 0.0, None  # Platzhalter, falls Abruf fehlschlägt

def main():
    fetch_lng_feedgas()

if __name__ == "__main__":
    main()
