import csv
import requests

CSV_URL = "https://ir.eia.gov/ngs/wngsr.csv"

def fetch_storage():
    r = requests.get(CSV_URL)
    r.raise_for_status()

    lines = r.text.splitlines()
    reader = csv.DictReader(lines)

    # Wir suchen die Zeile "Total Lower 48" → das ist der landesweite Gesamtwert
    for row in reader:
        if row.get("Region") == "Total Lower 48":
            value = row.get("Current_Storage")
            if value is None or value == "":
                raise ValueError("Spalte Current_Storage nicht gefunden oder leer")
            print(value)   # WICHTIG: Nur Zahl ausgeben
            return

    raise ValueError("Region 'Total Lower 48' nicht gefunden")

if __name__ == "__main__":
    fetch_storage()
