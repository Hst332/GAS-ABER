import csv
import requests
import os

# EIA WNGSR CSV URL (aktuellste Daten)
CSV_URL = "https://www.eia.gov/ngs/ngs.csv"  # Beispiel, bitte prüfen

def fetch_storage():
    r = requests.get(CSV_URL)
    r.raise_for_status()

    reader = csv.DictReader(r.text.splitlines())
    latest_row = next(reader)  # Erste Zeile = aktuellste

    # Spaltenname anpassen, z.B. "Total Working Gas"
    value = latest_row.get("Working Gas in Underground Storage (Bcf)")
    if value is None:
        raise ValueError("Spalte 'Working Gas in Underground Storage (Bcf)' nicht gefunden")
    
    os.environ["EIA_STORAGE"] = str(value)
    with open(os.environ["GITHUB_ENV"], "a") as f:
        f.write(f"EIA_STORAGE={value}\n")

if __name__ == "__main__":
    fetch_storage()
