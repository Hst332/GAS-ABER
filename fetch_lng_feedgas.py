import requests
import os

LAST_FILE = "lng_feedgas_last.txt"

# Beispiel-URL (ersetze bei Bedarf durch die korrekte aktuelle EIA-Quelle)
CSV_URL = "https://ir.eia.gov/ngd/ngd_weekly.csv"

def fetch_lng_feedgas():
    try:
        r = requests.get(CSV_URL, timeout=10)
        r.raise_for_status()
        lines = r.text.splitlines()
        # CSV auswerten – Annahme: erste Datenzeile nach Header enthält Wert und Datum
        header = lines[0].split(",")
        value_line = lines[1].split(",")
        value = float(value_line[1])
        date = value_line[0]
        # letzten Wert sichern
        with open(LAST_FILE, "w") as f:
            f.write(f"{value},{date}")
        return value, date
    except Exception:
        # fallback auf letzten bekannten Wert
        if os.path.exists(LAST_FILE):
            with open(LAST_FILE, "r") as f:
                last_value, last_date = f.read().split(",")
                return float(last_value), last_date
        else:
            # kein historischer Wert vorhanden
            return 0.0, None

def main():
    value, date = fetch_lng_feedgas()
    print(f"{value}")

if __name__ == "__main__":
    main()
