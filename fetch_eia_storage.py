import csv
import requests

CSV_URL = "https://ir.eia.gov/ngs/wngsr.csv"

def debug_headers():
    r = requests.get(CSV_URL)
    r.raise_for_status()

    lines = r.text.splitlines()
    reader = csv.reader(lines)

    headers = next(reader)
    print("HEADERS:", headers)

if __name__ == "__main__":
    debug_headers()
