import csv
import requests

CSV_URL = "https://ir.eia.gov/ngs/wngsr.csv"

r = requests.get(CSV_URL)
r.raise_for_status()

lines = r.text.splitlines()
reader = csv.DictReader(lines)

regions = set(row.get("Region") for row in reader)

print("Gefundene Regionsnamen:")
for reg in regions:
    print(reg)
