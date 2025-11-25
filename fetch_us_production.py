# fetch_us_production.py
import os
import requests

EIA_API_KEY = os.environ.get("EIA_API_KEY")
if not EIA_API_KEY:
    raise ValueError("EIA_API_KEY environment variable is not set")

# Series ID für wöchentliche US-Natural-Gas-Produktion
SERIES_ID = "NG.N9010US2.W"
URL = f"https://api.eia.gov/v2/seriesid/{SERIES_ID}?api_key={EIA_API_KEY}"

def fetch_us_production():
    resp = requests.get(URL)
    resp.raise_for_status()
    data = resp.json()

    # Die letzte verfügbare Periode
    latest = data["response"]["data"][0]
    value = latest["value"]
    date = latest["period"]

    return value, date

if __name__ == "__main__":
    value, date = fetch_us_production()
    print(f"US_PRODUCTION={value}")
    # Optional: direkt ins GitHub Environment schreiben
    with open(os.environ.get("GITHUB_ENV", ".env"), "a") as f:
        f.write(f"US_PRODUCTION={value}\n")
