import requests
import os

# Beispiel API für US Production, EIA API v2
API_KEY = os.environ.get("EIA_API_KEY")
URL = f"https://api.eia.gov/v2/natural-gas/production/data/?api_key={API_KEY}&frequency=weekly&sort[0][column]=period&sort[0][direction]=desc&length=1"

def fetch_us_production():
    r = requests.get(URL)
    r.raise_for_status()
    data = r.json()

    latest = data.get("response", {}).get("data", [{}])[0]
    value = latest.get("value")
    if value is None:
        raise ValueError("US Production Wert nicht gefunden")

    os.environ["US_PRODUCTION"] = str(value)
    with open(os.environ["GITHUB_ENV"], "a") as f:
        f.write(f"US_PRODUCTION={value}\n")

if __name__ == "__main__":
    fetch_us_production()
