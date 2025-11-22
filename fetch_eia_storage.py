import os
import requests
import json

EIA_URL = (
    "https://api.eia.gov/v2/natural-gas/storage/wngstp/data/"
    "?frequency=weekly&data=value&sort=period:desc&offset=0&length=1"
)

API_KEY = os.getenv("EIA_API_KEY")

def fetch_eia_storage():
    if not API_KEY:
        raise ValueError("EIA_API_KEY not found in environment")

    url = f"{EIA_URL}&api_key={API_KEY}"
    print("Requesting:", url.replace(API_KEY, "***"))  # debug

    r = requests.get(url)
    r.raise_for_status()

    data = r.json()
    row = data["response"]["data"][0]

    value = row["value"]
    date = row["period"]

    return value, date


if __name__ == "__main__":
    try:
        value, date = fetch_eia_storage()
        print(f"EIA Storage Latest: {value} Bcf ({date})")
    except Exception as e:
        print("ERROR:", e)

