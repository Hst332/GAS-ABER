import os
import requests

API_KEY = os.getenv("EIA_API_KEY")

def fetch_eia_storage():
    url = (
        "https://api.eia.gov/v2/natural-gas/stor/wngsr/data/"
        f"?api_key={API_KEY}"
        "&frequency=weekly"
        "&data=value"
        "&sort=period:desc"
        "&offset=0"
        "&length=1"
    )

    r = requests.get(url)
    r.raise_for_status()
    data = r.json()

    # Defensive check
    if "response" not in data or "data" not in data["response"]:
        raise ValueError("Unerwartetes EIA API Format")

    latest = data["response"]["data"][0]
    value = latest["value"]
    date = latest["period"]

    return value, date


if __name__ == "__main__":
    value, date = fetch_eia_storage()
    print(f"EIA_STORAGE_VALUE={value}")
    print(f"EIA_STORAGE_DATE={date}")
