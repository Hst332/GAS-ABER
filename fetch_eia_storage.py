import os
import requests

API_KEY = os.getenv("EIA_API_KEY")

def fetch_eia_storage():
    url = (
        "https://api.eia.gov/v2/natural-gas/stor/wngsr/data/"
        "?frequency=weekly"
        "&facets[region][]=N5010"       # <<< WICHTIG: Lower 48 total
        "&data=value"
        "&sort=period:desc"
        "&offset=0"
        "&length=1"
        f"&api_key={API_KEY}"
    )

    r = requests.get(url)
    r.raise_for_status()
    data = r.json()

    if "response" not in data or "data" not in data["response"]:
        raise ValueError("Unerwartetes API Format: " + str(data))

    entry = data["response"]["data"][0]
    value = entry["value"]
    date = entry["period"]

    return value, date


if __name__ == "__main__":
    value, date = fetch_eia_storage()
    print(f"EIA_STORAGE_VALUE={value}")
    print(f"EIA_STORAGE_DATE={date}")
