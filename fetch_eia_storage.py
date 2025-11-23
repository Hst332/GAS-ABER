import os
import requests
import json

def fetch_eia_storage():
    api_key = os.getenv("EIA_API_KEY", "")
    if not api_key:
        print("ERROR: Missing API key")
        return None, None

    url = (
        "https://api.eia.gov/v2/natural-gas/wngsr/data/"
        "?frequency=weekly"
        "&sort[0][column]=period"
        "&sort[0][direction]=desc"
        "&offset=0"
        "&length=1"
        f"&api_key={api_key}"
    )

    print(f"DEBUG_URL: {url}")

    r = requests.get(url)
    print(f"DEBUG_STATUS: {r.status_code}")
    print(f"DEBUG_RAW_RESPONSE: {r.text}")

    if r.status_code != 200:
        return "ERROR", None

    try:
        data = r.json()
        value = data["response"]["data"][0]["value"]
        period = data["response"]["data"][0]["period"]
        return value, period
    except Exception as e:
        print("PARSE_ERROR:", e)
        return "ERROR", None


if __name__ == "__main__":
    value, date = fetch_eia_storage()

    if value == "ERROR":
        print("ERROR")
    else:
        print(value)
