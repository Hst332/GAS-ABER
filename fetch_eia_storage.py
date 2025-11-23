import os
import requests
import json

def fetch_eia_storage():
    api_key = os.getenv("EIA_API_KEY", "")
    if not api_key:
        print("ERROR: Missing API key")
        return None, None

    # Old but STILL ACTIVE + correct
    url = (
        "https://api.eia.gov/series/"
        f"?api_key={api_key}"
        "&series_id=NG.WKST.S"   # Total Working Gas in Storage (weekly)
    )

    r = requests.get(url)

    print("DEBUG_URL:", url)
    print("DEBUG_STATUS:", r.status_code)
    print("DEBUG_RAW_RESPONSE:", r.text)

    if r.status_code != 200:
        return "ERROR", None

    try:
        data = r.json()
        value = data["series"][0]["data"][0][1]  # Most recent storage value
        period = data["series"][0]["data"][0][0]  # Week
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
