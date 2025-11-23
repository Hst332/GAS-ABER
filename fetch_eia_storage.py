import os
import requests
import json

def fetch_eia_storage():
    api_key = os.getenv("EIA_API_KEY", "")
    if not api_key:
        print("ERROR: Missing API key")
        return None, None

    # Correct v2 backward-compatible endpoint
    url = (
        "https://api.eia.gov/v2/seriesid/NG.WKST.S"
        f"?api_key={api_key}"
    )

    r = requests.get(url)

    print("DEBUG_URL:", url)
    print("DEBUG_STATUS:", r.status_code)
    print("DEBUG_RAW_RESPONSE:", r.text)

    if r.status_code != 200:
        return "ERROR", None

    try:
        data = r.json()

        # New v2 structure:
        # data → response → data → [{ period, value }]
        latest = data["response"]["data"][0]
        value = latest["value"]
        period = latest["period"]

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
