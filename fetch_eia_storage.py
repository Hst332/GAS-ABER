import requests
import os
import sys
import json

def fetch_eia_storage():
    api_key = os.environ.get("EIA_API_KEY")
    if not api_key:
        print("ERROR: Missing EIA_API_KEY", file=sys.stderr)
        return None

    url = (
        "https://api.eia.gov/v2/natural-gas/stor/wngsr/data/"
        "?frequency=weekly"
        "&facets[region][]=N5010"
        "&data[0]=value"
        "&sort[0][column]=period"
        "&sort[0][direction]=desc"
        "&offset=0&length=1"
        f"&api_key={api_key}"
    )

    print("DEBUG_URL:", url, file=sys.stderr)

    try:
        r = requests.get(url)
        print("DEBUG_STATUS:", r.status_code, file=sys.stderr)
        print("DEBUG_RAW_RESPONSE:", r.text, file=sys.stderr)

        r.raise_for_status()
        data = r.json()

        if "response" not in data or "data" not in data["response"]:
            print("ERROR: Unexpected EIA response format", file=sys.stderr)
            return None

        entry = data["response"]["data"][0]
        return entry["value"]

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return None


if __name__ == "__main__":
    value = fetch_eia_storage()
    if value is None:
        print("ERROR")
    else:
        print(value)
