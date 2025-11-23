import os
import requests

def fetch_eia_storage():
    api_key = os.environ.get("EIA_API_KEY")
    if not api_key:
        raise ValueError("EIA_API_KEY environment variable is missing")

    url = f"https://api.eia.gov/v2/seriesid/NG.WKST.W?api_key={api_key}"
    
    r = requests.get(url)
    if r.status_code != 200:
        print("DEBUG_STATUS:", r.status_code)
        print("DEBUG_RAW_RESPONSE:", r.text)
        return "ERROR", None

    data = r.json()
    try:
        latest_point = data['response']['data'][0]
        value = latest_point['value']
        date = latest_point['period']
        return value, date
    except Exception as e:
        print("ERROR parsing response:", e)
        return "ERROR", None

if __name__ == "__main__":
    value, date = fetch_eia_storage()
    if value == "ERROR":
        print("ERROR")
    else:
        print(value)
