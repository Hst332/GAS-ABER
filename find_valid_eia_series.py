import os
import requests
import time

API_KEY = os.getenv("EIA_API_KEY")
if not API_KEY:
    raise RuntimeError("Bitte EIA_API_KEY als Environment-Variable setzen")

candidates = [
    "NG.WKST.S",  # Working Gas in Storage, Weekly – klassische SeriesID
    "NG.WKST.W",
    "NG.WKST",
    "NG.WST",
    "NG.WST.W",
    "NG.WKST.SA",
    "NG.WKST.USA"
]

def test_series(series_id):
    url = f"https://api.eia.gov/v2/seriesid/{series_id}?api_key={API_KEY}"
    try:
        r = requests.get(url, timeout=10)
    except Exception as e:
        return False, f"request‐error: {e}"
    status = r.status_code
    text = r.text[:500]
    if status == 200:
        try:
            j = r.json()
            if "response" in j and "data" in j["response"] and len(j["response"]["data"]) > 0:
                return True, f"OK: value={j['response']['data'][0].get('value')} period={j['response']['data'][0].get('period')}"
            else:
                return False, "200 but no data"
        except Exception as e:
            return False, f"parse‐error: {e}"
    else:
        return False, f"status {status}: {text}"

def main():
    print("Testing candidate SeriesIDs:")
    found_valid = False
    for sid in candidates:
        print(f"Testing {sid} …", end=" ")
        ok, msg = test_series(sid)
        if ok:
            print(f"✔ VALID: {sid} → {msg}")
            found_valid = True
            break
        else:
            print(f"✖ {sid} → {msg}")
        time.sleep(1)  # Be polite to API
    if not found_valid:
        print("Keiner der Kandidaten lieferte Daten — bitte Kandidatenliste erweitern.")

if __name__ == "__main__":
    main()
