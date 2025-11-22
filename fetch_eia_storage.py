import requests
import os


def fetch_eia_storage():
    """
    Holt den neuesten EIA Working Gas in Storage Wert (Lower 48).
    Gibt (wert, datum) zurück.
    """

    api_key = os.getenv("EIA_API_KEY")
    if not api_key:
        raise ValueError("EIA_API_KEY not found in environment variables!")

    url = (
        "https://api.eia.gov/v2/natural-gas/stor/wngsr/data/"
        f"?api_key={api_key}"
        "&frequency=weekly"
        "&data[0]=value"
        "&sort[0][column]=period&sort[0][direction]=desc"
        "&offset=0&length=1"
    )

    r = requests.get(url, timeout=20)
    r.raise_for_status()
    data = r.json()

    entry = data["response"]["data"][0]
    value = float(entry["value"])
    date = entry["period"]

    return value, date


if __name__ == "__main__":
    value, date = fetch_eia_storage()
    print(f"EIA Storage latest ({date}): {value}")
