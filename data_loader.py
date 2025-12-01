import os
import requests
import pandas as pd

FRED_API_KEY = os.getenv("FRED_API_KEY")
SERIES_ID = "DHHNGSP"

def load_gas_from_fred(start="2018-01-01"):
    if not FRED_API_KEY:
        raise RuntimeError("FRED_API_KEY not set")

    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": SERIES_ID,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": start
    }

    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()["observations"]

    df = pd.DataFrame(data)
    df = df[["date", "value"]]
    df.columns = ["Date", "Gas_Close"]

    df["Gas_Close"] = pd.to_numeric(df["Gas_Close"], errors="coerce")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.dropna()

    return df
