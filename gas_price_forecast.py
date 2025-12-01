import os
import requests
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

# =====================
# CONFIG
# =====================
START_DATE = "2018-01-01"

FRED_GAS = "DHHNGSP"     # Henry Hub Gas
FRED_OIL = "DCOILWTICO" # WTI Oil

# =====================
# HELPERS
# =====================
def fetch_fred(series_id, api_key):
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": START_DATE
    }

    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()

    df = pd.DataFrame(r.json()["observations"])
    df = df[["date", "value"]]
    df.columns = ["Date", series_id]
    df["Date"] = pd.to_datetime(df["Date"])
    df[series_id] = pd.to_numeric(df[series_id], errors="coerce")

    return df.dropna().sort_values("Date")


def fetch_weather_open_meteo(start, end):
    # Repräsentative US-Zone (Midwest)
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 39.0,
        "longitude": -96.0,
        "start_date": start,
        "end_date": end,
        "daily": "temperature_2m_mean",
        "timezone": "UTC"
    }

    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()["daily"]

    df = pd.DataFrame({
        "Date": pd.to_datetime(data["time"]),
        "Temp": data["temperature_2m_mean"]
    })

    df["HDD"] = np.maximum(18 - df["Temp"], 0)
    df["CDD"] = np.maximum(df["Temp"] - 18, 0)

    return df[["Date", "HDD", "CDD"]]


# =====================
# FEATURES
# =====================
def build_features(df):
    df = df.copy()

    df["Gas_Return"] = df["Gas"].pct_change()
    df["Oil_Return"] = df["Oil"].pct_change()

    for lag in range(1, 6):
        df[f"Gas_lag{lag}"] = df["Gas_Return"].shift(lag)
        df[f"Oil_lag{lag}"] = df["Oil_Return"].shift(lag)
        df[f"HDD_lag{lag}"] = df["HDD"].shift(lag)
        df[f"CDD_lag{lag}"] = df["CDD"].shift(lag)

    df["Momentum7"] = df["Gas"].pct_change(7).shift(1)
    df["Volatility5"] = df["Gas_Return"].rolling(5).std().shift(1)

    df["Target"] = (df["Gas"].shift(-1) > df["Gas"]).astype(int)

    return df.dropna()


# =====================
# MAIN
# =====================
def main():
    fred_key = os.getenv("FRED_API_KEY")
    if not fred_key:
        raise RuntimeError("FRED_API_KEY not set")

    print("Fetching gas & oil from FRED...")
    gas = fetch_fred(FRED_GAS, fred_key).rename(columns={FRED_GAS: "Gas"})
    oil = fetch_fred(FRED_OIL, fred_key).rename(columns={FRED_OIL: "Oil"})

    df = gas.merge(oil, on="Date", how="inner")

    print("Fetching weather...")
    weather = fetch_weather_open_meteo(
        df["Date"].min().strftime("%Y-%m-%d"),
        df["Date"].max().strftime("%Y-%m-%d"),
    )

    df = df.merge(weather, on="Date", how="inner")
    df = build_features(df)

    features = [c for c in df.columns if c not in ["Date", "Gas", "Target"]]
    X, y = df[features], df["Target"]

    print(f"Rows: {len(df)} | Features: {len(features)}")

    tss = TimeSeriesSplit(5)
    accs = []

    for i, (tr, te) in enumerate(tss.split(X)):
        model = RandomForestClassifier(
            n_estimators=400,
            max_depth=7,
            min_samples_leaf=25,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X.iloc[tr], y.iloc[tr])
        pred = model.predict(X.iloc[te])
        acc = accuracy_score(y.iloc[te], pred)
        accs.append(acc)
        print(f"Split {i}: {acc:.3f}")

    print("====================================")
    print(f"MEAN ACCURACY: {np.mean(accs):.3f}")
    print("====================================")


if __name__ == "__main__":
    main()
