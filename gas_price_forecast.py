import os
import requests
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

# ==============================
# CONFIG
# ==============================

FRED_SERIES = "DHHNGSP"
START_DATE = "2018-01-01"
OIL_CSV = "data/oil.csv"
WEATHER_CSV = "data/weather.csv"


# ==============================
# DATA LOADERS
# ==============================

def load_gas_from_fred():
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise RuntimeError("FRED_API_KEY not set")

    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": FRED_SERIES,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": START_DATE
    }

    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    obs = r.json()["observations"]

    df = pd.DataFrame(obs)
    df = df[["date", "value"]]
    df.columns = ["Date", "Gas_Close"]
    df["Gas_Close"] = pd.to_numeric(df["Gas_Close"], errors="coerce")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.dropna().sort_values("Date")

    return df


def load_local_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    return pd.read_csv(path, parse_dates=["Date"])


# ==============================
# FEATURE ENGINEERING
# ==============================

def build_features(df):
    df = df.copy()
    df["Gas_Return"] = df["Gas_Close"].pct_change()

    for l in range(1, 6):
        df[f"Gas_Return_lag{l}"] = df["Gas_Return"].shift(l)
        df[f"Oil_Return_lag{l}"] = df["Oil_Return"].shift(l)
        df[f"HDD_lag{l}"] = df["HDD"].shift(l)
        df[f"CDD_lag{l}"] = df["CDD"].shift(l)

    df["Momentum7"] = df["Gas_Close"].pct_change(7).shift(1)
    df["Volatility5"] = df["Gas_Return"].rolling(5).std().shift(1)

    df["Target"] = (df["Gas_Close"].shift(-1) > df["Gas_Close"]).astype(int)

    df = df.dropna()
    return df


# ==============================
# MAIN PIPELINE
# ==============================

def main():
    print("Loading online gas price...")
    gas = load_gas_from_fred()

    print("Loading oil & weather...")
    oil = load_local_csv(OIL_CSV)
    weather = load_local_csv(WEATHER_CSV)

    weather = weather.groupby("Date", as_index=False)[["HDD", "CDD"]].mean()

    oil["Oil_Return"] = oil["Oil_Close"].pct_change()

    df = gas.merge(oil[["Date", "Oil_Return"]], on="Date", how="inner")
    df = df.merge(weather, on="Date", how="inner")
    df = df.dropna().sort_values("Date")

    df = build_features(df)

    features = [c for c in df.columns if c not in ["Date", "Gas_Close", "Target"]]
    X = df[features]
    y = df["Target"]

    print(f"Dataset rows: {len(df)}, features: {len(features)}")

    tss = TimeSeriesSplit(n_splits=5)
    accs = []

    for i, (train, test) in enumerate(tss.split(X)):
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=20,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X.iloc[train], y.iloc[train])
        pred = model.predict(X.iloc[test])
        acc = accuracy_score(y.iloc[test], pred)
        accs.append(acc)
        print(f"Split {i} accuracy: {acc:.3f}")

    print("===================================")
    print(f"Mean accuracy: {np.mean(accs):.3f}")
    print("===================================")


if __name__ == "__main__":
    main()
