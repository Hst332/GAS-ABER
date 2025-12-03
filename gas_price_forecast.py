# ===================================================================
# gas_price_forecast.py
# Stable / Online / CI-safe (robuste Price-Loading + Storage Surprise)
# ===================================================================

import numpy as np
import pandas as pd
from datetime import datetime

import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
import fetch_eia_storage

# =======================
# SAFETY
# =======================
assert callable(str), "FATAL: built-in 'str' overwritten"

# =======================
# CONFIG
# =======================
START_DATE = "2015-01-01"
FORECAST_FILE = "forecast_output.txt"

SYMBOL_GAS = "NG=F"
SYMBOL_OIL = "CL=F"

PROB_THRESHOLD = 0.5

# =======================
# HELPERS
# =======================
def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols = []
    for c in df.columns:
        if isinstance(c, tuple):
            cols.append("_".join(str(x) for x in c if x))
        else:
            cols.append(str(c))
    df.columns = cols
    return df


def _find_close_column(df: pd.DataFrame) -> str:
    for c in df.columns:
        if c.lower() == "close":
            return c
    for c in df.columns:
        if "close" in c.lower():
            return c
    raise RuntimeError(f"No Close column found: {df.columns.tolist()}")

# =======================
# DATA LOADING
# =======================
def load_prices():
    print("[INFO] Downloading prices since", START_DATE)

    gas_raw = yf.download(
        SYMBOL_GAS,
        start=START_DATE,
        progress=False,
        auto_adjust=False,
    )
    oil_raw = yf.download(
        SYMBOL_OIL,
        start=START_DATE,
        progress=False,
        auto_adjust=False,
    )

    gas = flatten_columns(gas_raw)
    oil = flatten_columns(oil_raw)

    gas = gas[[_find_close_column(gas)]].rename(columns=lambda _: "Gas_Close")
    oil = oil[[_find_close_column(oil)]].rename(columns=lambda _: "Oil_Close")

    df = gas.join(oil, how="inner").sort_index().dropna()

    print("[INFO] loaded price dataframe:", df.shape)
    return df

# =======================
# FEATURES
# =======================
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Returns
    df["Gas_Return"] = df["Gas_Close"].pct_change()
    df["Oil_Return"] = df["Oil_Close"].pct_change()

    for l in range(1, 6):
        df[f"Gas_Return_lag{l}"] = df["Gas_Return"].shift(l)
        df[f"Oil_Return_lag{l}"] = df["Oil_Return"].shift(l)

    # =======================
    # EIA STORAGE SURPRISE
    # =======================
    try:
        storage = load_storage_data()
        storage = storage.sort_values("Date")

        storage["Storage_Change"] = storage["Storage"].diff()
        storage["Storage_Exp"] = storage["Storage_Change"].rolling(4).mean()
        storage["Storage_Surprise"] = (
            storage["Storage_Change"] - storage["Storage_Exp"]
        ).shift(1)

        df = df.merge(
            storage[["Date", "Storage_Surprise"]],
            left_index=True,
            right_on="Date",
            how="left",
        ).drop(columns=["Date"])

        df["Storage_Surprise"] = df["Storage_Surprise"].fillna(0.0)

        print("[INFO] Storage Surprise added")

    except Exception as e:
        print("[WARN] Storage data unavailable:", e)
        df["Storage_Surprise"] = 0.0

    # Target (next-day direction, no leak)
    df["Target"] = (df["Gas_Return"].shift(-1) > 0).astype(int)

    df = df.iloc[10:].dropna()
    return df

# =======================
# MODEL TRAINING
# =======================
def train_model(df, features):
    X = df[features]
    y = df["Target"]

    tscv = TimeSeriesSplit(n_splits=5)
    accs = []

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=20,
        random_state=42,
    )

    for tr, te in tscv.split(X):
        model.fit(X.iloc[tr], y.iloc[tr])
        accs.append(model.score(X.iloc[te], y.iloc[te]))

    model.fit(X, y)
    return model, float(np.mean(accs)), float(np.std(accs))

# =======================
# OUTPUT
# =======================
def write_forecast(prob_up, acc_mean, acc_std, last_date):
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    signal = "UP" if prob_up > PROB_THRESHOLD else "DOWN"

    txt = f"""
===================================
      NATURAL GAS PRICE FORECAST
===================================
Run time: {now}
Data date: {last_date}

Model CV Accuracy: {acc_mean:.2%} ± {acc_std:.2%}

Probability price goes UP: {prob_up:.2%}
Signal: {signal}
===================================
""".strip()

    with open(FORECAST_FILE, "w", encoding="utf-8") as f:
        f.write(txt)

# =======================
# MAIN
# =======================
def main():
    df = load_prices()
    df = build_features(df)

    if len(df) < 100:
        print("[WARN] Not enough data – skipping forecast")
        return

    features = [
        c for c in df.columns
        if c.startswith(("Gas_Return_lag", "Oil_Return_lag"))
    ] + ["Storage_Surprise"]

    model, acc_mean, acc_std = train_model(df, features)

    last_row = df.iloc[-1:]
    prob_up = model.predict_proba(last_row[features])[0][1]

    last_date = df.index[-1].date().isoformat()
    write_forecast(prob_up, acc_mean, acc_std, last_date)

    print("[OK] Forecast written to", FORECAST_FILE)

# =======================
# ENTRY
# =======================
if __name__ == "__main__":
    main()
