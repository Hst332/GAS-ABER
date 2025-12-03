# ===================================================================
# gas_price_forecast.py
# Stable / Online / CI-safe
# ===================================================================

import os
import numpy as np
import pandas as pd
from datetime import datetime

import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

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
# DATA LOADING
# =======================
def load_prices():
    print("[INFO] Downloading prices since", START_DATE)

    gas = yf.download(
        SYMBOL_GAS, start=START_DATE, progress=False, auto_adjust=False
    )[["Close"]]

    oil = yf.download(
        SYMBOL_OIL, start=START_DATE, progress=False, auto_adjust=False
    )[["Close"]]

    gas.rename(columns={"Close": "Gas_Close"}, inplace=True)
    oil.rename(columns={"Close": "Oil_Close"}, inplace=True)

    df = gas.join(oil, how="inner")
    df = df.sort_index()
    df.dropna(inplace=True)

    return df

# =======================
# FEATURES
# =======================
def build_features(df):
    df = df.copy()

    df["Gas_Return"] = df["Gas_Close"].pct_change()
    df["Oil_Return"] = df["Oil_Close"].pct_change()

    for l in range(1, 6):
        df[f"Gas_Return_lag{l}"] = df["Gas_Return"].shift(l)
        df[f"Oil_Return_lag{l}"] = df["Oil_Return"].shift(l)

    # NEXT DAY DIRECTION
    df["Target"] = (df["Gas_Return"].shift(-1) > 0).astype(int)

    df = df.iloc[10:].dropna()
    return df

# =======================
# MODEL TRAINING
# =======================
def train_model(df, features):
    if "Target" not in df.columns:
        raise RuntimeError("Target column missing")

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
# FORECAST OUTPUT
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
"""

    with open(FORECAST_FILE, "w", encoding="utf-8") as f:
        f.write(txt.strip())

# =======================
# MAIN
# =======================
def main():
    df = load_prices()
    df = build_features(df)

    # CI SAFETY: ensure enough data
    if df.shape[0] < 50:
        print("[WARN] Not enough data after feature build, skipping run.")
        return

    features = [c for c in df.columns if c.endswith(tuple(str(i) for i in range(1, 6)))]
    if not features:
        raise RuntimeError("No feature columns generated")

    model, acc_mean, acc_std = train_model(df, features)

    # LIVE FORECAST
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
