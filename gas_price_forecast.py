# ===================================================================
# gas_price_forecast.py
# Stable / Online / CI-safe
# ===================================================================

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
import yfinance as yf

# =======================
# SAFETY (Line ~20)
# =======================
assert callable(str), "FATAL: built-in 'str' overwritten"

# =======================
# CONFIG (Line ~25)
# =======================
START_DATE = "2015-01-01"
FORECAST_FILE = "forecast_output.txt"
SYMBOL_GAS = "NG=F"
SYMBOL_OIL = "CL=F"
PROB_THRESHOLD = 0.5

# =======================
# DATA LOADING (Line ~35)
# =======================
def load_prices():
    print("[INFO] Downloading prices since", START_DATE)
    gas = yf.download(SYMBOL_GAS, start=START_DATE, progress=False, auto_adjust=False)[["Close"]]
    oil = yf.download(SYMBOL_OIL, start=START_DATE, progress=False)[["Close"]]

    gas.rename(columns={"Close": "Gas_Close"}, inplace=True)
    oil.rename(columns={"Close": "Oil_Close"}, inplace=True)

    df = gas.join(oil, how="inner")
    df.dropna(inplace=True)
    return df

# =======================
# FEATURES (Line ~55)
# =======================
def build_features(df):
    df = df.copy()

    df["Gas_Return"] = df["Gas_Close"].pct_change()
    df["Oil_Return"] = df["Oil_Close"].pct_change()

    for l in range(1, 6):
        df[f"Gas_Return_lag{l}"] = df["Gas_Return"].shift(l)
        df[f"Oil_Return_lag{l}"] = df["Oil_Return"].shift(l)

    # Target: NEXT DAY DIRECTION (NO LEAK)
    df["Target"] = (df["Gas_Return"].shift(-1) > 0).astype(int)

    df = df.iloc[10:].dropna()
    return df

# =======================
# TRAINING (Line ~80)
# =======================
def train_model(df, features):
    X = df[features]
    y = df["Target"]

    tscv = TimeSeriesSplit(n_splits=5)
    accs = []

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=5,
        random_state=42
    )

    for tr, te in tscv.split(X):
        model.fit(X.iloc[tr], y.iloc[tr])
        pred = model.predict(X.iloc[te])
        accs.append(accuracy_score(y.iloc[te], pred))

    model.fit(X, y)
    return model, float(np.mean(accs)), float(np.std(accs))

# =======================
# FORECAST OUTPUT (Line ~115)
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
# MAIN (Line ~145)
# =======================
def main():
    df = load_prices()
    df = build_features(df)

def train_model(df, features):
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np

    if "Target" not in df.columns:
        raise RuntimeError("Target column missing")

    if len(features) == 0:
        raise RuntimeError("No feature columns created")

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

    for train_idx, test_idx in tscv.split(X):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        accs.append(model.score(X.iloc[test_idx], y.iloc[test_idx]))

    model.fit(X, y)

    return model, float(np.mean(accs)), float(np.std(accs))

    # ===========================
    # LIVE FORECAST (Line ~160)
    # ===========================
    last_row = df.iloc[-1:]                       # ✅ korrekt
    prob_up = model.predict_proba(
        last_row[features]
    )[0][1]                                       # ✅ korrekt

    last_date = df.index[-1].date().isoformat()  # ✅ KEIN str()

    write_forecast(prob_up, acc_mean, acc_std, last_date)

    print("[OK] Forecast written to", FORECAST_FILE)

# =======================
# ENTRY (Line ~180)
# =======================
if __name__ == "__main__":
    main()
