# ===================================================================
# gas_price_forecast.py
# Stable / Online / CI-safe
# ===================================================================

import numpy as np
import pandas as pd
from datetime import datetime

import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit

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
        SYMBOL_GAS,
        start=START_DATE,
        progress=False,
        auto_adjust=False,
        group_by="column",
    )

    oil = yf.download(
        SYMBOL_OIL,
        start=START_DATE,
        progress=False,
        auto_adjust=False,
        group_by="column",
    )

    # ✅ CRITICAL FIX: flatten columns (NO MultiIndex!)
    gas = gas.rename(columns={"Close": "Gas_Close"})[["Gas_Close"]]
    oil = oil.rename(columns={"Close": "Oil_Close"})[["Oil_Close"]]

    df = gas.join(oil, how="inner")
    df = df.sort_index()
    df = df.dropna()

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

    # ✅ NEXT DAY TARGET (NO LEAKAGE)
    df["Target"] = (df["Gas_Return"].shift(-1) > 0).astype(int)

    # Cleanup warmup rows
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

    # Safety for CI
    if len(df) < 100:
        print("[WARN] Not enough data – skipping forecast")
        return

    # ✅ GUARANTEED CORRECT FEATURE LIST
    features = [
        c for c in df.columns
        if c.startswith(("Gas_Return_lag", "Oil_Return_lag"))
    ]

    if not features:
        raise RuntimeError("No feature columns created")

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

