#!/usr/bin/env python3
"""
CODE A – Natural Gas Forecast
Ruhig. Robust. Professionell.
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime
import yfinance as yf

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

# =======================
# CONFIG
# =======================
START_DATE = "2014-01-01"
GAS_SYMBOL = "NG=F"

OUT_TXT = "forecast_output.txt"
OUT_JSON = "forecast_output.json"

UP_THRESHOLD = 0.60
DOWN_THRESHOLD = 0.40

# =======================
# DATA
# =======================
def load_gas_prices():
    df = yf.download(GAS_SYMBOL, start=START_DATE, auto_adjust=True, progress=False)
    df = df[["Close"]].rename(columns={"Close": "Gas_Close"})
    df.dropna(inplace=True)
    return df

def load_eia_storage():
    """
    Erwartet CSV:
    Date,Storage,FiveYearAvg
    """
    try:
        s = pd.read_csv("eia_storage.csv", parse_dates=["Date"])
        s.sort_values("Date", inplace=True)
        return s
    except Exception:
        return None

# =======================
# FEATURES
# =======================
def build_features(price_df, storage_df):
    df = price_df.copy()

    df["ret"] = df["Gas_Close"].pct_change()
    df["trend_5"] = df["Gas_Close"].pct_change(5)
    df["trend_20"] = df["Gas_Close"].pct_change(20)
    df["vol_10"] = df["ret"].rolling(10).std()

    df["Target"] = (df["ret"].shift(-1) > 0).astype(int)

    if storage_df is not None:
        storage_df["surprise"] = storage_df["Storage"] - storage_df["FiveYearAvg"]
        storage_df["surprise_z"] = (
            (storage_df["surprise"] - storage_df["surprise"].rolling(52).mean())
            / storage_df["surprise"].rolling(52).std()
        )
        storage_df = storage_df[["Date", "surprise_z"]]
        df = df.merge(storage_df, left_index=True, right_on="Date", how="left")
        df["surprise_z"].ffill(inplace=True)
        df.set_index("Date", inplace=True)
    else:
        df["surprise_z"] = 0.0

    df.dropna(inplace=True)
    return df

# =======================
# MODEL
# =======================
def train_model(df):
    features = ["trend_5", "trend_20", "vol_10", "surprise_z"]
    X = df[features]
    y = df["Target"]

    tscv = TimeSeriesSplit(5)
    acc = []

    for tr, te in tscv.split(X):
        m = LogisticRegression(max_iter=200)
        m.fit(X.iloc[tr], y.iloc[tr])
        acc.append(accuracy_score(y.iloc[te], m.predict(X.iloc[te])))

    model = LogisticRegression(max_iter=200)
    model.fit(X, y)

    return model, features, float(np.mean(acc)), float(np.std(acc))

# =======================
# OUTPUT
# =======================
def write_output(res):
    with open(OUT_JSON, "w") as f:
        json.dump(res, f, indent=2)

    with open(OUT_TXT, "w") as f:
        f.write("===================================\n")
        f.write("   NATURAL GAS FORECAST – CODE A\n")
        f.write("===================================\n")
        f.write(f"Run time (UTC): {res['run_time']}\n")
        f.write(f"Data date     : {res['data_date']}\n\n")
        f.write(f"Model CV      : {res['cv_mean']:.2%} ± {res['cv_std']:.2%}\n\n")
        f.write(f"Prob UP       : {res['prob_up']:.2%}\n")
        f.write(f"Prob DOWN     : {res['prob_down']:.2%}\n")
        f.write(f"Signal        : {res['signal']}\n")
        f.write("===================================\n")

# =======================
# MAIN
# =======================
def main():
    prices = load_gas_prices()
    storage = load_eia_storage()

    df = build_features(prices, storage)
    model, features, cv_mean, cv_std = train_model(df)

    last = df.iloc[-1:]
    prob_up = model.predict_proba(last[features])[0][1]

    if prob_up >= UP_THRESHOLD:
        signal = "UP"
    elif prob_up <= DOWN_THRESHOLD:
        signal = "DOWN"
    else:
        signal = "NO_TRADE"

    res = {
        "run_time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "data_date": last.index[0].date().isoformat(),
        "prob_up": round(prob_up, 4),
        "prob_down": round(1 - prob_up, 4),
        "signal": signal,
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "sources": {
            "prices": "Yahoo NG=F",
            "storage": "EIA" if storage is not None else "missing"
        }
    }

    write_output(res)
    print("[OK] CODE A finished")

if __name__ == "__main__":
    main()
