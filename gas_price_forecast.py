#!/usr/bin/env python3
# ===================================================================
# gas_price_forecast.py
# Stable / Online / CI-safe
# ===================================================================

import numpy as np
import pandas as pd
from datetime import datetime

import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# =======================
# OPTIONAL LOADERS
# =======================
try:
    import fetch_eia_storage
except Exception:
    fetch_eia_storage = None

try:
    import fetch_lng_feedgas
except Exception:
    fetch_lng_feedgas = None

# =======================
# CONFIG
# =======================
START_DATE = "2015-01-01"
SYMBOL_GAS = "NG=F"
SYMBOL_OIL = "CL=F"
PROB_THRESHOLD = 0.5

# =======================
# HELPERS
# =======================
def flatten_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [c[0] for c in df.columns]
    return df

# =======================
# DATA
# =======================
def load_prices():
    gas = yf.download(SYMBOL_GAS, start=START_DATE, auto_adjust=True, progress=False)
    oil = yf.download(SYMBOL_OIL, start=START_DATE, auto_adjust=True, progress=False)

    gas = flatten_columns(gas)[["Close"]].rename(columns={"Close": "Gas_Close"})
    oil = flatten_columns(oil)[["Close"]].rename(columns={"Close": "Oil_Close"})

    return gas.join(oil, how="inner").dropna()

# =======================
# FEATURES
# =======================
def build_features(df):
    df = df.copy()

    # --- Returns ---
    df["Gas_Return"] = df["Gas_Close"].pct_change()
    df["Oil_Return"] = df["Oil_Close"].pct_change()

    for l in range(1, 6):
        df[f"Gas_Return_lag{l}"] = df["Gas_Return"].shift(l)
        df[f"Oil_Return_lag{l}"] = df["Oil_Return"].shift(l)

    # >>> ADDED: TREND REGIME (NO LEAK)
    df["Gas_MA_50"] = df["Gas_Close"].rolling(50).mean()
    df["Gas_MA_200"] = df["Gas_Close"].rolling(200).mean()
    df["Trend_Regime"] = (df["Gas_MA_50"] > df["Gas_MA_200"]).astype(int)

    # >>> ADDED: VOLATILITY REGIME
    df["Gas_Vol_20"] = df["Gas_Return"].rolling(20).std()
    df["Gas_Vol_252"] = df["Gas_Return"].rolling(252).std()
    df["High_Vol_Regime"] = (df["Gas_Vol_20"] > df["Gas_Vol_252"]).astype(int)

    # ---------- Storage Surprise ----------
    df["Storage_Surprise_Z"] = 0.0
    if fetch_eia_storage is not None:
        try:
            storage = fetch_eia_storage.load_storage_data()
            storage = storage.sort_values("Date")
            storage["Change"] = storage["Storage"].diff()
            storage["Exp"] = storage["Change"].rolling(5).mean()
            storage["Surprise"] = (storage["Change"] - storage["Exp"]).shift(1)

            roll = storage["Surprise"].rolling(52)
            storage["Storage_Surprise_Z"] = (
                (storage["Surprise"] - roll.median())
                / (roll.quantile(0.75) - roll.quantile(0.25))
            ).shift(1)

            df = df.merge(
                storage[["Date", "Storage_Surprise_Z"]],
                left_index=True,
                right_on="Date",
                how="left",
            ).drop(columns=["Date"])

            df["Storage_Surprise_Z"] = df["Storage_Surprise_Z"].fillna(0.0)
        except Exception as e:
            print("[WARN] Storage unavailable:", e)

    # ---------- LNG Feedgas ----------
    df["LNG_Feedgas_Surprise_Z"] = 0.0
    if fetch_lng_feedgas is not None:
        try:
            feedgas = fetch_lng_feedgas.load_lng_feedgas()
            feedgas = feedgas.sort_values("Date")
            feedgas["Change"] = feedgas["Feedgas"].diff()
            feedgas["Exp"] = feedgas["Change"].rolling(4).mean()
            feedgas["Surprise"] = (feedgas["Change"] - feedgas["Exp"]).shift(1)

            roll = feedgas["Surprise"].rolling(52)
            feedgas["LNG_Feedgas_Surprise_Z"] = (
                (feedgas["Surprise"] - roll.mean()) / roll.std()
            ).shift(1)

            df = df.merge(
                feedgas[["Date", "LNG_Feedgas_Surprise_Z"]],
                left_index=True,
                right_on="Date",
                how="left",
            ).drop(columns=["Date"])

            df["LNG_Feedgas_Surprise_Z"] = df["LNG_Feedgas_Surprise_Z"].fillna(0.0)
        except Exception as e:
            print("[WARN] LNG Feedgas unavailable:", e)

    # --- Target ---
    df["Target"] = (df["Gas_Return"].shift(-1) > 0).astype(int)
    return df.dropna()

# =======================
# MODEL
# =======================
def train_model(df, features):
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=20,
        random_state=42,
    )
    model.fit(df[features], df["Target"])
    return model
# =======================
# WALK-FORWARD BACKTEST
# =======================
def walk_forward_backtest(
    df: pd.DataFrame,
    features: list,
    train_window: int = 750,
):
    """
    True walk-forward backtest (1-step ahead).
    Returns DataFrame with equity curve.
    """
    records = []

    for t in range(train_window, len(df) - 1):
        train = df.iloc[t - train_window : t]
        test = df.iloc[t : t + 1]

        model = train_model(train, features)

        prob_up = model.predict_proba(test[features])[0][1]

        # --- apply SAME regime rules ---
        trend_ok = test["Trend_Regime"].iloc[0] == 1
        vol_ok = test["High_Vol_Regime"].iloc[0] == 0

        signal = int(prob_up > PROB_THRESHOLD and trend_ok and vol_ok)

        ret = df["Gas_Return"].iloc[t + 1]
        pnl = signal * ret

        records.append(
            {
                "Date": df.index[t],
                "Signal": signal,
                "Return": ret,
                "PnL": pnl,
            }
        )

    res = pd.DataFrame(records).set_index("Date")
    res["Equity"] = (1 + res["PnL"]).cumprod()
    return res

# =======================
# MAIN
# =======================
def main():
    df = build_features(load_prices())

    if len(df) < 300:
        print("[WARN] Not enough data")
        return

    features = [
        c for c in df.columns
        if c.startswith(("Gas_Return_lag", "Oil_Return_lag"))
    ] + [
        "Storage_Surprise_Z",
        "LNG_Feedgas_Surprise_Z",
        "Trend_Regime",          # >>> ADDED
        "High_Vol_Regime",       # >>> ADDED
    ]

    model = train_model(df, features)

    last = df.iloc[-1:]
    prob_up = model.predict_proba(last[features])[0][1]

    # >>> ADDED: REGIME FILTER (SAFE)
    trend_ok = last["Trend_Regime"].iloc[0] == 1
    vol_ok = last["High_Vol_Regime"].iloc[0] == 0

    signal = (
        "UP"
        if prob_up > PROB_THRESHOLD and trend_ok and vol_ok
        else "DOWN"
    )

    print(
        "\n[RESULT]",
        f"Probability UP: {prob_up:.3f}",
        "| Trend OK:", trend_ok,
        "| Vol OK:", vol_ok,
        "| Signal:", signal,
    )

# =======================
# ENTRY
# =======================
if __name__ == "__main__":
    main()
