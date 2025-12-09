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

    df["Gas_Return"] = df["Gas_Close"].pct_change()
    df["Oil_Return"] = df["Oil_Close"].pct_change()

    for l in range(1, 6):
        df[f"Gas_Return_lag{l}"] = df["Gas_Return"].shift(l)
        df[f"Oil_Return_lag{l}"] = df["Oil_Return"].shift(l)

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
# ROLLING PERMUTATION IMPORTANCE
# =======================
def rolling_permutation_importance_ts(df, features, window=500, step=25):
    rows = []

    for start in range(0, len(df) - window, step):
        train = df.iloc[start : start + window]
        X, y = train[features], train["Target"]

        model = RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=20, random_state=42
        ).fit(X, y)

        baseline = model.score(X, y)
        row = {"Date": train.index[-1], "_baseline": baseline}

        for f in features:
            Xp = X.copy()
            Xp[f] = np.random.permutation(Xp[f].values)
            row[f] = baseline - model.score(Xp, y)

        rows.append(row)

    return pd.DataFrame(rows).set_index("Date")

# =======================
# FEATURE SELECTION
# =======================
def select_stable_features(roll_imp, min_presence=0.6, min_median=0.001):
    keep = []
    for f in roll_imp.columns:
        if f == "_baseline":
            continue
        s = roll_imp[f].dropna()
        if (s > 0).mean() >= min_presence and s.median() >= min_median:
            keep.append(f)
    return keep

# =======================
# PERMUTATION IMPORTANCE (FINAL)
# =======================
def permutation_importance_ts(model, df, features, test_size=250):
    test = df.iloc[-test_size:]
    X, y = test[features], test["Target"]
    baseline = accuracy_score(y, model.predict(X))

    out = {}
    for f in features:
        Xp = X.copy()
        Xp[f] = np.random.permutation(Xp[f].values)
        out[f] = baseline - accuracy_score(y, model.predict(Xp))
    return out

# =======================
# MAIN
# =======================
def main():
    df = build_features(load_prices())

    if len(df) < 200:
        print("[WARN] Not enough data")
        return

    features = [
        c for c in df.columns
        if c.startswith(("Gas_Return_lag", "Oil_Return_lag"))
    ] + ["Storage_Surprise_Z", "LNG_Feedgas_Surprise_Z"]

    roll_imp = rolling_permutation_importance_ts(df, features)
    stable = select_stable_features(roll_imp)

    if not stable:
        print("[WARN] No stable features – using all")
        stable = features

    print("\n[INFO] Stable features:")
    for f in stable:
        print(" ", f)

    model = train_model(df, stable)
    prob_up = model.predict_proba(df.iloc[-1:][stable])[0][1]

    perm = permutation_importance_ts(model, df, stable)
    print("\n[INFO] Permutation importance:")
    for f, v in sorted(perm.items(), key=lambda x: x[1], reverse=True):
        print(f"{f:<30} {v:+.4f}")

    print("\n[RESULT]",
          f"Probability UP: {prob_up:.3f}",
          "| Signal:", "UP" if prob_up > PROB_THRESHOLD else "DOWN")

# =======================
# ENTRY
# =======================
if __name__ == "__main__":
    main()
