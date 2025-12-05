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
from sklearn.model_selection import TimeSeriesSplit

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
FORECAST_FILE = "forecast_output.txt"

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
    
def print_feature_importance(model, features, top_n=20):
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]

    print("\n[INFO] Feature importance (RandomForest):")
    for i in order[:top_n]:
        print(f"{features[i]:<30} {importances[i]:.4f}")

# =======================
# DATA
# =======================
def load_prices():
    gas = yf.download(SYMBOL_GAS, start=START_DATE, auto_adjust=True, progress=False)
    oil = yf.download(SYMBOL_OIL, start=START_DATE, auto_adjust=True, progress=False)

    gas = flatten_columns(gas)[["Close"]].rename(columns={"Close": "Gas_Close"})
    oil = flatten_columns(oil)[["Close"]].rename(columns={"Close": "Oil_Close"})

    df = gas.join(oil, how="inner").dropna()
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

    # ============================
    # STORAGE SURPRISE
    # ============================
    df["Storage_Surprise_Z"] = 0.0

    if fetch_eia_storage is not None:
        try:
            if hasattr(fetch_eia_storage, "load_storage_data"):
                storage = fetch_eia_storage.load_storage_data()
            else:
                raise RuntimeError("No storage loader")

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
                left_index=True, right_on="Date", how="left"
            ).drop(columns=["Date"])

            df["Storage_Surprise_Z"] = df["Storage_Surprise_Z"].fillna(0.0)

        except Exception as e:
            print("[WARN] Storage unavailable:", e)

    # ============================
    # LNG FEEDGAS SURPRISE
    # ============================
    df["LNG_Feedgas_Surprise_Z"] = 0.0

    if fetch_lng_feedgas is not None:
        try:
            if hasattr(fetch_lng_feedgas, "load_lng_feedgas"):
                feedgas = fetch_lng_feedgas.load_lng_feedgas()
            else:
                raise RuntimeError("No feedgas loader")

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
                left_index=True, right_on="Date", how="left"
            ).drop(columns=["Date"])

            df["LNG_Feedgas_Surprise_Z"] = df["LNG_Feedgas_Surprise_Z"].fillna(0.0)

        except Exception as e:
            print("[WARN] LNG Feedgas unavailable:", e)

    # ============================
    # TARGET
    # ============================
    df["Target"] = (df["Gas_Return"].shift(-1) > 0).astype(int)
    df = df.dropna()

    return df

# =======================
# MODEL
# =======================
def train_model(df, features):
    X = df[features]
    y = df["Target"]

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=20,
        random_state=42,
    )

    model.fit(X, y)
    return model
from sklearn.metrics import accuracy_score

def permutation_importance_ts(model, df, features, test_size=250):
    """
    Permutation importance on last test_size observations (time-safe)
    """
    df_test = df.iloc[-test_size:].copy()
    X = df_test[features]
    y = df_test["Target"]

    baseline = accuracy_score(y, model.predict(X))
    scores = {}

    rng = np.random.default_rng(42)

    for f in features:
        X_perm = X.copy()
        X_perm[f] = rng.permutation(X_perm[f].values)
        acc = accuracy_score(y, model.predict(X_perm))
        scores[f] = baseline - acc

    return scores

# =======================
# MAIN
# =======================
def main():
    df = load_prices()
    df = build_features(df)

    if df is None or len(df) < 200:
        print("[WARN] Not enough data to train model")
        return

    features = (
        [c for c in df.columns if isinstance(c, str) and c.startswith(("Gas_Return_lag", "Oil_Return_lag"))]
        + ["Storage_Surprise_Z", "LNG_Feedgas_Surprise_Z"]
    )

    # Train model (current version returns ONLY model)
    model = train_model(df, features)

    # Forecast
    last_row = df.iloc[-1:]
    prob_up = model.predict_proba(last_row[features])[0][1]

    # Permutation importance
    perm = permutation_importance_ts(model, df, features)

    print("\n[INFO] Permutation importance (accuracy drop):")
    for f, v in sorted(perm.items(), key=lambda x: x
