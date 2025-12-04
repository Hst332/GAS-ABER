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
# OPTIONAL STORAGE IMPORT (SAFE)
# =======================
try:
    import fetch_eia_storage
except Exception:
    fetch_eia_storage = None
    
def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [c[0] for c in df.columns]  # nimmt "Close" aus ('Close','NG=F')
    return df
    
# =======================
# SCALE STORAGE SURPRISE (NO LEAK)
# =======================
roll = df["Storage_Surprise"].rolling(52)

df["Storage_Surprise_Z"] = (
    (df["Storage_Surprise"] - roll.mean()) / roll.std()
).shift(1)

df["Storage_Surprise_Z"] = df["Storage_Surprise_Z"].replace(
    [np.inf, -np.inf], 0.0
).fillna(0.0)

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
        SYMBOL_GAS, start=START_DATE, progress=False, auto_adjust=True
    )
    oil = yf.download(
        SYMBOL_OIL, start=START_DATE, progress=False, auto_adjust=True
    )

    gas = flatten_columns(gas)
    oil = flatten_columns(oil)

    gas = gas[["Close"]].rename(columns={"Close": "Gas_Close"})
    oil = oil[["Close"]].rename(columns={"Close": "Oil_Close"})

    df = gas.join(oil, how="inner")
    df = df.sort_index().dropna()

    print("[INFO] loaded price dataframe:", df.shape)
    return df


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

    # --- Storage Surprise (OPTIONAL) ---
    df["Storage_Surprise"] = 0.0

    if fetch_eia_storage is not None:
        try:
            if hasattr(fetch_eia_storage, "load_storage_data"):
                storage = fetch_eia_storage.load_storage_data()
            elif hasattr(fetch_eia_storage, "load_eia_storage"):
                storage = fetch_eia_storage.load_eia_storage()
            else:
                raise RuntimeError("No suitable storage loader found")

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
                how="left"
            ).drop(columns=["Date"])

            df["Storage_Surprise"] = df["Storage_Surprise"].fillna(0.0)
            print("[INFO] Storage Surprise loaded")

        except Exception as e:
            print("[WARN] Storage data unavailable:", str(e))
            df["Storage_Surprise"] = 0.0

    # --- Target (NO LEAK) ---
    df["Target"] = (df["Gas_Return"].shift(-1) > 0).astype(int)

    df = df.dropna()
    return df

# =======================
# MODEL
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

    if df is None or len(df) < 200:
        print("[WARN] Not enough usable data")
        return

    features = (
        [c for c in df.columns if c.startswith(("Gas_Return_lag", "Oil_Return_lag"))]
        + ["Storage_Surprise_Z"]
    )

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
