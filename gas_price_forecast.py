# ===================================================================
# gas_price_forecast.py
# Stable / Online / CI-safe
# Adds: Storage Surprise + LNG Feedgas Surprise (both leak-free & scaled)
# ===================================================================

import numpy as np
import pandas as pd
from datetime import datetime

import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit

# =======================
# OPTIONAL EXTERNAL LOADERS (SAFE)
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
    """Make sure columns are simple strings (yfinance MultiIndex-safe)."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [c[0] for c in df.columns]
    return df

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
# FEATURE IMPORTANCE
# =======================
def print_feature_importance(model, features):
    print("\n[INFO] Feature importance (descending):")

    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]

    for i in order:
        print(
            f"  {features[i]:<25} : {importances[i]:.4f}"
        )

# =======================
# FEATURE ENGINEERING
# =======================
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Returns ---
    df["Gas_Return"] = df["Gas_Close"].pct_change()
    df["Oil_Return"] = df["Oil_Close"].pct_change()

    for l in range(1, 6):
        df[f"Gas_Return_lag{l}"] = df["Gas_Return"].shift(l)
        df[f"Oil_Return_lag{l}"] = df["Oil_Return"].shift(l)

    # ============================
    # STORAGE SURPRISE (OPTIONAL)
    # ============================
    df["Storage_Surprise_Z"] = 0.0

    if fetch_eia_storage is not None:
        try:
            if hasattr(fetch_eia_storage, "load_storage_data"):
                storage = fetch_eia_storage.load_storage_data()
            elif hasattr(fetch_eia_storage, "load_eia_storage"):
                storage = fetch_eia_storage.load_eia_storage()
            else:
                raise RuntimeError("No suitable storage loader found")

            # Expect storage: DataFrame with Date (datetime) and Storage numeric column
            storage = storage.sort_values("Date").reset_index(drop=True)
            storage["Storage_Change"] = storage["Storage"].diff()
            storage["Exp_4w"] = storage["Storage_Change"].rolling(4).mean()
            storage["Exp_8w"] = storage["Storage_Change"].rolling(8).mean()

            storage["Storage_Exp"] = 0.5 * storage["Exp_4w"] + 0.5 * storage["Exp_8w"]

            storage["Storage_Surprise"] = (storage["Storage_Change"] - storage["Storage_Exp"]).shift(1)

            # merge to main df (left index = prices index)
            df = df.merge(
                storage[["Date", "Storage_Surprise"]],
                left_index=True, right_on="Date", how="left"
            ).drop(columns=["Date"])
            df["Storage_Surprise"] = df["Storage_Surprise"].fillna(0.0)

            # z-score scale using 52-period rolling of surprises (avoid future leak)
            roll = df["Storage_Surprise"].rolling(52)
            df["Storage_Surprise_Z"] = ((df["Storage_Surprise"] - roll.mean()) / roll.std()).shift(1)
            df["Storage_Surprise_Z"] = df["Storage_Surprise_Z"].replace([np.inf, -np.inf], 0.0).fillna(0.0)

            print("[INFO] Storage Surprise loaded & scaled")

        except Exception as e:
            print("[WARN] Storage data unavailable:", str(e))
            df["Storage_Surprise_Z"] = 0.0

    # =================================
    # LNG Feedgas Surprise (OPTIONAL)
    # =================================
    df["LNG_Feedgas_Surprise_Z"] = 0.0

    if fetch_lng_feedgas is not None:
        try:
            # expected loader returns DataFrame with Date and Feedgas (numeric)
            if hasattr(fetch_lng_feedgas, "load_lng_feedgas"):
                feedgas = fetch_lng_feedgas.load_lng_feedgas()
            elif hasattr(fetch_lng_feedgas, "load_feedgas"):
                feedgas = fetch_lng_feedgas.load_feedgas()
            else:
                raise RuntimeError("No suitable LNG feedgas loader found")

            feedgas = feedgas.sort_values("Date").reset_index(drop=True)
            # compute change (use diff), expectation (rolling mean of last 4 periods),
            # surprise = actual - expected shifted by 1 to avoid leak
            feedgas["Feedgas_Change"] = feedgas["Feedgas"].diff()
            feedgas["Feedgas_Exp"] = feedgas["Feedgas_Change"].rolling(4).mean()
            feedgas["Feedgas_Surprise"] = (feedgas["Feedgas_Change"] - feedgas["Feedgas_Exp"]).shift(1)

            # Merge into df
            df = df.merge(
                feedgas[["Date", "Feedgas_Surprise"]],
                left_index=True, right_on="Date", how="left"
            ).drop(columns=["Date"])
            df["Feedgas_Surprise"] = df["Feedgas_Surprise"].fillna(0.0)

            # Scale by rolling 52 (Z score) and shift to avoid leak
            roll = storage["Storage_Surprise"].rolling(52)
            storage["Storage_Surprise_Z"] = (
            (storage["Storage_Surprise"] - roll.median())
            / (roll.quantile(0.75) - roll.quantile(0.25))
            ).shift(1)

            df["LNG_Feedgas_Surprise_Z"] = ((df["Feedgas_Surprise"] - roll_f.mean()) / roll_f.std()).shift(1)
            df["LNG_Feedgas_Surprise_Z"] = df["LNG_Feedgas_Surprise_Z"].replace([np.inf, -np.inf], 0.0).fillna(0.0)

            print("[INFO] LNG Feedgas Surprise loaded & scaled")

        except Exception as e:
            print("[WARN] LNG feedgas unavailable:", str(e))
            df["LNG_Feedgas_Surprise_Z"] = 0.0

    # --- Target (NEXT DAY, NO LEAK) ---
    df["Target"] = (df["Gas_Return"].shift(-1) > 0).astype(int)

    df = df.dropna()
    return df

# =======================
# MODEL
# =======================
def train_model(df: pd.DataFrame, features: list):
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
        [c for c in df.columns if isinstance(c, str) and c.startswith(("Gas_Return_lag", "Oil_Return_lag"))]
        + ["Storage_Surprise_Z", "LNG_Feedgas_Surprise_Z"]
    )

    model, acc_mean, acc_std = train_model(df, features)
    print_feature_importance(model, features)

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
