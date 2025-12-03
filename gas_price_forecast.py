# ===================================================================
# gas_price_forecast.py
# Stable / Online / CI-safe (robuste Price-Loading)
# ===================================================================

import numpy as np
import pandas as pd
from datetime import datetime

import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from fetch_eia_storage import load_storage_data   

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
    """
    Ensure df.columns are plain strings regardless of MultiIndex or tuple columns.
    If a column is a tuple, join by '_' and strip spaces.
    """
    df = df.copy()
    new_cols = []
    for c in df.columns:
        if isinstance(c, tuple):
            # join tuple elements which are not empty / None
            joined = "_".join([str(x) for x in c if x is not None and str(x) != ""])
            new_cols.append(joined)
        else:
            new_cols.append(str(c))
    df.columns = new_cols
    return df

def _find_close_column(df: pd.DataFrame) -> str:
    """
    Return the name of a column to be used as 'Close' price.
    Strategy (in order):
      1) exact 'Close'
      2) any column that endswith '_Close' (case-insensitive)
      3) any column that equals 'Adj Close' or contains 'adjclose' (some yfinance variants)
      4) any column that contains 'close' substring (case-insensitive)
      5) raise informative error listing df.columns
    """
    cols = list(df.columns)
    lower_map = {c: c.lower() for c in cols}

    # 1) exact Close
    for c in cols:
        if c == "Close":
            return c

    # 2) endswith _Close or endswith Close (case-insensitive)
    for c in cols:
        if lower_map[c].endswith("_close") or lower_map[c].endswith("close"):
            return c

    # 3) common alternative names
    for c in cols:
        if "adj close" in lower_map[c] or "adjclose" in lower_map[c]:
            return c

    # 4) fallback any column that contains 'close'
    for c in cols:
        if "close" in lower_map[c]:
            return c

    # 5) nothing found -> raise with helpful debug
    raise RuntimeError(
        "Could not find a 'Close' column in dataframe. Columns present:\n  "
        + ", ".join(cols)
        + "\nTip: inspect the downloaded DataFrames from yfinance."
    )

# =======================
# DATA LOADING (robust)
# =======================
def load_prices():
    print("[INFO] Downloading prices since", START_DATE)

    # download raw frames
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

    # flatten columns to strings
    gas = flatten_columns(gas_raw)
    oil = flatten_columns(oil_raw)

    # find the correct 'Close' column robustly
    try:
        gas_close_col = _find_close_column(gas)
    except Exception as e:
        print("[ERROR] gas dataframe columns:", gas.columns.tolist())
        raise

    try:
        oil_close_col = _find_close_column(oil)
    except Exception as e:
        print("[ERROR] oil dataframe columns:", oil.columns.tolist())
        raise

    # rename selected close columns to canonical names and keep only them
    gas = gas[[gas_close_col]].rename(columns={gas_close_col: "Gas_Close"})
    oil = oil[[oil_close_col]].rename(columns={oil_close_col: "Oil_Close"})

    # join, sort, drop missing
    df = gas.join(oil, how="inner")
    df = df.sort_index().dropna()

    # quick sanity print for CI logs
    print("[INFO] loaded price dataframe:", df.shape, "columns:", df.columns.tolist())

    return df

# =======================
# FEATURES
# =======================
def build_features(df):
    # =======================
# EIA STORAGE (SAFE)
# =======================
try:
    storage = load_storage_data()  # erwartet: Date, Storage
    storage = storage.sort_values("Date")

    storage["Storage_Change"] = storage["Storage"].diff()

    # Market expectation = rolling mean of last 4 weeks
    storage["Storage_Exp"] = storage["Storage_Change"].rolling(4).mean()

    # Surprise = actual - expectation (SHIFTED to avoid leak)
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

except Exception as e:
    print("[WARN] Storage data unavailable:", e)
    df["Storage_Surprise"] = 0.0

    df = df.copy()

    df["Gas_Return"] = df["Gas_Close"].pct_change()
    df["Oil_Return"] = df["Oil_Close"].pct_change()

    for l in range(1, 6):
        df[f"Gas_Return_lag{l}"] = df["Gas_Return"].shift(l)
        df[f"Oil_Return_lag{l}"] = df["Oil_Return"].shift(l)

    # NEXT DAY TARGET (NO LEAK)
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

    if len(df) < 100:
        print("[WARN] Not enough data – skipping forecast")
        return

    # safe feature selection
    features = [
        c for c in df.columns
        if isinstance(c, str) and c.startswith(("Gas_Return_lag", "Oil_Return_lag"))
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
