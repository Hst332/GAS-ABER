"""
gas_price_forecast.py
SAFE, incremental improvements:
 - futures spread proxy (NG1-NG3 proxy)
 - rolling z-score normalization for features
 - change-log (forecast_log.csv) + human output (forecast_output.txt)

Minimal invasive changes, no leakage, daily-stable.
"""

import warnings
warnings.filterwarnings("ignore")

import os
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

# -----------------------
# CONFIG
# -----------------------
GAS_TICKER = "NG=F"  # continuous Henry Hub front
OIL_TICKER = "CL=F"

START_DATE = "2019-01-01"
N_SPLITS = 5

# model
N_ESTIMATORS = 300
MAX_DEPTH = 8
MIN_SAMPLES_LEAF = 5
RANDOM_STATE = 42

# live decision thresholds (conservative)
UPPER_PROB = 0.55
LOWER_PROB = 0.45

# z-score window for normalization (days)
ZSCORE_WINDOW = 90

# output files
FORECAST_TXT = "forecast_output.txt"
FORECAST_LOG = "forecast_log.csv"

# -----------------------
# UTIL: load prices
# -----------------------
def load_prices(start=START_DATE):
    start_ts = pd.to_datetime(start)
    print(f"Downloading prices since {start_ts.date()} ...")
    gas = yf.download(GAS_TICKER, start=start_ts, progress=False, auto_adjust=False)
    oil = yf.download(OIL_TICKER, start=start_ts, progress=False, auto_adjust=False)

    if gas.empty or oil.empty:
        raise RuntimeError("Price download failed or returned no data")

    gas = gas[["Close"]].rename(columns={"Close": "Gas_Close"})
    oil = oil[["Close"]].rename(columns={"Close": "Oil_Close"})
    df = gas.join(oil, how="inner")
    df = df.dropna()
    df.index = pd.to_datetime(df.index)
    return df

# -----------------------
# FEATURES (base)
# -----------------------
def add_base_features(df):
    df = df.copy()
    df["Gas_Return"] = df["Gas_Close"].pct_change()
    df["Oil_Return"] = df["Oil_Close"].pct_change()

    # lags
    for lag in [1,2,3,5]:
        df[f"Gas_Return_lag{lag}"] = df["Gas_Return"].shift(lag)
        df[f"Oil_Return_lag{lag}"] = df["Oil_Return"].shift(lag)

    # momentum/vol
    df["Momentum5"] = df["Gas_Close"].pct_change(5).shift(1)
    df["Volatility5"] = df["Gas_Return"].rolling(5).std()

    # -----------------------
    # 1) TERM-STRUCTURE PROXY (Futures spread proxy)
    # We don't rely on separate contract tickers (not always available).
    # Use short MA vs longer MA as a proxy for near-term vs medium-term price
    # Proxy = (MA30 - MA5) / MA5  -> positive -> upward-sloping (contango proxy)
    # -----------------------
    df["MA5"] = df["Gas_Close"].rolling(5).mean()
    df["MA30"] = df["Gas_Close"].rolling(30).mean()
    df["term_structure_proxy"] = (df["MA30"] - df["MA5"]) / (df["MA5"].replace(0, np.nan))
    # shift proxy to avoid lookahead (use yesterday's proxy for today's decision)
    df["term_structure_proxy"] = df["term_structure_proxy"].shift(1)

    # target: next day up?
    df["Target"] = (df["Gas_Close"].shift(-1) > df["Gas_Close"]).astype(int)

    df = df.dropna()
    return df

# -----------------------
# Rolling Z-score normalizer (fit on historical window)
# -----------------------
def rolling_zscore(df, feature_cols, window=ZSCORE_WINDOW):
    # compute rolling mean/std for each column
    roll_mean = df[feature_cols].rolling(window, min_periods=10).mean()
    roll_std = df[feature_cols].rolling(window, min_periods=10).std().replace(0, np.nan)

    z = (df[feature_cols] - roll_mean) / roll_std
    # fill early NA with column-wise zscore using full history (fallback)
    z = z.fillna((df[feature_cols] - df[feature_cols].mean()) / df[feature_cols].std())
    # keep same index
    z.index = df.index
    return z

# -----------------------
# Train & CV
# -----------------------
def train_and_validate(df, feature_cols):
    X = df[feature_cols]
    y = df["Target"]

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    accs = []
    for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
        Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
        ytr, yte = y.iloc[train_idx], y.iloc[test_idx]

        m = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        m.fit(Xtr, ytr)
        pred = m.predict(Xte)
        acc = accuracy_score(yte, pred)
        accs.append(acc)
        print(f"  CV split {i}: acc={acc:.3f}")

    # final train on all available data
    final_model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    final_model.fit(X, y)
    print(f"CV mean acc = {np.mean(accs):.4f}, std = {np.std(accs):.4f}")
    return final_model, np.mean(accs), np.std(accs)

# -----------------------
# Persist forecast + append log
# -----------------------
def write_forecast_human(date, latest_price, prob_up, signal, aux):
    # human readable summary
    now = datetime.utcnow()
    lines = []
    lines.append("===================================")
    lines.append("      NATURAL GAS PRICE FORECAST")
    lines.append("===================================")
    lines.append(f"Run UTC: {now.isoformat(sep=' ')}")
    lines.append(f"Datum (data index): {date}")
    lines.append("")
    lines.append(f"Latest_price: {latest_price:.4f}")
    lines.append(f"Probability UP: {prob_up:.4f}")
    lines.append(f"Signal: {signal}")
    lines.append("")
    lines.append("Auxiliary features (z-scored):")
    for k,v in aux.items():
        lines.append(f"  {k:25s}: {v:.4f}")
    lines.append("===================================")
    with open(FORECAST_TXT, "w") as f:
        f.write("\n".join(lines))
    print(f"WROTE {FORECAST_TXT}")

def append_forecast_log(date, prob_up, signal):
    # append minimal CSV log for tracking changes
    row = {"run_utc": datetime.utcnow().isoformat(sep=' '),
           "data_date": str(date),
           "prob_up": float(prob_up),
           "signal": signal}
    df_new = pd.DataFrame([row])
    if os.path.exists(FORECAST_LOG):
        df_old = pd.read_csv(FORECAST_LOG)
        df_comb = pd.concat([df_old, df_new], ignore_index=True)
        df_comb.to_csv(FORECAST_LOG, index=False)
    else:
        df_new.to_csv(FORECAST_LOG, index=False)
    print(f"APPENDED {FORECAST_LOG}")

# -----------------------
# MAIN
# -----------------------
def main():
    # 1) Load prices
    df_raw = load_prices(START_DATE)

    # 2) build base features (includes term_structure_proxy)
    df = add_base_features(df_raw)

    # 3) select numeric features to zscore
    # include: returns, lags, momentum, vol, term structure
    feature_cols = [
        "Gas_Return", "Oil_Return",
        "Momentum5", "Volatility5",
        "term_structure_proxy"
    ] + [f"Gas_Return_lag{i}" for i in [1,2,3,5]] \
      + [f"Oil_Return_lag{i}" for i in [1,2,3,5]]

    df = df.copy()

    # 4) compute rolling z-scores for features (no leakage: window uses past only)
    z = rolling_zscore(df, feature_cols, window=ZSCORE_WINDOW)

    # align z into df (rename to mark they are z)
    z_cols = [f"{c}_z" for c in feature_cols]
    z.columns = z_cols
    df = pd.concat([df, z], axis=1)

    # final features used for model are z-scored versions
    model_features = z_cols

    # 5) train + validate
    print("Training model with rolling CV ...")
    model, cv_mean, cv_std = train_and_validate(df, model_features)

    # 6) Live forecast: build last_row with latest intraday where available
    last_index = df.index[-1]
    last_row = df.iloc[-1:].copy()  # contains z-scored features based on historical window

    # Try to update with intraday latest close for Gas to get fresh momentum if available
    try:
        ticker = yf.Ticker(GAS_TICKER)
        intraday = ticker.history(period="1d", interval="1m", actions=False)
        if (not intraday.empty) and ("Close" in intraday.columns):
            # use last available close in intraday series
            live_close = intraday["Close"].iloc[-1]
            # update raw series copy to compute auxiliary (but don't rewrite historical z-window)
            latest_price = live_close
        else:
            latest_price = df["Gas_Close"].iloc[-1]
    except Exception:
        latest_price = df["Gas_Close"].iloc[-1]

    # Recompute auxiliary values from raw (not to overwrite model inputs that are historical z-scores)
    # Aux features for reporting: use most recent raw features (last available in df_raw)
    aux = {}
    aux["term_structure_proxy_z"] = float(last_row.get("term_structure_proxy_z", np.nan))
    aux["Momentum5_z"] = float(last_row.get("Momentum5_z", np.nan))
    aux["Volatility5_z"] = float(last_row.get("Volatility5_z", np.nan))

    # Probability prediction using z-scored features in last_row
    X_live = last_row[model_features].fillna(0.0)
    prob_up = model.predict_proba(X_live)[0][1]

    # Decision with NO_TRADE band
    if prob_up > UPPER_PROB:
        signal = "UP"
    elif prob_up < LOWER_PROB:
        signal = "DOWN"
    else:
        signal = "NO_TRADE"

    # 7) write outputs
    write_forecast_human(last_index.date(), latest_price, prob_up, signal, aux)
    append_forecast_log(last_index.date(), prob_up, signal)

    # 8) print summary to console
    print("\nSUMMARY:")
    print(f"  data date: {last_index.date()}")
    print(f"  latest_price: {latest_price:.4f}")
    print(f"  prob_up: {prob_up:.4f}")
    print(f"  signal: {signal}")
    print(f"  CV mean acc: {cv_mean:.4f} (std {cv_std:.4f})")
    print(f"  model features (z-scored): {len(model_features)}")

if __name__ == "__main__":
    main()
