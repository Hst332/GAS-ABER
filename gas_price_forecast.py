#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robustes gas_price_forecast.py
- Saubere Fehlerbehandlung wenn Daten fehlen
- Rolling Z-score, term-structure proxy, intraday best-effort
- Persistenter human-readable forecast + append-log
- Designed to run in CI (GitHub Actions) or lokal
"""

import os
import warnings
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

# ----------------------------
# CONFIG
# ----------------------------
GAS_TICKER = "NG=F"
OIL_TICKER = "CL=F"
START_DATE = "2015-01-01"   # moved back to ensure enough history
ZSCORE_WINDOW = 90
N_SPLITS = 5

N_ESTIMATORS = 300
MAX_DEPTH = 8
MIN_SAMPLES_LEAF = 5
RANDOM_STATE = 42

UPPER_PROB = 0.55
LOWER_PROB = 0.45

FORECAST_TXT = "forecast_output.txt"
FORECAST_LOG = "forecast_log.csv"

# ----------------------------
# UTIL: download prices (safe)
# ----------------------------
def download_close_series(ticker, start):
    # try to download; accept Close or Adj Close
    df = yf.download(ticker, start=start, progress=False, auto_adjust=False)
    if df.empty:
        raise RuntimeError(f"yfinance returned empty DataFrame for {ticker}")
    # prefer 'Close'; fall back to 'Adj Close' or last available column
    if "Close" in df.columns:
        series = df["Close"].rename(ticker)
    elif "Adj Close" in df.columns:
        series = df["Adj Close"].rename(ticker)
    else:
        # pick last numeric column
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if not numeric_cols:
            raise RuntimeError(f"No numeric columns for ticker {ticker}")
        series = df[numeric_cols[-1]].rename(ticker)
    series.index = pd.to_datetime(series.index)
    return series

def load_prices(start=START_DATE):
    print(f"[INFO] Downloading price series since {start} ...")
    gas_s = download_close_series(GAS_TICKER, start)
    oil_s = download_close_series(OIL_TICKER, start)

    # inner join on index (dates)
    df = pd.concat([gas_s, oil_s], axis=1, join="inner")
    df.columns = ["Gas_Close", "Oil_Close"]
    df = df.sort_index()
    df = df.dropna()
    if df.empty:
        raise RuntimeError("After join the price DataFrame is empty.")
    print(f"[INFO] price rows available: {len(df)} ({df.index[0].date()} -> {df.index[-1].date()})")
    return df

# ----------------------------
# FEATURES
# ----------------------------
def add_base_features(df):
    df = df.copy()

    # percent returns (no fill)
    df["Gas_Return"] = df["Gas_Close"].pct_change()
    df["Oil_Return"] = df["Oil_Close"].pct_change()

    # lags
    for lag in [1, 2, 3, 5]:
        df[f"Gas_Return_lag{lag}"] = df["Gas_Return"].shift(lag)
        df[f"Oil_Return_lag{lag}"] = df["Oil_Return"].shift(lag)

    # momentum & volatility
    df["Momentum5"] = df["Gas_Close"].pct_change(5).shift(1)
    df["Volatility5"] = df["Gas_Return"].rolling(5).std()

    # term-structure proxy (MA30 vs MA5) - shifted to avoid lookahead
    df["MA5"] = df["Gas_Close"].rolling(5).mean()
    df["MA30"] = df["Gas_Close"].rolling(30).mean()
    df["term_structure_proxy"] = (df["MA30"] - df["MA5"]) / df["MA5"]
    df["term_structure_proxy"] = df["term_structure_proxy"].shift(1)

    # TARGET: next-day up (use shift(-1) on Gas_Close to avoid leaking tomorrow)
    df["Target"] = (df["Gas_Close"].shift(-1) > df["Gas_Close"]).astype(int)

    # drop rows with NA caused by rolling/shifts
    df = df.dropna()

    return df

# ----------------------------
# Rolling z-score normalizer
# ----------------------------
def rolling_zscore(df, cols, window=ZSCORE_WINDOW):
    # returns DataFrame with same index and columns named "<col>_z"
    roll_mean = df[cols].rolling(window, min_periods=10).mean()
    roll_std = df[cols].rolling(window, min_periods=10).std().replace(0, np.nan)
    z = (df[cols] - roll_mean) / roll_std
    # fill initial NaNs with global zscore fallback to avoid completely NaN early-region
    fallback = (df[cols] - df[cols].mean()) / (df[cols].std().replace(0, np.nan))
    z = z.fillna(fallback)
    z.columns = [c + "_z" for c in cols]
    return z

# ----------------------------
# train + CV + final model
# ----------------------------
def train_and_validate(df, feat_cols_z):
    # check existence
    if "Target" not in df.columns:
        raise RuntimeError("DataFrame has no 'Target' column. Can't train.")
    X = df[feat_cols_z].fillna(0.0)
    y = df["Target"].astype(int)

    if len(X) < 50:
        raise RuntimeError(f"Not enough rows to train (need >=50, have {len(X)})")

    # TimeSeries CV
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
        preds = m.predict(Xte)
        acc = accuracy_score(yte, preds)
        accs.append(acc)
        print(f"[CV] split {i}: acc={acc:.4f}, test_rows={len(test_idx)}")

    # final model on full dataset
    final = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    final.fit(X, y)
    print(f"[CV] mean acc={np.mean(accs):.4f} std={np.std(accs):.4f}")
    return final, np.mean(accs), np.std(accs)

# ----------------------------
# write human & append log
# ----------------------------
def write_forecast_text(date, latest_price, prob_up, signal, aux_map):
    now = datetime.utcnow().isoformat(sep=" ")
    lines = []
    lines.append("===================================")
    lines.append("      NATURAL GAS PRICE FORECAST")
    lines.append("===================================")
    lines.append(f"Run UTC: {now}")
    lines.append(f"Data index date: {date}")
    lines.append("")
    lines.append(f"Latest price: {latest_price:.4f}")
    lines.append(f"Probability UP: {prob_up:.4f}")
    lines.append(f"Signal: {signal}")
    lines.append("")
    lines.append("Aux (z-scored):")
    for k, v in aux_map.items():
        lines.append(f"  {k:25s}: {v:.4f}")
    lines.append("===================================")
    with open(FORECAST_TXT, "w") as f:
        f.write("\n".join(lines))
    print(f"[OUT] wrote {FORECAST_TXT}")

def append_forecast_log(date, prob_up, signal):
    row = {
        "run_utc": datetime.utcnow().isoformat(sep=" "),
        "data_date": str(date),
        "prob_up": float(prob_up),
        "signal": signal
    }
    df_new = pd.DataFrame([row])
    if os.path.exists(FORECAST_LOG):
        try:
            df_old = pd.read_csv(FORECAST_LOG)
            df_comb = pd.concat([df_old, df_new], ignore_index=True)
            df_comb.to_csv(FORECAST_LOG, index=False)
        except Exception as e:
            # fallback: overwrite with only new row
            df_new.to_csv(FORECAST_LOG, index=False)
    else:
        df_new.to_csv(FORECAST_LOG, index=False)
    print(f"[OUT] appended {FORECAST_LOG}")

# ----------------------------
# MAIN
# ----------------------------
def main():
    try:
        df_prices = load_prices(START_DATE)
    except Exception as e:
        print("[ERROR] loading prices:", e)
        raise SystemExit(1)

    # build features
    df = add_base_features(df_prices)
    if "Target" not in df.columns:
        print("[ERROR] 'Target' column missing after feature build. aborting.")
        raise SystemExit(2)

    # select numeric columns to z-score
    base_numeric_feats = [
        "Gas_Return", "Oil_Return", "Momentum5", "Volatility5", "term_structure_proxy"
    ] + [f"Gas_Return_lag{l}" for l in [1,2,3,5]] + [f"Oil_Return_lag{l}" for l in [1,2,3,5]]

    # ensure these features exist in df
    available_feats = [c for c in base_numeric_feats if c in df.columns]
    if len(available_feats) < 6:
        print(f"[ERROR] Not enough base features available for model (found {len(available_feats)}).")
        raise SystemExit(3)

    z = rolling_zscore(df, available_feats, window=ZSCORE_WINDOW)
    # attach z columns
    df = pd.concat([df, z], axis=1)

    # model features: all z-scored versions that exist
    model_features = [c + "_z" for c in available_feats if (c + "_z") in df.columns]
    if len(model_features) == 0:
        print("[ERROR] No z-scored features available for model.")
        raise SystemExit(4)

    # training + CV
    try:
        model, cv_mean, cv_std = train_and_validate(df, model_features)
    except Exception as e:
        print("[ERROR] training failed:", e)
        raise SystemExit(5)

    # LIVE forecast: use last available z-row as model input
    last_index = df.index[-1]
    last_row_z = df.loc[[last_index], model_features].fillna(0.0)

    # attempt intraday price to refresh latest_price (best-effort)
    latest_price = df["Gas_Close"].iloc[-1]
    try:
        intraday = yf.Ticker(GAS_TICKER).history(period="1d", interval="1m", actions=False)
        if (not intraday.empty) and ("Close" in intraday.columns):
            latest_price = float(intraday["Close"].iloc[-1])
    except Exception:
        # ignore intraday errors (network/timeouts) — use last daily close
        pass

    # aux map for human output (take z-values if available)
    aux_map = {}
    for f in ["term_structure_proxy", "Momentum5", "Volatility5"]:
        key = f + "_z"
        aux_map[f] = float(df.iloc[-1].get(key, np.nan))

    # predict
    prob_up = model.predict_proba(last_row_z)[0][1]

    if prob_up > UPPER_PROB:
        signal = "UP"
    elif prob_up < LOWER_PROB:
        signal = "DOWN"
    else:
        signal = "NO_TRADE"

    # outputs
    write_forecast_text(last_index.date(), latest_price, prob_up, signal, aux_map)
    append_forecast_log(last_index.date(), prob_up, signal)

    # console summary
    print("=== SUMMARY ===")
    print(f"data_date: {last_index.date()}")
    print(f"latest_price: {latest_price:.4f}")
    print(f"prob_up: {prob_up:.4f}")
    print(f"signal: {signal}")
    print(f"cv_mean_acc: {cv_mean:.4f} (std {cv_std:.4f})")
    print(f"model_features_count: {len(model_features)}")

if __name__ == "__main__":
    main()
