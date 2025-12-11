#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gas_price_forecast.py
Daily automated run (GitHub Actions) - robust fetching + EIA fallback + clear outputs.
Outputs:
 - forecast_output.txt (human readable)
 - forecast_output.json (machine readable)
 - logs printed to stdout (CI logs)
Usage: python gas_price_forecast.py
Env:
 - EIA_API_KEY (optional) : if set, attempt to query EIA storage series
"""
from __future__ import annotations
import os
import sys
import json
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import requests

# external libs that should be installed (requirements.txt)
try:
    import yfinance as yf
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import accuracy_score
except Exception as e:
    print("Missing Python packages:", e)
    print("Install requirements.txt before running.")
    raise

# -------------------------
# Config
# -------------------------
START_DATE = "2015-01-01"
SYMBOL_GAS = "NG=F"   # Natural Gas continuous future / Yahoo
SYMBOL_OIL = "CL=F"   # WTI future / Yahoo
FORECAST_TXT = "forecast_output.txt"
FORECAST_JSON = "forecast_output.json"
PROB_THRESHOLD = 0.5

# EIA storage series id for US working gas in storage (weekly)
# We will attempt to use EIA API series: "NG.RNGWHHD.W" (example) or user can set a series in code
# For safety we use a common storage series id used in some scripts: "NG.RNGWHHD.W" is placeholder.
EIA_STORAGE_SERIES = os.getenv("EIA_STORAGE_SERIES", "NG.RNGWHHD.W")
EIA_API_KEY = os.getenv("EIA_API_KEY", None)

# Local cache file for storage (if remote fails)
STORAGE_CACHE = "storage_cache.csv"

# -------------------------
# Utilities
# -------------------------
def utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")

def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        cols = []
        for c in df.columns:
            if isinstance(c, tuple):
                # pick first non-empty element, join otherwise
                s = "_".join([str(x) for x in c if x is not None and str(x) != ""])
                cols.append(s)
            else:
                cols.append(str(c))
        df = df.copy()
        df.columns = cols
    return df

# -------------------------
# Fetchers
# -------------------------
def fetch_prices_yahoo(start=START_DATE) -> Tuple[pd.DataFrame, Dict[str,str]]:
    """Download gas + oil closes from yfinance. Returns df and sources info."""
    sources = {}
    print("[INFO] Fetching prices from Yahoo Finance...")
    gas = yf.download(SYMBOL_GAS, start=start, progress=False, auto_adjust=True)
    oil = yf.download(SYMBOL_OIL, start=start, progress=False, auto_adjust=True)

    gas = flatten_columns(gas)
    oil = flatten_columns(oil)

    # pick column named 'Close' or fallback
    def pick_close(df: pd.DataFrame) -> str:
        for c in df.columns:
            if str(c).lower() == "close":
                return c
        # fallback: first numeric column
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                return c
        raise RuntimeError("No close-like column found in price DataFrame")

    gc = pick_close(gas)
    oc = pick_close(oil)

    gas = gas[[gc]].rename(columns={gc: "Gas_Close"})
    oil = oil[[oc]].rename(columns={oc: "Oil_Close"})
    df = gas.join(oil, how="inner")
    df = df.sort_index().dropna()

    sources["gas"] = f"yahoo:{SYMBOL_GAS}"
    sources["oil"] = f"yahoo:{SYMBOL_OIL}"
    print(f"[INFO] Prices fetched: rows={len(df)}, columns={list(df.columns)}")
    return df, sources

def fetch_eia_storage_weekly(series_id: str = EIA_STORAGE_SERIES, api_key: Optional[str] = EIA_API_KEY) -> Tuple[Optional[pd.DataFrame], Dict[str,str]]:
    """
    Attempt to fetch EIA series via API (series/observations).
    If api_key is None, skip and return (None, info) - caller will use cache fallback.
    Returns (df or None, sources dict)
    df columns: Date (pd.Timestamp), Storage (float)
    """
    sources = {}
    if not api_key:
        sources["storage"] = "EIA:SKIPPED(no API key)"
        return None, sources

    url = "https://api.eia.gov/series/"
    params = {"api_key": api_key, "series_id": series_id}
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        if "series" not in data or not data["series"]:
            sources["storage"] = f"EIA:ERROR(no series returned)"
            return None, sources
        obs = data["series"][0].get("data", [])
        # obs is list of [date_string, value] with date like '2025-11-14'
        df = pd.DataFrame(obs, columns=["date", "value"])
        df["Date"] = pd.to_datetime(df["date"])
        df["Storage"] = pd.to_numeric(df["value"], errors="coerce")
        df = df[["Date", "Storage"]].sort_values("Date").reset_index(drop=True)
        sources["storage"] = f"EIA:{series_id}"
        return df, sources
    except Exception as e:
        sources["storage"] = f"EIA:ERROR({str(e)})"
        return None, sources

def read_storage_cache(path: str = STORAGE_CACHE) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, parse_dates=["Date"])
        if "Storage" in df.columns:
            return df[["Date","Storage"]].sort_values("Date").reset_index(drop=True)
    except Exception:
        pass
    return None

def write_storage_cache(df: pd.DataFrame, path: str = STORAGE_CACHE) -> None:
    try:
        df[["Date","Storage"]].to_csv(path, index=False)
    except Exception:
        pass

# -------------------------
# Feature & model helpers
# -------------------------
def prepare_features(df_prices: pd.DataFrame, storage_df: Optional[pd.DataFrame], feedgas_df: Optional[pd.DataFrame]=None) -> Tuple[pd.DataFrame, Dict[str,Any]]:
    """
    Build features. Returns (df, meta) where meta contains which sources were used & indicators.
    Important: no leakage — target = next day direction (shift -1)
    """
    df = df_prices.copy()
    meta: Dict[str, Any] = {"sources": {}, "notes": []}

    # returns
    df["Gas_Return"] = df["Gas_Close"].pct_change()
    df["Oil_Return"] = df["Oil_Close"].pct_change()
    for l in range(1,6):
        df[f"Gas_Return_lag{l}"] = df["Gas_Return"].shift(l)
        df[f"Oil_Return_lag{l}"] = df["Oil_Return"].shift(l)

    # optional storage: compute Surprise as (change - rolling expectation) shifted 1
    df["Storage_Surprise"] = 0.0
    if storage_df is not None and not storage_df.empty:
        # align weekly storage to daily index: forward-fill backward-fill as appropriate
        s = storage_df.copy()
        s = s.set_index("Date").sort_index()
        s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D"))
        s["Storage"] = s["Storage"].ffill().bfill()
        s = s.reset_index().rename(columns={"index":"Date"})
        # compute weekly-like changes on original weekly points: use original storage_df for changes
        sd = storage_df.copy().sort_values("Date").reset_index(drop=True)
        sd["Change"] = sd["Storage"].diff()
        sd["Exp"] = sd["Change"].rolling(4, min_periods=1).mean()
        sd["Surprise"] = (sd["Change"] - sd["Exp"]).shift(1)
        # map surprise to daily by forward-fill
        sd_daily = sd.set_index("Date")[["Surprise"]].reindex(df.index, method="ffill")
        sd_daily = sd_daily.fillna(0.0)
        df["Storage_Surprise"] = sd_daily["Surprise"].values
        meta["sources"]["storage"] = "used"
    else:
        meta["sources"]["storage"] = "missing"

    # optional feedgas: same pattern
    df["Feedgas_Surprise"] = 0.0
    if feedgas_df is not None and not feedgas_df.empty:
        fg = feedgas_df.copy().sort_values("Date").reset_index(drop=True)
        fg["Change"] = fg["Feedgas"].diff()
        fg["Exp"] = fg["Change"].rolling(4, min_periods=1).mean()
        fg["Surprise"] = (fg["Change"] - fg["Exp"]).shift(1)
        fg_daily = fg.set_index("Date")[["Surprise"]].reindex(df.index, method="ffill").fillna(0.0)
        df["Feedgas_Surprise"] = fg_daily["Surprise"].values
        meta["sources"]["feedgas"] = "used"
    else:
        meta["sources"]["feedgas"] = "missing"

    # scale surprises robustly (Z-like using rolling median/IQR) and shift to avoid leak
    try:
        roll = pd.Series(df["Storage_Surprise"]).rolling(window=52, min_periods=1)
        med = roll.median()
        iqr = roll.quantile(0.75) - roll.quantile(0.25)
        df["Storage_Surprise_Z"] = ((df["Storage_Surprise"] - med) / (iqr.replace(0, np.nan))).shift(1).fillna(0.0).replace([np.inf, -np.inf], 0.0)
    except Exception:
        df["Storage_Surprise_Z"] = 0.0

    try:
        rollf = pd.Series(df["Feedgas_Surprise"]).rolling(window=52, min_periods=1)
        df["Feedgas_Surprise_Z"] = ((df["Feedgas_Surprise"] - rollf.mean()) / (rollf.std().replace(0, np.nan))).shift(1).fillna(0.0).replace([np.inf, -np.inf], 0.0)
    except Exception:
        df["Feedgas_Surprise_Z"] = 0.0

    # regime features (no leakage)
    df["Gas_MA_50"] = df["Gas_Close"].rolling(50).mean()
    df["Gas_MA_200"] = df["Gas_Close"].rolling(200).mean()
    df["Trend_Regime"] = (df["Gas_MA_50"] > df["Gas_MA_200"]).astype(int)

    df["Gas_Vol_20"] = df["Gas_Return"].rolling(20).std()
    df["Gas_Vol_252"] = df["Gas_Return"].rolling(252).std()
    df["High_Vol_Regime"] = (df["Gas_Vol_20"] > df["Gas_Vol_252"]).astype(int)

    # target: next day direction (no leak)
    df["Target"] = (df["Gas_Return"].shift(-1) > 0).astype(int)

    df = df.dropna().copy()
    meta["n_rows"] = len(df)
    return df, meta

def train_and_cv(df: pd.DataFrame, features: list) -> Tuple[RandomForestClassifier, float, float]:
    X = df[features]
    y = df["Target"]
    tscv = TimeSeriesSplit(n_splits=5)
    accs = []
    model = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=5, random_state=42)
    for tr, te in tscv.split(X):
        model.fit(X.iloc[tr], y.iloc[tr])
        pred = model.predict(X.iloc[te])
        accs.append(accuracy_score(y.iloc[te], pred))
    # final fit on all
    model.fit(X, y)
    return model, float(np.mean(accs)), float(np.std(accs))
def write_outputs(res):
    """
    Writes forecast_output.txt and forecast_output.json
    from the dictionary 'res' returned by your forecast logic.
    Safe defaults included to avoid NameErrors.
    """

    import json

    # Extract with fallbacks
    prob_up_raw = res.get("prob_up_raw", 0.5)
    prob_up_adj = res.get("prob_up_adj", prob_up_raw)
    prob_down_adj = 1 - prob_up_adj
    confidence = res.get("confidence", 0.0)
    data_date = res.get("data_date", "N/A")
    now_utc = res.get("now_utc", "N/A")
    signal = res.get("signal", "UNKNOWN")

   

    # ---------------- JSON ----------------
    with open("forecast_output.json", "w") as jf:
        json.dump(
            {
                "timestamp_utc": now_utc,
                "data_date": data_date,
                "prob_up_raw": prob_up_raw,
                "prob_up_adj": prob_up_adj,
                "prob_down_adj": prob_down_adj,
                "confidence": confidence,
                "signal": signal,
            },
            jf,
            indent=2
        )

    print("[OK] Outputs written: forecast_output.txt forecast_output.json")

# -------------------------
# Main run logic
# -------------------------
def run_one_cycle() -> Dict[str,Any]:
    out: Dict[str, Any] = {"run_time_utc": utc_now_str(), "data": {}, "model": {}}

    # 1) prices
    try:
        df_prices, src_prices = fetch_prices_yahoo()
        out["data"].update(src_prices)
    except Exception as e:
        raise RuntimeError("Failed to fetch prices: " + str(e))

    # 2) EIA storage attempt
    storage_df = None
    feedgas_df = None
    storage_sources = {}
    # Try EIA
    sd, ssrc = fetch_eia_storage_weekly()
    storage_sources.update(ssrc)
    if sd is not None:
        storage_df = sd
        write_storage_cache(sd)  # update cache
        out["data"]["storage_fetch"] = ssrc.get("storage", "unknown")
    else:
        # fallback to cache
        cached = read_storage_cache()
        if cached is not None:
            storage_df = cached
            out["data"]["storage_fetch"] = "cache"
            out["data"]["storage_note"] = "Used cached storage due to EIA failure"
        else:
            out["data"]["storage_fetch"] = ssrc.get("storage", "missing")
            out["data"]["storage_note"] = "No storage available (EIA and cache missing). Influence reduced."

    # We do not implement feedgas live fetch in core script (could plug fetch_lng_feedgas module)
    out["data"]["feedgas_fetch"] = "not_fetched"

    # 3) prepare features
    df, meta = prepare_features(df_prices, storage_df, feedgas_df)
    out["meta"] = meta

    # 4) features list
    feature_cols = [c for c in df.columns if c.startswith("Gas_Return_lag") or c.startswith("Oil_Return_lag")]
    # add surprises/regimes
    add_cols = []
    if "Storage_Surprise_Z" in df.columns:
        add_cols.append("Storage_Surprise_Z")
    if "Feedgas_Surprise_Z" in df.columns:
        add_cols.append("Feedgas_Surprise_Z")
    add_cols += ["Trend_Regime", "High_Vol_Regime"]
    features = feature_cols + [c for c in add_cols if c in df.columns]

    if len(df) < 50 or len(features) < 3:
        raise RuntimeError("Not enough data/features to train safely")

    # 5) train and cross-validate
    model, cv_mean, cv_std = train_and_cv(df, features)
    out["model"]["cv_mean"] = cv_mean
    out["model"]["cv_std"] = cv_std
    out["model"]["n_features"] = len(features)
    out["model"]["features"] = features

    # 6) live forecast: last row
    last_row = df.iloc[-1:]
    prob_up = float(model.predict_proba(last_row[features])[0][1])
    prob_down = 1.0 - prob_up

    out["model"]["prob_up_raw"] = prob_up
    out["model"]["prob_down_raw"] = prob_down

    # 7) Confidence adjustment depending on missing data
    # base confidence starts from CV accuracy
    base_conf = cv_mean
    conf = base_conf
    influence_notes = []
    # penalize if storage missing (because storage matters for fundamentals)
    storage_fetch = out["data"].get("storage_fetch", "missing")
    if storage_fetch != "used":
        # decrease confidence by 8% if cached used, 12% if completely missing — heuristics
        if storage_fetch == "cache":
            conf -= 0.08
            influence_notes.append("storage_from_cache: -8% conf")
        else:
            conf -= 0.12
            influence_notes.append("storage_missing: -12% conf")
    # feedgas not present -> small penalty
    if out["data"].get("feedgas_fetch", "not_fetched") != "used":
        conf -= 0.02
        influence_notes.append("feedgas_missing: -2% conf")

    # clamp
    conf = max(0.0, min(1.0, conf))
    out["model"]["confidence"] = conf
    out["model"]["confidence_notes"] = influence_notes

    # 8) final adjusted probabilities (note: this is heuristic weighting for display)
    # We will present the raw model prob and an adjusted prob weighted by confidence (not altering model)
    adjusted_prob_up = prob_up * conf + 0.5 * (1.0 - conf)  # move to 0.5 when very low confidence
    adjusted_prob_down = 1.0 - adjusted_prob_up

    out["model"]["prob_up_adjusted"] = adjusted_prob_up
    out["model"]["prob_down_adjusted"] = adjusted_prob_down

    # 9) produce human-friendly assessment
    out["assessment"] = {
        "statement": f"Probability UP: {adjusted_prob_up:.2%}, DOWN: {adjusted_prob_down:.2%}",
        "signal": "UP" if adjusted_prob_up > adjusted_prob_down else "DOWN",
    }

    # 10) data snapshot for output (latest numeric values)
    last_prices = last_row[["Gas_Close","Oil_Close"]].iloc[0].to_dict()
    out["latest"] = {"Gas_Close": float(last_prices["Gas_Close"]), "Oil_Close": float(last_prices["Oil_Close"]), "date": str(df.index[-1].date())}
    # storage snapshot if present
    if storage_df is not None and not storage_df.empty:
        out["latest"]["storage_latest"] = {"date": str(storage_df["Date"].iloc[-1].date()), "value": float(storage_df["Storage"].iloc[-1])}
    else:
        out["latest"]["storage_latest"] = None

    return out

# ============ WRITE OUTPUT FILES ============

# Safety defaults — falls einzelne Werte nicht vorhanden sind
prob_up_raw = locals().get("prob_up_raw", 0.5)
prob_up_adj = locals().get("prob_up_adj", prob_up_raw)
confidence  = locals().get("model_confidence", 0.0)
data_date   = locals().get("data_date", "N/A")
now_utc     = locals().get("now_utc", "N/A")

prob_down_adj = 1 - prob_up_adj

# -------- JSON OUTPUT --------
json.dump(
    {
        "timestamp_utc": now_utc,
        "data_date": data_date,
        "prob_up_raw": prob_up_raw,
        "prob_up_adj": prob_up_adj,
        "prob_down_adj": prob_down_adj,
        "confidence": confidence,
        "signal": signal,
    },
    open("forecast_output.json", "w"),
    indent=2
)

print("[OK] Outputs written: forecast_output.txt forecast_output.json")

# -------------------------
# Entrypoint
# -------------------------
def main():
    try:
        res = run_one_cycle()
        write_outputs(res)
        print("[OK] Outputs written:", FORECAST_TXT, FORECAST_JSON)
    except Exception as e:
        print("[ERROR] Run failed:", str(e))
        # write minimal failure file
        with open(FORECAST_TXT, "w", encoding="utf-8") as f:
            f.write(f"FAILED: {str(e)}\n")
        raise

if __name__ == "__main__":
    main()
with open("forecast_output.txt", "w") as f:
    f.write(f"Forecast Run: {datetime.now()}\n")
    f.write(f"Probability UP: {prob_up:.2%}\n")
    f.write(f"Probability DOWN: {1-prob_up:.2%}\n")
