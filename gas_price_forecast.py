#!/usr/bin/env python3
"""
gas_price_forecast.py
Final, robust, CI-safe version.

Features:
 - Loads gas & oil prices from Yahoo (yfinance)
 - Optional: loads EIA storage and LNG feedgas via fetch_eia_storage / fetch_lng_feedgas modules
 - Builds leak-free features (lags, storage/feedgas surprise, scaled)
 - Validates via TimeSeriesSplit, trains RandomForest, outputs forecast and diagnostics
 - Writes forecast_output.txt and forecast_output.json
"""
from __future__ import annotations

import os
import json
import math
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# ML
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

# prices
import yfinance as yf

# optional loaders (if present in repo)
try:
    import fetch_eia_storage  # optional module expected to provide load_storage_data()
except Exception:
    fetch_eia_storage = None

try:
    import fetch_lng_feedgas  # optional module expected to provide load_lng_feedgas()
except Exception:
    fetch_lng_feedgas = None

# -----------------------
# Config
# -----------------------
START_DATE = "2015-01-01"
SYMBOL_GAS = "NG=F"
SYMBOL_OIL = "CL=F"
FORECAST_FILE_TXT = "forecast_output.txt"
FORECAST_FILE_JSON = "forecast_output.json"
PROB_THRESHOLD = 0.5
# -----------------------
# Capital Management
# -----------------------
START_CAPITAL = 5000.0
WEEKLY_TARGET_EUR = 250.0
MAX_RISK_PER_TRADE = 0.02      # 2% vom Kapital
MAX_LEVERAGE = 5               # z.B. Futures / CFD Faktor
MIN_TRADE_EUR = 25.0


# -----------------------
# Helpers
# -----------------------
def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """If yfinance returns MultiIndex columns, flatten to simple strings."""
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) and len(c) > 0 else str(c) for c in df.columns]
    else:
        df.columns = [str(c) for c in df.columns]
    return df

def try_to_datetime_col(df: pd.DataFrame, col: str = "Date") -> pd.DataFrame:
    """Ensure a Date column is datetime (if exists)."""
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

# -----------------------
# Load Prices
# -----------------------
def load_prices(start_date: str = START_DATE) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Download gas and oil prices from yfinance and return dataframe with columns:
      Gas_Close, Oil_Close
    Also return a small dict 'sources' describing where values are from.
    """
    sources = {"gas": f"yahoo:{SYMBOL_GAS}", "oil": f"yahoo:{SYMBOL_OIL}"}
    print("[INFO] Fetching prices from Yahoo Finance...")
    gas_raw = yf.download(SYMBOL_GAS, start=start_date, progress=False, auto_adjust=True)
    oil_raw = yf.download(SYMBOL_OIL, start=start_date, progress=False, auto_adjust=True)

    gas = flatten_columns(gas_raw)
    oil = flatten_columns(oil_raw)

    # find Close column robustly
    if "Close" not in gas.columns:
        # try any column containing 'close'
        cand = [c for c in gas.columns if "close" in c.lower()]
        if cand:
            close_g = cand[0]
        else:
            raise RuntimeError(f"Cannot find Close column in gas dataframe: {gas.columns.tolist()}")
    else:
        close_g = "Close"

    if "Close" not in oil.columns:
        cand = [c for c in oil.columns if "close" in c.lower()]
        if cand:
            close_o = cand[0]
        else:
            raise RuntimeError(f"Cannot find Close column in oil dataframe: {oil.columns.tolist()}")
    else:
        close_o = "Close"

    gas = gas[[close_g]].rename(columns={close_g: "Gas_Close"})
    oil = oil[[close_o]].rename(columns={close_o: "Oil_Close"})

    df = gas.join(oil, how="inner")
    df = df.sort_index().dropna(how="any")
    print(f"[INFO] Prices fetched: rows={len(df)}, columns={list(df.columns)}")
    return df, sources

# -----------------------
# External data loaders (optional)
# -----------------------
def load_storage_optional() -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Attempt to load storage DataFrame via fetch_eia_storage.
    Expected to return DataFrame with columns ['Date','Storage'] where Date is parseable.
    Returns (df, note). On failure returns (None, error_message).
    """
    if fetch_eia_storage is None:
        return None, "module_missing"
    # try a few expected function names
    loader = None
    for name in ("load_storage_data", "load_eia_storage", "load_storage"):
        if hasattr(fetch_eia_storage, name):
            loader = getattr(fetch_eia_storage, name)
            break
    if loader is None:
        return None, "no_loader_fn"
    try:
        s = loader()
        s = try_to_datetime_col(s, "Date")
        # ensure numeric Storage column exists
        if "Storage" not in s.columns:
            return None, "no_Storage_col"
        s = s.sort_values("Date").reset_index(drop=True)
        return s, "ok"
    except Exception as e:
        return None, f"error:{e}"

def load_feedgas_optional() -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Attempt to load LNG feedgas (sendout) DataFrame via fetch_lng_feedgas.
    Expected columns: ['Date','Feedgas'].
    """
    if fetch_lng_feedgas is None:
        return None, "module_missing"
    loader = None
    for name in ("load_lng_feedgas", "load_feedgas", "load_lng"):
        if hasattr(fetch_lng_feedgas, name):
            loader = getattr(fetch_lng_feedgas, name)
            break
    if loader is None:
        return None, "no_loader_fn"
    try:
        f = loader()
        f = try_to_datetime_col(f, "Date")
        if "Feedgas" not in f.columns:
            return None, "no_Feedgas_col"
        f = f.sort_values("Date").reset_index(drop=True)
        return f, "ok"
    except Exception as e:
        return None, f"error:{e}"

# -----------------------
# Feature Engineering
# -----------------------
def add_lags(df: pd.DataFrame, col: str, n: int = 5) -> pd.DataFrame:
    for i in range(1, n + 1):
        df[f"{col}_lag{i}"] = df[col].shift(i)
    return df

def scale_robust_z(series: pd.Series, window: int = 52) -> pd.Series:
    """
    Robust scaling: (x - median) / (q75 - q25). Returns series shifted by 1 to avoid leakage.
    Fills NaN/inf with 0.
    """
    roll_med = series.rolling(window).median()
    roll_q75 = series.rolling(window).quantile(0.75)
    roll_q25 = series.rolling(window).quantile(0.25)
    iqr = (roll_q75 - roll_q25).replace(0, np.nan)
    z = (series - roll_med) / iqr
    z = z.shift(1)  # shift to avoid leak
    z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return z

def build_features(df_prices: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Build features:
      - Gas_Return, Oil_Return
      - lags 1..5
      - Storage_Surprise_Z (optional)
      - Feedgas_Surprise_Z (optional)
      - additional simple features (momentum)
    Returns (df_features, meta)
    """
    df = df_prices.copy()
    meta = {"sources": {}, "notes": []}

    # basic returns
    df["Gas_Return"] = df["Gas_Close"].pct_change()
    df["Oil_Return"] = df["Oil_Close"].pct_change()

    # lags
    for l in range(1, 6):
        df[f"Gas_Return_lag{l}"] = df["Gas_Return"].shift(l)
        df[f"Oil_Return_lag{l}"] = df["Oil_Return"].shift(l)

    # momentum & moving averages (no leak: use shift where needed)
    df["Momentum5"] = df["Gas_Close"].pct_change(5).shift(1)  # use past info only
    df["SMA10"] = df["Gas_Close"].rolling(10).mean().shift(1)
    df["Volatility5"] = df["Gas_Return"].rolling(5).std().shift(1)

    # Target: NEXT DAY direction (no leak)
    df["Target"] = (df["Gas_Return"].shift(-1) > 0).astype(int)

       # Optional: storage surprise
    storage_df, storage_note = load_storage_optional()
    meta["sources"]["storage"] = storage_note

    if storage_df is None:
        df["Storage_Surprise_Z"] = 0.0
        meta["notes"].append("storage_missing")
    else:
        storage_df["Storage_Change"] = storage_df["Storage"].diff()
        storage_df["Storage_Exp"] = storage_df["Storage_Change"].rolling(4).mean()
        storage_df["Storage_Surprise"] = (
            storage_df["Storage_Change"] - storage_df["Storage_Exp"]
        ).shift(1)

        storage_df["Storage_Surprise_Z"] = scale_robust_z(
            storage_df["Storage_Surprise"].fillna(0.0), window=52
        )

        storage_df = storage_df[["Date", "Storage_Surprise_Z"]].rename(
            columns={"Date": "merge_Date"}
        )
        df.index.name = "merge_Date"
        left = df.reset_index()
       
        df = (
           left.merge(storage_df, on="merge_Date", how="left")
           .set_index("merge_Date")
       )
        left = df.reset_index().rename(columns={"index": "merge_Date"})
        df = (
            left.merge(storage_df, on="merge_Date", how="left")
            .set_index("merge_Date")
        )
    if "Storage_Surprise_Z" not in df.columns:
        df["Storage_Surprise_Z"] = 0.0
        df["Storage_Surprise_Z"] = df["Storage_Surprise_Z"].ffill().fillna(0.0)
        meta["notes"].append("storage_loaded")

        df["Days_Since_Storage"] = (
            df["Storage_Surprise_Z"]
            .ne(0)
            .astype(int)
            .groupby((df["Storage_Surprise_Z"] != 0).cumsum())
            .cumcount()
            .clip(0, 7)
        )

    # Optional: feedgas surprise
    feedgas_df, feedgas_note = load_feedgas_optional()
    meta["sources"]["feedgas"] = feedgas_note

    if feedgas_df is None:
        df["LNG_Feedgas_Surprise_Z"] = 0.0
        meta["notes"].append("feedgas_missing")
    else:
        feedgas_df["Feedgas_Change"] = feedgas_df["Feedgas"].diff()
        feedgas_df["Feedgas_Exp"] = feedgas_df["Feedgas_Change"].rolling(4).mean()
        feedgas_df["Feedgas_Surprise"] = (
            feedgas_df["Feedgas_Change"] - feedgas_df["Feedgas_Exp"]
        ).shift(1)

        feedgas_df["LNG_Feedgas_Surprise_Z"] = scale_robust_z(
            feedgas_df["Feedgas_Surprise"].fillna(0.0), window=52
        )

        feedgas_df = feedgas_df[["Date", "LNG_Feedgas_Surprise_Z"]].rename(
            columns={"Date": "merge_Date"}
        )

        left = df.reset_index().rename(columns={"index": "merge_Date"})
        df = (
            left.merge(feedgas_df, on="merge_Date", how="left")
            .set_index("merge_Date")
        )
        df["LNG_Feedgas_Surprise_Z"] = (
            df["LNG_Feedgas_Surprise_Z"].ffill().fillna(0.0)
        )
        meta["notes"].append("feedgas_loaded")

        df["Days_Since_Feedgas"] = (
            df["LNG_Feedgas_Surprise_Z"]
            .ne(0)
            .astype(int)
            .groupby((df["LNG_Feedgas_Surprise_Z"] != 0).cumsum())
            .cumcount()
            .clip(0, 7)
        )
    # --- Phase 2.1: market persistence for fundamentals ---

    # Storage persistence (EIA effect lasts several days)
    if "Storage_Surprise_Z" not in df.columns:
        df["Storage_Surprise_Z"] = 0.0

    df["Days_Since_Storage"] = (
        df["Storage_Surprise_Z"]
        .ne(0)
        .astype(int)
        .groupby((df["Storage_Surprise_Z"] != 0).cumsum())
        .cumcount()
        .clip(0, 7)
    )

    # LNG Feedgas persistence (flow regime)
    if "LNG_Feedgas_Surprise_Z" not in df.columns:
        df["LNG_Feedgas_Surprise_Z"] = 0.0

    df["Days_Since_Feedgas"] = (
        df["LNG_Feedgas_Surprise_Z"]
        .ne(0)
        .astype(int)
        .groupby((df["LNG_Feedgas_Surprise_Z"] != 0).cumsum())
        .cumcount()
        .clip(0, 7)
    )
    # --- Phase 2.2: seasonal regime weighting ---

    # Winter sensitivity: Nov–Mar
    winter_months = df.index.month.isin([11, 12, 1, 2, 3]).astype(int)
    df["Winter_Factor"] = 1.0 + 0.6 * winter_months

    # Apply winter weighting to storage
    df["Storage_Surprise_Z"] = df["Storage_Surprise_Z"] * df["Winter_Factor"]
    # --- Phase 2.2 B: LNG regime weighting ---

    lng_active = (df["Days_Since_Feedgas"] < 3).astype(int)
    df["LNG_Regime_Factor"] = 1.0 + 0.5 * lng_active

    df["LNG_Feedgas_Surprise_Z"] = (
        df["LNG_Feedgas_Surprise_Z"] * df["LNG_Regime_Factor"]
    )
    # --- Phase 2.3: Weather proxy via volatility & momentum ---

    df["Cold_Weather_Proxy"] = (
        df["Volatility5"].rolling(3).mean() *
        df["Momentum5"].abs()
    ).shift(1)

    df["Cold_Weather_Proxy_Z"] = scale_robust_z(
        df["Cold_Weather_Proxy"].fillna(0.0), window=52
    )
    # --- Phase 2.4: Regime Interactions (Weather, Storage, LNG) ---

    # Interaction Weather + Storage
    df["Weather_Storage_Interaction"] = df["Cold_Weather_Proxy_Z"] * df["Storage_Surprise_Z"]

    # Interaction Winter + LNG Feedgas
    df["Winter_LNG_Interaction"] = df["Cold_Weather_Proxy_Z"] * df["LNG_Feedgas_Surprise_Z"]

    # Interaction between Storage and LNG (if Feedgas available)
    if "LNG_Feedgas_Surprise_Z" in df.columns:
        df["Storage_LNG_Interaction"] = df["Storage_Surprise_Z"] * df["LNG_Feedgas_Surprise_Z"]
    # --- Phase 2.5 A: Volatility Regime Gate ---
    
    # Volatility regime (market stress proxy)
    df["High_Vol_Regime"] = (df["Volatility5"] > df["Volatility5"].rolling(252).median()).astype(int)
    
    # Gate storage & LNG impact by volatility
    df["Storage_Effective"] = df["Storage_Surprise_Z"] * df["High_Vol_Regime"]
    df["LNG_Effective"] = df["LNG_Feedgas_Surprise_Z"] * df["High_Vol_Regime"]
    
    meta["notes"].append("volatility_regime_gate")
    # --- Phase 2.5 B: Directional Regime (Backwardation / Contango Proxy) ---
    df["Directional_Regime"] = (
    (df["Gas_Close"] > df["Gas_Close"].rolling(60).mean())
    .shift(1)
    .fillna(0)
    .astype(int)
)
      
    # Gate storage & LNG by directional regime
    df["Storage_Directional"] = df["Storage_Effective"] * df["Directional_Regime"]
    df["LNG_Directional"] = df["LNG_Effective"] * df["Directional_Regime"]
    
    meta["notes"].append("directional_regime_gate")

    # final cleanup
    df = df.dropna(subset=["Gas_Close", "Oil_Close", "Target"])
    meta["rows_after"] = len(df)
   # -----------------------
    # Phase 2.6: Feature Pruning (low-variance filter)
    # -----------------------
    feature_cols = [
        c for c in df.columns
        if c not in ("Target", "Gas_Close", "Oil_Close")
    ]

    low_var_features = [
        c for c in feature_cols
        if df[c].std() < 1e-6
    ]

    if low_var_features:
        df = df.drop(columns=low_var_features)
        meta["notes"].append(f"pruned:{len(low_var_features)}")


    return df, meta


# -----------------------
# Model training + CV
# -----------------------
def train_and_validate(df: pd.DataFrame, feature_cols: List[str], n_splits: int = 5) -> Tuple[RandomForestClassifier, float, float]:
    """
    Train with TimeSeriesSplit CV. Return (fitted_model_on_full_data, cv_mean, cv_std)
    """
    if "Target" not in df.columns:
        raise RuntimeError("Target column missing")

    X = df[feature_cols].fillna(0.0)
    y = df["Target"]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    accs = []

    for tr, te in tscv.split(X):
        clf = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=5, random_state=42)
        clf.fit(X.iloc[tr], y.iloc[tr])
        pred = clf.predict(X.iloc[te])
        accs.append(accuracy_score(y.iloc[te], pred))

    # train final on full data
    final_clf = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=5, random_state=42)
    final_clf.fit(X, y)
    return final_clf, float(np.mean(accs)), float(np.std(accs))

# -----------------------
# Permutation importance helpers
# -----------------------
def permutation_importance_ts(model, df: pd.DataFrame, features: List[str], test_size: int = 250, rng_seed: int = 42) -> Dict[str, float]:
    """Permutation importance on recent test_size rows (time-safe)."""
    if len(df) < test_size:
        test = df.copy()
    else:
        test = df.iloc[-test_size:].copy()
    X = test[features].fillna(0.0)
    y = test["Target"]
    baseline = accuracy_score(y, model.predict(X))
    rng = np.random.default_rng(rng_seed)
    out = {}
    for f in features:
        Xp = X.copy()
        Xp[f] = rng.permutation(Xp[f].values)
        out[f] = baseline - accuracy_score(y, model.predict(Xp))
    return out

def rolling_permutation_importance_ts(df: pd.DataFrame, features: List[str], window: int = 500, step: int = 25) -> pd.DataFrame:
    """
    Rolling permutation importance across windows.
    Returns DataFrame indexed by window end date with columns = features + "_baseline".
    """
    rows = []
    for start in range(0, max(0, len(df) - window), step):
        end = start + window
        if end > len(df):
            break
        train = df.iloc[start:end]
        X = train[features].fillna(0.0)
        y = train["Target"]
        clf = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=5, random_state=42)
        clf.fit(X, y)
        baseline = clf.score(X, y)
        row = {"Date": train.index[-1], "_baseline": baseline}
        for f in features:
            Xp = X.copy()
            Xp[f] = np.random.permutation(Xp[f].values)
            row[f] = baseline - clf.score(Xp, y)
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    df_imp = pd.DataFrame(rows).set_index("Date")
    return df_imp

# -----------------------
# Outputs writer
# -----------------------
def load_capital_state():
    if not os.path.exists(CAPITAL_STATE_FILE):
        return {
            "capital": START_CAPITAL,
            "weekly_pnl": 0.0,
            "week_id": datetime.utcnow().isocalendar().week,
            "last_trade_pnl": 0.0
        }
    with open(CAPITAL_STATE_FILE, "r") as f:
        return json.load(f)

def save_capital_state(state):
    with open(CAPITAL_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

def write_outputs(result: Dict, txt_path: str = FORECAST_FILE_TXT, json_path: str = FORECAST_FILE_JSON) -> None:
    """
    result dict expected keys:
      - run_time, data_date, sources, meta, prob_up_raw, prob_up_adj, prob_down_adj, confidence, model_cv_mean, model_cv_std, numeric_snapshot, notes
    """
    now_str = result.get("run_time", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"))
    lines = []
    lines.append("===================================")
    lines.append("      NATURAL GAS PRICE FORECAST")
    lines.append("===================================")
    lines.append(f"Run time (UTC): {now_str}")
    lines.append(f"Data date     : {result.get('data_date','-')}")
    lines.append("")
    lines.append("Sources fetched:")
    for k, v in result.get("sources", {}).items():
        lines.append(f"  {k:12}: {v}")
    lines.append(f"model_cv     : {result.get('model_cv_mean',0):.2%} ± {result.get('model_cv_std',0):.2%}")
    lines.append("")
    lines.append(f"Model raw prob UP : {result.get('prob_up_raw',0.0):.2%}")
    lines.append(f"Adjusted prob UP  : {result.get('prob_up_adj',0.0):.2%}")
    lines.append(f"Adjusted prob DOWN: {result.get('prob_down_adj',0.0):.2%}")
    lines.append(f"Model confidence  : {result.get('confidence',0.0):.2%}")
    lines.append(f"Confidence notes  : {', '.join(result.get('notes',[]))}")
    lines.append("")
    lines.append("Latest numeric snapshot:")
    snap = result.get("numeric_snapshot", {})
    lines.append(f"  Gas_Close : {snap.get('Gas_Close', 'NA')}")
    lines.append(f"  Oil_Close : {snap.get('Oil_Close', 'NA')}")
    lines.append(f"  Storage(latest): {snap.get('Storage_latest', 'NOT AVAILABLE')}")
    lines.append("")
    lines.append("Assessment:")
    lines.append(f"  Probability UP: {result.get('prob_up_adj',0.0):.2%}, DOWN: {result.get('prob_down_adj',0.0):.2%}")
    lines.append(f"  Signal: {result.get('signal','-')}")
     # Phase 4C: trading output details
    if "signal_strength" in result:
        lines.append(f"  Signal strength : {result['signal_strength']}")
    if "position_size" in result:
        lines.append(f"  Position size   : {result['position_size']:.2f}")
    if "risk_cap" in result:
        lines.append(f"  Risk cap        : {result['risk_cap']:.2f}")
    lines.append(f"  Trade bias      : {result.get('trade_bias')}")
    lines.append(f"  Final position  : {result.get('final_position')}")
    lines.append(f"  Execution OK    : {result.get('execution_ok')}")
    lines.append(f"Capital (€)       : {result['capital']}")
    lines.append(f"Position (€)      : {result['trade_eur']}")
    lines.append(f"Notional (€)      : {result['trade_notional']}")
    lines.append(f"Weekly target (€) : {result['weekly_target']}")
    lines.append(f"Weekly PnL (€)    : {result['weekly_pnl']}")
    lines.append("===================================")

    # write txt
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # write json
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"[OK] Outputs written: {txt_path} {json_path}")

# -----------------------
# Main orchestration
# -----------------------
def main():
    # 1) load prices
    try:
        df_prices, sources = load_prices()
    except Exception as e:
        print("[ERROR] loading prices:", e)
        raise

    # 2) features
    df, meta = build_features(df_prices)
    # merge sources into meta
    meta_sources = {"prices_gas": sources.get("gas"), "prices_oil": sources.get("oil"),
                    "storage": meta.get("sources", {}).get("storage"), "feedgas": meta.get("sources", {}).get("feedgas")}

    # 3) ensure enough rows
    if len(df) < 120:
        print("[WARN] Not enough data after feature build (rows={})".format(len(df)))
        # still write an informative output with baseline 50/50
        res = {
            "run_time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "data_date": df.index[-1].date().isoformat() if len(df) else None,
            "sources": meta_sources,
            "meta": meta,
            "prob_up_raw": 0.5,
            "prob_up_adj": 0.5,
            "prob_down_adj": 0.5,
            "confidence": 0.0,
            "model_cv_mean": 0.0,
            "model_cv_std": 0.0,
            "numeric_snapshot": {
                "Gas_Close": df["Gas_Close"].iloc[-1] if len(df) else None,
                "Oil_Close": df["Oil_Close"].iloc[-1] if len(df) else None,
                "Storage_latest": None
            },
            "notes": ["not_enough_data"] ,
            "signal": "UNKNOWN"
        }
        write_outputs(res)
        return

    # 4) feature list
    feature_cols = [c for c in df.columns if isinstance(c, str) and (c.startswith("Gas_Return_lag") or c.startswith("Oil_Return_lag"))]
    # include optional surprises
    for extra in ("Days_Since_Storage", "Days_Since_Feedgas"):
        if extra in df.columns:
           feature_cols.append(extra)
    if "Storage_Surprise_Z" in df.columns:
        feature_cols.append("Storage_Surprise_Z")
    if "LNG_Feedgas_Surprise_Z" in df.columns:
        feature_cols.append("LNG_Feedgas_Surprise_Z")
    if "Storage_Surprise_Z" in df.columns:
        feature_cols += ["Storage_Surprise_Z", "Days_Since_Storage"]
    if "LNG_Feedgas_Surprise_Z" in df.columns:
        feature_cols += ["LNG_Feedgas_Surprise_Z", "Days_Since_Feedgas"]
    # also technical momentum / vol
    for extra in ("Momentum5","SMA10","Volatility5"):
        if extra in df.columns:
            feature_cols.append(extra)

    # Deduplicate and keep order
    seen = set()
    feature_cols = [x for x in feature_cols if not (x in seen or seen.add(x))]

    # 5) train & validate
    model, cv_mean, cv_std = train_and_validate(df, feature_cols, n_splits=5)

    # 6) predict last row
    last_row = df.iloc[-1:]
    prob_up_raw = float(model.predict_proba(last_row[feature_cols].fillna(0.0))[0][1])

    # 7) adjust probability & confidence based on missing data (simple heuristic)
    notes = []
    confidence = 1.0
    prob_up_adj = prob_up_raw

    # if storage or feedgas missing, reduce confidence and nudge probability towards 50%
    if meta.get("sources", {}).get("storage") != "ok":
        notes.append("storage_missing")
        confidence *= 0.88  # reduce confidence
        # nudge to 50% by blending
        prob_up_adj = 0.6 * prob_up_adj + 0.4 * 0.5

    if meta.get("sources", {}).get("feedgas") != "ok":
        notes.append("feedgas_missing")
        confidence *= 0.95
        prob_up_adj = 0.7 * prob_up_adj + 0.3 * 0.5

    # clip
    prob_up_adj = float(max(0.0, min(1.0, prob_up_adj)))
    prob_down_adj = 1.0 - prob_up_adj
    # Phase 3A: Trade / No-Trade filter
    trade_allowed = True
    if 0.48 <= prob_up_adj <= 0.52:
        trade_allowed = False
        notes.append("no_trade_neutral_prob")
    
    if confidence < 0.85:
        trade_allowed = False
        notes.append("no_trade_low_conf")
    
    if not trade_allowed:
        signal = "NO_TRADE"
     
    # determine signal
    signal = "UP" if prob_up_adj > PROB_THRESHOLD else "DOWN"
 
      # Phase 3B: Signal strength classification
    signal_strength = "NEUTRAL"
    
    if signal == "UP":
        if prob_up_adj >= 0.58 and confidence >= 0.90:
            signal_strength = "STRONG_UP"
        else:
            signal_strength = "WEAK_UP"
    
    elif signal == "DOWN":
        if prob_up_adj <= 0.42 and confidence >= 0.90:
            signal_strength = "STRONG_DOWN"
        else:
            signal_strength = "WEAK_DOWN"


    # numeric snapshot
    numeric_snapshot = {
        "Gas_Close": float(last_row["Gas_Close"].iloc[0]),
        "Oil_Close": float(last_row["Oil_Close"].iloc[0]),
        "Storage_latest": None,
    }
    # if storage loaded, get latest value if available
    if "Storage_Surprise_Z" in df.columns and fetch_eia_storage is not None:
        try:
            storage_df, _ = load_storage_optional()
            if storage_df is not None and len(storage_df):
                numeric_snapshot["Storage_latest"] = float(storage_df["Storage"].iloc[-1])
        except Exception:
            numeric_snapshot["Storage_latest"] = None

    # package result
    result = {
        "run_time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "data_date": last_row.index[0].date().isoformat(),
        "sources": meta_sources,
        "meta": meta,
        "prob_up_raw": prob_up_raw,
        "prob_up_adj": prob_up_adj,
        "prob_down_adj": prob_down_adj,
        "confidence": confidence,
        "model_cv_mean": cv_mean,
        "model_cv_std": cv_std,
        "numeric_snapshot": numeric_snapshot,
        "notes": notes,
        "signal": signal,
        "signal_strength": signal_strength,
    }

    # 8) permutation importance (final diagnostic)
    try:
        perm = permutation_importance_ts(model, df, feature_cols, test_size=250)
        result["perm_importance"] = perm
    except Exception as e:
        result["perm_importance"] = {}
        result["notes"].append(f"perm_error:{e}")
    # -----------------------
    # Risk per Signal Strength
    # -----------------------
    SIGNAL_RISK_MAP = {
        "STRONG_UP": 1.0,
        "WEAK_UP": 0.5,
        "WEAK_DOWN": 0.5,
        "STRONG_DOWN": 1.0,
        "NO_TRADE": 0.0
    }
    
    risk_factor = SIGNAL_RISK_MAP.get(signal_strength, 0.0)

     

     # Phase 4B skipped – superseded by Phase 3D
    position_size = 0.0

     # Phase 3E: risk cap based on model quality
    risk_cap = 1.0

    if cv_mean < 0.52:
        risk_cap *= 0.5
    if confidence < 0.7:
        risk_cap *= 0.5

    position_size = position_size * risk_cap
    position_size = max(-1.0, min(1.0, position_size))

    result["position_size"] = position_size
    result["risk_cap"] = risk_cap
    # -----------------------
    # Phase 5A: Trend Regime Filter
    # -----------------------
    trend_up = (
        df["Gas_Close"].iloc[-1]
        > df["Gas_Close"].rolling(50).mean().iloc[-1]
    )
    
    if signal == "UP" and not trend_up:
        position_size *= 0.5
        notes.append("counter_trend_up")
    
    elif signal == "DOWN" and trend_up:
        position_size *= 0.5
        notes.append("counter_trend_down")
    
    result["position_size"] = position_size
    result["trend_regime"] = "UPTREND" if trend_up else "DOWNTREND"
    #===== A2: Historical Signal Reconstruction =====
    hist = df.copy()
    
    probs = model.predict_proba(hist[feature_cols].fillna(0.0))[:, 1]
    hist["Prob_UP"] = probs
    
    # Historical signal
    hist["Hist_Signal"] = np.where(hist["Prob_UP"] > 0.5, "UP", "DOWN")
    
    # Historical signal strength
    hist["Hist_Strength"] = "NEUTRAL"
    hist.loc[hist["Prob_UP"] >= 0.58, "Hist_Strength"] = "STRONG_UP"
    hist.loc[(hist["Prob_UP"] >= 0.50) & (hist["Prob_UP"] < 0.58), "Hist_Strength"] = "WEAK_UP"
    hist.loc[(hist["Prob_UP"] <= 0.42), "Hist_Strength"] = "STRONG_DOWN"
    hist.loc[(hist["Prob_UP"] > 0.42) & (hist["Prob_UP"] < 0.50), "Hist_Strength"] = "WEAK_DOWN"
    # ===== A3: Historical Position Sizing =====
    hist["Hist_Position"] = 0.0
    
    hist.loc[hist["Hist_Strength"] == "STRONG_UP", "Hist_Position"] = 1.0
    hist.loc[hist["Hist_Strength"] == "WEAK_UP", "Hist_Position"] = 0.5
    hist.loc[hist["Hist_Strength"] == "WEAK_DOWN", "Hist_Position"] = -0.5
    hist.loc[hist["Hist_Strength"] == "STRONG_DOWN", "Hist_Position"] = -1.0
    # ===== A4: Clean Backtest =====
    hist["Strategy_Return"] = hist["Hist_Position"].shift(1) * hist["Gas_Return"]
    
    result["backtest"] = {
        "cum_pnl": float(hist["Strategy_Return"].cumsum().iloc[-1]),
        "avg_daily_return": float(hist["Strategy_Return"].mean()),
        "hit_rate": float((hist["Strategy_Return"] > 0).mean()),
    }

    
     # -----------------------
     # Phase 6B: Execution-Ready Signal
     # -----------------------
    final_position = position_size
      
    execution_ok = abs(final_position) >= 0.1 and trade_allowed
    # ===== A6: Go-Live Safety Brake =====
    if result.get("backtest", {}).get("hit_rate", 1.0) < 0.48:
        execution_ok = False
        result["notes"].append("execution_blocked_low_hit_rate")
     # -----------------------
     # Capital-based Position Sizing
     # -----------------------
    ACCOUNT_CAPITAL_EUR = capital
    risk_eur = ACCOUNT_CAPITAL_EUR * MAX_RISK_PER_TRADE

    position_eur = risk_eur * risk_factor * confidence

          
    trade_bias = (
        "LONG" if final_position > 0 else
        "SHORT" if final_position < 0 else
        "FLAT"
    )
      
    result["final_position"] = round(final_position, 2)
    result["trade_bias"] = trade_bias
    result["execution_ok"] = execution_ok
    result["final_position"] = round(position_size, 2)
    result["backtest_note"] = "Backtest is diagnostic only. No feedback into model."
    result["capital"] = ACCOUNT_CAPITAL_EUR
    result["position_eur"] = round(position_eur, 2)
    result["weekly_target_eur"] = round(
    ACCOUNT_CAPITAL_EUR * TARGET_WEEKLY_RETURN, 2
)
    capital = capital_state["capital"]   
    result["capital"] = round(capital, 2)
    result["trade_eur"] = round(trade_eur, 2)
    result["trade_notional"] = round(trade_notional, 2)
    result["weekly_target"] = WEEKLY_TARGET_EUR
    result["weekly_pnl"] = round(weekly_pnl, 2)
    ACCOUNT_CAPITAL_EUR = START_CAPITAL
    TARGET_WEEKLY_RETURN = WEEKLY_TARGET_EUR / START_CAPITAL
    CAPITAL_STATE_FILE = "capital_state.json"

    # -----------------------
    # Phase 7A: Trade sizing (€)
    # -----------------------
    capital_state = load_capital_state()
    
    capital = capital_state["capital"]
    weekly_pnl = capital_state["weekly_pnl"]
    
    risk_eur = capital * MAX_RISK_PER_TRADE
    base_trade_eur = risk_eur * abs(final_position)
    
    trade_eur = max(base_trade_eur, MIN_TRADE_EUR)
    
    # Leverage berücksichtigen
    trade_notional = trade_eur * MAX_LEVERAGE
    remaining_target = WEEKLY_TARGET_EUR - weekly_pnl


    # Wenn wir zurückliegen → leicht aggressiver
    if remaining_target > 0:
        urgency_factor = min(1.5, remaining_target / WEEKLY_TARGET_EUR + 1)
    else:
        urgency_factor = 0.8  # defensiver nach Zielerreichung
    
    trade_notional *= urgency_factor


   
    # 9) write outputs
    write_outputs(result)
# -----------------------
# Entrypoint
# -----------------------
if __name__ == "__main__":
    main()
