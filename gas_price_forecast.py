#!/usr/bin/env python3
# ===================================================================
# gas_price_forecast.py
# Stable / Online / CI-safe
# Adds: Kill-switch comparison (on/off) with walk-forward equity exports
# ===================================================================

import os
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

# Kill-switch params (configurable)
MAX_DRAWDOWN = -0.12  # when equity/peak -1 <= MAX_DRAWDOWN -> stop trading
REENTRY_LEVEL = 0.95  # equity must recover to peak * REENTRY_LEVEL to re-enable trading

# =======================
# HELPERS
# =======================
def flatten_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [c[0] for c in df.columns]
    return df

def safe_mkdir(path):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass

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

    # --- Trend & Vol Regimes ---
    df["Gas_MA_50"] = df["Gas_Close"].rolling(50).mean()
    df["Gas_MA_200"] = df["Gas_Close"].rolling(200).mean()
    df["Trend_Regime"] = (df["Gas_MA_50"] > df["Gas_MA_200"]).astype(int)

    df["Gas_Vol_20"] = df["Gas_Return"].rolling(20).std()
    df["Gas_Vol_252"] = df["Gas_Return"].rolling(252).std()
    df["High_Vol_Regime"] = (df["Gas_Vol_20"] > df["Gas_Vol_252"]).astype(int)

    # --- Storage Surprise (optional) ---
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

    # --- LNG Feedgas Surprise (optional) ---
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
# WALK-FORWARD BACKTEST (with toggle for kill-switch)
# =======================
def walk_forward_backtest(df, features, train_window=750, enable_kill_switch=True):
    """
    Walk-forward 1-step ahead backtest.
    enable_kill_switch: if True, apply the drawdown-based kill-switch logic.
    Returns: DataFrame with Date index and columns Signal, Return, PnL, Equity, Trading (bool)
    """
    records = []

    equity = 1.0
    peak = 1.0
    trading_enabled = True

    for t in range(train_window, len(df) - 1):
        train = df.iloc[t - train_window : t]
        test = df.iloc[t : t + 1]

        model = train_model(train, features)
        prob_up = model.predict_proba(test[features])[0][1]

        trend_ok = test["Trend_Regime"].iloc[0] == 1
        vol_ok = test["High_Vol_Regime"].iloc[0] == 0

        signal = int(
            trading_enabled
            and (prob_up > PROB_THRESHOLD)
            and trend_ok
            and vol_ok
        )

        ret = df["Gas_Return"].iloc[t + 1]
        pnl = signal * ret
        equity *= (1 + pnl)
        peak = max(peak, equity)
        drawdown = equity / peak - 1

        # apply kill-switch only if enabled
        if enable_kill_switch:
            if trading_enabled and drawdown <= MAX_DRAWDOWN:
                trading_enabled = False
            elif (not trading_enabled) and (equity >= peak * REENTRY_LEVEL):
                trading_enabled = True

        records.append(
            {
                "Date": df.index[t],
                "Signal": signal,
                "Return": ret,
                "PnL": pnl,
                "Equity": equity,
                "Trading": trading_enabled,
                "Drawdown": drawdown,
            }
        )

    out = pd.DataFrame(records).set_index("Date")
    return out

# =======================
# UTIL: simple stats
# =======================
def backtest_stats(bt_df):
    if bt_df.empty:
        return {}
    total_return = bt_df["Equity"].iloc[-1] - 1
    hit_rate = (bt_df["PnL"] > 0).mean()
    max_dd = (bt_df["Equity"] / bt_df["Equity"].cummax() - 1).min()
    return {"total_return": total_return, "hit_rate": hit_rate, "max_dd": max_dd}

# =======================
# MAIN (runs both variants and compares)
# =======================
def main():
    out_dir = "backtest_outputs"
    safe_mkdir(out_dir)

    df = build_features(load_prices())

    if len(df) < 300:
        print("[WARN] Not enough data")
        return

    features = [
        c for c in df.columns if c.startswith(("Gas_Return_lag", "Oil_Return_lag"))
    ] + [
        "Storage_Surprise_Z",
        "LNG_Feedgas_Surprise_Z",
        "Trend_Regime",
        "High_Vol_Regime",
    ]

    print("[INFO] Features used:", features)

    # 1) Run with Kill-Switch (default behaviour)
    print("\n[RUN] Walk-forward with Kill-Switch ENABLED")
    bt_with = walk_forward_backtest(df, features, train_window=750, enable_kill_switch=True)
    stats_with = backtest_stats(bt_with)
    bt_with.to_csv(os.path.join(out_dir, "walkforward_with_killswitch.csv"))

    # 2) Run without Kill-Switch
    print("\n[RUN] Walk-forward with Kill-Switch DISABLED")
    bt_without = walk_forward_backtest(df, features, train_window=750, enable_kill_switch=False)
    stats_without = backtest_stats(bt_without)
    bt_without.to_csv(os.path.join(out_dir, "walkforward_without_killswitch.csv"))

    # Print comparison
    print("\n=== COMPARISON: with vs without kill-switch ===")
    print(f"With  - total_return: {stats_with['total_return']:.2%}, hit_rate: {stats_with['hit_rate']:.2%}, max_dd: {stats_with['max_dd']:.2%}")
    print(f"Without - total_return: {stats_without['total_return']:.2%}, hit_rate: {stats_without['hit_rate']:.2%}, max_dd: {stats_without['max_dd']:.2%}")

    # Save comparison CSV
    comp = pd.DataFrame([{"variant":"with_kill","total_return":stats_with["total_return"],"hit_rate":stats_with["hit_rate"],"max_dd":stats_with["max_dd"]},
                         {"variant":"without_kill","total_return":stats_without["total_return"],"hit_rate":stats_without["hit_rate"],"max_dd":stats_without["max_dd"]}])
    comp.to_csv(os.path.join(out_dir, "kill_switch_comparison.csv"), index=False)

    # Write a short human-readable file
    with open(os.path.join(out_dir, "kill_switch_summary.txt"), "w", encoding="utf-8") as f:
        f.write("Kill-switch comparison\n")
        f.write(f"With  : total_return={stats_with['total_return']:.6f}, hit_rate={stats_with['hit_rate']:.6f}, max_dd={stats_with['max_dd']:.6f}\n")
        f.write(f"Without: total_return={stats_without['total_return']:.6f}, hit_rate={stats_without['hit_rate']:.6f}, max_dd={stats_without['max_dd']:.6f}\n")

    print("\n[OK] Backtest outputs written to", out_dir)

# =======================
# ENTRY
# =======================
if __name__ == "__main__":
    main()
