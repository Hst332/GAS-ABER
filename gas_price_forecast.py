#!/usr/bin/env python3
# ===================================================================
# gas_price_forecast.py
# Stable / Online / CI-safe
# D1: Kill-switch Sweep
# D2: Probability-based Position Sizing
# D3: Equity Curve Plots
# ===================================================================

import os
import numpy as np
import pandas as pd
from datetime import datetime

# =======================
# OPTIONAL PLOTTING (CI-SAFE)
# =======================
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
except Exception:
    PLOTTING_AVAILABLE = False

import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

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

# >>> D1: Kill-switch sweep grid
KILL_SWITCH_GRID = [-0.05, -0.10, -0.15, -0.20]
REENTRY_LEVEL = 0.95

# >>> D2: Position sizing
MAX_POSITION = 1.0   # full notional

# =======================
# HELPERS
# =======================
def flatten_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df

def safe_mkdir(path):
    os.makedirs(path, exist_ok=True)

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

    # --- Regimes ---
    df["Gas_MA_50"] = df["Gas_Close"].rolling(50).mean()
    df["Gas_MA_200"] = df["Gas_Close"].rolling(200).mean()
    df["Trend_Regime"] = (df["Gas_MA_50"] > df["Gas_MA_200"]).astype(int)

    df["Gas_Vol_20"] = df["Gas_Return"].rolling(20).std()
    df["Gas_Vol_252"] = df["Gas_Return"].rolling(252).std()
    df["High_Vol_Regime"] = (df["Gas_Vol_20"] > df["Gas_Vol_252"]).astype(int)

    # --- optional fundamentals ---
    df["Storage_Surprise_Z"] = 0.0
    df["LNG_Feedgas_Surprise_Z"] = 0.0

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
# WALK-FORWARD BACKTEST
# =======================
def walk_forward_backtest(
    df,
    features,
    train_window=750,
    max_drawdown=None,
):
    equity = 1.0
    peak = 1.0
    trading = True
    rows = []

    for t in range(train_window, len(df) - 1):
        train = df.iloc[t - train_window : t]
        test = df.iloc[t : t + 1]

        model = train_model(train, features)
        prob_up = model.predict_proba(test[features])[0][1]

        # >>> D2: Position sizing
        position = max(0.0, min(MAX_POSITION, (prob_up - 0.5) * 2))

        trend_ok = test["Trend_Regime"].iloc[0] == 1
        vol_ok = test["High_Vol_Regime"].iloc[0] == 0

        position *= (trend_ok and vol_ok and trading)

        ret = df["Gas_Return"].iloc[t + 1]
        pnl = position * ret
        equity *= (1 + pnl)
        peak = max(peak, equity)
        drawdown = equity / peak - 1

        # >>> D1: Kill-switch
        if max_drawdown is not None:
            if trading and drawdown <= max_drawdown:
                trading = False
            elif not trading and equity >= peak * REENTRY_LEVEL:
                trading = True

        rows.append(
            {
                "Date": df.index[t],
                "Equity": equity,
                "PnL": pnl,
                "Position": position,
                "Drawdown": drawdown,
            }
        )

    out = pd.DataFrame(rows).set_index("Date")
    return out

# =======================
# STATS
# =======================
def stats(bt):
    return {
        "TotalReturn": bt["Equity"].iloc[-1] - 1,
        "MaxDD": (bt["Equity"] / bt["Equity"].cummax() - 1).min(),
        "HitRate": (bt["PnL"] > 0).mean(),
    }

# =======================
# PLOTS (D3)
# =======================
def plot_equity(bt, title, path):
    if not PLOTTING_AVAILABLE:
        return

    bt["Equity"].plot(figsize=(10, 4))
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

# =======================
# MAIN
# =======================
def main():
    if PLOTTING_AVAILABLE:
    safe_mkdir("outputs")

    df = build_features(load_prices())

    features = [
        c for c in df.columns
        if c.startswith(("Gas_Return_lag", "Oil_Return_lag"))
    ] + [
        "Trend_Regime",
        "High_Vol_Regime",
    ]

    results = []

    # =======================
    # D1: Kill-switch sweep
    # =======================
    for dd in KILL_SWITCH_GRID:
        bt = walk_forward_backtest(df, features, max_drawdown=dd)
        s = stats(bt)
        s["MaxDrawdownParam"] = dd
        results.append(s)

        plot_equity(
            bt,
            f"Equity Curve (Kill DD = {dd:.0%})",
            f"outputs/equity_dd_{int(abs(dd)*100)}.png",
        )

    res_df = pd.DataFrame(results).sort_values(
        ["MaxDD", "TotalReturn"], ascending=[False, False]
    )
    res_df.to_csv("outputs/kill_switch_sweep.csv", index=False)

    # =======================
    # Best model
    # =======================
    best_dd = res_df.iloc[0]["MaxDrawdownParam"]

    print("\n[INFO] Kill-switch sweep result:")
    print(res_df)

    print(f"\n[INFO] Best MAX_DRAWDOWN = {best_dd:.0%}")

    # =======================
    # Final run with best DD
    # =======================
    bt_final = walk_forward_backtest(df, features, max_drawdown=best_dd)
    plot_equity(bt_final, "FINAL EQUITY (BEST CONFIG)", "outputs/equity_final.png")

    print("\n[FINAL STATS]")
    print(stats(bt_final))


# =======================
# ENTRY
# =======================
if __name__ == "__main__":
    main()
