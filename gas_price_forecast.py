#!/usr/bin/env python3
# ===================================================================
# gas_price_forecast.py
# Stable / Online / CI-safe
# Adds:
#  E) transaction costs & slippage
#  F) position sizing via vol-targeting + prob-based sizing
#  G) hard kill-switch (drawdown-based)
#  H) live-signal export (JSON)
# ===================================================================

import os
import json
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
# CONFIG (tweakable)
# =======================
START_DATE = "2015-01-01"
SYMBOL_GAS = "NG=F"
SYMBOL_OIL = "CL=F"
PROB_THRESHOLD = 0.5

# E) transaction costs & slippage defaults
TX_COST = 0.0005       # transaction cost per unit notional (0.05%)
SLIPPAGE = 0.0002      # slippage fraction applied on trade (0.02%)

# F) vol-targeting
TARGET_VOL_ANN = 0.20  # target annual vol (20%)
MAX_POSITION = 1.0     # cap notional

# D1/D3 config
KILL_SWITCH_GRID = [-0.05, -0.10, -0.15, -0.20]  # drawdown thresholds (negative)
REENTRY_LEVEL = 0.95   # re-enter when equity >= peak * REENTRY_LEVEL

# OUTPUTS folder
OUT_DIR = "outputs"

# =======================
# HELPERS
# =======================
def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)

def flatten_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [c[0] for c in df.columns]
    return df

def now_utc_iso():
    return datetime.utcnow().isoformat() + "Z"

# =======================
# DATA
# =======================
def load_prices():
    gas = yf.download(SYMBOL_GAS, start=START_DATE, auto_adjust=True, progress=False)
    oil = yf.download(SYMBOL_OIL, start=START_DATE, auto_adjust=True, progress=False)

    gas = flatten_columns(gas)[["Close"]].rename(columns={"Close": "Gas_Close"})
    oil = flatten_columns(oil)[["Close"]].rename(columns={"Close": "Oil_Close"})

    df = gas.join(oil, how="inner").dropna()
    print("[INFO] loaded prices:", df.shape)
    return df

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

    # regimes
    df["Gas_MA_50"] = df["Gas_Close"].rolling(50).mean()
    df["Gas_MA_200"] = df["Gas_Close"].rolling(200).mean()
    df["Trend_Regime"] = (df["Gas_MA_50"] > df["Gas_MA_200"]).astype(int)

    df["Gas_Vol_20"] = df["Gas_Return"].rolling(20).std()
    df["Gas_Vol_252"] = df["Gas_Return"].rolling(252).std()
    df["High_Vol_Regime"] = (df["Gas_Vol_20"] > df["Gas_Vol_252"]).astype(int)

    # optional fundamentals placeholders (will be 0 if loaders missing)
    df["Storage_Surprise_Z"] = 0.0
    df["LNG_Feedgas_Surprise_Z"] = 0.0

    # Target (next day direction; no leak)
    df["Target"] = (df["Gas_Return"].shift(-1) > 0).astype(int)

    df = df.dropna()
    return df

# =======================
# MODEL
# =======================
def train_model(df, features):
    model = RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=20, random_state=42
    )
    model.fit(df[features], df["Target"])
    return model

# =======================
# POSITION SIZING (E+F)
# - combine prob-strength with vol-targeting
# - returns position in [0, MAX_POSITION]
# =======================
def compute_position(prob_up, realized_vol_daily, prev_position, tx_cost=TX_COST):
    # prob strength (0..1)
    strength = max(0.0, (prob_up - 0.5) * 2.0)

    # realized_vol_daily -> annualize
    if (realized_vol_daily is None) or np.isnan(realized_vol_daily) or realized_vol_daily <= 0:
        vol_pos = 0.0
    else:
        realized_vol_ann = realized_vol_daily * np.sqrt(252.0)
        vol_target_mult = TARGET_VOL_ANN / realized_vol_ann if realized_vol_ann > 0 else 0.0
        vol_pos = vol_target_mult

    # combined
    raw = strength * vol_pos

    # cap and ensure non-negative
    pos = max(0.0, min(MAX_POSITION, raw))

    # account roughly for transaction cost by slightly reducing size if switching a lot
    # (we simply penalize immediate size relative to prev position to reduce churning)
    delta = abs(pos - (prev_position or 0.0))
    # simple shrink when delta large (this is a heuristic)
    if delta > 0.5:
        pos = pos * (1.0 - min(0.25, delta * 0.25))

    return pos

# =======================
# Walk-forward backtest with:
# - transaction costs & slippage (E)
# - vol-targeting + prob-based sizing (F)
# - hard kill-switch (G)
# - outputs equity series
# =======================
def walk_forward_backtest(df, features, train_window=750, max_drawdown=None, tx_cost=TX_COST, slippage=SLIPPAGE):
    equity = 1.0
    peak = 1.0
    trading = True
    prev_position = 0.0
    rows = []

    # precompute rolling realized vol (daily) using window=20
    df["RealizedVolDaily"] = df["Gas_Return"].rolling(20).std().fillna(method="bfill")

    for t in range(train_window, len(df) - 1):
        train = df.iloc[t - train_window : t]
        test = df.iloc[t : t + 1]

        model = train_model(train, features)
        prob_up = model.predict_proba(test[features])[0][1]

        # realized vol at time t (use prior window)
        realized_vol_daily = float(df["RealizedVolDaily"].iloc[t]) if not np.isnan(df["RealizedVolDaily"].iloc[t]) else None

        # compute raw position (0..MAX) by combining prob & vol-target
        position = compute_position(prob_up, realized_vol_daily, prev_position)

        # apply regime & trading flag
        trend_ok = bool(test["Trend_Regime"].iloc[0] == 1)
        vol_ok = bool(test["High_Vol_Regime"].iloc[0] == 0)
        if not (trend_ok and vol_ok and trading):
            position = 0.0

        # apply transaction cost & slippage when position changes
        delta_pos = position - prev_position
        # transaction cost reduces equity immediately proportional to change in notional
        tx_cost_amount = abs(delta_pos) * tx_cost * equity
        # slippage approximated as slippage fraction of notional times sign of trade (cost)
        slippage_amount = abs(delta_pos) * slippage * equity

        # market return next day
        ret = float(df["Gas_Return"].iloc[t + 1])

        # PnL before costs = position * ret
        pnl = position * ret

        # subtract costs (immediately)
        pnl_after_costs = pnl - (tx_cost_amount / equity) - (slippage_amount / equity)
        # Update equity (multiplicative)
        equity *= (1 + pnl_after_costs)

        # update peak/drawdown and trading kill-switch
        peak = max(peak, equity)
        drawdown = equity / peak - 1.0

        if max_drawdown is not None:
            if trading and drawdown <= max_drawdown:
                trading = False
            elif not trading and equity >= peak * REENTRY_LEVEL:
                trading = True

        rows.append(
            {
                "Date": df.index[t],
                "Equity": equity,
                "PnL": pnl_after_costs,
                "Position": position,
                "Drawdown": drawdown,
                "ProbUp": prob_up,
                "RealizedVolDaily": realized_vol_daily,
                "TxCostUsd": tx_cost_amount,
                "SlippageUsd": slippage_amount,
            }
        )

        prev_position = position

    out = pd.DataFrame(rows).set_index("Date")
    return out

# =======================
# STATS
# =======================
def stats(bt):
    if bt.empty:
        return {"TotalReturn": 0.0, "MaxDD": 0.0, "HitRate": 0.0}
    total_return = bt["Equity"].iloc[-1] - 1.0
    maxdd = (bt["Equity"] / bt["Equity"].cummax() - 1.0).min()
    hit_rate = (bt["PnL"] > 0).mean()
    return {"TotalReturn": float(total_return), "MaxDD": float(maxdd), "HitRate": float(hit_rate)}

# =======================
# PLOTTING (optional)
# =======================
def plot_equity(bt, title, path):
    if not PLOTTING_AVAILABLE:
        return
    safe_mkdir(os.path.dirname(path) or ".")
    bt["Equity"].plot(figsize=(10, 4))
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

# =======================
# Live signal export (H)
# =======================
def write_live_signal(prob_up, signal, position, acc=None, out_dir=OUT_DIR):
    safe_mkdir(out_dir)
    payload = {
        "timestamp": now_utc_iso(),
        "prob_up": float(prob_up),
        "signal": str(signal),
        "position": float(position),
        "cv_accuracy": None if acc is None else float(acc),
    }
    with open(os.path.join(out_dir, "last_signal.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    # also write a small human log
    with open(os.path.join(out_dir, "last_signal.txt"), "w", encoding="utf-8") as f:
        f.write(json.dumps(payload))

# =======================
# MAIN
# =======================
def main():
    safe_mkdir(OUT_DIR)

    df = build_features(load_prices())

    if len(df) < 300:
        print("[WARN] Not enough data")
        return

    features = [c for c in df.columns if c.startswith(("Gas_Return_lag", "Oil_Return_lag"))] + [
        "Trend_Regime", "High_Vol_Regime", "Storage_Surprise_Z", "LNG_Feedgas_Surprise_Z"
    ]

    # Sweep kill-switch thresholds (D1)
    results = []
    for dd in KILL_SWITCH_GRID:
        bt = walk_forward_backtest(df, features, train_window=750, max_drawdown=dd, tx_cost=TX_COST, slippage=SLIPPAGE)
        s = stats(bt)
        s["MaxDrawdownParam"] = dd
        results.append(s)

        # plot per option if plotting available
        plot_equity(bt, f"Equity (DD={dd:.0%})", os.path.join(OUT_DIR, f"equity_dd_{int(abs(dd)*100)}.png"))

    res_df = pd.DataFrame(results).sort_values("TotalReturn", ascending=False).reset_index(drop=True)
    res_df.to_csv(os.path.join(OUT_DIR, "kill_switch_sweep.csv"), index=False)

    best_dd = float(res_df.iloc[0]["MaxDrawdownParam"])
    print("[INFO] kill-switch sweep results:\n", res_df)
    print(f"[INFO] selected best drawdown threshold: {best_dd:.0%}")

    # final run with best kill-switch
    bt_final = walk_forward_backtest(df, features, train_window=750, max_drawdown=best_dd, tx_cost=TX_COST, slippage=SLIPPAGE)
    plot_equity(bt_final, "FINAL EQUITY (BEST CONFIG)", os.path.join(OUT_DIR, "equity_final.png"))
    final_stats = stats(bt_final)
    print("[FINAL STATS]", final_stats)

    # compute last live signal and persist (H)
    # we train model on full data and compute last prediction
    model = train_model(df, features)
    last_row = df.iloc[-1:]
    prob_up = float(model.predict_proba(last_row[features])[0][1])

    # compute position using today's realized vol (use last value)
    realized_vol_daily = float(df["Gas_Return"].rolling(20).std().iloc[-1])
    position = compute_position(prob_up, realized_vol_daily, prev_position=0.0)

    trend_ok = bool(last_row["Trend_Regime"].iloc[0] == 1)
    vol_ok = bool(last_row["High_Vol_Regime"].iloc[0] == 0)
    position = position if (trend_ok and vol_ok) else 0.0
    signal = "UP" if prob_up > PROB_THRESHOLD and position > 0.0 else "DOWN"

    # write live signal JSON
    write_live_signal(prob_up, signal, position, acc=None, out_dir=OUT_DIR)
    print("[OK] Live signal written to", os.path.join(OUT_DIR, "last_signal.json"))

# =======================
# ENTRY
# =======================
if __name__ == "__main__":
    main()
