import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# =====================
# CONFIG
# =====================
OUTPUT_FILE = "forecast_output.txt"

# FIXED FUNDAMENTAL INPUTS (weekly / slow moving)
EIA_STORAGE = 3935.0
US_PRODUCTION = 94.8
LNG_FEEDGAS = 0.0
FUTURES_CURVE = 0.0
COT_MANAGED_MONEY = 0.0

# WEIGHTS (unchanged)
WEIGHTS = {
    "storage": 0.30,
    "production": 0.20,
    "lng": 0.15,
    "curve": 0.10,
    "cot": 0.10,
    "momentum": 0.15,   # ✅ NEW (small but meaningful)
}

# =====================
# HELPERS
# =====================
def fetch_gas_prices(days=7):
    data = yf.download("NG=F", period=f"{days}d", interval="1d", progress=False)
    closes = data["Close"].dropna()
    return closes

def scale_to_0_10(value, low, high):
    value = max(min(value, high), low)
    return 10 * (value - low) / (high - low + 1e-9)

# =====================
# MAIN
# =====================
def main():
    # LIVE GAS DATA
    gas_prices = fetch_gas_prices(7)

    if len(gas_prices) < 6:
        raise RuntimeError("Not enough gas price data")

    latest_price = gas_prices.iloc[-1]
    momentum_5d = (latest_price / gas_prices.iloc[-6] - 1) * 100

    # SCALE INPUTS
    storage_s = scale_to_0_10(EIA_STORAGE, 3200, 4300)
    prod_s = scale_to_0_10(US_PRODUCTION, 85, 105)
    lng_s = scale_to_0_10(LNG_FEEDGAS, 0, 15)
    curve_s = FUTURES_CURVE
    cot_s = COT_MANAGED_MONEY
    momentum_s = scale_to_0_10(momentum_5d, -10, 10)

    # WEIGHTED SCORE
    score = (
        storage_s * WEIGHTS["storage"] +
        prod_s * WEIGHTS["production"] +
        lng_s * WEIGHTS["lng"] +
        curve_s * WEIGHTS["curve"] +
        cot_s * WEIGHTS["cot"] +
        momentum_s * WEIGHTS["momentum"]
    )

    prob_up = min(max(score / 10, 0), 1)

    # WRITE OUTPUT
    with open(OUTPUT_FILE, "w") as f:
        f.write("===================================\n")
        f.write("      NATURAL GAS PRICE FORECAST\n")
        f.write("===================================\n")
        f.write(f"Datum: {datetime.utcnow()} UTC\n\n")
        f.write("Eingabewerte (0–10 Skala):\n")
        f.write(f"  EIA Storage         : {storage_s:.2f}\n")
        f.write(f"  US Production       : {prod_s:.2f}\n")
        f.write(f"  LNG Feedgas         : {lng_s:.2f}\n")
        f.write(f"  Futures Curve       : {curve_s:.2f}\n")
        f.write(f"  COT Managed Money   : {cot_s:.2f}\n")
        f.write(f"  Gas Momentum (5d)   : {momentum_s:.2f}\n\n")
        f.write(f"Gewichteter Score: {score:.2f}\n")
        f.write(f"Wahrscheinlichkeit, dass Gaspreis steigt: {prob_up*100:.1f}%\n")
        f.write("===================================\n")

if __name__ == "__main__":
    main()
