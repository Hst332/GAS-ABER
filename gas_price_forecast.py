# =========================================
# GAS PRICE DIRECTION FORECAST (SAFE VERSION)
# =========================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score


# =====================
# CONFIG
# =====================
GAS_TICKER = "NG=F"
OIL_TICKER = "CL=F"

START_DATE = "2019-01-01"
END_DATE = None  # today

N_SPLITS = 5
N_ESTIMATORS = 300
MAX_DEPTH = 6

UPPER_PROB = 0.55
LOWER_PROB = 0.45


# =====================
# LOAD DATA
# =====================
def load_prices():
    gas = yf.download(GAS_TICKER, start=START_DATE, end=END_DATE)
    oil = yf.download(OIL_TICKER, start=START_DATE, end=END_DATE)

    gas = gas[["Close"]].rename(columns={"Close": "Gas_Close"})
    oil = oil[["Close"]].rename(columns={"Close": "Oil_Close"})

    df = gas.join(oil, how="inner")
    df.dropna(inplace=True)
    return df


# =====================
# FEATURE ENGINEERING
# =====================
def build_features(df):
    df = df.copy()

    df["Gas_Return"] = df["Gas_Close"].pct_change()
    df["Oil_Return"] = df["Oil_Close"].pct_change()

    for lag in range(1, 6):
        df[f"Gas_Return_lag{lag}"] = df["Gas_Return"].shift(lag)
        df[f"Oil_Return_lag{lag}"] = df["Oil_Return"].shift(lag)

    df["Momentum5"] = df["Gas_Close"].pct_change(5).shift(1)
    df["Volatility5"] = df["Gas_Return"].rolling(5).std()

    # TARGET: next-day direction
    df["Target"] = (df["Gas_Return"].shift(-1) > 0).astype(int)

    df.dropna(inplace=True)
    return df


# =====================
# TRAIN MODEL
# =====================
def train_model(df, features):
    X = df[features]
    y = df["Target"]

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    accs = []

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        accs.append(accuracy_score(y_test, preds))

    model.fit(X, y)

    print(f"Rolling Accuracy: mean={np.mean(accs)*100:.2f}%  std={np.std(accs)*100:.2f}%")
    return model


# =====================
# MAIN
# =====================
def main():
    print("Loading price data...")
    df = load_prices()
    df = build_features(df)

    features = [
        "Gas_Return",
        "Oil_Return",
        "Momentum5",
        "Volatility5"
    ] + [f"Gas_Return_lag{i}" for i in range(1, 6)] \
      + [f"Oil_Return_lag{i}" for i in range(1, 6)]

    print("Training model...")
    model = train_model(df, features)

    # =====================
    # LIVE FORECAST (SAFE)
    # =====================
    print("Running LIVE forecast...")

    last_row = df.iloc[-1:].copy()

    try:
        live_price = yf.Ticker(GAS_TICKER).history(period="1d")["Close"].iloc[-1]
        last_row["Gas_Close"] = live_price
        last_row["Gas_Return"] = (live_price / df["Gas_Close"].iloc[-2]) - 1
    except Exception:
        pass  # fallback: letzter Schlusskurs

    last_row["Momentum5"] = (
        last_row["Gas_Close"].values[0] / df["Gas_Close"].iloc[-6] - 1
    )
    last_row["Volatility5"] = df["Gas_Return"].iloc[-5:].std()

    prob_up = model.predict_proba(last_row[features])[0][1]

    if prob_up > UPPER_PROB:
        signal = "UP"
    elif prob_up < LOWER_PROB:
        signal = "DOWN"
    else:
        signal = "NO_TRADE"

    print("\n=== LIVE FORECAST ===")
    print(f"Probability UP: {prob_up:.2%}")
    print(f"Signal: {signal}")

    with open("forecast_output.txt", "w") as f:
        f.write(f"Probability_UP={prob_up:.4f}\n")
        f.write(f"Signal={signal}\n")


if __name__ == "__main__":
    main()
