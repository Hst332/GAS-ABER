import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore")

# =====================
# CONFIG
# =====================
GAS_TICKER = "NG=F"    # Henry Hub Natural Gas
OIL_TICKER = "CL=F"    # Crude Oil
LOOKBACK_YEARS = 4

UPPER_PROB = 0.55      # CONFIDENCE FILTER (wichtig!!)
LOWER_PROB = 0.45

RANDOM_SEED = 42

# =====================
# LOAD ONLINE DATA
# =====================
def load_prices():
    start = pd.Timestamp.today() - pd.DateOffset(years=LOOKBACK_YEARS)

    gas = yf.download(GAS_TICKER, start=start, progress=False)[["Close"]]
    oil = yf.download(OIL_TICKER, start=start, progress=False)[["Close"]]

    gas.columns = ["Gas_Close"]
    oil.columns = ["Oil_Close"]

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

    for lag in [1, 2, 3, 5]:
        df[f"Gas_Return_lag{lag}"] = df["Gas_Return"].shift(lag)
        df[f"Oil_Return_lag{lag}"] = df["Oil_Return"].shift(lag)

    df["Momentum5"] = df["Gas_Close"].pct_change(5).shift(1)
    df["Volatility5"] = df["Gas_Return"].rolling(5).std()

    # TARGET: morgen UP oder DOWN
    df["Target"] = (df["Gas_Return"].shift(-1) > 0).astype(int)

    df.dropna(inplace=True)
    return df

# =====================
# TRAIN MODEL
# =====================
def train_model(df, features):
    X = df[features]
    y = df["Target"]

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )

    # Rolling validation (nur Reporting)
    tscv = TimeSeriesSplit(n_splits=5)
    accs = []

    for train, test in tscv.split(X):
        model.fit(X.iloc[train], y.iloc[train])
        preds = model.predict(X.iloc[test])
        accs.append(accuracy_score(y.iloc[test], preds))

    print(f"Rolling Accuracy: mean={np.mean(accs)*100:.2f}%")

    # Final train on full data
    model.fit(X, y)
    return model

# =====================
# MAIN
# =====================
def main():
    print("Loading online gas & oil prices...")
    df = load_prices()

    df = build_features(df)

    features = [
        "Gas_Return",
        "Oil_Return",
        "Gas_Return_lag1",
        "Gas_Return_lag2",
        "Gas_Return_lag3",
        "Gas_Return_lag5",
        "Oil_Return_lag1",
        "Oil_Return_lag2",
        "Oil_Return_lag3",
        "Oil_Return_lag5",
        "Momentum5",
        "Volatility5",
    ]

    print("Training model...")
    model = train_model(df, features)

    # =====================
    # LIVE FORECAST (NUR SACHLICH SINNVOLL)
    # =====================
    last_row = df.iloc[-1:]
    prob_up = model.predict_proba(last_row[features])[0][1]

    if prob_up > UPPER_PROB:
        signal = "UP"
    elif prob_up < LOWER_PROB:
        signal = "DOWN"
    else:
        signal = "NO_TRADE"

    output = (
        f"Date: {last_row.index[0].date()}\n"
        f"Probability UP: {prob_up:.3f}\n"
        f"Signal: {signal}\n"
    )

    print("\n=== LIVE FORECAST ===")
    print(output)

    with open("forecast_output.txt", "w") as f:
        f.write(output)

if __name__ == "__main__":
    main()
