# =========================
# GAS PRICE FORECAST v1
# CI-STABLE VERSION
# =========================

import os
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance

# =========================
# PATHS
# =========================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"

def resolve_csv(filename: str) -> Path:
    p1 = DATA_DIR / filename
    p2 = BASE_DIR / filename
    if p1.exists():
        return p1
    if p2.exists():
        return p2
    raise FileNotFoundError(
        f"{filename} not found. Looked in {p1} and {p2}"
    )

GAS_CSV = resolve_csv("gas.csv")
OIL_CSV = resolve_csv("oil.csv")
WEATHER_CSV = resolve_csv("weather.csv")


# =========================
# SAFE CSV LOAD
# =========================

def read_csv_safe(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Required for CI.")
    return pd.read_csv(path)

# =========================
# LOAD DATA
# =========================

def load_data():
    print("Loading local CSVs...")
    gas = read_csv_safe(GAS_CSV)
    oil = read_csv_safe(OIL_CSV)
    weather = read_csv_safe(WEATHER_CSV)

    for df in (gas, oil, weather):
        df["Date"] = pd.to_datetime(df["Date"])

    weather = weather.groupby("Date").mean(numeric_only=True).reset_index()

    df = gas.merge(oil, on="Date", how="inner")
    df = df.merge(weather, on="Date", how="inner")

    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df

# =========================
# FEATURE ENGINEERING
# =========================

def build_features(df):
    df["Gas_Return"] = df["Gas_Close"].pct_change()
    df["Oil_Return"] = df["Oil_Close"].pct_change()

    for lag in range(1, 6):
        df[f"Gas_Return_lag{lag}"] = df["Gas_Return"].shift(lag)
        df[f"Oil_Return_lag{lag}"] = df["Oil_Return"].shift(lag)
        df[f"HDD_lag{lag}"] = df["HDD"].shift(lag)
        df[f"CDD_lag{lag}"] = df["CDD"].shift(lag)
        df[f"TempAvg_lag{lag}"] = df["Temp_Avg"].shift(lag)

    df["Momentum5"] = df["Gas_Close"].pct_change(5).shift(1)
    df["Volatility5"] = df["Gas_Return"].rolling(5).std()
    df["SMA10"] = df["Gas_Close"].rolling(10).mean()

    df["Target"] = (df["Gas_Return"].shift(-1) > 0).astype(int)

    df.dropna(inplace=True)
    return df

# =========================
# MODEL & VALIDATION
# =========================

def run_model(df):
    feature_cols = [c for c in df.columns if c not in ("Date", "Target")]

    X = df[feature_cols]
    y = df["Target"]

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=5,
        max_features="log2",
        random_state=42,
        n_jobs=-1
    )

    tscv = TimeSeriesSplit(n_splits=5)
    accs = []

    print("\nRunning TimeSeriesSplit validation")
    for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[test_idx])
        acc = accuracy_score(y.iloc[test_idx], preds)
        accs.append(acc)
        print(f"Split {i} accuracy: {acc:.3f}")

    print(f"\nMean accuracy: {np.mean(accs):.3f}")

    # Train full model
    model.fit(X, y)

    perm = permutation_importance(
        model, X, y, n_repeats=10, random_state=42, n_jobs=-1
    )

    imp = (
        pd.DataFrame({
            "feature": X.columns,
            "importance": perm.importances_mean
        })
        .sort_values("importance", ascending=False)
        .head(20)
    )

    print("\nTop features:")
    print(imp)

# =========================
# MAIN
# =========================

def main():
    df = load_data()
    print(f"Rows before features: {len(df)}")

    df = build_features(df)
    print(f"Rows after features: {len(df)}")

    run_model(df)

if __name__ == "__main__":
    main()
