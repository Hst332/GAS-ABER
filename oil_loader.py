# oil_loader.py
import yfinance as yf
import pandas as pd

SYMBOL_OIL = "CL=F"  # WTI Crude Oil Futures

def load_oil_from_yahoo(start="2018-01-01"):
    df = yf.download(
        SYMBOL_OIL,
        start=start,
        progress=False,
        auto_adjust=True
    )["Close"].to_frame("Oil_Close")

    df = df.reset_index()
    df.columns = ["Date", "Oil_Close"]
    df["Date"] = pd.to_datetime(df["Date"])
    df["Oil_Close"] = pd.to_numeric(df["Oil_Close"], errors="coerce")

    df = df.dropna().sort_values("Date")

    # =========================
    # ✅ ML-FEATURES (sicher)
    # =========================
    df["Oil_Return"] = df["Oil_Close"].pct_change()

    for lag in [1, 2, 3, 5, 7]:
        df[f"Oil_Return_lag{lag}"] = df["Oil_Return"].shift(lag)

    df = df.dropna().reset_index(drop=True)
    return df
