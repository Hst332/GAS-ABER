#!/usr/bin/env python3
"""
gas_backtest_v8_autodiscover.py

Automatische EIA Series-Suche (Storage / Production / LNG feedgas) + robustes backtest-pipeline.
SET EIA_API_KEY before running.

Usage:
 - Put gas.csv, oil.csv, weather.csv in same folder.
 - Set EIA_API_KEY below.
 - python gas_backtest_v8_autodiscover.py

CAUTION: This is an analysis tool; no guarantees on real trading performance.
"""

import os, sys, json, time
from datetime import datetime, timedelta
import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.inspection import permutation_importance

# -----------------------
# USER: set your EIA key here
# -----------------------
EIA_API_KEY = "<PUT_YOUR_EIA_API_KEY_HERE>"

# Local CSV filenames
GAS_CSV = "gas.csv"
OIL_CSV = "oil.csv"
WEATHER_CSV = "weather.csv"

OUTDIR = "analysis_outputs_v8"
os.makedirs(OUTDIR, exist_ok=True)

# -----------------------
# Helpers
# -----------------------
def read_csv_safe(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df

def find_date_col(df):
    for c in ['Date','date','observation_date','DATE']:
        if c in df.columns:
            return c
    # fallback: any column name containing 'date'
    for c in df.columns:
        if 'date' in c.lower():
            return c
    raise ValueError("No date column found")

def standardize_dfs():
    gas = read_csv_safe(GAS_CSV)
    oil = read_csv_safe(OIL_CSV)
    weather = read_csv_safe(WEATHER_CSV)

    # Date cols
    gas_date_col = find_date_col(gas); gas = gas.rename(columns={gas_date_col:'Date'})
    oil_date_col = find_date_col(oil); oil = oil.rename(columns={oil_date_col:'Date'})
    weather_date_col = find_date_col(weather); weather = weather.rename(columns={weather_date_col:'Date'})

    gas['Date'] = pd.to_datetime(gas['Date'])
    oil['Date'] = pd.to_datetime(oil['Date'])
    weather['Date'] = pd.to_datetime(weather['Date'])

    # Price column inference
    def infer_price_col(df, prefer=['DHHNGSP','Gas_Close','Close','PRICE','Value']):
        for p in prefer:
            if p in df.columns:
                return p
        numcols = df.select_dtypes(include=[np.number]).columns.tolist()
        # if exactly 1 numeric -> assume it's price
        if len(numcols)==1:
            return numcols[0]
        # else try common names
        for cand in ['DHHNGSP','DCOILWTICO','Oil_Close','Gas_Close','Close','PRICE','VALUE','Value']:
            if cand in df.columns:
                return cand
        raise ValueError("Cannot infer price column in dataframe. Columns: " + str(df.columns.tolist()))

    gas_price_col = infer_price_col(gas, ['DHHNGSP','Gas_Close','Close','PRICE'])
    oil_price_col = infer_price_col(oil, ['DCOILWTICO','Oil_Close','Close','PRICE'])

    gas = gas.rename(columns={gas_price_col:'Gas_Close'})
    oil = oil.rename(columns={oil_price_col:'Oil_Close'})

    return gas[['Date','Gas_Close']].copy(), oil[['Date','Oil_Close']].copy(), weather.copy()

# -----------------------
# EIA v2 search helper
# -----------------------
def eia_search(query, api_key, rows=20):
    """
    Uses EIA Open Data v2 search endpoint to find series by keyword.
    Returns JSON of results.
    """
    if not api_key:
        print("No EIA API key set; skipping EIA search.")
        return None
    # v2 search endpoint: https://api.eia.gov/v2/series?api_key=KEY&search=...
    url = "https://api.eia.gov/v2/series"
    params = {
        "api_key": api_key,
        "series_id": "",  # not searching by id; we'll use 'search' param below via 'q' style
        "page": 1,
        "per_page": rows,
        "facets": "",
        "sort": "series_id",
        "frequency": "weekly,monthly,daily"  # let API filter
    }
    # EIA v2 uses 'keywords' param in some doc; fallback: use q param appended
    # Build a basic query via 'q' parameter
    params_simple = {"api_key": api_key, "q": query, "per_page": rows}
    try:
        r = requests.get(url, params=params_simple, timeout=20)
    except Exception as e:
        print("EIA search request failed:", e)
        return None
    if r.status_code != 200:
        print("EIA search HTTP", r.status_code, r.text[:300])
        return None
    jd = r.json()
    # jd likely contains 'response' -> 'data'
    hits = []
    if isinstance(jd, dict):
        # try extract list
        if 'response' in jd and 'data' in jd['response']:
            hits = jd['response']['data']
        elif 'data' in jd:
            # some endpoints return 'data' directly
            hits = jd['data']
        else:
            # fallback: return full json
            hits = jd
    return hits

def eia_get_series_by_id(series_id, api_key):
    url = "https://api.eia.gov/series/"
    params = {"api_key": api_key, "series_id": series_id}
    try:
        r = requests.get(url, params=params, timeout=20)
    except Exception as e:
        print("EIA series request failed:", e)
        return None
    if r.status_code != 200:
        print("EIA series HTTP", r.status_code)
        return None
    jd = r.json()
    if 'series' not in jd:
        return None
    s = jd['series'][0]
    df = pd.DataFrame(s['data'], columns=['Date','Value'])
    # robust date parsing
    def parse_date(x):
        x = str(x)
        for fmt in ("%Y%m%d","%Y-%m-%d","%Y%m","%Y"):
            try:
                return pd.to_datetime(x, format=fmt)
            except Exception:
                continue
        return pd.to_datetime(x, errors='coerce')
    df['Date'] = df['Date'].apply(parse_date)
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    return df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)

# -----------------------
# Auto-discover candidate series
# -----------------------
def autodiscover_series(api_key):
    if not api_key:
        print("No API key: cannot autodiscover EIA series. Returning Nones.")
        return None, None, None

    # Search keywords and fallback lists
    searches = {
        "storage": ["working gas in underground storage", "weekly working gas", "working gas in storage"],
        "production": ["natural gas marketed production", "dry natural gas production", "natural gas production US"],
        "lng": ["lng feedgas", "pipeline deliveries to lnd terminals", "liquefaction feedgas", "lng sendout", "sendout"]
    }

    found = {}
    for kind, keys in searches.items():
        candidates = []
        for k in keys:
            hits = eia_search(k, api_key, rows=30)
            if not hits:
                continue
            # hits might be dict/list; normalize
            if isinstance(hits, list):
                for h in hits:
                    # each h expected to have 'series_id' and 'name'
                    sid = h.get('series_id') or h.get('series_id')
                    name = h.get('name') or h.get('description') or h.get('series_id')
                    if sid:
                        candidates.append((sid, name))
            elif isinstance(hits, dict):
                # sometimes the structure differs; try to read top-level 'data'
                for item in hits.values():
                    if isinstance(item, list):
                        for it in item:
                            sid = it.get('series_id')
                            name = it.get('name') or it.get('description')
                            if sid:
                                candidates.append((sid, name))
        # deduplicate candidates, keep those with likely keywords
        uniq = []
        seen = set()
        for sid,name in candidates:
            if sid in seen: continue
            seen.add(sid)
            uniq.append((sid,name))
        found[kind] = uniq[:10]
    # pick top candidate heuristically: prefer series_id containing 'STOR' or 'STORAGE' for storage, 'PROD'/'NATGAS' for production, 'LNG' for lng
    pick = {}
    for k, cand in found.items():
        pick[k] = None
        for sid,name in cand:
            s_low = sid.lower() + " " + (name or "").lower()
            if k=='storage' and ('stor' in s_low or 'working' in s_low):
                pick[k] = sid; break
            if k=='production' and ('prod' in s_low or 'production' in s_low):
                pick[k] = sid; break
            if k=='lng' and ('lng' in s_low or 'feed' in s_low):
                pick[k] = sid; break
        if pick[k] is None and cand:
            pick[k] = cand[0][0]
    return pick.get('storage'), pick.get('production'), pick.get('lng')

# -----------------------
# load and run pipeline
# -----------------------
def load_and_run(eia_api_key):
    gas, oil, weather = standardize_dfs()
    print("Local ranges: gas", gas['Date'].min(), "->", gas['Date'].max(), "rows", len(gas))
    print("             oil", oil['Date'].min(), "->", oil['Date'].max(), "rows", len(oil))
    print("         weather", weather['Date'].min(), "->", weather['Date'].max(), "rows", len(weather))

    # autodiscover
    storage_sid, prod_sid, lng_sid = autodiscover_series(eia_api_key)
    print("Auto-discovered series IDs (may be None):")
    print(" storage:", storage_sid)
    print(" production:", prod_sid)
    print(" lng_feedgas:", lng_sid)

    # fetch series data if found
    storage_df = eia_get_series_by_id(storage_sid, eia_api_key) if storage_sid and eia_api_key else None
    prod_df = eia_get_series_by_id(prod_sid, eia_api_key) if prod_sid and eia_api_key else None
    lng_df = eia_get_series_by_id(lng_sid, eia_api_key) if lng_sid and eia_api_key else None

    # upsample weekly to daily if needed
    def upsample_weekly(df):
        if df is None: return None
        df = df.set_index('Date').resample('D').ffill().reset_index()
        return df

    storage_daily = upsample_weekly(storage_df) if storage_df is not None else None
    prod_daily = upsample_weekly(prod_df) if prod_df is not None else None
    lng_daily  = upsample_weekly(lng_df) if lng_df is not None else None

    # Merge
    df = gas.merge(oil, on='Date', how='left')
    # aggregate weather: if multiple regions average by date
    if 'Region' in weather.columns:
        # ensure Temp_Avg exists
        if 'Temp_Avg' not in weather.columns and 'Temp_Avg' not in weather.columns:
            if 'Temp_Min' in weather.columns and 'Temp_Max' in weather.columns:
                weather['Temp_Avg'] = (weather['Temp_Min'] + weather['Temp_Max']) / 2.0
        agg_cols = [c for c in ['Temp_Avg','HDD','CDD'] if c in weather.columns]
        if len(agg_cols)>0:
            weather_agg = weather.groupby('Date')[agg_cols].mean().reset_index()
        else:
            weather_agg = weather.groupby('Date').mean().reset_index()
    else:
        weather_agg = weather

    df = df.merge(weather_agg.drop_duplicates(subset=['Date']), on='Date', how='left')

    if storage_daily is not None:
        storage_daily = storage_daily.rename(columns={'Value':'Storage'})
        df = df.merge(storage_daily[['Date','Storage']], on='Date', how='left')
    if prod_daily is not None:
        prod_daily = prod_daily.rename(columns={'Value':'Production'})
        df = df.merge(prod_daily[['Date','Production']], on='Date', how='left')
    if lng_daily is not None:
        lng_daily = lng_daily.rename(columns={'Value':'LNG_feedgas'})
        df = df.merge(lng_daily[['Date','LNG_feedgas']], on='Date', how='left')

    # basic preprocessing
    df = df.sort_values('Date').reset_index(drop=True)
    df['Oil_Close'] = pd.to_numeric(df['Oil_Close'], errors='coerce').ffill().bfill()
    df['Gas_Close'] = pd.to_numeric(df['Gas_Close'], errors='coerce')
    df['Gas_Return'] = df['Gas_Close'].pct_change()
    df['Oil_Return'] = df['Oil_Close'].pct_change()

    # Storage surprise
    if 'Storage' in df.columns:
        df['Storage'] = pd.to_numeric(df['Storage'], errors='coerce')
        df['Storage_delta'] = df['Storage'].diff()
        df['Storage_4w_mean'] = df['Storage_delta'].rolling(4).mean()
        df['Storage_surprise'] = df['Storage_delta'] - df['Storage_4w_mean']
        df['Storage_surprise'] = df['Storage_surprise'].fillna(0.0)

    # Feature engineering: lags
    NLAGS = 7
    cols_for_lags = ['Gas_Return','Oil_Return','Storage_surprise','Production','LNG_feedgas','HDD','CDD','Temp_Avg']
    for col in cols_for_lags:
        if col in df.columns:
            for i in range(1, NLAGS+1):
                df[f"{col}_lag{i}"] = df[col].shift(i)

    # tech features
    df['Momentum7'] = df['Gas_Close'].pct_change(7).shift(1)
    df['Volatility7'] = df['Gas_Return'].rolling(7).std().shift(1)
    df['SMA10'] = df['Gas_Close'].rolling(10).mean().shift(1)

    df['Next_Return'] = df['Gas_Close'].pct_change().shift(-1)
    TH = 0.003  # ignore tiny moves
    df['Target'] = np.where(df['Next_Return'] > TH, 1, np.where(df['Next_Return'] < -TH, 0, np.nan))
    df = df.dropna(subset=['Target']).reset_index(drop=True)
    df['Target'] = df['Target'].astype(int)

    # pick features
    features = [c for c in df.columns if (('_lag' in c) or c in ['Momentum7','Volatility7','SMA10','Oil_Return','Storage_surprise','Production','LNG_feedgas','HDD','CDD','Temp_Avg'])]
    df = df.dropna(subset=features).reset_index(drop=True)
    print("Built dataset rows:", len(df), "features:", len(features))

    # TimeSeriesSplit training
    tscv = TimeSeriesSplit(n_splits=5)
    accs = []; cms = []
    for train_idx, test_idx in tscv.split(df):
        X_train = df.iloc[train_idx][features].values; y_train = df.iloc[train_idx]['Target'].values
        X_test  = df.iloc[test_idx][features].values;  y_test  = df.iloc[test_idx]['Target'].values
        scaler = StandardScaler().fit(X_train)
        X_train_s = scaler.transform(X_train); X_test_s = scaler.transform(X_test)
        mdl = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1)
        mdl.fit(X_train_s,y_train)
        pred = mdl.predict(X_test_s)
        accs.append(accuracy_score(y_test,pred))
        cms.append(confusion_matrix(y_test,pred))
    # final model for importances
    X_full = df[features].values; y_full = df['Target'].values
    scaler_full = StandardScaler().fit(X_full); X_full_s = scaler_full.transform(X_full)
    final_mdl = RandomForestClassifier(n_estimators=500, max_depth=10, random_state=42, n_jobs=-1)
    final_mdl.fit(X_full_s, y_full)
    perm = permutation_importance(final_mdl, X_full_s, y_full, n_repeats=30, random_state=42, n_jobs=-1)
    perm_df = pd.DataFrame({'feature':features, 'perm_mean': perm.importances_mean, 'perm_std': perm.importances_std}).sort_values('perm_mean', ascending=False)

    # leak rudimentary check: shuffle target acc
    shuffled = df.copy()
    shuffled['Target'] = np.random.permutation(shuffled['Target'].values)
    Xs = shuffled[features].values; ys = shuffled['Target'].values
    scaler_s = StandardScaler().fit(Xs); Xs_s = scaler_s.transform(Xs)
    mdl_s = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=1, n_jobs=-1)
    # quick split
    split = int(len(shuffled)*0.7)
    mdl_s.fit(Xs_s[:split], ys[:split])
    ypred_s = mdl_s.predict(Xs_s[split:])
    shuffle_acc = accuracy_score(ys[split:], ypred_s)

    # Save outputs
    df.to_csv(os.path.join(OUTDIR,"merged_features_targets_v8.csv"), index=False)
    perm_df.to_csv(os.path.join(OUTDIR,"feature_perm_importance_v8.csv"), index=False)
    summary = {"rows": len(df), "n_features": len(features), "tscv_accs": accs, "confusion_matrices":[cm.tolist() for cm in cms], "shuffle_acc": shuffle_acc, "top_perm": perm_df.head(25).to_dict(orient='records')}
    with open(os.path.join(OUTDIR,"summary_v8.json"), "w") as f:
        json.dump(summary,f, default=str, indent=2)

    print("\n--- SUMMARY ---")
    print("Rows:", summary['rows'])
    print("Features:", summary['n_features'])
    print("TS CV accs:", [round(a,4) for a in summary['tscv_accs']])
    print("Shuffle acc (sanity):", round(shuffle_acc,4))
    print("Top permutation importance (top 15):\n", perm_df.head(15).to_string(index=False))
    print("Outputs in", OUTDIR)

if __name__ == "__main__":
    if EIA_API_KEY == "" or "<PUT_YOUR" in EIA_API_KEY:
        print("EIA_API_KEY not set. Script will still run but EIA autodiscover will be skipped (weaker features).")
    load_and_run(EIA_API_KEY)

