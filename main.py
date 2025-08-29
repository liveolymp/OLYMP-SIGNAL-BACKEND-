import os
import time
from typing import List
from fastapi import FastAPI, HTTPException, Query
import requests
import pandas as pd
import numpy as np

# === YOUR TWELVEDATA API KEY (embedded per your request) ===
TWELVEDATA_API_KEY = "a24ff933811047d994b9e76f1e9d7280"

if not TWELVEDATA_API_KEY:
    print("WARNING: TWELVEDATA_API_KEY not set. Set env var TWELVEDATA_API_KEY before production.")

app = FastAPI(title="Olymp Signal Backend")

# --- helper: fetch candles from TwelveData ---
def fetch_td_series(symbol: str, interval: str = "1min", outputsize: int = 200) -> pd.DataFrame:
    base = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "format": "JSON",
        "apikey": TWELVEDATA_API_KEY,
    }
    r = requests.get(base, params=params, timeout=15)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"TwelveData error: {r.status_code}")
    j = r.json()
    if "values" not in j:
        raise HTTPException(status_code=502, detail=f"TwelveData response error: {j}")
    df = pd.DataFrame(j["values"]).iloc[::-1].reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["datetime"])
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# --- Indicators ---
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, 1e-8))
    rsi_vals = 100 - (100 / (1 + rs))
    return rsi_vals

def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(window=length).mean()

def awesome_oscillator(df: pd.DataFrame, s=5, l=34) -> pd.Series:
    median = (df['high'] + df['low']) / 2
    ao = median.rolling(window=s).mean() - median.rolling(window=l).mean()
    return ao

def accelerator_oscillator(df: pd.DataFrame) -> pd.Series:
    ao = awesome_oscillator(df)
    ao_sma5 = ao.rolling(window=5).mean()
    acc = ao - ao_sma5
    return acc

def momentum(series: pd.Series, length: int = 10) -> pd.Series:
    return series.diff(periods=length)

# ZigZag simple implementation
def zigzag(df: pd.DataFrame, pct: float = 0.3):
    close = df['close'].values
    n = len(close)
    zz = [np.nan] * n
    last_pivot = close[0]
    last_type = None
    threshold = pct / 100.0
    for i in range(1, n):
        change = (close[i] - last_pivot) / (last_pivot if last_pivot!=0 else 1)
        if last_type in (None, 'low') and change >= threshold:
            zz[i] = 1
            last_pivot = close[i]
            last_type = 'high'
        elif last_type in (None, 'high') and change <= -threshold:
            zz[i] = -1
            last_pivot = close[i]
            last_type = 'low'
    return pd.Series(zz, index=df.index)

# --- Signal logic ---
def evaluate_signals(df: pd.DataFrame,
                     zigzag_pct: float = 0.3,
                     rsi_period: int = 14,
                     rsi_threshold: float = 52,
                     momentum_len: int = 10):
    df = df.copy()
    df['rsi'] = rsi(df['close'], period=rsi_period)
    df['ao'] = awesome_oscillator(df)
    df['acc'] = accelerator_oscillator(df)
    df['mom'] = momentum(df['close'], length=momentum_len)
    df['zz'] = zigzag(df, pct=zigzag_pct)

    if len(df) < 10:
        raise HTTPException(status_code=400, detail="Not enough data to compute signals")

    i = len(df) - 1
    prev = i - 1
    prev2 = i - 2

    latest = df.iloc[i]
    prev_row = df.iloc[prev]
    prev2_row = df.iloc[prev2]

    zz_recent = df['zz'].iloc[-3:]
    zigzag_upper = zz_recent.isin([1]).any()
    zigzag_lower = zz_recent.isin([-1]).any()

    acc_flip_up = (prev_row['acc'] < 0) and (latest['acc'] > 0) and (latest['close'] > latest['open'])
    acc_flip_down = (prev_row['acc'] > 0) and (latest['acc'] < 0) and (latest['close'] < latest['open'])

    rsi_up = (latest['rsi'] > prev_row['rsi']) and (latest['rsi'] >= rsi_threshold)
    rsi_down = (latest['rsi'] < prev_row['rsi']) and (latest['rsi'] <= (100 - rsi_threshold))

    mom_flip_up = (prev_row['mom'] < 0) and (latest['mom'] > 0) and (latest['close'] > latest['open'])
    mom_flip_down = (prev_row['mom'] > 0) and (latest['mom'] < 0) and (latest['close'] < latest['open'])

    buy = all([zigzag_upper, acc_flip_up, rsi_up, mom_flip_up])
    sell = all([zigzag_lower, acc_flip_down, rsi_down, mom_flip_down])

    signal = None
    if buy:
        signal = "BUY"
    elif sell:
        signal = "SELL"
    else:
        signal = "HOLD"

    details = {
        "signal": signal,
        "zigzag_upper": bool(zigzag_upper),
        "zigzag_lower": bool(zigzag_lower),
        "acc_flip_up": bool(acc_flip_up),
        "acc_flip_down": bool(acc_flip_down),
        "rsi_up": bool(rsi_up),
        "rsi_down": bool(rsi_down),
        "mom_flip_up": bool(mom_flip_up),
        "mom_flip_down": bool(mom_flip_down),
        "latest_rsi": float(latest['rsi']),
        "latest_acc": float(latest['acc']),
        "latest_mom": float(latest['mom']),
        "latest_close": float(latest['close']),
        "latest_open": float(latest['open']),
    }

    return details

# --- API endpoints ---
@app.get("/health")
def health():
    return {"status": "ok", "timestamp": int(time.time())}

@app.get("/signal")
def get_signal(symbol: str = Query(..., description="TwelveData symbol e.g. EUR/USD or EURUSD"),
               interval: str = "1min",
               zigzag_pct: float = 0.3,
               rsi_period: int = 14,
               rsi_threshold: float = 52,
               momentum_len: int = 10):
    df = fetch_td_series(symbol, interval=interval, outputsize=200)
    result = evaluate_signals(df,
                              zigzag_pct=zigzag_pct,
                              rsi_period=rsi_period,
                              rsi_threshold=rsi_threshold,
                              momentum_len=momentum_len)
    return result

@app.get("/batch_signals")
def batch_signals(symbols: str = Query(..., description="comma separated symbols, e.g. EUR/USD,GBP/USD,USD/JPY")):
    out = {}
    s_list = [s.strip() for s in symbols.split(",") if s.strip()]
    for s in s_list:
        try:
            df = fetch_td_series(s, interval="1min", outputsize=200)
            out[s] = evaluate_signals(df)
        except Exception as e:
            out[s] = {"error": str(e)}
    return out
