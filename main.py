import os
import time
from typing import List
from fastapi import FastAPI, HTTPException, Query
import requests
import pandas as pd
import numpy as np

# === YOUR TWELVEDATA API KEY ===
TWELVEDATA_API_KEY = "a24ff933811047d994b9e76f1e"

# === FASTAPI APP ===
app = FastAPI(title="AI Signal Engine", description="Fancy Futuristic Signal API 🚀")

# === TECHNICAL INDICATORS ===
def momentum(series, length=10):
    return series.diff(length)

def rsi(series, length=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(length).mean()
    avg_loss = pd.Series(loss).rolling(length).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def ao(high, low, short=5, long=34):
    median_price = (high + low) / 2
    short_ma = median_price.rolling(short).mean()
    long_ma = median_price.rolling(long).mean()
    return short_ma - long_ma

def zigzag(df, deviation=5):
    df["zz"] = np.nan
    direction = None
    last_pivot_idx = 0
    last_pivot_price = df["close"].iloc[0]

    for i in range(1, len(df)):
        price = df["close"].iloc[i]
        if direction is None:
            if price >= last_pivot_price * (1 + deviation / 100):
                direction = "UP"
                last_pivot_price = price
                last_pivot_idx = i
                df.loc[df.index[i], "zz"] = price
            elif price <= last_pivot_price * (1 - deviation / 100):
                direction = "DOWN"
                last_pivot_price = price
                last_pivot_idx = i
                df.loc[df.index[i], "zz"] = price
        elif direction == "UP" and price <= last_pivot_price * (1 - deviation / 100):
            direction = "DOWN"
            last_pivot_price = price
            last_pivot_idx = i
            df.loc[df.index[i], "zz"] = price
        elif direction == "DOWN" and price >= last_pivot_price * (1 + deviation / 100):
            direction = "UP"
            last_pivot_price = price
            last_pivot_idx = i
            df.loc[df.index[i], "zz"] = price

    return df

# === FETCH DATA FROM TWELVEDATA ===
def fetch_data(symbol: str, interval: str = "1min", outputsize: int = 100):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&apikey={TWELVEDATA_API_KEY}&outputsize={outputsize}"
    response = requests.get(url).json()
    if "values" not in response:
        raise HTTPException(status_code=400, detail="Error fetching data")
    df = pd.DataFrame(response["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    return df.reset_index(drop=True)

# === SIGNAL GENERATOR ===
def generate_signal(df):
    df["ao"] = ao(df["high"], df["low"])
    df["mom"] = momentum(df["close"], length=10)
    df["rsi"] = rsi(df["close"], length=14)
    df = zigzag(df)

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    # Conditions
    acc_flip_up = latest["ao"] > 0 and prev["ao"] <= 0
    acc_flip_down = latest["ao"] < 0 and prev["ao"] >= 0
    momentum_up = latest["mom"] > 0
    momentum_down = latest["mom"] < 0
    rsi_up = latest["rsi"] > 55
    rsi_down = latest["rsi"] < 45

    zz_direction = "UP" if latest["close"] >= df["zz"].dropna().iloc[-1] else "DOWN"

    # Signal logic
    signal = "HOLD"
    if acc_flip_up and momentum_up and rsi_up and zz_direction == "UP":
        signal = "BUY"
    elif acc_flip_down and momentum_down and rsi_down and zz_direction == "DOWN":
        signal = "SELL"

    return {
        "signal": signal,
        "ao": float(latest["ao"]),
        "momentum": float(latest["mom"]),
        "rsi": float(latest["rsi"]),
        "zigzag": zz_direction,
        "price": float(latest["close"])
    }

# === API ENDPOINT ===
@app.get("/signal")
def get_signal(symbol: str = Query(...), interval: str = Query("1min")):
    df = fetch_data(symbol, interval)
    signal_data = generate_signal(df)
    return {
        "symbol": symbol,
        "interval": interval,
        "signal": signal_data["signal"],
        "details": signal_data
    }
