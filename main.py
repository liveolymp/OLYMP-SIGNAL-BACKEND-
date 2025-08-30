import os
import time
import requests
import pandas as pd
import numpy as np
from typing import Dict, List
from fastapi import FastAPI
import uvicorn

app = FastAPI()

TWELVEDATA_API_KEY = "a24ff933811047d994b9e76f1e"
SYMBOLS = [
    "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF",
    "AUD/USD", "NZD/USD", "USD/CAD", "EUR/JPY"
]
INTERVAL = "1min"

# === Indicators ===
def calculate_rsi(data: pd.Series, period: int = 14):
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=period-1, adjust=False).mean()
    avg_loss = loss.ewm(com=period-1, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_momentum(data: pd.Series, period: int = 10):
    return data.diff(period)

def calculate_ao(df: pd.DataFrame):
    median_price = (df["high"] + df["low"]) / 2
    sma5 = median_price.rolling(5).mean()
    sma34 = median_price.rolling(34).mean()
    return sma5 - sma34

def calculate_zigzag(df: pd.DataFrame, deviation: float = 0.5):
    df['zigzag'] = np.where(df['close'] > df['close'].shift(1) * (1 + deviation/100), 1,
                     np.where(df['close'] < df['close'].shift(1) * (1 - deviation/100), -1, 0))
    return df['zigzag']

# === Signal Logic ===
def check_signal(df: pd.DataFrame):
    df["RSI"] = calculate_rsi(df["close"])
    df["Momentum"] = calculate_momentum(df["close"])
    df["AO"] = calculate_ao(df)
    df["Zigzag"] = calculate_zigzag(df)

    latest = df.iloc[-1]

    acc_flip = latest["AO"] > 0
    mom_flip = latest["Momentum"] > 0
    rsi_ok = 50 < latest["RSI"] < 60
    zigzag_up = latest["Zigzag"] == 1

    if acc_flip and mom_flip and rsi_ok and zigzag_up:
        return "BUY"
    elif not acc_flip and not mom_flip and latest["RSI"] < 50 and latest["Zigzag"] == -1:
        return "SELL"
    else:
        return None

def get_data(symbol: str, interval: str = INTERVAL, outputsize: int = 100):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&apikey={TWELVEDATA_API_KEY}&outputsize={outputsize}"
    r = requests.get(url)
    data = r.json()
    if "values" not in data:
        return None
    df = pd.DataFrame(data["values"])
    df = df.astype({"open": float, "high": float, "low": float, "close": float})
    df = df[::-1].reset_index(drop=True)
    return df

def scan_markets():
    signals = {}
    for symbol in SYMBOLS:
        df = get_data(symbol)
        if df is not None:
            signal = check_signal(df)
            if signal:
                signals[symbol] = signal
    return signals

# === API Endpoint ===
@app.get("/signals")
def get_signals():
    return {"signals": scan_markets()}

# === Run Server ===
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
