import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from forex_python.converter import CurrencyRates

from config import (
    TWELVE_DATA_API_KEY,
    CACHE_DIR,
    CACHE_EXPIRY_SECONDS,
    MAX_RETRIES,
    RETRY_WAIT,
    TIMEOUT,
    BASE_URL,
)
from utils import log, ensure_dir

# Ensure necessary directories exist
ensure_dir(CACHE_DIR)
ensure_dir("data/historical")

def is_cache_fresh(filepath: str, max_age_sec: int) -> bool:
    if not os.path.exists(filepath):
        return False
    mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
    age = datetime.utcnow() - mtime
    return age < timedelta(seconds=max_age_sec)

def get_ohlc_data(symbol: str, interval: str = "1min", outputsize: int = 500) -> pd.DataFrame:
    log(f"[DataLoader] Getting data for {symbol}")
    formatted = symbol.replace("/", "").upper()
    cache_file = os.path.join(CACHE_DIR, f"{formatted}.csv")

    # 1) Load fresh cache
    if is_cache_fresh(cache_file, CACHE_EXPIRY_SECONDS):
        try:
            df = pd.read_csv(cache_file)
            df['datetime'] = pd.to_datetime(df['datetime'])
            log(f"[DataLoader] ⚡ Loaded fresh data from cache: {formatted}")
            return df
        except Exception as e:
            log(f"[DataLoader] ⚠️ Failed loading cache: {e}")

    # 2) Twelve Data API (use the original symbol, with slash)
    symbol_param = symbol.upper()  # e.g. "EUR/USD"
    url = (
        f"{BASE_URL}/time_series?"
        f"symbol={symbol_param}&interval={interval}&outputsize={outputsize}&"
        f"apikey={TWELVE_DATA_API_KEY}"
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, timeout=TIMEOUT)
            data = r.json()
            if data.get('status') == 'error':
                log(f"[DataLoader] ❌ Twelve Data API error: {data.get('message')}")
            elif 'values' in data:
                df = pd.DataFrame(data['values'])
                df.columns = [c.lower() for c in df.columns]
                df['datetime'] = pd.to_datetime(df['datetime'])
                df = df.sort_values('datetime')
                df[['open','high','low','close']] = df[['open','high','low','close']].astype(float)
                # Save to cache
                try:
                    df[['datetime','open','high','low','close']].to_csv(cache_file, index=False)
                except Exception as e:
                    log(f"[DataLoader] ⚠️ Failed to save cache: {e}")
                log(f"[DataLoader] ✅ Loaded data from Twelve Data: {formatted}")
                return df
        except Exception as e:
            log(f"[DataLoader] ❌ Twelve Data request failed (attempt {attempt}): {e}")
        time.sleep(RETRY_WAIT)

    # 3) Forex-python fallback
    try:
        base, quote = symbol.split('/')
        rate = CurrencyRates().get_rate(base, quote)
        now = datetime.utcnow()
        df = pd.DataFrame([{
            'datetime': now,
            'open': rate,
            'high': rate,
            'low': rate,
            'close': rate
        }])
        log(f"[DataLoader] ⚠️ Using spot rate from forex-python for {symbol}")
        return df
    except Exception as e:
        log(f"[DataLoader] ❌ Forex Python failed: {e}")

    # 4) Local historical data
    try:
        hist = f"data/historical/{formatted}.csv"
        if os.path.exists(hist):
            df = pd.read_csv(hist)
            df.columns = [c.lower() for c in df.columns]
            if 'datetime' not in df.columns and 'date' in df.columns and 'time' in df.columns:
                df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
            else:
                df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.sort_values('datetime')
            df[['open','high','low','close']] = df[['open','high','low','close']].astype(float)
            log(f"[DataLoader] 🗃️ Loaded local historical data: {formatted}")
            return df
    except Exception as e:
        log(f"[DataLoader] ❌ Local fallback failed: {e}")

    log(f"[DataLoader] ❌ Failed to load any data for {symbol}")
    return pd.DataFrame()
