import os
import time
import requests
import pandas as pd
from config import (
    BASE_URL, TWELVE_DATA_API_KEY,
    INTERVAL, OUTPUT_SIZE,
    MAX_RETRIES, RETRY_WAIT, TIMEOUT,
    CACHE_EXPIRY_SECONDS, CACHE_DIR
)
from utils import log


def _cache_file_path(pair: str) -> str:
    """Generate a safe cache filepath for a given pair."""
    filename = pair.replace("/", "_") + ".csv"
    return os.path.join(CACHE_DIR, filename)


def _is_cache_fresh(filepath: str) -> bool:
    """Check if the cache file exists and is within expiry window."""
    if not os.path.exists(filepath):
        return False
    age = time.time() - os.path.getmtime(filepath)
    return age < CACHE_EXPIRY_SECONDS


def _save_cache_atomic(df: pd.DataFrame, filepath: str):
    """Save DataFrame to CSV atomically to prevent corruption."""
    temp_path = filepath + ".tmp"
    df.to_csv(temp_path)
    os.replace(temp_path, filepath)
    log(f"[Cache] Saved data atomically to cache: {filepath}")


def fetch_data(pair: str) -> pd.DataFrame:
    """
    Fetch OHLCV data for a currency pair using Twelve Data API with caching and retries.

    Returns:
        pd.DataFrame indexed by datetime with columns: open, high, low, close, volume (if any).
    """
    cache_file = _cache_file_path(pair)

    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Load from cache if fresh
    if _is_cache_fresh(cache_file):
        try:
            log(f"[Cache] Loading fresh cached data for {pair}")
            df = pd.read_csv(cache_file, parse_dates=["datetime"], index_col="datetime")
            if df.empty:
                raise ValueError("Cached data is empty")
            return df
        except Exception as e:
            log(f"[Cache] Failed to load cache for {pair}: {e} -- Will fetch fresh data")

    # Fetch fresh data with retry and backoff
    url = f"{BASE_URL}/time_series"
    params = {
        "symbol": pair,
        "interval": INTERVAL,
        "outputsize": OUTPUT_SIZE,
        "apikey": TWELVE_DATA_API_KEY,
        "format": "JSON"
    }
    retries = 0
    wait_time = RETRY_WAIT

    while retries < MAX_RETRIES:
        try:
            log(f"[API] Fetching data for {pair}, attempt {retries + 1}")
            response = requests.get(url, params=params, timeout=TIMEOUT)
            response.raise_for_status()
            data_json = response.json()

            if data_json.get("status") == "error":
                msg = data_json.get("message", "").lower()
                if "run out of api credits" in msg or "rate limit" in msg:
                    log(f"[API] Rate limit hit for {pair}, backing off {wait_time} sec...")
                    time.sleep(wait_time)
                    retries += 1
                    wait_time = min(wait_time * 2, 60)  # cap backoff at 60 seconds
                    continue
                else:
                    raise ValueError(f"Twelve Data API error: {data_json.get('message')}")

            values = data_json.get("values")
            if not values:
                raise ValueError(f"No data returned for {pair}")

            # Data comes reversed, oldest last; reverse for chronological order
            df = pd.DataFrame(values[::-1])
            df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)
            df["datetime"] = pd.to_datetime(df["datetime"])
            df.set_index("datetime", inplace=True)

            # Save cache atomically
            try:
                _save_cache_atomic(df, cache_file)
            except Exception as e:
                log(f"[Cache] Failed to save cache for {pair}: {e}")

            return df.copy()

        except Exception as e:
            log(f"[API] Error fetching data for {pair} attempt {retries + 1}: {e}")
            retries += 1
            if retries >= MAX_RETRIES:
                log(f"[API] Max retries reached for {pair}, raising exception.")
                raise
            log(f"[API] Backing off {wait_time} sec before retry...")
            time.sleep(wait_time)
            wait_time = min(wait_time * 2, 60)

    # If somehow loop exits without return, raise error
    raise RuntimeError(f"Failed to fetch data for {pair} after {MAX_RETRIES} retries")
