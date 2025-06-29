# === forex_bot_project/modules/data_fetcher.py ===

import os
import time
import json
import requests
import pandas as pd
from datetime import datetime
import config

class ForexDataFetcher:
    def __init__(self, cache_dir="cached_data"):
        self.api_key = config.ALPHA_VANTAGE_API_KEY
        self.base_url = "https://www.alphavantage.co/query"
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def fetch_pair(self, pair):
        from_symbol, to_symbol = pair.split("/")
        cache_file = os.path.join(self.cache_dir, f"{from_symbol}_{to_symbol}.json")

        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                data = json.load(f)
        else:
            params = {
                "function": "FX_INTRADAY",
                "from_symbol": from_symbol,
                "to_symbol": to_symbol,
                "interval": "5min",
                "outputsize": "compact",
                "apikey": self.api_key
            }
            response = requests.get(self.base_url, params=params)
            data = response.json()

            if "Note" in data:
                print(f"API Notice: {data['Note']}")
                return None
            if "Error Message" in data:
                print(f"API Error: {data['Error Message']}")
                return None
            if "Time Series FX (5min)" not in data:
                print(f"No 5min time series data for {pair}")
                return None

            with open(cache_file, "w") as f:
                json.dump(data, f)

            time.sleep(12)  # Respect Alpha Vantage free-tier rate limits

        df = pd.DataFrame(data["Time Series FX (5min)"]).T
        df = df.astype(float)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df.rename(columns={
            "1. open": "open",
            "2. high": "high",
            "3. low": "low",
            "4. close": "close"
        }, inplace=True)
        return df

    def fetch_all_pairs(self):
        results = {}
        for pair in config.CURRENCY_PAIRS:
            df = self.fetch_pair(pair)
            results[pair] = df
        return results
