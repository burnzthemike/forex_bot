import time
import pandas as pd
import numpy as np

from data_loader import get_ohlc_data
from sentiment import fetch_news_sentiment
from feature_engineering import add_advanced_features
from config import CURRENCY_PAIRS, BASE_URL
from utils import log

# Retry settings
MAX_RETRIES = 3
RETRY_WAIT = 1  # seconds

def score_pair(pair: str) -> float:
    """
    Score a currency pair by combining:
      • ATR volatility (higher is better)
      • ADX trend strength (higher is better)
      • News sentiment (positive boosts score)
    """
    formatted_pair = pair.replace("/", "").upper()

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            df = get_ohlc_data(pair)
            if df is None or df.empty or len(df) < 30:
                return -np.inf

            df = add_advanced_features(df)
            latest = df.iloc[-1]

            atr = latest.get("atr", 0.0)
            adx = latest.get("adx", 0.0)
            sentiment = fetch_news_sentiment(formatted_pair)
            sentiment_score = np.clip(sentiment, -1.0, 1.0)

            # Normalize each component
            atr_score = atr / 0.01
            adx_score = adx / 25.0
            sentiment_score_scaled = sentiment_score + 1.0  # shift to [0,2]

            # Weighted sum
            score = (
                0.5 * atr_score +
                0.4 * adx_score +
                0.1 * sentiment_score_scaled
            )

            log(f"[TrendScanner] Score for {pair}: {score:.2f}")
            return score

        except Exception as e:
            log(f"[TrendScanner] Attempt {attempt} error scoring {pair}: {e}")
            time.sleep(RETRY_WAIT)

    log(f"[TrendScanner] All attempts failed for {pair}, assigning -inf")
    return -np.inf

def rank_pairs() -> list[str]:
    scores = {pair: score_pair(pair) for pair in CURRENCY_PAIRS}
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ranked_pairs = [pair for pair, score in ranked if score > -np.inf]
    log(f"[TrendScanner] Ranked pairs: {ranked_pairs}")
    return ranked_pairs
