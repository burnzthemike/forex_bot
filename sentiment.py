import requests
import datetime
import re
import time
import threading
import json
import os
import signal
import sys
import pandas as pd
from collections import Counter, deque
from typing import Dict, Tuple
from config import NEWSAPI_KEY, NEWSAPI_URL, SENTIMENT_WINDOW
from utils import log

CACHE_FILE = "sentiment_cache.json"
POS_WORDS = {"good", "positive", "gain", "up", "strong", "bull", "beat"}
NEG_WORDS = {"bad", "negative", "loss", "down", "weak", "bear", "miss"}

_lock = threading.Lock()
_sentiment_cache: Dict[str, Dict[str, any]] = {}
_request_queue: deque[Tuple[str, int]] = deque()
_worker_thread = None
_worker_running = False

CALL_INTERVAL_SECONDS = 15
CACHE_TTL_SECONDS = 3600
CACHE_AUTOSAVE_INTERVAL = 300

def normalize_query(q: str) -> str:
    return re.sub(r'[^A-Za-z0-9]', '', q).upper()

def load_cache():
    global _sentiment_cache
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                raw_cache = json.load(f)
                _sentiment_cache = {
                    k: {"timestamp": datetime.datetime.fromisoformat(v["timestamp"]), "score": v["score"]}
                    for k, v in raw_cache.items()
                }
            log(f"[Sentiment] Loaded cache ({len(_sentiment_cache)} entries)")
        except Exception as e:
            log(f"[Sentiment] Failed to load cache: {e}")
    else:
        log("[Sentiment] No cache found, starting fresh.")

def save_cache():
    with _lock:
        try:
            to_save = {
                k: {"timestamp": v["timestamp"].isoformat(), "score": v["score"]}
                for k, v in _sentiment_cache.items()
            }
            with open(CACHE_FILE, "w") as f:
                json.dump(to_save, f)
            log(f"[Sentiment] Cache saved ({len(to_save)} entries)")
        except Exception as e:
            log(f"[Sentiment] Cache save failed: {e}")

def sentiment_worker():
    global _worker_running
    last_autosave = time.time()

    while _worker_running:
        if not _request_queue:
            time.sleep(1)
            continue

        norm_query, days = _request_queue.popleft()
        now = datetime.datetime.utcnow()

        # Defensive handling for 'days' type
        if isinstance(days, pd.Timestamp):
            days = 1
        elif not isinstance(days, int):
            try:
                days = int(days)
            except:
                days = 1

        with _lock:
            cache_entry = _sentiment_cache.get(norm_query)
            if cache_entry and (now - cache_entry["timestamp"]).total_seconds() < CACHE_TTL_SECONDS:
                continue

            from_date = now - datetime.timedelta(days=days)

        params = {
            "q": norm_query,
            "from": from_date.isoformat(),
            "to": now.isoformat(),
            "apiKey": NEWSAPI_KEY,
            "pageSize": 50,
            "language": "en"
        }

        try:
            log(f"[Sentiment] Fetching news for '{norm_query}'")
            resp = requests.get(NEWSAPI_URL, params=params, timeout=10)
            resp.raise_for_status()
            articles = resp.json().get("articles", [])

            if not articles:
                score = 0.0
            else:
                pos = neg = 0
                for art in articles:
                    title = art.get("title", "").lower()
                    words = Counter(title.split())
                    pos += sum(words[w] for w in POS_WORDS)
                    neg += sum(words[w] for w in NEG_WORDS)
                score = (pos - neg) / len(articles)

            with _lock:
                _sentiment_cache[norm_query] = {"timestamp": datetime.datetime.utcnow(), "score": score}
            log(f"[Sentiment] Updated sentiment for '{norm_query}': {score:.2f}")

        except requests.exceptions.HTTPError as e:
            log(f"[Sentiment] HTTP error for '{norm_query}': {e}")
        except Exception as e:
            log(f"[Sentiment] Unexpected error: {e}")

        time.sleep(CALL_INTERVAL_SECONDS)

        if time.time() - last_autosave > CACHE_AUTOSAVE_INTERVAL:
            save_cache()
            last_autosave = time.time()

def start_worker():
    global _worker_thread, _worker_running
    if _worker_thread is None or not _worker_thread.is_alive():
        _worker_running = True
        _worker_thread = threading.Thread(target=sentiment_worker, daemon=True)
        _worker_thread.start()
        log("[Sentiment] Worker thread started")

def stop_worker():
    global _worker_running
    _worker_running = False
    if _worker_thread:
        _worker_thread.join()
        log("[Sentiment] Worker thread stopped")

def fetch_news_sentiment(query: str, days: int = SENTIMENT_WINDOW) -> float:
    norm_query = normalize_query(query)
    now = datetime.datetime.utcnow()

    with _lock:
        cache_entry = _sentiment_cache.get(norm_query)
        if cache_entry and (now - cache_entry["timestamp"]).total_seconds() < CACHE_TTL_SECONDS:
            log(f"[Sentiment] Cache hit for '{query}'")
            return cache_entry["score"]

        if norm_query not in [q for q, _ in _request_queue]:
            log(f"[Sentiment] Queuing '{query}' for background sentiment fetch")
            _request_queue.append((norm_query, days))

    return 0.0  # Neutral until updated

def shutdown_handler(signum, frame):
    log(f"\n[Sentiment] Shutdown signal received ({signum}). Saving cache...")
    stop_worker()
    save_cache()
    log("[Sentiment] Cleanup complete. Exiting.")
    sys.exit(0)

# Graceful shutdown
signal.signal(signal.SIGINT, shutdown_handler)
signal.signal(signal.SIGTERM, shutdown_handler)

# Initialize
load_cache()
start_worker()
