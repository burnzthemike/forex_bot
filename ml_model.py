import numpy as np
import pandas as pd
import joblib
import os
import pandas_ta as ta
from sklearn.linear_model import LogisticRegression
from sentiment import fetch_news_sentiment
from feature_engineering import compute_volatility, compute_trend_strength
from config import EMA_FAST, EMA_SLOW, VOLATILITY_WINDOW
from utils import log

MODEL_PATH = "ml_model.joblib"
TRAINING_LOG = "training_data.csv"

def feature_names():
    return sorted(["ema_cross", "macd_hist", "rsi", "atr", "sentiment", "trend_strength"])

def model_exists():
    return os.path.exists(MODEL_PATH)

def extract_features(pair: str, df: pd.DataFrame) -> np.ndarray:
    """
    Extracts input feature vector for ML model from given dataframe.
    """
    try:
        close = df["close"]

        ema_fast = close.ewm(span=EMA_FAST).mean()
        ema_slow = close.ewm(span=EMA_SLOW).mean()

        try:
            macd = ta.macd(close)
            macd_hist = macd["MACDh_12_26_9"].iloc[-1] if "MACDh_12_26_9" in macd else 0
        except Exception:
            macd_hist = 0

        try:
            rsi = ta.rsi(close, length=14)
            rsi_val = rsi.iloc[-1] if not rsi.isnull().all() else 50
        except Exception:
            rsi_val = 50

        atr = compute_volatility(df, VOLATILITY_WINDOW)
        trend = compute_trend_strength(df, EMA_FAST, EMA_SLOW)
        sentiment = fetch_news_sentiment(pair.replace("/", " "), days=1)

        features = {
            "ema_cross": int(ema_fast.iloc[-2] < ema_slow.iloc[-2] and ema_fast.iloc[-1] > ema_slow.iloc[-1]) -
                         int(ema_fast.iloc[-2] > ema_slow.iloc[-2] and ema_fast.iloc[-1] < ema_slow.iloc[-1]),
            "macd_hist": macd_hist,
            "rsi": rsi_val,
            "atr": atr.iloc[-1] if not atr.isnull().all() else 0,
            "trend_strength": trend.iloc[-1] if not trend.isnull().all() else 0,
            "sentiment": sentiment
        }

        return np.array([features[k] for k in feature_names()], dtype=np.float32).reshape(1, -1)

    except Exception as e:
        log(f"[ML] Feature extraction failed for {pair}: {e}")
        return np.zeros((1, len(feature_names())), dtype=np.float32)

def log_training_sample(pair: str, df: pd.DataFrame, pnl: float):
    """
    Logs a single training sample to CSV with features + label.
    Label is 1 if pnl > 0 else 0.
    """
    try:
        X = extract_features(pair, df).flatten()
        label = int(pnl > 0)
        row = list(X) + [label]
        header_needed = not os.path.exists(TRAINING_LOG)

        df_out = pd.DataFrame([row], columns=feature_names() + ["label"])
        df_out.to_csv(TRAINING_LOG, mode="a", header=header_needed, index=False)

        log(f"[ML] Logged training sample for {pair} | PnL={pnl:.2f} | Label={label}")

    except Exception as e:
        log(f"[ML] Failed to log training sample for {pair}: {e}")

def train_model():
    """
    Loads data from CSV and trains a logistic regression model.
    Saves to disk if successful.
    """
    try:
        if not os.path.exists(TRAINING_LOG):
            log("[ML] No training log found.")
            return

        data = pd.read_csv(TRAINING_LOG)
        if data.empty:
            log("[ML] No training data found.")
            return

        X = data[feature_names()]
        y = data["label"]

        if y.nunique() < 2:
            log("[ML] Not enough class diversity to train model.")
            return

        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)

        joblib.dump(model, MODEL_PATH)
        log("[ML] Model trained and saved successfully.")

    except Exception as e:
        log(f"[ML] Training failed: {e}")

def load_model():
    """
    Loads the trained model from disk. Trains one if none found.
    """
    try:
        if not model_exists():
            log("[ML] No model found. Training new model...")
            train_model()

        model = joblib.load(MODEL_PATH)
        log("[ML] Model loaded successfully.")
        return model

    except Exception as e:
        log(f"[ML] Failed to load model: {e}")
        return None

def predict_signal(pair: str, df: pd.DataFrame) -> int:
    """
    Predict trading signal using trained model:
    1 = buy, -1 = sell, 0 = hold/error.
    """
    try:
        model = load_model()
        if model is None:
            return 0

        features = extract_features(pair, df)
        pred = model.predict(features)[0]
        return 1 if pred == 1 else -1

    except Exception as e:
        log(f"[ML] Prediction failed for {pair}: {e}")
        return 0

if __name__ == "__main__":
    train_model()
