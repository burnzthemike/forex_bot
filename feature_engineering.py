import pandas as pd
import pandas_ta as ta
from utils import log
from config import EMA_FAST, EMA_SLOW, VOLATILITY_WINDOW, MOMENTUM_WINDOW

def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds advanced TA indicators to price dataframe:
    - RSI (default 50)
    - EMA fast/slow
    - MACD & Signal
    - ATR (volatility)
    - ROC (momentum)
    - Normalized EMA trend strength
    """
    data = df.copy()
    try:
        if "close" not in data.columns:
            raise KeyError("Missing 'close' column in dataframe")

        # RSI
        try:
            data["rsi"] = ta.rsi(data["close"], length=14).fillna(50)
        except Exception:
            data["rsi"] = 50

        # EMAs
        data["ema_fast"] = ta.ema(data["close"], length=EMA_FAST).fillna(method='bfill')
        data["ema_slow"] = ta.ema(data["close"], length=EMA_SLOW).fillna(method='bfill')

        # MACD & Signal
        try:
            macd = ta.macd(data["close"], fast=EMA_FAST, slow=EMA_SLOW, signal=9)
            data["macd"] = macd.get(f"MACD_{EMA_FAST}_{EMA_SLOW}_9", 0).fillna(0)
            data["macd_signal"] = macd.get(f"MACDs_{EMA_FAST}_{EMA_SLOW}_9", 0).fillna(0)
        except Exception:
            data["macd"] = 0
            data["macd_signal"] = 0

        # ATR
        try:
            data["atr"] = ta.atr(data["high"], data["low"], data["close"], length=VOLATILITY_WINDOW).fillna(0)
        except Exception:
            data["atr"] = 0

        # ROC
        try:
            data["roc"] = ta.roc(data["close"], length=MOMENTUM_WINDOW).fillna(0)
        except Exception:
            data["roc"] = 0

        # Trend strength
        try:
            ema_f = data["ema_fast"]
            ema_s = data["ema_slow"].replace(0, 1e-8)
            data["trend_strength"] = ((ema_f - ema_s) / ema_s).fillna(0)
        except Exception:
            data["trend_strength"] = 0

        # Clean NaNs
        data.fillna(method="bfill", inplace=True)
        data.fillna(method="ffill", inplace=True)
        data.fillna(0, inplace=True)

    except Exception as e:
        log(f"[FeatureEngineering] Error adding advanced features: {e}")

    return data


def compute_volatility(df: pd.DataFrame, window: int) -> pd.Series:
    """
    Computes ATR-based volatility over a rolling window.
    """
    try:
        return ta.atr(df["high"], df["low"], df["close"], length=window).fillna(0)
    except Exception as e:
        log(f"[FeatureEngineering] Failed to compute ATR: {e}")
        return pd.Series(0, index=df.index)


def compute_momentum(df: pd.DataFrame, window: int) -> pd.Series:
    """
    Computes Rate of Change (ROC) as momentum measure.
    """
    try:
        return ta.roc(df["close"], length=window).fillna(0)
    except Exception as e:
        log(f"[FeatureEngineering] Failed to compute ROC: {e}")
        return pd.Series(0, index=df.index)


def compute_trend_strength(df: pd.DataFrame, fast: int, slow: int) -> pd.Series:
    """
    Computes normalized EMA divergence (fast - slow / slow).
    """
    try:
        ema_f = df["close"].ewm(span=fast, adjust=False).mean()
        ema_s = df["close"].ewm(span=slow, adjust=False).mean().replace(0, 1e-8)
        return ((ema_f - ema_s) / ema_s).fillna(0)
    except Exception as e:
        log(f"[FeatureEngineering] Failed to compute trend strength: {e}")
        return pd.Series(0, index=df.index)
