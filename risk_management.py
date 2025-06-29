# risk_management.py

import pandas as pd
import numpy as np
from config import MAX_RISK_PER_TRADE, MAX_DRAWDOWN
from utils import log

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculates the Average True Range (ATR) for volatility estimation.
    Returns a pandas Series with ATR values.
    """
    try:
        high = df["high"]
        low = df["low"]
        close = df["close"]

        hl = high - low
        hc = np.abs(high - close.shift(1))
        lc = np.abs(low - close.shift(1))

        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr

    except Exception as e:
        log(f"[Risk] Failed to calculate ATR: {e}")
        return pd.Series(dtype=float)

def position_size(equity: float, atr_value: float, risk_per_trade: float = MAX_RISK_PER_TRADE) -> float:
    """
    Calculates position size based on ATR and allowed risk per trade.
    Returns size in units (e.g., lots, contracts, shares).
    """
    try:
        if atr_value <= 0 or equity <= 0:
            return 0.0

        dollar_risk = equity * risk_per_trade
        size = dollar_risk / atr_value
        return round(size, 2)  # rounding to avoid excessive precision

    except Exception as e:
        log(f"[Risk] Error calculating position size: {e}")
        return 0.0

def check_drawdown(equity: float, peak_equity: float, max_allowed: float = MAX_DRAWDOWN) -> bool:
    """
    Checks if the drawdown from peak equity exceeds the max allowed.
    Returns True if drawdown breach occurs.
    """
    try:
        if peak_equity <= 0:
            return False
        drawdown_pct = (peak_equity - equity) / peak_equity
        return drawdown_pct > max_allowed
    except Exception as e:
        log(f"[Risk] Error checking drawdown: {e}")
        return False
