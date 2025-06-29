import pandas as pd
import pandas_ta as ta
from sklearn.linear_model import LogisticRegression
import numpy as np
from config import EMA_FAST, EMA_SLOW
from typing import Optional


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical indicators on price data.
    Returns DataFrame with new indicator columns and drops NaNs.
    """
    df = df.copy()

    # Compute EMAs
    df['ema_fast'] = ta.ema(df['close'], length=EMA_FAST)
    df['ema_slow'] = ta.ema(df['close'], length=EMA_SLOW)

    # Compute MACD and MACD signal
    macd = ta.macd(df['close'])
    df['macd'] = macd['MACD_12_26_9']
    df['macd_signal'] = macd['MACDs_12_26_9']

    # Compute RSI, ROC, Momentum
    df['rsi'] = ta.rsi(df['close'], length=14)
    df['roc'] = ta.roc(df['close'], length=10)
    df['momentum'] = ta.mom(df['close'], length=10)

    df.dropna(inplace=True)

    return df


def generate_ml_signals(df: pd.DataFrame) -> int:
    """
    Generate buy/sell/hold signal (1/-1/0) from logistic regression model
    trained on recent data with EMA crossover as target.
    Returns 0 if insufficient data or class diversity.
    """

    df = compute_features(df)

    features = ['ema_fast', 'ema_slow', 'macd', 'macd_signal', 'rsi', 'roc', 'momentum']

    if any(col not in df.columns for col in features):
        print("[ML] Missing required features for ML signal.")
        return 0

    X = df[features].values
    y = np.where(df['ema_fast'] > df['ema_slow'], 1, 0)  # target: EMA fast > EMA slow

    # Use last 50 samples for training
    window = 50
    if len(df) < window:
        print("[ML] Not enough data points for ML training.")
        return 0

    y_recent = y[-window:]
    if len(np.unique(y_recent)) < 2:
        print("[ML] Not enough class diversity in training data, skipping ML signal.")
        return 0

    model = LogisticRegression(max_iter=1000)
    model.fit(X[-window:], y_recent)

    pred_prob = model.predict_proba(X[-1].reshape(1, -1))[0]

    if pred_prob[1] > 0.6:
        return 1
    elif pred_prob[1] < 0.4:
        return -1
    else:
        return 0


def decide(pair: Optional[str], df: pd.DataFrame) -> int:
    """
    Combine EMA crossover, MACD crossover, RSI filter, and ML signals to
    return final trading decision: 1 (buy), -1 (sell), or 0 (hold).

    Parameters:
        pair (Optional[str]): currency pair symbol, currently unused but kept for future use.
        df (pd.DataFrame): OHLC data with close prices.

    Returns:
        int: trading signal (1=buy, -1=sell, 0=hold)
    """

    min_rows_required = max(EMA_SLOW, 26, 14) + 10  # enough data for indicators
    if df is None or len(df) < min_rows_required:
        return 0  # insufficient data

    df = compute_features(df)

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # EMA crossover signal
    ema_cross = 0
    if last['ema_fast'] > last['ema_slow'] and prev['ema_fast'] <= prev['ema_slow']:
        ema_cross = 1
    elif last['ema_fast'] < last['ema_slow'] and prev['ema_fast'] >= prev['ema_slow']:
        ema_cross = -1

    # MACD crossover signal
    macd_cross = 0
    if last['macd'] > last['macd_signal'] and prev['macd'] <= prev['macd_signal']:
        macd_cross = 1
    elif last['macd'] < last['macd_signal'] and prev['macd'] >= prev['macd_signal']:
        macd_cross = -1

    # RSI filter (avoid overbought/oversold)
    rsi_filter = not (last['rsi'] > 70 or last['rsi'] < 30)

    # ML-based signal
    ml_signal = generate_ml_signals(df)

    # Final decision logic:
    # If RSI filter passed and EMA & MACD agree, use that signal
    # Otherwise fallback to ML signal if RSI filter passed
    if rsi_filter:
        if ema_cross != 0 and ema_cross == macd_cross:
            return ema_cross
        else:
            return ml_signal
    else:
        return 0
