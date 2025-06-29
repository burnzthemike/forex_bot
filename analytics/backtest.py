import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any
from trade_logger import log_trade
from strategy import decide
from ml_model import extract_features, load_model
from rl_agent import QLearningAgent, discretize_state
from execution_simulator import simulate_execution
from utils import calculate_drawdown, get_position_size
from sentiment import fetch_news_sentiment
from risk_management import position_size as risk_position_size

from config import (
    POSITION_SIZE,
    COMMISSION_PER_TRADE,
    MAX_HOLD_TIME_MINUTES
)

def backtest(
    df: pd.DataFrame,
    pair: str,
    starting_equity: float = 10_000
) -> Dict[str, Any]:
    df = df.copy()
    equity = starting_equity
    equity_curve = [equity]
    agent = QLearningAgent()
    ml_model = load_model()

    open_trade: Optional[Dict] = None
    entry_time: Optional[pd.Timestamp] = None

    for i in range(2, len(df)):
        try:
            window = df.iloc[:i + 1].copy()
            close_now = window["close"].iloc[-1]
            close_prev = window["close"].iloc[-2]
            price_change_pct = (close_now - close_prev) / close_prev if close_prev != 0 else 0

            # Sentiment filtering
            sentiment_score = fetch_news_sentiment(pair.replace("/", ""), days=1)
            if sentiment_score < -0.3:
                continue  # Skip bearish sentiment

            # Generate signals
            strategy_signal = decide(pair, window)
            ml_features = extract_features(pair, window)
            ml_probs = ml_model.predict_proba(ml_features)[0]
            ml_confidence = ml_probs[1]
            ml_signal_raw = ml_model.predict(ml_features)[0]
            ml_signal = 1 if ml_signal_raw == 1 else -1

            # Calculate indicator crosses
            ema_cross = int(window['ema_fast'].iloc[-1] > window['ema_slow'].iloc[-1]) - \
                        int(window['ema_fast'].iloc[-2] > window['ema_slow'].iloc[-2])
            macd_cross = int(window['macd'].iloc[-1] > window['macd_signal'].iloc[-1]) - \
                         int(window['macd'].iloc[-2] > window['macd_signal'].iloc[-2])
            volatility = window["atr"].iloc[-1] if "atr" in window.columns else 0
            rsi_now = window["rsi"].iloc[-1] if "rsi" in window.columns else 50

            state = discretize_state(price_change_pct, rsi_now, ema_cross, macd_cross, volatility)
            rl_action = agent.choose_action(state)

            # Combine signals conservatively
            final_signal = 0
            if sentiment_score > -0.3:
                if ml_signal == rl_action != 0:
                    final_signal = ml_signal
                elif ml_signal == strategy_signal != 0:
                    final_signal = ml_signal

            # Position size with risk management and ML confidence scaling
            base_pos_size = risk_position_size(equity, volatility)
            adjusted_pos_size = min(base_pos_size * ml_confidence, POSITION_SIZE)

            # Close any open trade before opening new one
            if open_trade is not None:
                entry_price = open_trade["entry_price"]
                trade_signal = open_trade["signal"]

                simulated_exit = simulate_execution(pair, close_now, -trade_signal)
                simulated_entry = simulate_execution(pair, entry_price, trade_signal)

                change_pct = (simulated_exit - simulated_entry) / simulated_entry if simulated_entry != 0 else 0
                pnl = trade_signal * change_pct * open_trade["position_size"] - COMMISSION_PER_TRADE
                equity += pnl
                equity_curve.append(equity)
                dd = calculate_drawdown(equity_curve)

                log_trade(
                    symbol=pair,
                    entry_time=entry_time,
                    exit_time=df.index[i],
                    entry_price=simulated_entry,
                    exit_price=simulated_exit,
                    direction=trade_signal,
                    pnl=pnl,
                    position_size=open_trade["position_size"],
                    drawdown=dd,
                    strategy_signal=strategy_signal,
                    rl_signal=rl_action,
                    final_signal=final_signal
                )

                open_trade = None
                entry_time = None

            # Open new trade if signal present
            if final_signal != 0:
                simulated_entry = simulate_execution(pair, close_now, final_signal)
                open_trade = {
                    "entry_price": simulated_entry,
                    "position_size": adjusted_pos_size,
                    "signal": final_signal
                }
                entry_time = df.index[i]

        except Exception as e:
            print(f"[Backtest] Error at index {i}: {e}")
            continue

    # Final performance metrics
    curve = pd.Series(equity_curve)
    returns = curve.pct_change().dropna()
    total_return = curve.iloc[-1] / curve.iloc[0] - 1
    sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() else 0
    max_dd = (curve / curve.cummax() - 1).min()

    return {
        "Total Return (%)": round(total_return * 100, 2),
        "Sharpe Ratio": round(sharpe, 2),
        "Max Drawdown (%)": round(max_dd * 100, 2),
        "Equity Curve": curve
    }

if __name__ == "__main__":
    from data import fetch_data
    from feature_engineering import add_advanced_features

    pair = "GBP/USD"
    df = fetch_data(pair)
    if df is not None and not df.empty:
        df = add_advanced_features(df)
        results = backtest(df, pair)
        print("===== BACKTEST SUMMARY =====")
        for k, v in results.items():
            if k != "Equity Curve":
                print(f"{k}: {v}")
