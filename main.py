import os
import time
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

# Force early .env load and validate Twelve Data key
load_dotenv()
TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY")
if not TWELVE_DATA_API_KEY or '"' in TWELVE_DATA_API_KEY:
    raise ValueError("❌ Invalid or missing TWELVE_DATA_API_KEY in your .env file. Remove any quotes.")

npNaN = np.nan

from trade_logger import log_trade, record_trade_minimal
from retrain_model import retrain_model
from data_loader import get_ohlc_data
from strategy import decide
from utils import log, calculate_drawdown
from dashboard import shared_data, start_dashboard
from sentiment import fetch_news_sentiment

from config import (
    CURRENCY_PAIRS, INTERVAL, OUTPUT_SIZE,
    POSITION_SIZE, MAX_DRAWDOWN,
    DASHBOARD_REFRESH, PAPER_TRADING,
    REWARD_CLIP_MIN, REWARD_CLIP_MAX, REWARD_SMOOTHING_ALPHA,
    DRAWDOWN_PENALTY_WEIGHT, MAX_RISK_PER_TRADE,
    TOP_PAIR_COUNT, COMMISSION_PER_TRADE,
    LIVE_TRADING
)

from rl_agent import QLearningAgent, discretize_state
from risk_management import position_size, check_drawdown
from exit_strategy import check_time_exit
from feature_engineering import add_advanced_features
from ml_model import load_model, log_training_sample, extract_features
from trend_scanner import rank_pairs
from execution_simulator import simulate_execution

# MT5 Integration
from connectors.mt5_connector import (
    initialize_mt5, place_order, close_position, shutdown_mt5, get_current_price
)

API_CALL_INTERVAL = 8

def run_trading_engine():
    """Main trading loop with ML, RL, and strategy fusion. Dashboard and live trading enabled."""
    log("[Engine] Initializing... Live trading is ON" if LIVE_TRADING else "[Engine] Running in paper trading mode.")
    start_dashboard()

    equity = 10_000.0
    peak_equity = equity
    history = [equity]
    last_retrain_date = datetime.utcnow().date()
    agent = QLearningAgent()
    ml_model = load_model()

    prev_state = {}
    reward_ema = {}
    open_trades = {}

    if LIVE_TRADING and not initialize_mt5():
        log("❌ [MT5] Initialization failed. Aborting.")
        return

    try:
        while True:
            log("🔄 Engine heartbeat — new cycle")
            today = datetime.utcnow().date()

            # Retrain daily
            if today != last_retrain_date:
                log("📚 Retraining ML model...")
                retrain_model()
                ml_model = load_model()
                last_retrain_date = today

            # Pair selection
            ranked_pairs = rank_pairs()
            selected_pairs = ranked_pairs[:TOP_PAIR_COUNT]
            log(f"📈 Top pairs this cycle: {selected_pairs}")

            for pair in selected_pairs:
                start = time.time()
                try:
                    df = get_ohlc_data(pair, INTERVAL, OUTPUT_SIZE)
                    if df.empty or len(df) < 30:
                        log(f"⚠️ {pair}: Not enough data. Skipping.")
                        continue

                    df = add_advanced_features(df)
                    sentiment = fetch_news_sentiment(pair.replace("/", ""))
                    if sentiment < -0.3:
                        log(f"😐 {pair}: Negative sentiment ({sentiment:.2f}), skipping")
                        continue

                    strat_sig = decide(pair, df)
                    feats = extract_features(pair, df)
                    probs = ml_model.predict_proba(feats)[0]
                    ml_conf = probs[1]
                    ml_raw = ml_model.predict(feats)[0]
                    ml_sig = 1 if ml_raw == 1 else -1

                    close_now = df["close"].iloc[-1]
                    close_prev = df["close"].iloc[-2]
                    price_pct = (close_now - close_prev) / close_prev
                    state = discretize_state(
                        price_pct,
                        df["rsi"].iloc[-1],
                        int(df["ema_fast"].iloc[-1] > df["ema_slow"].iloc[-1]) - int(df["ema_fast"].iloc[-2] > df["ema_slow"].iloc[-2]),
                        int(df["macd"].iloc[-1] > df["macd_signal"].iloc[-1]) - int(df["macd"].iloc[-2] > df["macd_signal"].iloc[-2]),
                        df["atr"].iloc[-1]
                    )
                    rl_act = agent.choose_action(state)

                    final_sig = 0
                    if sentiment > -0.3 and (ml_sig == rl_act != 0 or ml_sig == strat_sig != 0):
                        final_sig = ml_sig

                    pos_size = position_size(equity, df["atr"].iloc[-1])
                    dd = calculate_drawdown(history)
                    adj_size = min(pos_size * max(0.2, 1 + dd) * ml_conf, POSITION_SIZE)

                    log(f"{pair}: Strat={strat_sig:+d} RL={rl_act:+d} ML={ml_sig:+d} | Final={final_sig:+d} | Price={close_now:.5f} | Pos={adj_size:.2f} | Sent={sentiment:.2f}")

                    trade = open_trades.get(pair)
                    now = datetime.utcnow()

                    # EXIT
                    if trade:
                        entry_price = trade["entry_price"]
                        entry_time = trade["entry_time"]
                        signal = trade["signal"]
                        size = trade["position_size"]

                        exec_entry = simulate_execution(pair, entry_price, signal) if not LIVE_TRADING else entry_price
                        exec_exit = simulate_execution(pair, close_now, -signal) if not LIVE_TRADING else close_now
                        pnl = signal * ((exec_exit - exec_entry) / exec_entry) * size - COMMISSION_PER_TRADE

                        sl = (exec_exit - entry_price)/entry_price <= -2 * df["atr"].iloc[-1] / entry_price
                        tp = (exec_exit - entry_price)/entry_price >= 2 * df["atr"].iloc[-1] / entry_price
                        tte = check_time_exit(entry_time, now)

                        if sl or tp or tte:
                            if LIVE_TRADING:
                                close_position(pair)

                            equity += pnl
                            peak_equity = max(peak_equity, equity)
                            dd = calculate_drawdown(history + [equity])
                            history.append(equity)

                            shared_data["equity_curve"] = history.copy()
                            shared_data["latest_metrics"] = {"Equity": equity, "Drawdown": dd}

                            log_training_sample(pair, df, pnl)
                            log_trade(pair, entry_time, now, exec_entry, exec_exit, signal, pnl, size, dd, strat_sig, rl_act, final_sig)
                            record_trade_minimal(entry_time, now, pnl, strat_sig, rl_act)

                            shaped = pnl * (1.1 if tp else 1) - DRAWDOWN_PENALTY_WEIGHT * abs(min(dd, 0))
                            clipped = np.clip(np.tanh(shaped / max(1, size)), REWARD_CLIP_MIN, REWARD_CLIP_MAX)
                            reward_ema[pair] = REWARD_SMOOTHING_ALPHA * clipped + (1 - REWARD_SMOOTHING_ALPHA) * reward_ema.get(pair, 0)
                            agent.learn(prev_state[pair], signal, reward_ema[pair], state)

                            open_trades.pop(pair)

                    # OPEN
                    elif final_sig != 0 and adj_size > 0:
                        entry_price = get_current_price(pair) if LIVE_TRADING else simulate_execution(pair, close_now, final_sig)
                        if LIVE_TRADING:
                            place_order(pair, final_sig, adj_size)

                        open_trades[pair] = {
                            "entry_price": entry_price,
                            "entry_time": now,
                            "position_size": adj_size,
                            "signal": final_sig
                        }
                        log(f"{pair} ✅ OPEN TRADE: {final_sig:+d} @ {entry_price:.5f} size {adj_size:.2f}")

                    prev_state[pair] = state

                except Exception as e:
                    log(f"❌ Error processing {pair}: {e}")

                elapsed = time.time() - start
                if elapsed < API_CALL_INTERVAL:
                    time.sleep(API_CALL_INTERVAL - elapsed)

            if check_drawdown(equity, peak_equity):
                log("🟥 Max drawdown hit — shutting down engine.")
                break

            log(f"⏳ Waiting {DASHBOARD_REFRESH} sec before next cycle...")
            time.sleep(DASHBOARD_REFRESH)

    finally:
        if LIVE_TRADING:
            shutdown_mt5()
        log("🛑 Engine shutdown complete.")

if __name__ == "__main__":
    run_trading_engine()
