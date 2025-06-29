import random
from typing import Optional, Dict
from utils import log

try:
    from config import SPREAD_PIPS, SLIPPAGE_PIPS, PAIR_PIP_SIZES
except ImportError:
    # Fallback defaults for standalone testing
    SPREAD_PIPS = 0.8       # typical spread in pips
    SLIPPAGE_PIPS = 0.2     # typical slippage in pips
    PAIR_PIP_SIZES = {
        "EUR/USD": 0.0001,
        "USD/JPY": 0.01,
        "GBP/USD": 0.0001,
        "USD/CHF": 0.0001,
        "USD/CAD": 0.0001,
    }

def get_pip_size(pair: str) -> float:
    """
    Return pip size for the given currency pair.
    Defaults to 0.0001 unless overridden (e.g., JPY pairs).
    """
    return PAIR_PIP_SIZES.get(pair, 0.0001)

def simulate_execution(pair: str, raw_price: float, direction: int) -> float:
    """
    Simulate realistic execution price incorporating spread and slippage.

    Parameters:
        pair (str): currency pair like "EUR/USD"
        raw_price (float): current mid-market price
        direction (int): 1 for buy (ask), -1 for sell (bid), 0 for hold/no trade

    Returns:
        float: executed price adjusted for spread and slippage
    """
    pip_size = get_pip_size(pair)
    spread_cost = SPREAD_PIPS * pip_size
    slippage = random.uniform(0, SLIPPAGE_PIPS) * pip_size

    if direction == 1:
        executed_price = raw_price + spread_cost + slippage
    elif direction == -1:
        executed_price = raw_price - spread_cost - slippage
    else:
        executed_price = raw_price

    log(f"[ExecutionSimulator] {pair} {'BUY' if direction == 1 else 'SELL' if direction == -1 else 'HOLD'} | "
        f"Raw: {raw_price:.5f} | Executed: {executed_price:.5f} | Spread: {spread_cost:.5f} | Slippage: {slippage:.5f}")

    return executed_price

def simulate_trade(pair: str, entry_price: float, exit_price: float, direction: int):
    """
    Simulate the full trade with realistic execution prices at entry and exit.

    Parameters:
        pair (str): currency pair like "EUR/USD"
        entry_price (float): mid-market entry price
        exit_price (float): mid-market exit price
        direction (int): 1 for long/buy, -1 for short/sell

    Returns:
        tuple: (pnl: float, executed_entry_price: float, executed_exit_price: float)
    """
    executed_entry = simulate_execution(pair, entry_price, direction)
    executed_exit = simulate_execution(pair, exit_price, -direction)

    pnl = (executed_exit - executed_entry) * direction
    return pnl, executed_entry, executed_exit

def simulate_backtest_trade(df, signal: int, index: int, position_size: float) -> Optional[Dict]:
    """
    Unified backtest-compatible execution simulator.

    Parameters:
        df (pd.DataFrame): DataFrame containing OHLCV data.
        signal (int): -1 = short, 0 = hold, 1 = long.
        index (int): current row index in df.
        position_size (float): number of units traded.

    Returns:
        dict or None: trade record with execution details, or None if no trade.
    """
    if signal == 0 or index + 1 >= len(df):
        return None

    pair = df.attrs.get("pair", "EUR/USD")

    entry_price = df.iloc[index]["close"]
    exit_price = df.iloc[index + 1]["close"]

    executed_entry = simulate_execution(pair, entry_price, signal)
    executed_exit = simulate_execution(pair, exit_price, -signal)

    pnl = (executed_exit - executed_entry) * signal * position_size

    trade_record = {
        "pair": pair,
        "entry_index": index,
        "exit_index": index + 1,
        "entry_time": df.index[index],
        "exit_time": df.index[index + 1],
        "entry_price": executed_entry,
        "exit_price": executed_exit,
        "direction": signal,
        "pnl": pnl,
        "position_size": position_size
    }

    log(f"[BacktestTrade] Index {index} | Pair: {pair} | Direction: {'LONG' if signal == 1 else 'SHORT'} | "
        f"Entry: {executed_entry:.5f} | Exit: {executed_exit:.5f} | Position Size: {position_size} | PnL: {pnl:.5f}")

    return trade_record

if __name__ == "__main__":
    # Quick standalone test
    pnl, entry, exit = simulate_trade("EUR/USD", 1.1000, 1.1050, 1)
    print(f"Simulated PnL: {pnl:.5f}, Entry Exec: {entry:.5f}, Exit Exec: {exit:.5f}")
