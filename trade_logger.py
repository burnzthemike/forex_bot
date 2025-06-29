# trade_logger.py

import csv
import os
import threading
from datetime import datetime
from typing import Union

TRADE_LOG_FILE = "trades.csv"
_lock = threading.Lock()

# Full headers
CSV_HEADERS = [
    "timestamp", "symbol", "entry_time", "exit_time",
    "entry_price", "exit_price", "direction", "pnl",
    "position_size", "drawdown", "strategy_signal",
    "rl_signal", "final_signal"
]

def _ensure_log_file():
    """Make sure the CSV exists and has the correct header row."""
    if not os.path.exists(TRADE_LOG_FILE):
        with open(TRADE_LOG_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADERS)
    else:
        # verify header
        with open(TRADE_LOG_FILE, "r") as f:
            first = f.readline().strip().split(",")
        if first != CSV_HEADERS:
            print("[Logger] ⚠️  Header mismatch in trades.csv; please check.")

def log_trade(
    symbol: str,
    entry_time: Union[datetime, str],
    exit_time: Union[datetime, str],
    entry_price: float,
    exit_price: float,
    direction: int,
    pnl: float,
    position_size: float,
    drawdown: float,
    strategy_signal: int,
    rl_signal: int,
    final_signal: int
):
    """
    Log a full trade record, thread-safe.
    """
    _ensure_log_file()

    # ISO-format datetimes
    if isinstance(entry_time, datetime):
        entry_time = entry_time.isoformat()
    if isinstance(exit_time, datetime):
        exit_time = exit_time.isoformat()
    timestamp = datetime.utcnow().isoformat()

    row = [
        timestamp,
        symbol,
        entry_time,
        exit_time,
        f"{entry_price:.5f}",
        f"{exit_price:.5f}",
        direction,
        f"{pnl:.2f}",
        f"{position_size:.2f}",
        f"{drawdown:.4f}",
        strategy_signal,
        rl_signal,
        final_signal
    ]

    with _lock:
        try:
            with open(TRADE_LOG_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)
            print(f"[Logger] ✅ Trade logged: {symbol} PnL={pnl:.2f}")
        except Exception as e:
            print(f"[Logger] ❌ Failed to log trade: {e}")

def record_trade_minimal(
    entry_time: Union[datetime, str],
    exit_time: Union[datetime, str],
    pnl: float,
    strategy_signal: int,
    rl_signal: int
):
    """
    For analytics-only CSV (overwrite header if absent).  
    Columns: entry_time, exit_time, pnl, strategy_signal, rl_signal
    """
    minimal_file = "trades_minimal.csv"
    header = ["entry_time", "exit_time", "pnl", "strategy_signal", "rl_signal"]
    exists = os.path.exists(minimal_file)

    if not exists:
        with open(minimal_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

    # format
    if isinstance(entry_time, datetime):
        entry_time = entry_time.isoformat()
    if isinstance(exit_time, datetime):
        exit_time = exit_time.isoformat()

    with _lock:
        try:
            with open(minimal_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([entry_time, exit_time, f"{pnl:.2f}", strategy_signal, rl_signal])
            print(f"[Logger] 🔖 Minimal trade recorded: pnl={pnl:.2f}")
        except Exception as e:
            print(f"[Logger] ❌ Failed to record minimal trade: {e}")
