import os
import datetime
import csv
from typing import List

from config import LOG_FILE, EQUITY_LOG, MAX_RISK_PER_TRADE, BASE_URL


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def log(msg: str) -> None:
    now = datetime.datetime.utcnow().isoformat(timespec="microseconds")
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{now} | {msg}\n")
    except Exception as e:
        print(f"[LogError] Failed to write to log file: {e}")
    print(f"{now} | {msg}")


def calculate_drawdown(equity_curve: List[float]) -> float:
    peak = float('-inf')
    max_dd = 0.0
    for equity in equity_curve:
        if equity > peak:
            peak = equity
        dd = (equity - peak) / peak if peak != 0 else 0
        if dd < max_dd:
            max_dd = dd
    return max_dd


def get_position_size(equity: float) -> float:
    return equity * MAX_RISK_PER_TRADE


def safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def append_equity_log(timestamp: datetime.datetime, equity: float, drawdown: float) -> None:
    file_exists = os.path.exists(EQUITY_LOG)
    try:
        with open(EQUITY_LOG, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "equity", "drawdown"])
            writer.writerow([timestamp.isoformat(timespec="microseconds"), equity, drawdown])
    except Exception as e:
        log(f"[EquityLogError] Failed to write equity log: {e}")