# simulate_trade_log.py (extended for testing)
from trade_logger import log_trade
from datetime import datetime, timedelta
import random

symbols = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD"]

for _ in range(100):
    entry_time = datetime.utcnow() - timedelta(minutes=random.randint(60, 1440))
    exit_time = entry_time + timedelta(minutes=random.randint(30, 120))
    entry_price = round(random.uniform(1.10, 1.30), 5)
    direction = random.choice([-1, 1])
    pnl = round(random.uniform(-30, 50), 2)
    exit_price = round(entry_price + (pnl / 1000) * direction, 5)
    pos_size = random.choice([500, 1000, 2000])
    drawdown = round(random.uniform(0.0, 0.05), 4)
    strategy = random.choice([-1, 0, 1])
    rl = random.choice([-1, 0, 1])
    final = random.choice([-1, 0, 1])
    
    log_trade(
        symbol=random.choice(symbols),
        entry_time=entry_time,
        exit_time=exit_time,
        entry_price=entry_price,
        exit_price=exit_price,
        direction=direction,
        pnl=pnl,
        position_size=pos_size,
        drawdown=drawdown,
        strategy_signal=strategy,
        rl_signal=rl,
        final_signal=final
    )
print("✅ Simulated 100+ trades.")
