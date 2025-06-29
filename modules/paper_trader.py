# === forex_bot_project/modules/paper_trader.py ===

import time
import config
from modules.data_fetcher import ForexDataFetcher
from modules.strategy import MultiStrategy

class PaperTrader:
    def __init__(self):
        self.fetcher = ForexDataFetcher()
        self.strategy = MultiStrategy()
        self.balance = 10000
        self.open_positions = {}

    def run(self):
        print("Starting paper trading...")
        while True:
            for pair in config.CURRENCY_PAIRS:
                df = self.fetcher.fetch_pair(pair)
                if df is None or df.empty:
                    continue
                signals = self.strategy.generate_signals(df)
                signal = signals["signal"].iloc[-1]
                price = signals["close"].iloc[-1]

                if signal == 1 and pair not in self.open_positions:
                    qty = self._calc_position_size(price)
                    self.open_positions[pair] = {
                        "entry": price,
                        "qty": qty,
                        "stop_loss": price * (1 - config.STOP_LOSS_MULTIPLIER * config.RISK_PER_TRADE),
                        "take_profit": price * (1 + config.TAKE_PROFIT_MULTIPLIER * config.RISK_PER_TRADE)
                    }
                    print(f"[{pair}] BUY @ {price:.5f} (Qty: {qty})")

                elif signal == -1 and pair in self.open_positions:
                    print(f"[{pair}] SELL @ {price:.5f} — Closed position.")
                    del self.open_positions[pair]

                elif pair in self.open_positions:
                    pos = self.open_positions[pair]
                    if price <= pos["stop_loss"]:
                        print(f"[{pair}] STOP LOSS hit @ {price:.5f}")
                        del self.open_positions[pair]
                    elif price >= pos["take_profit"]:
                        print(f"[{pair}] TAKE PROFIT hit @ {price:.5f}")
                        del self.open_positions[pair]

            time.sleep(60)

    def _calc_position_size(self, price):
        risk_amount = self.balance * config.RISK_PER_TRADE
        return round(risk_amount / price, 4)
