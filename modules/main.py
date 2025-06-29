# === forex_bot_project/main.py ===
from modules.logger import setup_logger
from modules.data_fetcher import ForexDataFetcher
from modules.strategy import EMARSIStrategy
from modules.backtester import Backtester
from modules.optimizer import StrategyOptimizer
from modules.paper_trader import PaperTrader
import config

def run_backtest():
    fetcher = ForexDataFetcher()
    data = fetcher.fetch_all_pairs()
    strategy = EMARSIStrategy()
    results = {}
    for pair, df in data.items():
        if df is not None and not df.empty:
            signals = strategy.generate_signals(df)
            metrics, _ = Backtester(df, signals).run()
            results[pair] = metrics
    for pair, metrics in results.items():
        print(f"{pair}: {metrics}")

def run_optimization():
    optimizer = StrategyOptimizer()
    optimizer.run_grid_search()

def run_paper_trading():
    trader = PaperTrader()
    trader.run()

if __name__ == "__main__":
    setup_logger()
    run_backtest()
    # Uncomment to run optimization or live testing
    # run_optimization()
    # run_paper_trading()
