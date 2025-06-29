# === optimize.py ===

import itertools
import config
from strategy import MultiStrategy
from backtest import backtest

def optimize(data_dict):
    """
    Perform grid search optimization over parameter grid.

    data_dict: dict of currency pair tuples -> DataFrame
    """

    best_params = None
    best_metrics = {"Sharpe Ratio": float("-inf")}
    grid = config.OPTIMIZATION_GRID

    param_names = list(grid.keys())
    param_values = [grid[name] for name in param_names]

    total_runs = 1
    for vals in param_values:
        total_runs *= len(vals)
    print(f"Starting optimization over {total_runs} parameter combinations...")

    for combination in itertools.product(*param_values):
        params = dict(zip(param_names, combination))
        strat = MultiStrategy(params)

        combined_returns = []

        # Backtest on all pairs
        for pair, df in data_dict.items():
            df_signals = strat.generate_signals(df)
            results = backtest(
                df_signals,
                initial_balance=config.BACKTEST_SETTINGS["initial_balance"],
                position_size=config.BACKTEST_SETTINGS["position_size"],
                slippage=config.BACKTEST_SETTINGS["slippage"],
                commission_per_trade=config.BACKTEST_SETTINGS["commission_per_trade"],
            )
            combined_returns.append(results["Sharpe Ratio"])

        avg_sharpe = sum(combined_returns) / len(combined_returns)

        if avg_sharpe > best_metrics["Sharpe Ratio"]:
            best_metrics["Sharpe Ratio"] = avg_sharpe
            best_params = params

    return best_params, best_metrics
