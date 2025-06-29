# === forex_bot_project/modules/backtester.py ===

import pandas as pd
import numpy as np
import config

class Backtester:
    def __init__(self, df, signal_df):
        self.df = df.copy()
        self.signals = signal_df.copy()
        self.initial_balance = 10000

    def run(self):
        df = self.df.join(self.signals["signal"], how="left")
        print("Unique signals in strategy:", df["signal"].unique())
        df["position"] = df["signal"].shift(1).fillna(0)
        df["returns"] = df["close"].pct_change().fillna(0)
        df["strategy_returns"] = df["position"] * df["returns"]

        df["equity"] = self.initial_balance * (1 + df["strategy_returns"]).cumprod()

        total_return = df["equity"].iloc[-1] / self.initial_balance - 1
        if df["strategy_returns"].std() == 0 or df["strategy_returns"].sum() == 0:
            sharpe = 0
            total_return = 0
            max_drawdown = 0
        else:
            sharpe = df["strategy_returns"].mean() / df["strategy_returns"].std() * np.sqrt(252 * 12)
            total_return = df["equity"].iloc[-1] / self.initial_balance - 1
            max_drawdown = (df["equity"].cummax() - df["equity"]).max() / df["equity"].cummax().max()

        metrics = {
            "Total Return": round(total_return * 100, 2),
            "Sharpe Ratio": round(sharpe, 2),
            "Max Drawdown": round(max_drawdown * 100, 2)
        }


        return metrics, df
