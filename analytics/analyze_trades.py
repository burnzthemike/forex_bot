# analyze_trades.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from utils import log
from config import EQUITY_LOG  # or define STARTING_EQUITY here

TRADES_CSV = "trades.csv"
STARTING_EQUITY = 10_000  # Should match your engine/backtest start

def load_trades():
    try:
        df = pd.read_csv(TRADES_CSV, parse_dates=["entry_time", "exit_time"])
        df.sort_values("exit_time", inplace=True)
        return df
    except Exception as e:
        log(f"[ANALYTICS] Error loading trades: {e}")
        return pd.DataFrame()

def compute_metrics(df):
    total = len(df)
    wins = df[df["pnl"] > 0]
    losses = df[df["pnl"] <= 0]
    win_rate = len(wins) / total if total else 0
    avg_win = wins["pnl"].mean() if not wins.empty else 0
    avg_loss = losses["pnl"].mean() if not losses.empty else 0
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

    daily = df.groupby(df["exit_time"].dt.date)["pnl"].sum()
    sharpe = (daily.mean() / daily.std() * np.sqrt(252)) if len(daily) > 1 and daily.std() != 0 else 0

    log("======== TRADE PERFORMANCE ========")
    log(f"Total Trades:         {total}")
    log(f"Win Rate:             {win_rate:.2%}")
    log(f"Average Win:          {avg_win:.2f}")
    log(f"Average Loss:         {avg_loss:.2f}")
    log(f"Expectancy:           {expectancy:.2f}")
    log(f"Sharpe Ratio (daily): {sharpe:.2f}")
    log("===================================")

def strategy_breakdown(df):
    if {"strategy_signal", "rl_signal"}.issubset(df.columns):
        df["source"] = np.where(df["rl_signal"] != 0, "RL", "Strategy")
        grouped = df.groupby("source")["pnl"]
        log("=== Attribution Breakdown ===")
        for src, pnl in grouped:
            count = len(pnl)
            log(f"{src:<10} | Trades: {count:<4} | Avg PnL: {pnl.mean():>6.2f} | Win Rate: { (pnl>0).mean():.2% }")
    else:
        log("[ANALYTICS] No strategy_signal/rl_signal columns for attribution.")

def plot_equity_curve(df):
    df = df.copy()
    df["equity"] = STARTING_EQUITY + df["pnl"].cumsum()
    plt.figure()
    plt.plot(df["exit_time"], df["equity"], label="Equity")
    plt.title("Equity Curve")
    plt.xlabel("Time")
    plt.ylabel("Equity")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_pnl_distribution(df):
    plt.figure()
    sns.histplot(df["pnl"], bins=30, kde=True)
    plt.title("PnL Distribution")
    plt.xlabel("Pnl")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_heatmap(df):
    # Only if we have at least one week of trades
    if df.empty:
        log("[ANALYTICS] No trades to plot heatmap.")
        return
    df = df.copy()
    df["day"] = df["exit_time"].dt.day_name()
    df["hour"] = df["exit_time"].dt.hour
    pivot = df.pivot_table(
        index="day", columns="hour", values="pnl", aggfunc="sum"
    ).reindex(["Monday","Tuesday","Wednesday","Thursday","Friday"]).fillna(0)
    if pivot.empty:
        log("[ANALYTICS] Heatmap pivot is empty.")
        return
    plt.figure(figsize=(10,5))
    sns.heatmap(pivot, center=0, annot=True, fmt=".1f")
    plt.title("PnL Heatmap by Day/Hour")
    plt.tight_layout()
    plt.show()

def plot_signal_vs_equity(df):
    if {"source","pnl"}.issubset(df.columns):
        df = df.copy()
        df["equity"] = STARTING_EQUITY + df["pnl"].cumsum()
        plt.figure()
        sns.lineplot(data=df, x="exit_time", y="equity", hue="source")
        plt.title("Equity by Signal Source")
        plt.tight_layout()
        plt.show()
    else:
        log("[ANALYTICS] Cannot plot signal vs equity, missing columns.")

def run_analysis():
    df = load_trades()
    if df.empty:
        log("[ANALYTICS] No trades to analyze.")
        return

    compute_metrics(df)
    strategy_breakdown(df)
    plot_equity_curve(df)
    plot_pnl_distribution(df)
    plot_heatmap(df)
    plot_signal_vs_equity(df)

if __name__ == "__main__":
    run_analysis()
