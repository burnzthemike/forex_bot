import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Template

# ─── CONFIG ────────────────────────────────────────────────────────────────────

TRADES_CSV    = "trades.csv"
REPORT_DIR    = "reports"
REPORT_HTML   = os.path.join(REPORT_DIR, "trading_report.html")

# Ensure output directory exists
os.makedirs(REPORT_DIR, exist_ok=True)


# ─── DATA LOADING & SUMMARY ────────────────────────────────────────────────────

def load_trades():
    """Load and preprocess the trades CSV."""
    if not os.path.exists(TRADES_CSV):
        print(f"[Report] No trade log found at '{TRADES_CSV}'.")
        return pd.DataFrame()

    df = pd.read_csv(TRADES_CSV, parse_dates=["entry_time", "exit_time"])
    df.sort_values("exit_time", inplace=True)
    # Mark which signal source “won”
    df["source"] = df.apply(lambda x: "RL" if x.get("rl_signal", 0) != 0 else "Strategy", axis=1)
    # Build cumulative equity curve
    df["equity"] = 10000 + df["pnl"].cumsum()
    return df

def compute_summary(df: pd.DataFrame) -> dict:
    """Compute a small dictionary of summary metrics."""
    total     = len(df)
    wins      = df[df.pnl > 0]
    losses    = df[df.pnl <= 0]
    win_rate  = len(wins) / total if total else 0
    avg_win   = wins.pnl.mean() if not wins.empty else 0
    avg_loss  = losses.pnl.mean() if not losses.empty else 0
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss
    sharpe     = df.pnl.mean() / (df.pnl.std() + 1e-6) * np.sqrt(252)

    final_equity = df["equity"].iat[-1] if total else 0

    return {
        "Total Trades":    total,
        "Win Rate":        f"{win_rate:.2%}",
        "Avg Win":         f"{avg_win:.2f}",
        "Avg Loss":        f"{avg_loss:.2f}",
        "Expectancy":      f"{expectancy:.2f}",
        "Sharpe Ratio":    f"{sharpe:.2f}",
        "Final Equity":    f"{final_equity:.2f}"
    }


# ─── PLOT GENERATION ────────────────────────────────────────────────────────────

def save_plot(fig, filename: str) -> str:
    """Save a Matplotlib figure into REPORT_DIR and return its relative path."""
    path = os.path.join(REPORT_DIR, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Report] Saved plot: {path}")
    return filename

def generate_plots(df: pd.DataFrame) -> dict:
    """Build all the charts and return a dict of filename keys."""
    plots = {}

    # Equity Curve
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.exit_time, df.equity, label="Equity")
    ax.set_title("Equity Curve")
    ax.set_xlabel("Time")
    ax.set_ylabel("Equity")
    ax.grid(True)
    plots["equity"] = save_plot(fig, "equity.png")

    # PnL Distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df.pnl, bins=30, kde=True, ax=ax)
    ax.set_title("PnL Distribution")
    ax.set_xlabel("Profit / Loss")
    plots["pnl"] = save_plot(fig, "pnl_dist.png")

    # Attribution: RL vs Strategy average PnL
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(
        data=df,
        x="source", y="pnl",
        estimator=np.mean,
        errorbar=None,
        ax=ax
    )
    ax.set_title("Avg PnL by Source")
    ax.set_ylabel("Average PnL")
    plots["attribution"] = save_plot(fig, "attribution.png")

    # Heatmap: PnL by Day & Hour
    df["day"]  = df.exit_time.dt.day_name()
    df["hour"] = df.exit_time.dt.hour
    pivot = (
        df.pivot_table(index="day", columns="hour", values="pnl", aggfunc="sum")
          .reindex(["Monday","Tuesday","Wednesday","Thursday","Friday"])
          .fillna(0)
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(pivot, cmap="RdYlGn", center=0, annot=True, fmt=".1f", ax=ax)
    ax.set_title("PnL Heatmap by Day/Hour")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Day of Week")
    plots["heatmap"] = save_plot(fig, "heatmap.png")

    return plots


# ─── HTML REPORT RENDERING ──────────────────────────────────────────────────────

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Forex Bot Performance Report</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 30px; background: #f8f9fa; color: #212529; }
    h1, h2 { color: #003366; }
    table.metric-table td { padding: 6px 12px; border-bottom: 1px solid #ccc; }
    .charts img { margin: 10px 0; width: 100%; max-width: 800px; border: 1px solid #ccc; }
  </style>
</head>
<body>
  <h1>📊 Forex Bot Trading Report</h1>
  <p>Generated: {{ date }}</p>

  <h2>📋 Summary</h2>
  <table class="metric-table">
    {% for key, val in summary.items() %}
      <tr><td><strong>{{ key }}</strong></td><td>{{ val }}</td></tr>
    {% endfor %}
  </table>

  <h2>📈 Charts</h2>
  <div class="charts">
    <h3>Equity Curve</h3>
    <img src="{{ plots.equity }}" alt="Equity Curve">
    <h3>PnL Distribution</h3>
    <img src="{{ plots.pnl }}" alt="PnL Distribution">
    <h3>Avg PnL by Source</h3>
    <img src="{{ plots.attribution }}" alt="Attribution">
    <h3>PnL Heatmap</h3>
    <img src="{{ plots.heatmap }}" alt="Heatmap">
  </div>
</body>
</html>
"""

def render_html(summary: dict, plots: dict):
    """Write out the final HTML report."""
    tpl = Template(HTML_TEMPLATE)
    html = tpl.render(
        date   = datetime.now().strftime("%Y-%m-%d %H:%M"),
        summary= summary,
        plots  = plots
    )
    with open(REPORT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[Report] HTML report generated at: {REPORT_HTML}")


# ─── MAIN ENTRY POINT ────────────────────────────────────────────────────────────

def generate_report():
    df = load_trades()
    if df.empty:
        print("[Report] No trades to report. Exiting.")
        return

    summary = compute_summary(df)
    plots   = generate_plots(df)
    render_html(summary, plots)


if __name__ == "__main__":
    generate_report()
