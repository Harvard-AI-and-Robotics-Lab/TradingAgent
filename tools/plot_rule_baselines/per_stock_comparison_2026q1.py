"""
Per-stock equity-curve comparison of rule-based baselines + agent models
(2026 Q1).

For each stock (AAPL, GOOG, NVDA) produces one line chart:
  x = trading dates, y = portfolio value ($), one line per strategy/agent.
"""

import os
import re
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── Configuration ─────────────────────────────────────────────────────
RESULTS_BASE = os.path.join(
    os.path.dirname(__file__), "..", "..",
    "baseline_analysis", "results", "2026q1", "baseline",
)
AGENT_BASE = os.path.join(os.path.dirname(__file__), "2026q1")

DATE_TAG = "2026-01-01_2026-02-19"
STOCKS = ["AAPL", "GOOG", "NVDA"]

# Rule-based strategies (dir_name → display label)
STRATEGIES = {
    "buy_hold": "Buy-Hold",
    "kdj":      "KDJ",
    "macd":     "MACD",
    "sma":      "SMA",
    "zmr":      "ZMR",
}

# Agent models (dir_name → display label)
AGENT_MODELS = {
    "gpt-4o-mini":           "TA (GPT-4o-mini)",
    "gpt-5-mini":            "TA (GPT-5-mini)",
    "grok-4-fast-reasoning": "TA (Grok-4)",
}

# ── Visual styles ─────────────────────────────────────────────────────
COLORS = {
    "Buy-Hold":          "#8491B4",
    "KDJ":               "#00A087",
    "MACD":              "#E64B35",
    "SMA":               "#4DBBD5",
    "ZMR":               "#F39B7F",
    "TA (GPT-4o-mini)":  "#3C5488",
    "TA (GPT-5-mini)":   "#7E6148",
    "TA (Grok-4)":       "#B09C85",
}

LINESTYLES = {
    "Buy-Hold":          "-",
    "KDJ":               "-",
    "MACD":              "--",
    "SMA":               "-.",
    "ZMR":               ":",
    "TA (GPT-4o-mini)":  "-",
    "TA (GPT-5-mini)":   "-",
    "TA (Grok-4)":       "-",
}

MARKERS = {
    "Buy-Hold":          "s",
    "KDJ":               "o",
    "MACD":              "^",
    "SMA":               "D",
    "ZMR":               "v",
    "TA (GPT-4o-mini)":  "*",
    "TA (GPT-5-mini)":   "P",
    "TA (Grok-4)":       "X",
}

LINEWIDTHS = {
    "Buy-Hold": 1.5, "KDJ": 1.5, "MACD": 1.5, "SMA": 1.5, "ZMR": 1.5,
    "TA (GPT-4o-mini)": 2.5, "TA (GPT-5-mini)": 2.5, "TA (Grok-4)": 2.5,
}

# Only show the actual evaluation window
EVAL_START = pd.Timestamp("2026-01-02", tz="US/Eastern")

OUT_DIR = os.path.dirname(__file__)


# ── Parsing: rule-based ───────────────────────────────────────────────
def parse_equity_curve(path: str) -> pd.Series:
    """Extract the equity curve (date -> equity $) from a result file."""
    dates, equities = [], []
    in_curve = False
    with open(path) as f:
        for line in f:
            if "_equity_curve:" in line:
                in_curve = True
                continue
            if in_curve:
                if line.startswith("_trades:") or line.startswith("["):
                    break
                stripped = line.strip()
                if not stripped or stripped.startswith("...") \
                        or stripped.startswith("Equity") \
                        or stripped.startswith("Date"):
                    continue
                match = re.match(
                    r"(\d{4}-\d{2}-\d{2})\s+\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}\s+([\d.]+)",
                    stripped,
                )
                if match:
                    dt = pd.Timestamp(match.group(1), tz="US/Eastern")
                    eq = float(match.group(2))
                    dates.append(dt)
                    equities.append(eq)
    if not dates:
        return pd.Series(dtype=float)
    return pd.Series(equities, index=pd.DatetimeIndex(dates), name="Equity")


# ── Parsing: agent portfolios ─────────────────────────────────────────
def parse_agent_equity(path: str) -> pd.Series:
    """Read final_portfolio.csv and return date -> portfolio_value Series."""
    df = pd.read_csv(path, parse_dates=["date"])
    idx = pd.DatetimeIndex(df["date"].dt.tz_localize("US/Eastern"))
    return pd.Series(df["portfolio_value"].values, index=idx, name="Equity")


# ── Loading ───────────────────────────────────────────────────────────
def load_equity_curves() -> dict:
    """Return nested dict: curves[stock][label] = pd.Series."""
    curves = {s: {} for s in STOCKS}

    # Rule-based
    for dir_name, label in STRATEGIES.items():
        for stock in STOCKS:
            fname = f"{dir_name}_results_{DATE_TAG}.txt"
            path = os.path.join(RESULTS_BASE, dir_name, stock, fname)
            if not os.path.exists(path):
                print(f"  [warn] missing: {path}")
                continue
            series = parse_equity_curve(path)
            if not series.empty:
                curves[stock][label] = series

    # Agent models
    for dir_name, label in AGENT_MODELS.items():
        for stock in STOCKS:
            path = os.path.join(AGENT_BASE, dir_name, stock, "final_portfolio.csv")
            if not os.path.exists(path):
                print(f"  [warn] missing agent: {path}")
                continue
            series = parse_agent_equity(path)
            if not series.empty:
                curves[stock][label] = series

    return curves


# ── Plotting ──────────────────────────────────────────────────────────
# Plot order: rule-based first (behind), agents on top
PLOT_ORDER = list(STRATEGIES.values()) + list(AGENT_MODELS.values())


def plot_stock_equity(stock: str, curves: dict):
    """Line chart: equity curves for all strategies + agents, one stock."""
    fig, ax = plt.subplots(figsize=(10, 5.5))

    for label in PLOT_ORDER:
        if label not in curves:
            continue
        series = curves[label]
        series = series[series.index >= EVAL_START]
        if series.empty:
            continue

        dates = series.index
        vals = series.values

        ax.plot(
            dates, vals,
            color=COLORS[label],
            linestyle=LINESTYLES[label],
            linewidth=LINEWIDTHS.get(label, 2),
            label=label,
            marker=MARKERS[label],
            markersize=5 if label.startswith("TA") else 4,
            markevery=max(1, len(dates) // 8),
            alpha=0.9,
        )

    # Reference line at $10,000
    ax.axhline(10000, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)

    # Axes formatting
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_minor_locator(mdates.DayLocator())
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)

    ax.set_xlabel("Date", fontsize=11, fontweight="bold")
    ax.set_ylabel("Portfolio Value ($)", fontsize=11, fontweight="bold")
    ax.set_title(
        f"{stock} \u2014 Equity Curves (2026 Q1)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=8, framealpha=0.9, edgecolor="gray",
              loc="best", ncol=2)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = os.path.join(OUT_DIR, f"per_stock_{stock}_2026q1.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {out}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    curves = load_equity_curves()

    for stock in STOCKS:
        print(f"\n{'='*60}")
        print(f"  {stock}")
        print(f"{'='*60}")
        for label in PLOT_ORDER:
            if label not in curves[stock]:
                continue
            series = curves[stock][label]
            clipped = series[series.index >= EVAL_START]
            if not clipped.empty:
                print(f"  {label:20s} | points={len(clipped):>3} | "
                      f"start=${clipped.iloc[0]:,.2f} | end=${clipped.iloc[-1]:,.2f}")

        plot_stock_equity(stock, curves[stock])

    print("\nDone.")
