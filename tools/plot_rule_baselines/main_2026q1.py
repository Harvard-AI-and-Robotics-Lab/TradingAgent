"""
Plot 2026Q1 baseline strategy results.

Reads result files from baseline_analysis/results/2026q1/baseline/ for the
2026-01-01 to 2026-02-19 period, and agent portfolio CSVs from
tools/plot_rule_baselines/2026q1/.

Extracts CR / ARR / SR / MDD per strategy per stock, and produces:
  1. 2x2 averaged-bar chart
  2. Per-stock grouped-bar chart
  3. Risk-return trade-off scatter
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ── Configuration ──────────────────────────────────────────────────────
RESULTS_BASE = os.path.join(
    os.path.dirname(__file__), "..", "..",
    "baseline_analysis", "results", "2026q1", "baseline",
)
AGENT_BASE = os.path.join(os.path.dirname(__file__), "2026q1")

DATE_TAG = "2026-01-01_2026-02-19"
STOCKS = ["AAPL", "GOOG", "NVDA"]

# Rule-based strategies (dir_name → display label)
STRATEGIES = {
    "buy_hold": "Buy-and-Hold",
    "macd":     "MACD",
    "kdj":      "KDJ",
    "zmr":      "ZMR",
    "sma":      "SMA",
}

# Agent models (dir_name → display label)
AGENT_MODELS = {
    "gpt-4o-mini":           "GPT-4o-mini\n(partial alloc.)",
    "gpt-5-mini":            "GPT-5-mini\n(partial alloc.)",
    "grok-4-fast-reasoning": "Grok-4\n(partial alloc.)",
}

# Metrics to extract (key in file → display name)
METRIC_KEYS = {
    "Return [%]":         "CR",
    "Return (Ann.) [%]":  "ARR",
    "Sharpe Ratio":       "SR",
    "Max. Drawdown [%]":  "MDD",
}

TRADING_DAYS_PER_YEAR = 252


# ── Parsing: rule-based ───────────────────────────────────────────────
def parse_result_file(path: str) -> dict:
    """Extract numeric metrics from a backtesting result text file."""
    metrics = {}
    with open(path) as f:
        for line in f:
            for key, alias in METRIC_KEYS.items():
                if line.startswith(key + ":"):
                    val_str = line.split(":", 1)[1].strip()
                    try:
                        val = float(val_str)
                    except ValueError:
                        val = float("nan")
                    metrics[alias] = val
                    break
    if "MDD" in metrics:
        metrics["MDD"] = abs(metrics["MDD"])
    return metrics


# ── Parsing: agent portfolios ─────────────────────────────────────────
def compute_metrics_from_csv(path: str) -> dict:
    """Compute CR / ARR / SR / MDD from a final_portfolio.csv file."""
    df = pd.read_csv(path, parse_dates=["date"])
    pv = df["portfolio_value"].values

    if len(pv) < 2:
        return {k: float("nan") for k in ["CR", "ARR", "SR", "MDD"]}

    # Cumulative return
    cr = (pv[-1] / pv[0] - 1) * 100

    # Annualised return
    n_days = len(pv)
    total_return = pv[-1] / pv[0]
    arr = (total_return ** (TRADING_DAYS_PER_YEAR / n_days) - 1) * 100

    # Sharpe ratio (annualised, from daily returns)
    daily_ret = np.diff(pv) / pv[:-1]
    if np.std(daily_ret) > 0:
        sr = (np.mean(daily_ret) / np.std(daily_ret)) * np.sqrt(TRADING_DAYS_PER_YEAR)
    else:
        sr = float("nan")

    # Max drawdown (positive magnitude)
    running_max = np.maximum.accumulate(pv)
    drawdowns = (pv - running_max) / running_max * 100
    mdd = abs(drawdowns.min())

    return {"CR": cr, "ARR": arr, "SR": sr, "MDD": mdd}


# ── Loading ───────────────────────────────────────────────────────────
def load_all_results() -> dict:
    """Return nested dict: data[strategy_label][stock] = {CR, ARR, SR, MDD}."""
    data = {}

    # Rule-based strategies
    for dir_name, label in STRATEGIES.items():
        data[label] = {}
        for stock in STOCKS:
            fname = f"{dir_name}_results_{DATE_TAG}.txt"
            path = os.path.join(RESULTS_BASE, dir_name, stock, fname)
            if not os.path.exists(path):
                print(f"  [warn] missing: {path}")
                continue
            data[label][stock] = parse_result_file(path)

    # Agent models
    for dir_name, label in AGENT_MODELS.items():
        data[label] = {}
        for stock in STOCKS:
            path = os.path.join(AGENT_BASE, dir_name, stock, "final_portfolio.csv")
            if not os.path.exists(path):
                print(f"  [warn] missing agent: {path}")
                continue
            data[label][stock] = compute_metrics_from_csv(path)

    return data


# ── Group / style maps ────────────────────────────────────────────────
GROUP = {
    "Buy-and-Hold":      "Market",
    "MACD":              "Rule-based",
    "KDJ":               "Rule-based",
    "ZMR":               "Rule-based",
    "SMA":               "Rule-based",
    "GPT-4o-mini\n(partial alloc.)":  "Agent",
    "GPT-5-mini\n(partial alloc.)":   "Agent",
    "Grok-4\n(partial alloc.)":       "Agent",
}

ALL_COLORS = [
    "#4DBBD5", "#E64B35", "#00A087", "#F39B7F", "#8491B4",  # rule-based
    "#3C5488", "#7E6148", "#B09C85",                         # agents
]


# ── Plotting ───────────────────────────────────────────────────────────
def plot_avg_panel(data: dict):
    """2x2 panel: averaged CR / ARR / SR / MDD across stocks."""
    metrics = ["CR", "ARR", "SR", "MDD"]
    titles = [
        "Avg Cumulative Return (CR) [%]",
        "Avg Annualized Return (ARR) [%]",
        "Avg Sharpe Ratio (SR) [%]",
        "Avg Max Drawdown (MDD) [%]",
    ]

    rows = []
    for method, assets in data.items():
        vals = {}
        for m in metrics:
            vs = [assets[s][m] for s in assets
                  if m in assets[s] and not np.isnan(assets[s][m])]
            vals[m] = np.mean(vs) if vs else float("nan")
        rows.append({"Method": method, **vals})

    df = pd.DataFrame(rows).set_index("Method")
    print(df)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    x = np.arange(len(df.index))
    colors = ALL_COLORS[: len(x)]

    for ax, m, t in zip(axes, metrics, titles):
        bars = ax.bar(x, df[m].values, color=colors)
        ax.set_title(t, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(df.index, rotation=30, ha="right", fontsize=8)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.5)

        for bar, val in zip(bars, df[m].values):
            if np.isnan(val):
                continue
            fmt = f"{val:.2f}%" if m != "SR" else f"{val:.3f}"
            yoff = 3 if val >= 0 else -12
            va = "bottom" if val >= 0 else "top"
            ax.annotate(
                fmt,
                xy=(bar.get_x() + bar.get_width() / 2, val),
                xytext=(0, yoff),
                textcoords="offset points",
                ha="center", va=va, fontsize=7, fontweight="bold",
            )

    fig.suptitle(
        "2026 Q1 Strategies (Averaged Across AAPL, GOOG, NVDA)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "baselines_2026q1_avg.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")
    plt.close(fig)


def plot_per_stock(data: dict):
    """Grouped bar chart: each stock side-by-side for CR and MDD."""
    methods = list(data.keys())
    n_methods = len(methods)
    n_stocks = len(STOCKS)
    bar_width = 0.18
    x = np.arange(n_methods)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    stock_colors = ["#4DBBD5", "#E64B35", "#00A087"]

    for ax, metric, title in [
        (ax1, "CR", "Cumulative Return (CR) [%]"),
        (ax2, "MDD", "Max Drawdown (MDD) [%]"),
    ]:
        for i, stock in enumerate(STOCKS):
            vals = []
            for method in methods:
                v = data[method].get(stock, {}).get(metric, float("nan"))
                vals.append(v)
            offset = (i - n_stocks / 2 + 0.5) * bar_width
            bars = ax.bar(x + offset, vals, bar_width,
                          label=stock, color=stock_colors[i])

            for bar, val in zip(bars, vals):
                if np.isnan(val):
                    continue
                ax.annotate(
                    f"{val:.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2, val),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=6,
                )

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=8)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.5)

    fig.suptitle(
        "2026 Q1 Strategies \u2013 Per Stock",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "baselines_2026q1_per_stock.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")
    plt.close(fig)


def _scatter_tradeoff(ax, records, label_offsets):
    """Shared scatter rendering for tradeoff plots (MDD or SR on x-axis)."""
    group_to_marker = {
        "Market":     "<",
        "Rule-based": "<",
        "Agent":      "s",
        "Ours":       "X",
    }

    group_to_facecolor = {
        "Market":     "black",
        "Rule-based": "black",
        "Agent":      "#E64B35",
        "Ours":       "limegreen",
    }
    group_to_edgecolor = {
        "Market":     "black",
        "Rule-based": "black",
        "Agent":      "#E64B35",
        "Ours":       "black",
    }
    group_to_size = {
        "Market":     80,
        "Rule-based": 80,
        "Agent":      80,
        "Ours":       250,
    }

    for method, cr, xval, group in records:
        ax.scatter(
            xval, cr,
            s=group_to_size.get(group, 80),
            marker=group_to_marker.get(group, "o"),
            facecolor=group_to_facecolor.get(group, "black"),
            edgecolor=group_to_edgecolor.get(group, "black"),
            linewidth=1.5 if group == "Agent" else 1.0,
        )

        dx, dy = label_offsets.get(method, (0.15, 0.0))
        ax.annotate(
            method,
            xy=(xval, cr),
            xytext=(xval + dx, cr + dy),
            textcoords="data",
            arrowprops=dict(arrowstyle="-", lw=0.5, alpha=0.6),
            fontsize=9,
            fontweight="bold",
        )


def _tradeoff_legend():
    """Shared legend handles for tradeoff plots."""
    scatter_handles = [
        Line2D([0], [0], marker="<", linestyle="None",
               markerfacecolor="black", markeredgecolor="black",
               markersize=7, label="Baselines"),
        Line2D([0], [0], marker="s", linestyle="None",
               markerfacecolor="#E64B35", markeredgecolor="#E64B35",
               markersize=7, label="LLMs (Partial alloc.)"),
        Line2D([0], [0], marker="X", linestyle="None",
               markerfacecolor="limegreen", markeredgecolor="black",
               markersize=10, label="LLMs (Ours)"),
    ]
    line_handles = [
        Line2D([0], [0], linestyle="-", color="#3776ab", linewidth=1.5,
               label="Pareto frontier"),
        Line2D([0], [0], linestyle="--", color="#3776ab", linewidth=1.5,
               label="Linear trade-off fit"),
    ]
    return scatter_handles + line_handles


def plot_risk_return_tradeoff(data: dict):
    """
    Scatter plot: x = MDD, y = CR (averaged across stocks).
    Pareto frontier, linear fit, annotated labels, clean spines, legend.
    """
    # Build records
    records = []
    for method, assets in data.items():
        crs  = [assets[s]["CR"]  for s in assets
                if not np.isnan(assets[s].get("CR", float("nan")))]
        mdds = [assets[s]["MDD"] for s in assets
                if not np.isnan(assets[s].get("MDD", float("nan")))]
        if not crs or not mdds:
            continue
        records.append((method, np.mean(crs), np.mean(mdds), GROUP[method]))

    # Pareto frontier
    pts_sorted = sorted(records, key=lambda t: t[2])
    frontier = []
    best_cr = -1e18
    for m, cr, mdd, _ in pts_sorted:
        if cr > best_cr:
            frontier.append((m, cr, mdd))
            best_cr = cr

    label_offsets = {
        "Buy-and-Hold":      (+0.3, -0.1),
        "MACD":              (+0.3, -0.1),
        "KDJ":               (+0.3, -0.1),
        "ZMR":               (+0.3, -0.1),
        "SMA":               (+0.3, -0.1),
        "GPT-4o-mini\n(partial alloc.)":  (+0.3, +0.5),
        "GPT-5-mini\n(partial alloc.)":   (+0.3, -0.7),
        "Grok-4\n(partial alloc.)":       (-2.0, +0.5),
    }

    fig, ax = plt.subplots(figsize=(7, 6.5))

    _scatter_tradeoff(ax, records, label_offsets)

    # Pareto frontier line
    fx = [mdd for (_, _, mdd) in frontier]
    fy = [cr  for (_, cr, _) in frontier]
    ax.plot(fx, fy, linewidth=1.25, color="#3776ab")

    # Linear fit
    x_arr = np.array([mdd for (_, _, mdd, _) in records])
    y_arr = np.array([cr  for (_, cr, _, _) in records])
    a, b = np.polyfit(x_arr, y_arr, 1)
    xx = np.linspace(x_arr.min() - 0.5, x_arr.max() + 0.5, 200)
    yy = a * xx + b
    ax.plot(xx, yy, linestyle="--", linewidth=1.25, color="#3776ab")

    ax.set_xlabel("Average Max Drawdown % (MDD) \u2193",
                  fontsize=10, fontweight="bold")
    ax.set_ylabel("Average Cumulative Return % (CR) \u2191",
                  fontsize=10, fontweight="bold")
    ax.set_title("Risk\u2013Return Trade-off  (2026 Q1, Averaged Across All Stocks)",
                 fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", linewidth=0.5, color="gray")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        handles=_tradeoff_legend(),
        loc="lower right",
        frameon=True, framealpha=0.95, edgecolor="gray", fontsize=9,
    )

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "risk_return_tradeoff_2026q1.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")
    plt.close(fig)


def plot_sr_return_tradeoff(data: dict):
    """
    Scatter plot: x = SR, y = CR (averaged across stocks).
    Pareto frontier, linear fit, annotated labels, clean spines, legend.
    """
    # Build records: (method, cr, sr, group)
    records = []
    for method, assets in data.items():
        crs = [assets[s]["CR"] for s in assets
               if not np.isnan(assets[s].get("CR", float("nan")))]
        srs = [assets[s]["SR"] for s in assets
               if not np.isnan(assets[s].get("SR", float("nan")))]
        if not crs or not srs:
            continue
        records.append((method, np.mean(crs), np.mean(srs), GROUP[method]))

    # Pareto frontier: maximize both SR and CR
    pts_sorted = sorted(records, key=lambda t: t[2])  # sort by SR asc
    frontier = []
    best_cr = -1e18
    for m, cr, sr, _ in pts_sorted:
        if cr > best_cr:
            frontier.append((m, cr, sr))
            best_cr = cr

    label_offsets = {
        "Buy-and-Hold":      (+0.15, -0.1),
        "MACD":              (+0.15, -0.1),
        "KDJ":               (+0.15, -0.1),
        "ZMR":               (+0.15, -0.1),
        "SMA":               (+0.15, -0.1),
        "GPT-4o-mini\n(partial alloc.)":  (+0.15, +0.5),
        "GPT-5-mini\n(partial alloc.)":   (+0.15, -0.7),
        "Grok-4\n(partial alloc.)":       (+0.15, +0.5),
    }

    fig, ax = plt.subplots(figsize=(7, 6.5))

    _scatter_tradeoff(ax, records, label_offsets)

    # Pareto frontier line
    fx = [sr for (_, _, sr) in frontier]
    fy = [cr for (_, cr, _) in frontier]
    ax.plot(fx, fy, linewidth=1.25, color="#3776ab")

    # Linear fit
    x_arr = np.array([sr for (_, _, sr, _) in records])
    y_arr = np.array([cr for (_, cr, _, _) in records])
    a, b = np.polyfit(x_arr, y_arr, 1)
    xx = np.linspace(x_arr.min() - 0.3, x_arr.max() + 0.3, 200)
    yy = a * xx + b
    ax.plot(xx, yy, linestyle="--", linewidth=1.25, color="#3776ab")

    ax.set_xlabel("Average Sharpe Ratio (SR) \u2191",
                  fontsize=10, fontweight="bold")
    ax.set_ylabel("Average Cumulative Return % (CR) \u2191",
                  fontsize=10, fontweight="bold")
    ax.set_title("SR\u2013Return Trade-off  (2026 Q1, Averaged Across All Stocks)",
                 fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", linewidth=0.5, color="gray")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        handles=_tradeoff_legend(),
        loc="lower right",
        frameon=True, framealpha=0.95, edgecolor="gray", fontsize=9,
    )

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "sr_return_tradeoff_2026q1.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    data = load_all_results()

    for method, assets in data.items():
        for stock, vals in assets.items():
            print(f"  {method:20s} | {stock} | CR={vals.get('CR', 'n/a'):>8} | "
                  f"ARR={vals.get('ARR', 'n/a'):>9} | SR={vals.get('SR', 'n/a'):>8} | "
                  f"MDD={vals.get('MDD', 'n/a'):>8}")

    plot_avg_panel(data)
    plot_per_stock(data)
    plot_risk_return_tradeoff(data)
    plot_sr_return_tradeoff(data)
