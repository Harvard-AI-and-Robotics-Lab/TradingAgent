#!/usr/bin/env python3
"""
Plot flowcharts that explain how the multi-agent pipeline builds high-consensus
signals and filters low-consensus signals.

Code grounding for this diagram:
- tradingagents/graph/trading_graph.py
- tradingagents/graph/setup_multiagent.py
- tradingagents/graph/conditional_logic.py
- tradingagents/agents/credibility_scorer/*.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# Visual system tuned for paper readability
COLORS = {
    "text": "#102A43",
    "stroke": "#334E68",
    "muted": "#627D98",
    "lane_top": "#F0F7FF",
    "lane_mid": "#F6FFF6",
    "lane_bottom": "#FFF8F0",
    "input": "#FFFFFF",
    "domain": "#FFFFFF",
    "scorer": "#FFFFFF",
    "debate": "#FFFFFF",
    "trader": "#FFFFFF",
    "high": "#FFFFFF",
    "high_edge": "#1E7A46",
    "low": "#FFFFFF",
    "low_edge": "#A61B1B",
    "neutral": "#FFFFFF",
}


def setup_canvas(fig, ax, xlim, ylim):
    fig.patch.set_facecolor("#FCFEFF")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")


def draw_lane(ax, x, y, w, h, title, fc):
    lane = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=0.8,
        edgecolor="#D9E2EC",
        facecolor=fc,
        zorder=0,
    )
    ax.add_patch(lane)
    title_key = title.strip().lower()
    is_emphasis_lane = title_key in {"orchestration layer", "consensus filtering layer"}
    if is_emphasis_lane:
        ax.text(
            x + 0.18,
            y + h - 0.28,
            title,
            fontsize=17,
            fontweight="bold",
            color=COLORS["text"],
            va="top",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFFFFF", edgecolor="none", alpha=0.9),
        )
    else:
        ax.text(x + 0.18, y + h - 0.3, title, fontsize=10, fontweight="bold", color=COLORS["muted"], va="top")


def draw_node(ax, x, y, w, h, text, fc="#FFFFFF", ec=None, fontsize=11, bold=False):
    edge = ec if ec is not None else COLORS["stroke"]

    # soft shadow
    shadow = FancyBboxPatch(
        (x + 0.06, y - 0.05),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=0,
        facecolor="#D9E2EC",
        alpha=0.45,
        zorder=1,
    )
    ax.add_patch(shadow)

    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.3,
        edgecolor=edge,
        facecolor=fc,
        zorder=2,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=COLORS["text"],
        fontweight="bold" if bold else "normal",
        zorder=3,
    )


def draw_arrow(ax, x1, y1, x2, y2, text="", text_offset=(0, 0), color=None, rad=0.0):
    c = color if color is not None else COLORS["stroke"]
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="Simple,head_length=7,head_width=7,tail_width=0.9",
        linewidth=0,
        facecolor=c,
        edgecolor=c,
        alpha=0.95,
        zorder=2,
    )
    ax.add_patch(arrow)

    if text:
        xm = (x1 + x2) / 2 + text_offset[0]
        ym = (y1 + y2) / 2 + text_offset[1]
        ax.text(xm, ym, text, fontsize=9, color=c, ha="center", va="center", fontweight="bold")


def make_overview_chart(output_path: Path):
    fig, ax = plt.subplots(figsize=(18, 10), dpi=300)
    setup_canvas(fig, ax, (0, 16), (0, 9))

    # Lanes
    draw_lane(ax, 0.2, 6.0, 15.6, 2.75, "Orchestration Layer", COLORS["lane_top"])
    draw_lane(ax, 0.2, 2.2, 15.6, 3.65, "Consensus Filtering Layer", COLORS["lane_mid"])

    # Top pipeline
    draw_node(ax, 0.65, 7.25, 2.25, 0.95, "1) Input\n(ticker, date)", fc=COLORS["input"], fontsize=12, bold=True)
    # Represent N agents as overlapped same-size boxes (stacked cards).
    draw_node(ax, 3.31, 7.31, 2.8, 0.95, "", fc=COLORS["domain"], fontsize=12, bold=True)
    draw_node(ax, 3.25, 7.28, 2.8, 0.95, "", fc=COLORS["domain"], fontsize=12, bold=True)
    draw_node(ax, 3.19, 7.25, 2.8, 0.95, "2) N-Agent Teams", fc=COLORS["domain"], fontsize=13, bold=True)
    draw_node(ax, 6.45, 7.25, 2.95, 0.95, "3) Credibility Scorer", fc=COLORS["scorer"], fontsize=13, bold=True)
    draw_node(ax, 9.80, 7.25, 2.35, 0.95, "4) Bull/Bear\nDebate", fc=COLORS["debate"], fontsize=12, bold=True)
    draw_node(ax, 12.55, 7.25, 2.75, 0.95, "5) Trader Decision", fc=COLORS["trader"], fontsize=12, bold=True)

    draw_arrow(ax, 2.9, 7.75, 3.19, 7.75)
    draw_arrow(ax, 5.99, 7.75, 6.45, 7.75)
    draw_arrow(ax, 9.40, 7.75, 9.80, 7.75)
    draw_arrow(ax, 12.15, 7.75, 12.55, 7.75)

    # Aggregate domain reports (less fragmented than 5 separate boxes)
    draw_node(ax, 3.95, 6.05, 5.10, 0.9, "Domain reports\n(Price | Fund. | Market | News | Social)",
              fc=COLORS["domain"], fontsize=12, bold=True)
    draw_arrow(ax, 4.6, 7.2, 6.5, 6.95, color=COLORS["muted"])
    draw_arrow(ax, 6.55, 6.95, 7.90, 7.20, color=COLORS["stroke"])

    # Consensus detail (5 columns, aligned with top row as much as possible)
    draw_node(ax, 0.65, 4.35, 2.25, 1.0, "Collect N reports", fc=COLORS["neutral"], fontsize=13, bold=True)  # col-1
    draw_node(ax, 3.35, 4.35, 2.8, 1.0, "Consensus filtering", fc=COLORS["neutral"], fontsize=13, bold=True)  # col-2
    draw_node(ax, 6.85, 4.70, 2.9, 0.85, "High consensus", fc=COLORS["high"], ec=COLORS["high_edge"], fontsize=12, bold=True)  # col-3
    draw_node(ax, 6.85, 3.35, 2.9, 0.85, "Low consensus", fc=COLORS["low"], ec=COLORS["low_edge"], fontsize=12, bold=True)  # col-3
    draw_node(ax, 10.05, 4.35, 2.6, 1.0, "Research Manager", fc=COLORS["scorer"], fontsize=12, bold=True)  # col-4
    draw_node(ax, 12.95, 4.35, 2.35, 1.0, "Trader output\nBUY / HOLD / SELL + %", fc=COLORS["trader"], fontsize=12, bold=True)  # col-5

    draw_arrow(ax, 2.9, 4.85, 3.35, 4.85)
    draw_arrow(ax, 6.15, 4.85, 6.85, 5.10, text="keep", color=COLORS["high_edge"])
    draw_arrow(ax, 6.15, 4.85, 6.85, 3.78, text="filter", color=COLORS["low_edge"])
    draw_arrow(ax, 9.75, 5.10, 10.05, 4.90, color=COLORS["high_edge"])
    draw_arrow(ax, 9.75, 3.78, 10.05, 4.70, color=COLORS["low_edge"], rad=0.08)
    draw_arrow(ax, 12.65, 4.85, 12.95, 4.85, text="final proposal")

    # Memory + reflection loop (important in trading_graph.py)
    draw_node(
        ax,
        0.65,
        2.45,
        2.95,
        0.9,
        "init_agent_state[\nlatest_long_short_memory]",
        fc=COLORS["neutral"],
        fontsize=10,
        bold=True,
    )
    draw_node(
        ax,
        10.05,
        2.45,
        2.95,
        0.9,
        "MemoryBank\n(add_trade_history)",
        fc=COLORS["neutral"],
        fontsize=11,
        bold=True,
    )
    draw_node(
        ax,
        6.55,
        2.45,
        3.20,
        0.9,
        "Trend metrics\nreturns + Sharpe @ 1/7/14/28d...",
        fc=COLORS["neutral"],
        fontsize=10.5,
        bold=True,
    )
    draw_node(
        ax,
        13.20,
        2.45,
        2.10,
        0.9,
        "Trading strategy\nreflection",
        fc=COLORS["neutral"],
        fontsize=11,
        bold=True,
    )
    draw_node(
        ax,
        13.20,
        1.35,
        2.10,
        0.85,
        "Role memories\n(bull/bear/trader/judge)",
        fc=COLORS["neutral"],
        fontsize=9.5,
        bold=True,
    )
    draw_arrow(ax, 14.10, 4.35, 11.55, 3.35, text="store today's decision", color=COLORS["muted"], rad=0.05)
    draw_arrow(ax, 10.05, 2.90, 9.75, 2.90, text="recompute historical\nreturns + Sharpe", color=COLORS["muted"])
    draw_arrow(ax, 9.75, 2.90, 13.20, 2.90, text="evaluate short/long strategy\nusing slope trend", color=COLORS["muted"], rad=-0.08)
    draw_arrow(ax, 14.25, 2.45, 14.25, 2.20, color=COLORS["muted"])
    draw_arrow(ax, 14.25, 1.78, 5.55, 6.95, text="reflected memory guidance", color=COLORS["muted"], rad=0.26)
    draw_arrow(ax, 13.00, 2.90, 3.60, 2.90, text="get_latest_long_short_memory()", color=COLORS["muted"], rad=0.0)
    draw_arrow(ax, 3.60, 2.90, 4.95, 6.95, text="next run context", color=COLORS["muted"], rad=0.18)

    ax.text(
        0.25,
        8.92,
        "Multi-Agent Consensus Pipeline",
        fontsize=22,
        fontweight="bold",
        color=COLORS["text"],
        ha="left",
        va="top",
    )
    # ax.text(0.25, 8.55, "Focus: N agents -> N reports -> consensus filtering", fontsize=13, color=COLORS["muted"], ha="left", va="top")
    ax.text(
        0.25,
        0.25,
        "Rule: if return/Sharpe slope > 0, strategy is improving; otherwise it is deteriorating or riskier. "
        "Reflection updates long/short memory for the next cycle.",
        fontsize=12,
        color=COLORS["muted"],
        ha="left",
    )

    fig.tight_layout(pad=1.1)
    fig.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def make_price_consensus_chart(output_path: Path):
    """Chart focused on price_scorer consensus threshold/outlier filtering."""
    fig, ax = plt.subplots(figsize=(15, 8), dpi=300)
    setup_canvas(fig, ax, (0, 13.5), (0, 7))

    draw_lane(ax, 0.2, 3.9, 13.1, 2.8, "Claim Mining + Consensus Scoring", COLORS["lane_top"])
    draw_lane(ax, 0.2, 0.35, 13.1, 3.25, "Filtering + Final Signal Generation", COLORS["lane_bottom"])

    draw_node(ax, 0.6, 5.1, 2.7, 1.0, "1) N agents -> N reports", fc=COLORS["input"], fontsize=13, bold=True)
    draw_node(ax, 3.8, 5.1, 2.2, 1.0, "2) Extract claims", fc=COLORS["neutral"], fontsize=13, bold=True)
    draw_node(ax, 6.4, 5.1, 2.2, 1.0, "3) Cluster signals", fc=COLORS["neutral"], fontsize=13, bold=True)
    draw_node(ax, 9.0, 5.1, 2.9, 1.0, "4) Consensus score", fc=COLORS["scorer"], fontsize=13, bold=True)

    draw_arrow(ax, 3.3, 5.6, 3.8, 5.6)
    draw_arrow(ax, 6.0, 5.6, 6.4, 5.6)
    draw_arrow(ax, 8.6, 5.6, 9.0, 5.6)

    draw_node(ax, 1.8, 2.5, 4.0, 0.95, "score >= 0.4 -> keep", fc=COLORS["high"], ec=COLORS["high_edge"], fontsize=13, bold=True)
    draw_node(ax, 6.35, 2.5, 5.1, 0.95, "remove outliers + aggregate", fc=COLORS["high"], ec=COLORS["high_edge"], fontsize=13, bold=True)
    draw_node(ax, 1.8, 1.1, 4.0, 0.95, "score < 0.4 -> drop", fc=COLORS["low"], ec=COLORS["low_edge"], fontsize=13, bold=True)
    draw_node(ax, 6.35, 1.1, 5.1, 0.95, "final_price_signal -> action + %", fc=COLORS["trader"], fontsize=13, bold=True)

    draw_arrow(ax, 10.9, 5.1, 3.8, 3.45, text="pass", color=COLORS["high_edge"], rad=0.02)
    draw_arrow(ax, 10.9, 5.1, 3.8, 2.05, text="fail", color=COLORS["low_edge"], rad=-0.04)
    draw_arrow(ax, 5.8, 2.97, 6.35, 2.97, color=COLORS["high_edge"])
    draw_arrow(ax, 11.45, 2.45, 11.45, 2.05)

    ax.text(0.25, 6.92, "Price Consensus Filter Logic", fontsize=21, fontweight="bold", color=COLORS["text"], va="top")
    ax.text(0.25, 6.55, "From tradingagents/agents/credibility_scorer/price_scorer.py", fontsize=13, color=COLORS["muted"], va="top")
    ax.text(0.25, 0.08, "Only sufficiently supported clusters pass; low-consensus clusters are dropped.", fontsize=12, color=COLORS["muted"])

    fig.tight_layout(pad=1.1)
    fig.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot multi-agent consensus flowcharts.")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("tools/plot_consensus/outputs"),
        help="Directory to save output figures.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    overview_png = args.outdir / "multiagent_consensus_flow_overview.png"
    overview_svg = args.outdir / "multiagent_consensus_flow_overview.svg"
    price_png = args.outdir / "price_consensus_filter_flow.png"
    price_svg = args.outdir / "price_consensus_filter_flow.svg"

    make_overview_chart(overview_png)
    make_overview_chart(overview_svg)
    make_price_consensus_chart(price_png)
    make_price_consensus_chart(price_svg)

    print("Saved:")
    print(f"- {overview_png}")
    print(f"- {overview_svg}")
    print(f"- {price_png}")
    print(f"- {price_svg}")


if __name__ == "__main__":
    main()
