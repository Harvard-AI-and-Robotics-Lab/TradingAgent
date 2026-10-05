import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# -------------------------
# 0) Prepare dataframe (order + averages)
# -------------------------
data = {
    'Agent': ['Grok-4', 'GPT-4o-mini', 'GPT-4o', 'GPT-5-mini', 'GPT-5',
              'Grok-4\n(partial alloc.)', 'GPT-4o-mini\n(partial alloc.)',
              'Gemini-2.5-lite\n(partial alloc.)', 'Claude-haiku-4-5\n(partial alloc.)',
              'Human\n(partial alloc.)'],
    'AAPL': [-7.47, -0.20, -5.17, -6.66, -0.36, -4.31, 2.34, 1.29, -3.68, -3.65],
    'GOOG': [9.08, 0.56, 8.39, 9.08, 5.03, 3.30, 6.15, 6.06, 1.94, 6.86],
    'NVDA': [87.29, 48.42, 89.82, 87.29, 83.20, 38.40, 15.64, 2.80, 7.46, 71.29],
}
df = pd.DataFrame(data)

order = [
    'Human\n(partial alloc.)', 'GPT-5', 'GPT-5-mini', 'GPT-4o', 'GPT-4o-mini', 'Grok-4',
    'GPT-4o-mini\n(partial alloc.)', 'Grok-4\n(partial alloc.)',
    'Gemini-2.5-lite\n(partial alloc.)', 'Claude-haiku-4-5\n(partial alloc.)'
]
df = df.set_index("Agent").loc[order].reset_index()
df["Average"] = df[["AAPL", "GOOG", "NVDA"]].mean(axis=1)

# -------------------------
# 1) API cost (normalize per stock-day)
# -------------------------
data_api = {
    "Grok-4": 1.83,
    "GPT-4o-mini": 10,
    "GPT-4o": 166.67,
    "GPT-5-mini": 30,
    "GPT-5": 150.00,
    "Grok-4\n(partial alloc.)": 1.83,
    "GPT-4o-mini\n(partial alloc.)": 10,
    "Gemini-2.5-lite\n(partial alloc.)": 3.66,
    "Claude-haiku-4-5\n(partial alloc.)": 100,
    "Human\n(partial alloc.)": 0,
}
api = np.array([data_api[a] for a in df["Agent"]], dtype=float) / (3 * 61)

# -------------------------
# 2) Visual encodings
#    - color by family (simple + consistent)
#    - hatch by stock
# -------------------------
def family_color(agent: str) -> str:
    if "Human" in agent:
        return "#ffe388"        # human
    if "(partial alloc.)" in agent:
        return "#F39B7F"        # partial alloc family
    return "#4DBBD5"            # full alloc family

agent_colors = [family_color(a) for a in df["Agent"]]

stock_hatch = {"AAPL": ".", "GOOG": "*", "NVDA": "o"}

# -------------------------
# 3) Plot
# -------------------------
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]

fig = plt.figure(figsize=(20, 6), constrained_layout=True)
gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 1.0])

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])

x = np.arange(len(df))
w = 0.22

# --- Top: grouped bars per stock ---
for j, stock in enumerate(["AAPL", "GOOG", "NVDA"]):
    ax1.bar(
        x + (j - 1) * w,
        df[stock].values,
        width=w,
        color=agent_colors,
        hatch=stock_hatch[stock],
        edgecolor="black",
        linewidth=0.7,
        alpha=0.85,
        label=stock,
        zorder=2
    )

    # (optional) only label NVDA bars to reduce clutter
    for i, v in enumerate(df[stock].values):
        ax1.text(i + (j - 1) * w, v + (1.5 if v >= 0 else -4.0), f"{v:.1f}",
                ha="center", va="bottom" if v >= 0 else "top",
                fontsize=9, fontweight="bold", alpha=0.9)

ax1.axhline(0, color="black", linewidth=0.9, alpha=0.6, zorder=1)
ax1.set_ylabel("Cumulative Return (CR) % ↑", fontsize=13, fontweight="bold")
ax1.set_title("Cumulative Return (CR) by Stock (2024 Q1)", fontsize=15, pad=13, fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels(df["Agent"], fontsize=12, rotation=15, ha="right", fontweight="bold")
ax1.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35, zorder=0)
ax1.set_axisbelow(True)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# --- Bottom: average bars ---
bars = ax2.bar(
    x, df["Average"].values,
    width=0.58,
    color=agent_colors,
    edgecolor="black",
    linewidth=0.7,
    alpha=0.85,
    zorder=2
)

# highlight best avg subtly (no giant arrows)
best_pos = int(np.argmax(df["Average"].values))
bars[best_pos].set_linewidth(2.2)
bars[best_pos].set_edgecolor("gold")

# average value labels (small + consistent)
for i, v in enumerate(df["Average"].values):
    ax2.text(i, v + 0.8, f"{v:.1f}%",
             ha="center", va="bottom",
             fontsize=10, fontweight="bold")

ax2.axhline(0, color="black", linewidth=0.9, alpha=0.6)
ax2.set_ylabel("Average CR % ↑", fontsize=13, fontweight="bold")
ax2.set_title("Average CR Across Stocks", fontsize=14, pad=13, fontweight="bold")
ax2.set_xticks(x)
ax2.set_xticklabels(df["Agent"], fontsize=12, rotation=15, ha="right", fontweight="bold")
ax2.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
ax2.set_axisbelow(True)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

# --- API cost overlay (clean line+dot) on secondary axis ---
ax2b = ax2.twinx()
ax2b.plot(x, api, linestyle="-", linewidth=3, marker="o",
          markersize=12, color="darkred", alpha=0.85, zorder=3)
ax2b.set_ylabel("API cost ($)\n(per stock-day)", fontsize=12, fontweight="bold", color="darkred")
ax2b.tick_params(axis="y", labelsize=10)
ax2b.grid(False)
ax2b.spines["top"].set_visible(False)

# If cost range is very skewed, uncomment log scale:
# ax2b.set_yscale("log")

# -------------------------
# 4) One clean legend: stocks (hatches) + API line
# -------------------------
stock_handles = [
    Patch(facecolor="white", edgecolor="black", hatch=stock_hatch["AAPL"], label="AAPL"),
    Patch(facecolor="white", edgecolor="black", hatch=stock_hatch["GOOG"], label="GOOG"),
    Patch(facecolor="white", edgecolor="black", hatch=stock_hatch["NVDA"], label="NVDA"),
]
api_handle = Line2D([0], [0], color="darkred", marker="o", linewidth=1.2,
                    markersize=5, label="API cost (right axis)")

ax1.legend(handles=stock_handles + [api_handle],
           loc="upper right", frameon=False, fontsize=12)

plt.savefig("performance_with_price_clean.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print("Saved: performance_with_price_clean.png")


########################################################################################
# MDD
# -------------------------
# 0) Prepare dataframe (order + averages)
# -------------------------
data = {
    'Agent': ['Grok-4', 'GPT-4o-mini', 'GPT-4o', 'GPT-5-mini', 'GPT-5',
              'Grok-4\n(partial alloc.)', 'GPT-4o-mini\n(partial alloc.)',
              'Gemini-2.5-lite\n(partial alloc.)', 'Claude-haiku-4-5\n(partial alloc.)',
              'Human\n(partial alloc.)'],
    'AAPL': [13.23, 4.40, 13.16, 13.19, 0.36, 9.52, 1.37, 0.65, 4.93, 8.07],
    'GOOG': [14.36, 3.97, 14.26, 14.36, 15.52, 0.88, 3.20, 0.89, 1.82, 9.29],
    'NVDA': [8.68, 7.43, 8.69, 8.68, 7.41, 4.61, 1.78, 3.78, 0.81, 7.35]
}
df = pd.DataFrame(data)

order = [
    'Human\n(partial alloc.)', 'GPT-5', 'GPT-5-mini', 'GPT-4o', 'GPT-4o-mini', 'Grok-4',
    'GPT-4o-mini\n(partial alloc.)', 'Grok-4\n(partial alloc.)',
    'Gemini-2.5-lite\n(partial alloc.)', 'Claude-haiku-4-5\n(partial alloc.)'
]
df = df.set_index("Agent").loc[order].reset_index()
df["Average"] = df[["AAPL", "GOOG", "NVDA"]].mean(axis=1)

# -------------------------
# 1) API cost (normalize per stock-day)
# -------------------------
data_api = {
    "Grok-4": 1.83,
    "GPT-4o-mini": 10,
    "GPT-4o": 166.67,
    "GPT-5-mini": 30,
    "GPT-5": 150.00,
    "Grok-4\n(partial alloc.)": 1.83,
    "GPT-4o-mini\n(partial alloc.)": 10,
    "Gemini-2.5-lite\n(partial alloc.)": 3.66,
    "Claude-haiku-4-5\n(partial alloc.)": 100,
    "Human\n(partial alloc.)": 0,
}
api = np.array([data_api[a] for a in df["Agent"]], dtype=float) / (3 * 61)

# -------------------------
# 2) Visual encodings
#    - color by family (simple + consistent)
#    - hatch by stock
# -------------------------
def family_color(agent: str) -> str:
    if "Human" in agent:
        return "#ffe388"        # human
    if "(partial alloc.)" in agent:
        return "#F39B7F"        # partial alloc family
    return "#4DBBD5"            # full alloc family

agent_colors = [family_color(a) for a in df["Agent"]]

stock_hatch = {"AAPL": ".", "GOOG": "*", "NVDA": "o"}

# -------------------------
# 3) Plot
# -------------------------
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]

fig = plt.figure(figsize=(20, 6), constrained_layout=True)
gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 1.0])

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])

x = np.arange(len(df))
w = 0.22

# --- Top: grouped bars per stock ---
for j, stock in enumerate(["AAPL", "GOOG", "NVDA"]):
    ax1.bar(
        x + (j - 1) * w,
        df[stock].values,
        width=w,
        color=agent_colors,
        hatch=stock_hatch[stock],
        edgecolor="black",
        linewidth=0.7,
        alpha=0.85,
        label=stock,
        zorder=2
    )

    # (optional) only label NVDA bars to reduce clutter
    for i, v in enumerate(df[stock].values):
        ax1.text(i + (j - 1) * w, v + (1.5 if v >= 0 else -4.0), f"{v:.1f}",
                ha="center", va="bottom" if v >= 0 else "top",
                fontsize=9, fontweight="bold", alpha=0.9)

ax1.axhline(0, color="black", linewidth=0.9, alpha=0.6, zorder=1)
ax1.set_ylabel("Max Drawdown (MDD) % ↑", fontsize=13, fontweight="bold")
ax1.set_title("Max Drawdown (MDD) by Stock (2024 Q1)", fontsize=15, pad=13, fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels(df["Agent"], fontsize=12, rotation=15, ha="right", fontweight="bold")
ax1.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35, zorder=0)
ax1.set_axisbelow(True)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# --- Bottom: average bars ---
bars = ax2.bar(
    x, df["Average"].values,
    width=0.58,
    color=agent_colors,
    edgecolor="black",
    linewidth=0.7,
    alpha=0.85,
    zorder=2
)

# highlight best avg subtly (no giant arrows)
best_pos = int(np.argmax(df["Average"].values))
bars[best_pos].set_linewidth(2.2)
bars[best_pos].set_edgecolor("gold")

# average value labels (small + consistent)
for i, v in enumerate(df["Average"].values):
    ax2.text(i, v + 0.8, f"{v:.1f}%",
             ha="center", va="bottom",
             fontsize=10, fontweight="bold")

ax2.axhline(0, color="black", linewidth=0.9, alpha=0.6)
ax2.set_ylabel("Average MDD % ↑", fontsize=13, fontweight="bold")
ax2.set_title("Average MDD Across Stocks", fontsize=14, pad=13, fontweight="bold")
ax2.set_xticks(x)
ax2.set_xticklabels(df["Agent"], fontsize=12, rotation=15, ha="right", fontweight="bold")
ax2.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
ax2.set_axisbelow(True)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

# --- API cost overlay (clean line+dot) on secondary axis ---
ax2b = ax2.twinx()
ax2b.plot(x, api, linestyle="-", linewidth=3, marker="o",
          markersize=12, color="darkred", alpha=0.85, zorder=3)
ax2b.set_ylabel("API cost ($)\n(per stock-day)", fontsize=12, fontweight="bold", color="darkred")
ax2b.tick_params(axis="y", labelsize=10)
ax2b.grid(False)
ax2b.spines["top"].set_visible(False)

# If cost range is very skewed, uncomment log scale:
# ax2b.set_yscale("log")

# -------------------------
# 4) One clean legend: stocks (hatches) + API line
# -------------------------
stock_handles = [
    Patch(facecolor="white", edgecolor="black", hatch=stock_hatch["AAPL"], label="AAPL"),
    Patch(facecolor="white", edgecolor="black", hatch=stock_hatch["GOOG"], label="GOOG"),
    Patch(facecolor="white", edgecolor="black", hatch=stock_hatch["NVDA"], label="NVDA"),
]
api_handle = Line2D([0], [0], color="darkred", marker="o", linewidth=1.2,
                    markersize=5, label="API cost (right axis)")

ax1.legend(handles=stock_handles + [api_handle],
           loc="upper right", frameon=False, fontsize=12)

plt.savefig("performance_with_price_clean_mdd.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print("Saved: performance_with_price_clean_mdd.png")