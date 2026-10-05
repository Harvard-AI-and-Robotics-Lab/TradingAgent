import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# --- your nested dict from earlier ---
data = {
    "Market": {
        "Buy-and-Hold": {
            "AAPL": {"CR": -5.90, "ARR": -22.21, "SR": -1.43, "MDD": 13.25},
            "GOOGL": {"CR": 8.75, "ARR": 41.40, "SR": 1.03, "MDD": 14.30},
            "NVDA": {"CR": 88.78, "ARR": 1280.30, "SR": 1.73, "MDD": 8.70},
        }
    },
    "Rule-based": {
        "MACD": {
            "AAPL": {"CR": -5.84, "ARR": -14.90, "SR": -1.77, "MDD": 8.17},
            "GOOGL": {"CR": 6.92, "ARR": 19.65, "SR": 1.83, "MDD": 1.49},
            "NVDA": {"CR": 13.36, "ARR": 39.95, "SR": 1.31, "MDD": 7.41},
        },
        "KDJ_RSI": {
            "AAPL": {"CR": 4.55, "ARR": 16.59, "SR": 1.09, "MDD": 4.98},
            "GOOGL": {"CR": 6.56, "ARR": 24.53, "SR": 1.45, "MDD": 5.24},
            "NVDA": {"CR": 0.00, "ARR": 0.00, "SR": 0.00, "MDD": 0.00},
        },
        "ZMR": {
            "AAPL": {"CR": -5.97, "ARR": -17.63, "SR": -1.24, "MDD": 13.23},
            "GOOGL": {"CR": 9.34, "ARR": 32.48, "SR": 1.87, "MDD": 5.12},
            "NVDA": {"CR": 0.00, "ARR": 0.00, "SR": 0.00, "MDD": 0.00},
        },
        "SMA": {
            "AAPL": {"CR": -1.57, "ARR": -4.32, "SR": -0.81, "MDD": 3.51},
            "GOOGL": {"CR": 1.07, "ARR": 3.04, "SR": 0.72, "MDD": 1.23},
            "NVDA": {"CR": 0.00, "ARR": 0.00, "SR": 0.00, "MDD": 0.00},
        },
    },
}

metrics = ["CR", "ARR", "SR", "MDD"]

# --- flatten + compute averages across stocks for each method ---
rows = []
for category, models in data.items():
    for model, assets in models.items():
        vals = {m: np.mean([assets[a][m] for a in assets]) for m in metrics}
        rows.append({"Category": category, "Method": model, **vals})

df_avg = pd.DataFrame(rows).set_index("Method")
print(df_avg)

# --- plot: 2x2 panel for CR/ARR/SR/MDD ---
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.ravel()

plot_metrics = ["CR", "ARR", "SR", "MDD"]
titles = ["Avg Cumulative Return (CR) ↑", "Avg Annualized Return (ARR) ↑",
          "Avg Sharpe Ratio (SR) ↑", "Avg Max Drawdown (MDD) ↓"]

x = np.arange(len(df_avg.index))

for ax, m, t in zip(axes, plot_metrics, titles):
    ax.bar(x, df_avg[m].values)
    ax.set_title(t)
    ax.set_xticks(x)
    ax.set_xticklabels(df_avg.index, rotation=25, ha="right")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5)

fig.tight_layout()
plt.show()


########################################################
metrics = ["CR", "MDD"]

rows = []
for category, models in data.items():
    for model, assets in models.items():
        avg_cr = np.mean([assets[a]["CR"] for a in assets])
        avg_mdd = np.mean([assets[a]["MDD"] for a in assets])
        rows.append({
            "Category": category,
            "Method": model,
            "CR": avg_cr,
            "MDD": avg_mdd
        })

df = pd.DataFrame(rows).set_index("Method")
print(df)


########################################################
data2_cr = {
    'Agent': ['Grok-4', 'GPT-4o-mini', 'GPT-4o', 'GPT-5-mini', 'GPT-5', 'Grok-4\n(partial alloc.)', 'GPT-4o-mini\n(partial alloc.)', 'Gemini-2.5-lite\n(partial alloc.)', 'Claude-haiku-4-5\n(partial alloc.)', 'Human\n(partial alloc.)'],
    'AAPL': [-7.47, -0.20, -5.17, -6.66, -0.36, -4.31, 2.34, 1.29, -3.68, -3.85],
    'GOOGL': [9.08, 0.56, 8.39, 9.08, 5.03, 3.30, 6.15, 6.06, 1.94, 6.38],
    'NVDA': [87.29, 48.42, 89.82, 87.29, 83.20, 38.40, 15.64, 2.80, 7.46, 65.99],
}

data2_sr = {
    'Agent': ['Grok-4', 'GPT-4o-mini', 'GPT-4o', 'GPT-5-mini', 'GPT-5', 'Grok-4\n(partial alloc.)', 'GPT-4o-mini\n(partial alloc.)', 'Gemini-2.5-lite\n(partial alloc.)', 'Claude-haiku-4-5\n(partial alloc.)', 'Human\n(partial alloc.)'],
    'AAPL': [-1.54, -0.70, -1.04, -1.38, -2.05, -1.36, 1.44, 2.91, -5.32, -1.48],
    'GOOGL': [1.45, 0.28, 1.37, 1.45, 0.89, 0.88, 2.92, 4.31, 1.72, 1.08],
    'NVDA': [5.46, 3.96, 5.58, 5.46, 5.33, 5.35, 5.86, 0.84, 4.27, 5.02],
}

data2_mdd = {
    'Agent': ['Grok-4', 'GPT-4o-mini', 'GPT-4o', 'GPT-5-mini', 'GPT-5', 'Grok-4\n(partial alloc.)', 'GPT-4o-mini\n(partial alloc.)', 'Gemini-2.5-lite\n(partial alloc.)', 'Claude-haiku-4-5\n(partial alloc.)', 'Human\n(partial alloc.)'],
    'AAPL': [13.23, 4.40, 13.16, 13.19, 0.36, 9.52, 1.37, 0.65, 4.93, 8.15],
    'GOOGL': [14.36, 3.97, 14.26, 14.36, 15.52, 0.88, 3.20, 0.89, 1.82, 9.49],
    'NVDA': [8.68, 7.43, 8.69, 8.68, 7.41, 4.61, 1.78, 3.78, 0.81, 7.36]
}

########################################################
data3_cr = {
    'Agent': ["+ High consensus\n(Ours)", "+ Low consensus\n(Ours)", "TrustTrade (Ours)\n(Memory & Reflection)", "TrustTrade (Ours)"],
    'AAPL': [-5.03, -2.77, -1.08, -1.30],
    'GOOGL': [7.73, 2.90, 4.41, 5.52],
    'NVDA': [76.59, 65.97, 59.37, 79.81],
}

data3_sr = {
    'Agent': ["+ High consensus\n(Ours)", "+ Low consensus\n(Ours)", "TrustTrade (Ours)\n(Memory & Reflection)", "TrustTrade (Ours)"],
    'AAPL': [-2.31, -1.41, -0.45, -0.75],
    'GOOGL': [1.58, 0.71, 1.21, 1.20],
    'NVDA': [5.31, 5.82, 4.74, 5.40],
}

data3_mdd = {
    'Agent': ["+ High consensus\n(Ours)", "+ Low consensus\n(Ours)", "TrustTrade (Ours)\n(Memory & Reflection)", "TrustTrade (Ours)"],
    'AAPL': [7.12, 5.05, 5.56, 4.62],
    'GOOGL': [10.65, 11.19, 8.15, 11.02],
    'NVDA': [8.67, 5.40, 8.26, 8.68],
}

for key in data3_cr.keys():
   data2_cr[key] = data2_cr[key] + data3_cr[key]
   data2_sr[key] = data2_sr[key] + data3_sr[key]
   data2_mdd[key] = data2_mdd[key] + data3_mdd[key]

# ---- CR mean ----
df_cr = pd.DataFrame(data2_cr)
df_cr["CR_mean"] = df_cr[["AAPL", "GOOGL", "NVDA"]].mean(axis=1)

# ---- SR mean ----
df_sr = pd.DataFrame(data2_sr)
df_sr["SR_mean"] = df_sr[["AAPL", "GOOGL", "NVDA"]].mean(axis=1)

# ---- MDD mean ----
df_mdd = pd.DataFrame(data2_mdd)
df_mdd["MDD_mean"] = df_mdd[["AAPL", "GOOGL", "NVDA"]].mean(axis=1)

# ---- merge results ----
df_mean = pd.DataFrame({
    "Agent": df_cr["Agent"],
    "CR_mean": df_cr["CR_mean"],
    "SR_mean": df_sr["SR_mean"],
    "MDD_mean": df_mdd["MDD_mean"]
})

# df_mean: columns Agent, CR_mean, MDD_mean
# df: baseline table with CR, MDD (already mean across assets)
# aims: your ordered list
aims = [
    "Human\n(partial alloc.)",

    # Market / Rule-based baselines
    "Buy-and-Hold",
    "MACD",
    "KDJ",
    "ZMR",
    "SMA",

    # LLMs (full allocation)
    "Grok-4",
    "GPT-4o-mini",
    "GPT-4o",
    "GPT-5-mini",
    "GPT-5",

    # LLMs & Human (partial allocation)
    "Grok-4\n(partial alloc.)",
    "GPT-4o-mini\n(partial alloc.)",
    "Gemini-2.5-lite\n(partial alloc.)",
    "Claude-haiku-4-5\n(partial alloc.)",

    # "+ High consensus\n(Ours)",
    # "+ Low consensus\n(Ours)",
    "TrustTrade (Ours)\n(Memory & Reflection)",
    "TrustTrade (Ours)"
]

# ---- Build a unified plotting dataframe (Method, CR, MDD, Group) ----
records = []

baseline_names = ["Buy-and-Hold", "MACD", "KDJ", "ZMR", "SMA"]
full_llm_names = ["Grok-4", "GPT-4o-mini", "GPT-4o", "GPT-5-mini", "GPT-5"]
ours_names = [
    "+ High consensus\n(Ours)", "+ Low consensus\n(Ours)",
    "TrustTrade (Ours)\n(Memory & Reflection)",
    "TrustTrade (Ours)"
]

for method in aims:
    if method in df.index:
        cr, mdd = float(df.loc[method, "CR"]), float(df.loc[method, "MDD"])
    elif method == "KDJ":
        cr, mdd = float(df.loc["KDJ_RSI", "CR"]), float(df.loc["KDJ_RSI", "MDD"])
        method = "KDJ"  # keep label
    else:
        row = df_mean[df_mean["Agent"] == method]
        if row.empty:
            continue
        cr, mdd = float(row["CR_mean"].values[0]), float(row["MDD_mean"].values[0])

    if method in ours_names:
        group = "Ours"
    elif "Human" in method:
        group = "Human"
    elif method in baseline_names:
        group = "Baselines"
    elif method in full_llm_names:
        group = "LLM Full Alloc"
    elif "partial" in method:
        group = "LLM Partial Alloc"
    else:
        raise ValueError(f"Unknown method: {method}")

    records.append((method, cr, mdd, group))

# --- Pareto frontier: maximize CR, minimize MDD ---
pts = [(m, cr, mdd) for (m, cr, mdd, g) in records]
pts_sorted = sorted(pts, key=lambda t: t[2])  # sort by MDD asc
frontier = []
best_cr = -1e18
for m, cr, mdd in pts_sorted:
    if cr > best_cr:
        frontier.append((m, cr, mdd))
        best_cr = cr
print(records)



# ---- Plot ----
fig, ax = plt.subplots(figsize=(5.7, 6.5))

# scatter (no manual colors required, but group separation helps contribution)
group_to_marker = {
    "Baselines": ["<", "black"],
    "LLM Full Alloc": ["o", "#4DBBD5"],
    "LLM Partial Alloc": ["s", "#E64B35"],
    "Human": "*",
    "Ours": ["X", "limegreen"],
}

group_to_marker_size = {
    "Human": 600,
    "Baselines": 80,
    "LLM Full Alloc": 80,
    "LLM Partial Alloc": 80,
    "Ours": 250,
}

# for method, cr, mdd, group in records:
#     ax.scatter(mdd, cr, s=130 if group in ["Ours","Human"] else 100,
#                marker=group_to_marker.get(group, "o"))
    # ax.text(mdd + 0.15, cr, method, fontsize=9)

label_offsets = {
    "Buy-and-Hold": (+0.3, 0.7),
    "SMA": (-0.25, -2.2),
    "KDJ": (0.3, -0.5),
    "MACD": (0.3, -0.5),
    "ZMR": (0.3, -0.5),
    "Claude-haiku-4-5\n(partial alloc.)": (0.1, -3.0),
    "Gemini-2.5-lite\n(partial alloc.)": (-2.1, 0.9),
    "GPT-4o-mini\n(partial alloc.)": (-1.0, 2.5),
    "Grok-4\n(partial alloc.)": (0.4, -1.5),
    "GPT-4o": (-0.9, 1.2),
    "GPT-4o-mini": (-2.3, 1.5),
    "GPT-5-mini": (-1.4, -2.2),
    "GPT-5": (-0.5, 1.5),
    "Grok-4": (0.6, -1.0),
    "+ High consensus\n(Ours)": (0.3, -2.0),
    "+ Low consensus\n(Ours)": (-2.0, 2.0),
    "TrustTrade (Ours)": (-5.2, -1.4),
    "TrustTrade (Ours)\n(Memory & Reflection)": (-6.5, -0.25),
    "Human\n(partial alloc.)": (0.6, -3.2),
}

for method, cr, mdd, group in records:
    if group == "Human":
        markerfacecolor = "#ffe388"
        markeredgecolor = "black"
        ax.scatter(mdd, cr, s=group_to_marker_size.get(group, 80),
                   marker=group_to_marker.get(group, "o"),
                   facecolor=markerfacecolor,
                   edgecolor=markeredgecolor,
                   linewidth=1.5,
                   alpha=1.0)
    else:
        marker = group_to_marker.get(group, "o")
        if isinstance(marker, list):
            marker, facecolor = marker
        else:
            facecolor = "black"
        if group == "Ours":
            markeredgecolor = "black"
        else:
            markeredgecolor = facecolor
        ax.scatter(mdd, cr, s=group_to_marker_size.get(group, 80),
                   marker=marker, facecolor=facecolor, edgecolor=markeredgecolor)

    dx, dy = label_offsets.get(method, (0.15, 0.0))
    ax.annotate(
        method,
        xy=(mdd, cr),
        xytext=(mdd + dx, cr + dy),
        textcoords="data",
        arrowprops=dict(arrowstyle="-", lw=0.5, alpha=0.6),
        fontsize=9,
        fontweight="bold"
    )


##############
# frontier line
fx = [mdd for (_, _, mdd) in frontier]
fy = [cr  for (_, cr, _) in frontier]
ax.plot(fx, fy, linewidth=1.25, color="#3776ab") # cyan

ax.set_xlabel("Average Max Drawdown % (MDD) ↓", fontsize=10, fontweight="bold")
ax.set_ylabel("Average Cumulative Return % (CR) ↑", fontsize=10, fontweight="bold")
ax.set_title("Risk–Return Trade-off (Averaged Across All Stocks)", fontsize=11, fontweight="bold")
ax.grid(True, linestyle="--", linewidth=0.5, color="gray")
# remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


########################################################
# x: MDD, y: CR (1D arrays)
x = np.array([mdd for (_, cr, mdd, _) in records])
y = np.array([cr  for (_, cr, mdd, _) in records])

# Fit y = a x + b
a, b = np.polyfit(x, y, 1)

# Plot regression line
xx = np.linspace(x.min(), x.max(), 200)
yy = a * xx + b
ax.plot(xx, yy, linestyle="--", linewidth=1.25, color="#3776ab")

# Optional: annotate
# ax.text(xx.max()-2.5, a*(xx.max()-2.5)+b, f"y = {a:.2f}x + {b:.2f}", fontsize=10)



########################################################
# anchor on human
human_row = df_mean[df_mean["Agent"] == "Human\n(partial alloc.)"]
cx = human_row["MDD_mean"].values[0]
cy = human_row["CR_mean"].values[0]

# ellipse parameters
width  = 2.47 # 3.5    # MDD direction (x)
height = 6.90 # 12.0   # CR direction (y)
# angle  = -6    # tilt toward upper-left (important!)
theta_rad = np.arctan(a)
theta_deg = np.degrees(a) - 180
angle = theta_deg  # or tweak sign depending on your axis direction
print("slope a =", a, theta_rad, "rotation angle =", theta_deg, "degrees")

# draw layered ellipses (outer → inner)
n_layers = 25
for i in range(n_layers):
    alpha = 0.05 + 0.02 * (1 - i / n_layers)
    scale = 1 + 0.05 * i

    e = patches.Ellipse(
        (cx, cy),
        width=width * scale,
        height=height * scale,
        angle=angle,
        facecolor="#f4d03f",   # soft yellow
        edgecolor=None,
        alpha=alpha,
        zorder=0
    )
    ax.add_patch(e)

# label
ax.text(
    cx + 0.2, cy - 8.5,
    "Human-aligned\ntrade-off region",
    fontsize=13,
    fontweight="bold",
    color="#e4af00",  # "#f4d03f"
)

# y ticks: 0, 5, 10, 15, 20, 25, 30
ax.set_yticks([-3, 0, 10, 20, 30, 33])
ax.set_xticks([-1, 0, 4, 8, 12, 15])


# legend
# ---- Scatter legend handles ----
scatter_handles = [
    Line2D([0], [0],
           marker="*", linestyle="None",
           markerfacecolor="#ffe388",
           markeredgecolor="black",
           markersize=12,
           label="Human (Partial alloc.)"),

    Line2D([0], [0],
           marker="<", linestyle="None",
           markerfacecolor="black", markeredgecolor="black",
           markersize=7,
           label="Rule-based Baselines"),

    Line2D([0], [0],
           marker="o", linestyle="None",
           markerfacecolor="#4DBBD5", markeredgecolor="#4DBBD5",
           markersize=7,
           label="LLMs (Full alloc.)"),

    Line2D([0], [0],
           marker="s", linestyle="None",
           markerfacecolor="#E64B35", markeredgecolor="#E64B35",
           markersize=7,
           label="LLMs (Partial alloc.)"),

    Line2D([0], [0],
           marker="X", linestyle="None",
           markerfacecolor="limegreen", markeredgecolor="black",
           markersize=9,
           label="LLMs (Ours)"),
]

# ---- Line legend handles ----
line_handles = [
    Line2D([0], [0],
           linestyle="-", color="#3776ab", linewidth=1.5,
           label="Pareto frontier"),

    Line2D([0], [0],
           linestyle="--", color="#3776ab", linewidth=1.5,
           label="Linear trade-off fit"),
]

# ---- Combine legends ----
handles = scatter_handles + line_handles

ax.legend(
    handles=handles,
    loc="lower right",
    frameon=True,
    framealpha=0.95,
    edgecolor="gray",
    fontsize=9
)

plt.tight_layout()
plt.savefig('risk_return_tradeoff.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Figure saved as 'risk_return_tradeoff.png'")
plt.close(fig)


########################################################
# NEW FIGURE: Sharpe Ratio vs Cumulative Return
########################################################

# ---- Build records: (method, cr, sr, group) ----
records_sr = []

# rename map: df_avg uses "KDJ_RSI", aims uses "KDJ"
baseline_sr_map = {
    "Buy-and-Hold": "Buy-and-Hold",
    "MACD": "MACD",
    "KDJ": "KDJ_RSI",
    "ZMR": "ZMR",
    "SMA": "SMA",
}

for method in aims:
    # Baselines: pull from df_avg (which has SR)
    idx_key = baseline_sr_map.get(method)
    if idx_key and idx_key in df_avg.index:
        cr = float(df_avg.loc[idx_key, "CR"])
        sr = float(df_avg.loc[idx_key, "SR"])
    else:
        row = df_mean[df_mean["Agent"] == method]
        if row.empty:
            continue
        cr = float(row["CR_mean"].values[0])
        sr = float(row["SR_mean"].values[0])

    if method in ours_names:
        group = "Ours"
    elif "Human" in method:
        group = "Human"
    elif method in baseline_names:
        group = "Baselines"
    elif method in full_llm_names:
        group = "LLM Full Alloc"
    elif "partial" in method:
        group = "LLM Partial Alloc"
    else:
        raise ValueError(f"Unknown method: {method}")

    records_sr.append((method, cr, sr, group))

# --- Pareto frontier: maximize both SR and CR ---
pts_sorted_sr = sorted(records_sr, key=lambda t: t[2])  # sort by SR asc
frontier_sr = []
best_cr_sr = -1e18
for m, cr, sr, _ in pts_sorted_sr:
    if cr > best_cr_sr:
        frontier_sr.append((m, cr, sr))
        best_cr_sr = cr

# ---- Label offsets (hand-tuned for SR x-axis) ----
label_offsets_sr = {
    "Buy-and-Hold": (+0.0, +1.5),
    "SMA": (0.15, 0.0),
    "KDJ": (0.15, 0.0),
    "MACD": (0.15, 0.0),
    "ZMR": (0.15, 0.0),
    "Claude-haiku-4-5\n(partial alloc.)": (+0.5, -1.5),
    "Gemini-2.5-lite\n(partial alloc.)": (-1.1, +1.0),
    "GPT-4o-mini\n(partial alloc.)": (-0.45, +1.4),
    "Grok-4\n(partial alloc.)": (+0.2, -0.5),
    "GPT-4o": (+0.2, +0.15),
    "GPT-4o-mini": (-0.5, +0.5),
    "GPT-5-mini": (+0.3, -0.5),
    "GPT-5": (-0.2, -1.2),
    "Grok-4": (+0.3, -0.9),
    "TrustTrade (Ours)": (+0.3, -1.0),
    "TrustTrade (Ours)\n(Memory & Reflection)": (0.25, -1.3),
    "Human\n(partial alloc.)": (+0.3, -0.4),
}

# ---- Plot ----
fig2, ax2 = plt.subplots(figsize=(5.7, 6.5))

for method, cr, sr, group in records_sr:
    if group == "Human":
        ax2.scatter(sr, cr, s=group_to_marker_size.get(group, 80),
                    marker=group_to_marker.get(group, "o"),
                    facecolor="#ffe388", edgecolor="black",
                    linewidth=1.5, alpha=1.0)
    else:
        marker = group_to_marker.get(group, "o")
        if isinstance(marker, list):
            marker, facecolor = marker
        else:
            facecolor = "black"
        markeredgecolor = "black" if group == "Ours" else facecolor
        ax2.scatter(sr, cr, s=group_to_marker_size.get(group, 80),
                    marker=marker, facecolor=facecolor,
                    edgecolor=markeredgecolor)

    dx, dy = label_offsets_sr.get(method, (0.1, 0.0))
    ax2.annotate(
        method,
        xy=(sr, cr),
        xytext=(sr + dx, cr + dy),
        textcoords="data",
        arrowprops=dict(arrowstyle="-", lw=0.5, alpha=0.6),
        fontsize=9,
        fontweight="bold",
    )

# Pareto frontier line
fx2 = [sr for (_, _, sr) in frontier_sr]
fy2 = [cr for (_, cr, _) in frontier_sr]
ax2.plot(fx2, fy2, linewidth=1.25, color="#3776ab")

# Linear fit
x_sr = np.array([sr for (_, _, sr, _) in records_sr])
y_cr = np.array([cr for (_, cr, _, _) in records_sr])
a2, b2 = np.polyfit(x_sr, y_cr, 1)
xx2 = np.linspace(x_sr.min() - 0.3, x_sr.max() + 0.3, 200)
yy2 = a2 * xx2 + b2
ax2.plot(xx2, yy2, linestyle="--", linewidth=1.25, color="#3776ab")

# Human-aligned ellipse
human_row_sr = df_mean[df_mean["Agent"] == "Human\n(partial alloc.)"]
cx2 = human_row_sr["SR_mean"].values[0]
cy2 = human_row_sr["CR_mean"].values[0]

width2  = 0.5
height2 = 6.50
theta_rad2 = np.arctan(a2)
theta_deg2 = np.degrees(a2)
angle2 = -3

n_layers2 = 25
for i in range(n_layers2):
    alpha2 = 0.05 + 0.02 * (1 - i / n_layers2)
    scale2 = 1 + 0.05 * i
    e2 = patches.Ellipse(
        (cx2, cy2),
        width=width2 * scale2,
        height=height2 * scale2,
        angle=angle2,
        facecolor="#f4d03f",
        edgecolor=None,
        alpha=alpha2,
        zorder=0,
    )
    ax2.add_patch(e2)

ax2.text(
    cx2 + 0.15, cy2 - 7.0,
    "Human-aligned\ntrade-off region",
    fontsize=13, fontweight="bold", color="#e4af00",
)

ax2.set_xlabel("Average Sharpe Ratio (SR) % \u2191", fontsize=10, fontweight="bold")
ax2.set_ylabel("Average Cumulative Return % (CR) \u2191", fontsize=10, fontweight="bold")
ax2.set_title("SR\u2013Return Trade-off (Averaged Across All Stocks)",
              fontsize=11, fontweight="bold")
ax2.grid(True, linestyle="--", linewidth=0.5, color="gray")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

ax2.legend(
    handles=scatter_handles + line_handles,
    loc="lower right",
    frameon=True, framealpha=0.95, edgecolor="gray", fontsize=9,
)
ax2.x_ticks = [-0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
ax2.set_xticks(ax2.x_ticks)

plt.tight_layout()
out_sr = os.path.join(os.path.dirname(__file__), "sr_return_tradeoff.png")
plt.savefig(out_sr, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Figure saved as '{out_sr}'")
plt.close(fig2)