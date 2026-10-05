import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Setup figure with two subplots
fig = plt.figure(figsize=(27, 9))
gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.4)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])

# Each AGENT gets its own color (matching plot_figs_v2 palette)
agent_colors = ["#ffe388", "#4DBBD5", "#4DBBD5", "#4DBBD5", "#4DBBD5", "#4DBBD5", "#F39B7F" , "#F39B7F" , "#F39B7F" , "#F39B7F"]

# Each STOCK gets its own hatch pattern
stock_patterns = {
    'AAPL': '.',     # Diagonal lines
    'GOOGL': '*',    # Stars/dots
    'NVDA': 'o'      # No pattern (solid)
}

### Data extracted from your Overleaf Table 1 (Online Agents)
data = {
    'Agent': ['Grok-4', 'GPT-4o-mini', 'GPT-4o', 'GPT-5-mini', 'GPT-5', 'Grok-4\n(partial alloc.)', 'GPT-4o-mini\n(partial alloc.)', 'Gemini-2.5-lite\n(partial alloc.)', 'Claude-haiku-4-5\n(partial alloc.)', 'Human\n(partial alloc.)'],
    'AAPL': [-7.47, -0.20, -5.17, -6.66, -0.36, -4.31, 2.34, 1.29, -3.68, -4.46],
    'GOOGL': [9.08, 0.56, 8.39, 9.08, 5.03, 3.30, 6.15, 6.06, 1.94, 1.98],
    'NVDA': [87.29, 48.42, 89.82, 87.29, 83.20, 38.40, 15.64, 2.80, 7.46, 71.13],
}

# API prices: in
data_api = {
    "Grok-4": 1.83,
    "GPT-4o-mini": 10, # 0.15 + 0.60, # per 1M tokens, 10 / 1k calls
    "GPT-4o": 166.67, # 2.5 + 10.0,
    "GPT-5-mini": 30, # 0.25 + 2.0,
    "GPT-5": 150.00, # 1.25 + 10.0,
    "Grok-4\n(partial alloc.)": 1.83,
    "GPT-4o-mini\n(partial alloc.)": 10,
    "Gemini-2.5-lite\n(partial alloc.)": 3.66,
    "Claude-haiku-4-5\n(partial alloc.)": 100,
    "Human\n(partial alloc.)": 0,
}

# price / 3 * 61
for key in data_api.keys():
    data_api[key] = data_api[key] / (3 * 61) # 3 stocks, 61 trading days

order = ['Human\n(partial alloc.)', 'GPT-5', 'GPT-5-mini', 'GPT-4o', 'GPT-4o-mini', 'Grok-4', 'GPT-4o-mini\n(partial alloc.)', 'Grok-4\n(partial alloc.)', 'Gemini-2.5-lite\n(partial alloc.)', 'Claude-haiku-4-5\n(partial alloc.)']
index = [data['Agent'].index(agent) for agent in order]
for key in data.keys():
    if key == 'Agent':
        data['Agent'] = order
        continue
    data[key] = [data[key][i] for i in index]

df = pd.DataFrame(data)
# Calculate average performance per agent
df['Average'] = df[['AAPL', 'GOOGL', 'NVDA']].mean(axis=1)

# Main plot - grouped by agent
x = np.arange(len(df['Agent']))
width = 0.2

# Set style to match plot_figs_v2
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

# Create bars - each agent uses the same color, stocks distinguished by patterns
bars1 = ax1.bar(x - width, df['AAPL'], width, label='AAPL',
                color=[agent_colors[i] for i in range(len(df))],
                hatch=stock_patterns['AAPL'],
                edgecolor='black', linewidth=0.8, alpha=0.8)

bars2 = ax1.bar(x, df['GOOGL'], width, label='GOOGL',
                color=[agent_colors[i] for i in range(len(df))],
                hatch=stock_patterns['GOOGL'],
                edgecolor='black', linewidth=0.8, alpha=0.8)

bars3 = ax1.bar(x + width, df['NVDA'], width, label='NVDA',
                color=[agent_colors[i] for i in range(len(df))],
                hatch=stock_patterns['NVDA'],
                edgecolor='black', linewidth=0.8, alpha=0.8)

# Add value labels on bars for NVDA (highest values)
for j, k in enumerate(['AAPL', 'GOOGL', 'NVDA']):
    for i, v in enumerate(df[k]):
        v_ = v + 2 if v > 0 else v - 8
        ax1.text(i + (j-1) *width, v_, f'{v:.1f}', ha='center', va='bottom',
                 fontsize=10, fontweight='bold')

# Formatting main plot
ax1.set_ylabel('Cumulative Return (CR) % $\\uparrow$', fontsize=14, fontweight='normal')
# ax1.set_title('Performance Comparison of LLM Agents Across Stocks (2024 Q1)',
            #   fontsize=16, fontweight='normal', pad=20)
ax1.set_title('Cumulative Return (CR) % $\\uparrow$',
              fontsize=16, fontweight='normal', pad=20)
ax1.set_xticks(x)
ax1.set_xticklabels(df['Agent'], fontsize=14, fontweight='bold')
ax1.set_yticks([-20, 0, 20, 40, 60, 80, 100])
# ax1.legend(fontsize=14, loc='upper left', frameon=False)
ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
ax1.set_axisbelow(True)
ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.spines['left'].set_linewidth(1.2)
ax1.spines['bottom'].set_linewidth(1.2)
ax1.tick_params(axis='both', which='major', labelsize=13)

# Add vertical separator lines between agents for clarity
for i in range(len(x) - 1):
    ax1.axvline(x=i + 0.5, color='gray', linestyle=':', linewidth=1, alpha=0.3)

# Second plot - Average performance per agent
bars_avg = ax2.bar(x, df['Average'], width=0.6,
                   color=agent_colors, edgecolor='black', linewidth=0.8, alpha=0.8)

# Highlight best and worst performing agents with thicker borders
best_idx = df['Average'].idxmax()
worst_idx = df['Average'].idxmin()
bars_avg[best_idx].set_edgecolor('gold')  # Dark blue for best
bars_avg[best_idx].set_linewidth(2.5)
bars_avg[worst_idx].set_edgecolor('gold')  # Red for worst
bars_avg[worst_idx].set_linewidth(2.5)

# Add value labels on average bars
for i, v in enumerate(df['Average']):
    ax2.text(i, v + 1.5, f'{v:.1f}%', ha='center', va='bottom',
             fontsize=10, fontweight='bold')

# Add annotations for best/worst
ax2.annotate('Best Overall', xy=(best_idx, df['Average'].iloc[best_idx]),
            xytext=(best_idx, df['Average'].iloc[best_idx] - 10),
            ha='center', fontsize=10, fontweight='bold', color='gold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.3),
            arrowprops=dict(arrowstyle='->', color='black', lw=2))

ax2.annotate('Worst Overall', xy=(worst_idx, df['Average'].iloc[worst_idx]),
            xytext=(worst_idx, df['Average'].iloc[worst_idx] - 10),
            ha='center', fontsize=10, fontweight='bold', color='gold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.3),
            arrowprops=dict(arrowstyle='->', color='black', lw=2))

# Formatting average plot to match style
ax2.set_ylabel('Average CR % $\\uparrow$', fontsize=14, fontweight='normal')
# ax2.set_xlabel('LLM Agent', fontsize=16, fontweight='normal')
# ax2.set_title('Average Performance Across All Stocks', fontsize=16, fontweight='normal')
ax2.set_title('Average Cumulative Return (CR) % $\\uparrow$', fontsize=16, fontweight='normal')
ax2.set_xticks(x)
ax2.set_xticklabels(df['Agent'], fontsize=14, fontweight='bold')
ax2.set_yticks([0, 10, 20, 30, 40])
ax2.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
ax2.set_axisbelow(True)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['left'].set_linewidth(1.2)
ax2.spines['bottom'].set_linewidth(1.2)
ax2.tick_params(axis='both', which='major', labelsize=13)

# plt.tight_layout()
# plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
# # plt.show()
# plt.close()

# Print summary statistics
print("\n=== Agent Performance Summary ===")
print(f"\nBest Overall Agent: {df['Agent'].iloc[best_idx]} (Avg: {df['Average'].iloc[best_idx]:.2f}%)")
print(f"Worst Overall Agent: {df['Agent'].iloc[worst_idx]} (Avg: {df['Average'].iloc[worst_idx]:.2f}%)")
print(f"\nPerformance Range: {df['Average'].max() - df['Average'].min():.2f}%")
print("\nFull Rankings:")
ranked = df.sort_values('Average', ascending=False)[['Agent', 'AAPL', 'GOOGL', 'NVDA', 'Average']]
for idx, row in enumerate(ranked.itertuples(), 1):
    print(f"{idx}. {row.Agent:15s} | AAPL: {row.AAPL:6.2f}% | GOOGL: {row.GOOGL:6.2f}% | NVDA: {row.NVDA:6.2f}% | Avg: {row.Average:6.2f}%")

# 收集所有 handles / labels（任选一个子图即可，或手动）
handles, labels = ax1.get_legend_handles_labels()

# 全局 legend：顶部居中
fig.legend(
    handles,
    labels,
    loc="upper right",
    ncol=len(labels),
    fontsize=14,
    frameon=False,
    bbox_to_anchor=(0.9, 0.9)
)

# 给 legend 留空间
plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.tight_layout()
plt.savefig('performance_comparison_v12_1.png', dpi=300, bbox_inches='tight')
# plt.show()
plt.close()


# ------------------------------------------------------------

# Setup figure with two subplots
fig = plt.figure(figsize=(20, 7.35))
gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.4)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])

### MDD

# Data extracted from your Overleaf Table 1 (Online Agents)
data_mdd = {
    'Agent': ['Grok-4', 'GPT-4o-mini', 'GPT-4o', 'GPT-5-mini', 'GPT-5', 'Grok-4\n(partial alloc.)', 'GPT-4o-mini\n(partial alloc.)', 'Gemini-2.5-lite\n(partial alloc.)', 'Claude-haiku-4-5\n(partial alloc.)', 'Human\n(partial alloc.)'],
    'AAPL': [13.23, 4.40, 13.16, 13.19, 0.36, 9.52, 1.37, 0.65, 4.93, 8.39],
    'GOOGL': [14.36, 3.97, 14.26, 14.36, 15.52, 0.88, 3.20, 0.89, 1.82, 8.73],
    'NVDA': [8.68, 7.43, 8.69, 8.68, 7.41, 4.61, 1.78, 3.78, 0.81, 7.78]
}

for key in data_mdd.keys():
    if key == 'Agent':
        data_mdd['Agent'] = order
        continue
    data_mdd[key] = [data_mdd[key][i] for i in index]

df = pd.DataFrame(data_mdd)
# Calculate average performance per agent
df['Average'] = df[['AAPL', 'GOOGL', 'NVDA']].mean(axis=1)


# Create bars - each agent uses the same color, stocks distinguished by patterns
bars1 = ax1.bar(x - width, df['AAPL'], width, label='AAPL',
                color=[agent_colors[i] for i in range(len(df))],
                hatch=stock_patterns['AAPL'],
                edgecolor='black', linewidth=0.8, alpha=0.8)

bars2 = ax1.bar(x, df['GOOGL'], width, label='GOOGL',
                color=[agent_colors[i] for i in range(len(df))],
                hatch=stock_patterns['GOOGL'],
                edgecolor='black', linewidth=0.8, alpha=0.8)

bars3 = ax1.bar(x + width, df['NVDA'], width, label='NVDA',
                color=[agent_colors[i] for i in range(len(df))],
                hatch=stock_patterns['NVDA'],
                edgecolor='black', linewidth=0.8, alpha=0.8)

# Add value labels on bars for NVDA (highest values)
for j, k in enumerate(['AAPL', 'GOOGL', 'NVDA']):
    for i, v in enumerate(df[k]):
        v_ = v + 0.4 if v > 0 else v - 8
        ax1.text(i + (j-1) *width, v_, f'{v:.1f}', ha='center', va='bottom',
                 fontsize=10, fontweight='bold')

# Formatting main plot
ax1.set_ylabel('Max Drawdown (MDD) %  $\\downarrow$', fontsize=14, fontweight='normal')
# ax1.set_title('Performance Comparison of LLM Agents Across Stocks (2024 Q1)',
            #   fontsize=16, fontweight='normal', pad=20)
ax1.set_title('Max Drawdown (MDD) % $\\downarrow$', fontsize=16, fontweight='normal', pad=20)
ax1.set_xticks(x)
ax1.set_xticklabels(df['Agent'], fontsize=14, fontweight='bold')
# ax1.set_yticks([-20, 0, 20, 40, 60, 80, 100])
# ax1.legend(fontsize=14, loc='upper left', frameon=False)
ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
ax1.set_axisbelow(True)
ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.spines['left'].set_linewidth(1.2)
ax1.spines['bottom'].set_linewidth(1.2)
ax1.tick_params(axis='both', which='major', labelsize=13)

# Add vertical separator lines between agents for clarity
for i in range(len(x) - 1):
    ax1.axvline(x=i + 0.5, color='gray', linestyle=':', linewidth=1, alpha=0.3)

# Second plot - Average performance per agent
bars_avg = ax2.bar(x, df['Average'], width=0.6,
                   color=agent_colors, edgecolor='black', linewidth=0.8, alpha=0.8)

# Highlight best and worst performing agents with thicker borders
best_idx = df['Average'].idxmax()
worst_idx = df['Average'].idxmin()
bars_avg[best_idx].set_edgecolor('gold')  # Dark blue for best
bars_avg[best_idx].set_linewidth(2.5)
bars_avg[worst_idx].set_edgecolor('gold')  # Red for worst
bars_avg[worst_idx].set_linewidth(2.5)

# Add value labels on average bars
for i, v in enumerate(df['Average']):
    ax2.text(i, v + 0.5, f'{v:.1f}%', ha='center', va='bottom',
             fontsize=10, fontweight='bold')

# Add annotations for best/worst
ax2.annotate('Worst Overall', xy=(best_idx, df['Average'].iloc[best_idx]),
            xytext=(best_idx, df['Average'].iloc[best_idx] - 3),
            ha='center', fontsize=10, fontweight='bold', color='gold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.3),
            arrowprops=dict(arrowstyle='->', color='black', lw=2))

ax2.annotate('Best Overall', xy=(worst_idx, df['Average'].iloc[worst_idx]),
            xytext=(worst_idx, df['Average'].iloc[worst_idx] - 3),
            ha='center', fontsize=10, fontweight='bold', color='gold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.3),
            arrowprops=dict(arrowstyle='->', color='black', lw=2))

# Formatting average plot to match style
ax2.set_ylabel('Average MDD % $\\downarrow$', fontsize=14, fontweight='normal')
# ax2.set_xlabel('LLM Agent', fontsize=16, fontweight='normal')
# ax2.set_title('Average Performance Across All Stocks', fontsize=16, fontweight='normal')
ax2.set_title('Average Max Drawdown (MDD) % $\\downarrow$', fontsize=16, fontweight='normal')
ax2.set_xticks(x)
ax2.set_xticklabels(df['Agent'], fontsize=14, fontweight='bold')
ax2.set_yticks([0, 5, 10, 15])
ax2.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
ax2.set_axisbelow(True)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['left'].set_linewidth(1.2)
ax2.spines['bottom'].set_linewidth(1.2)
ax2.tick_params(axis='both', which='major', labelsize=13)

# 收集所有 handles / labels（任选一个子图即可，或手动）
handles, labels = ax1.get_legend_handles_labels()

# 全局 legend：顶部居中
fig.legend(
    handles,
    labels,
    loc="upper right",
    ncol=len(labels),
    fontsize=14,
    frameon=False,
    bbox_to_anchor=(0.9, 0.9)
)

# 给 legend 留空间
plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.tight_layout()
plt.savefig('performance_comparison_v12_2.png', dpi=300, bbox_inches='tight')
# plt.show()
plt.close()

# Print summary statistics
print("\n=== Agent Performance Summary ===")
print(f"\nBest Overall Agent: {df['Agent'].iloc[best_idx]} (Avg: {df['Average'].iloc[best_idx]:.2f}%)")
print(f"Worst Overall Agent: {df['Agent'].iloc[worst_idx]} (Avg: {df['Average'].iloc[worst_idx]:.2f}%)")
print(f"\nPerformance Range: {df['Average'].max() - df['Average'].min():.2f}%")
print("\nFull Rankings:")
ranked = df.sort_values('Average', ascending=False)[['Agent', 'AAPL', 'GOOGL', 'NVDA', 'Average']]
for idx, row in enumerate(ranked.itertuples(), 1):
    print(f"{idx}. {row.Agent:15s} | AAPL: {row.AAPL:6.2f}% | GOOGL: {row.GOOGL:6.2f}% | NVDA: {row.NVDA:6.2f}% | Avg: {row.Average:6.2f}%")