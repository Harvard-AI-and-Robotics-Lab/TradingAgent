import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data from Excel file
# excel_file = 'TA_abl_analysts.xlsx'
# df = pd.read_excel(excel_file, sheet_name='NVDA', fontsize=14)

# Extract NVDA data
nvda_data = {
    'All_Data': {'CR': 45.3, 'ARR': 367.5, 'SR': 3.8, 'MDD': 7.4},
    'Fundamental_Only': {'CR': 46.2, 'ARR': 379.6, 'SR': 4.1, 'MDD': 5.0},
    'Market_Only': {'CR': 16.9, 'ARR': 90.8, 'SR': 3.2, 'MDD': 3.5},
    'News_Only': {'CR': 37.0, 'ARR': 266.7, 'SR': 3.3, 'MDD': 7.4},
    'Sentiment_Only': {'CR': 44.9, 'ARR': 363.4, 'SR': 4.0, 'MDD': 3.5}
}

# Extract AAPL data
aapl_data = {
    'All_Data': {'CR': 2.1, 'ARR': 8.9, 'SR': 1.8, 'MDD': 0.6},
    'Fundamental_Only': {'CR': -0.5, 'ARR': -2.2, 'SR': -0.2, 'MDD': 4.1},
    'Market_Only': {'CR': -1.4, 'ARR': -5.8, 'SR': -0.7, 'MDD': 4.9},
    'News_Only': {'CR': 0.2, 'ARR': 0.7, 'SR': 0.1, 'MDD': 3.1},
    'Sentiment_Only': {'CR': -2.3, 'ARR': -9.1, 'SR': -0.9, 'MDD': 4.1}
}

goog_data = {
    'All_Data': {'CR': -1.73, 'ARR': -6.96, 'SR': -0.445, 'MDD': 7.74},
    'Fundamental_Only': {'CR': -4.85, 'ARR': -18.58, 'SR': -0.815, 'MDD': 17.35},
    'Market_Only': {'CR': 12.67, 'ARR': 63.69, 'SR': 4.007, 'MDD': 2.57},
    'News_Only': {'CR': -5.72, 'ARR': -21.6, 'SR': -1.435, 'MDD': 7.74},
    'Sentiment_Only': {'CR': 9.0, 'ARR': 42.75, 'SR': 3.796, 'MDD': 1.87},
}

# NVDA's specific feature analysis
nvda_fundamentals = {
    'All_Data': {'CR': 87.3, 'ARR': 1236.1, 'SR': 5.5, 'MDD': 8.7},
    'Fundamental_Only': {'CR': 81.6, 'ARR': 1076.8, 'SR': 5.2, 'MDD': 8.7}
}

nvda_market = {
    'All_Data': {'CR': 88.1, 'ARR': 1259.1, 'SR': 5.5, 'MDD': 8.7},
    'Market_Only': {'CR': 101.9, 'ARR': 1720.9, 'SR': 7.4, 'MDD': 6.3}
}

nvda_news = {
    'All_Data': {'CR': 87.3, 'ARR': 1236.1, 'SR': 5.5, 'MDD': 8.7},
    'News_Only': {'CR': 87.3, 'ARR': 1236.1, 'SR': 5.5, 'MDD': 8.7}
}

nvda_sentiment = {
    'All_Data': {'CR': 96.0, 'ARR': 1512.5, 'SR': 6.0, 'MDD': 7.4},
    'Sentiment_Only': {'CR': 89.8, 'ARR': 1312.0, 'SR': 5.6, 'MDD': 8.7}
}

nvda_analysts = {
    'All_Data': {key: np.mean([
        nvda_fundamentals['All_Data'][key],
        nvda_market['All_Data'][key],
        nvda_news['All_Data'][key],
        nvda_sentiment['All_Data'][key]
    ]) for key in nvda_fundamentals['All_Data'].keys()},
    'Fundamental_Only': nvda_fundamentals['Fundamental_Only'],
    'Market_Only': nvda_market['Market_Only'],
    'News_Only': nvda_news['News_Only'],
    'Sentiment_Only': nvda_sentiment['Sentiment_Only'],
}

# AAPL's specific feature analysis
aapl_fundamentals = {
    'All_Data': {'CR': -6.7, 'ARR': -25.0, 'SR': -1.4, 'MDD': 13.1},
    'Fundamental_Only': {'CR': -7.5, 'ARR': -27.4, 'SR': -1.5, 'MDD': 13.2}
}

aapl_market = {
    'All_Data': {'CR': -2.5, 'ARR': -9.9, 'SR': -0.8, 'MDD': 7.2},
    'Market_Only': {'CR': -5.9, 'ARR': -22.1, 'SR': -1.8, 'MDD': 10.5}
}

aapl_news = {
    'All_Data': {'CR': -12.6, 'ARR': -42.6, 'SR': -4.8, 'MDD': 12.6},
    'News_Only': {'CR': -8.8, 'ARR': -31.8, 'SR': -2.9, 'MDD': 13.2}
}

aapl_sentiment = {
    'All_Data': {'CR': -6.2, 'ARR': -23.2, 'SR': -1.3, 'MDD': 13.8},
    'Sentiment_Only': {'CR': -9.3, 'ARR': -33.2, 'SR': -2.3, 'MDD': 12.1}
}

aapl_analysts = {
    'All_Data': {key: np.mean([
        aapl_fundamentals['All_Data'][key],
        aapl_market['All_Data'][key],
        aapl_news['All_Data'][key],
        aapl_sentiment['All_Data'][key]
    ]) for key in aapl_fundamentals['All_Data'].keys()},
    'Fundamental_Only': aapl_fundamentals['Fundamental_Only'],
    'Market_Only': aapl_market['Market_Only'],
    'News_Only': aapl_news['News_Only'],
    'Sentiment_Only': aapl_sentiment['Sentiment_Only'],
}

# GOOG's specific feature analysis
goog_fundamentals = {
    'All_Data': {'CR': 9.08, 'ARR': 43.21, 'SR': 1.452, 'MDD': 14.36},
    'Fundamental_Only': {'CR': 9.08, 'ARR': 43.21, 'SR': 1.452, 'MDD': 14.36},
}

goog_market = {
    'All_Data': {'CR': 9.08, 'ARR': 43.21, 'SR': 1.452, 'MDD': 14.36},
    'Market_Only': {'CR': 3.32, 'ARR': 14.43, 'SR': 0.705, 'MDD': 10.37},
}

goog_news = {
    'All_Data': {'CR': -2.78, 'ARR': -10.98, 'SR': -0.336, 'MDD': 13.6},
    'News_Only': {'CR': 3.07, 'ARR': 13.32, 'SR': 0.613, 'MDD': 14.36},
}

goog_sentiment = {
    'All_Data': {'CR': 17.48, 'ARR': 94.55, 'SR': 3.098, 'MDD': 11.65},
    'Sentiment_Only': {'CR': 9.08, 'ARR': 43.21, 'SR': 1.452, 'MDD': 14.36},
}

goog_analysts = {
    'All_Data': {key: np.mean([
        goog_fundamentals['All_Data'][key],
        goog_market['All_Data'][key],
        goog_news['All_Data'][key],
        goog_sentiment['All_Data'][key]
    ]) for key in goog_fundamentals['All_Data'].keys()},
    'Fundamental_Only': goog_fundamentals['Fundamental_Only'],
    'Market_Only': goog_market['Market_Only'],
    'News_Only': goog_news['News_Only'],
    'Sentiment_Only': goog_sentiment['Sentiment_Only'],
}

# Researcher vs Trader comparison
nvda_researcher = {
    'All_Data': {'CR': 38.9, 'ARR': 289.0, 'SR': 3.4, 'MDD': 7.4},
    'Fundamental_Only': {'CR': 11.2, 'ARR': 54.8, 'SR': 2.3, 'MDD': 5.0},
    'Market_Only': {'CR': 13.7, 'ARR': 70.1, 'SR': 2.9, 'MDD': 3.2},
    'News_Only': {'CR': 33.9, 'ARR': 233.7, 'SR': 3.1, 'MDD': 7.4},
    'Sentiment_Only': {'CR': 42.9, 'ARR': 337.5, 'SR': 3.9, 'MDD': 3.5}
}

nvda_trader = {
    'All_Data': {'CR': 40.8, 'ARR': 311.0, 'SR': 3.5, 'MDD': 7.4},
    'Fundamental_Only': {'CR': 47.0, 'ARR': 390.9, 'SR': 3.9, 'MDD': 8.7},
    'Market_Only': {'CR': 18.6, 'ARR': 102.3, 'SR': 3.4, 'MDD': 3.2},
    'News_Only': {'CR': 27.0, 'ARR': 168.4, 'SR': 2.5, 'MDD': 7.4},
    'Sentiment_Only': {'CR': 52.6, 'ARR': 473.1, 'SR': 4.5, 'MDD': 3.5}
}

aapl_researcher = {
    'All_Data': {'CR': 2.1, 'ARR': 8.9, 'SR': 1.8, 'MDD': 0.6},
    'Fundamental_Only': {'CR': -7.5, 'ARR': -27.4, 'SR': -1.5, 'MDD': 13.2},
    'Market_Only': {'CR': 1.1, 'ARR': 4.8, 'SR': 0.9, 'MDD': 1.8},
    'News_Only': {'CR': 1.2, 'ARR': 4.8, 'SR': 0.7, 'MDD': 2.2},
    'Sentiment_Only': {'CR': -2.3, 'ARR': -9.1, 'SR': -0.9, 'MDD': 4.1}
}

aapl_trader = {
    'All_Data': {'CR': 2.1, 'ARR': 8.9, 'SR': 1.8, 'MDD': 0.6},
    'Fundamental_Only': {'CR': 0.4, 'ARR': 1.8, 'SR': 0.2, 'MDD': 3.8},
    'Market_Only': {'CR': 0.9, 'ARR': 3.6, 'SR': 0.6, 'MDD': 3.1},
    'News_Only': {'CR': -4.5, 'ARR': -17.3, 'SR': -1.8, 'MDD': 7.6},
    'Sentiment_Only': {'CR': -1.8, 'ARR': -7.1, 'SR': -0.7, 'MDD': 4.1}
}

# Researcher vs Trader comparison
goog_researcher = {
    'All_Data': {'CR': -3.68, 'ARR': -14.36, 'SR': -1.094, 'MDD': 7.74},
    'Fundamental_Only': {'CR': 1.62, 'ARR': 6.85, 'SR': 0.49, 'MDD': 10.5},
    'Market_Only': {'CR': 12.67, 'ARR': 63.69, 'SR': 4.007, 'MDD': 2.57},
    'News_Only': {'CR': -6.5, 'ARR': -24.23, 'SR': -1.717, 'MDD': 8.32},
    'Sentiment_Only': {'CR': 9.44, 'ARR': 45.16, 'SR': 3.998, 'MDD': 1.87},
}

goog_trader = {
    'All_Data': {'CR': -1.99, 'ARR': -7.95, 'SR': -0.541, 'MDD': 6.88},
    'Fundamental_Only': {'CR': 3.11, 'ARR': 13.49, 'SR': 0.857, 'MDD': 9.19},
    'Market_Only': {'CR': 15.01, 'ARR': 78.18, 'SR': 4.642, 'MDD': 2.57},
    'News_Only': {'CR': -3.74, 'ARR': -14.58, 'SR': -0.895, 'MDD': 7.26},
    'Sentiment_Only': {'CR': 11.44, 'ARR': 56.45, 'SR': 4.356, 'MDD': 1.86},
}


def plot_ablation_study_research_trader_risk():
    """Plot the ablation study results"""

    # Set style
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

    # Create figure with subplots
    fig, axes = plt.subplots(3, 3, figsize=(25, 12.5))

    width = 0.12

    # Plot 5: Fund, Market, News, Sentiment, Researcher , Trader, Final on CR
    features_short = ['Analysts', 'Researcher', 'Trader', 'Risk Manager']
    data_agents = ["Fundamental_Only", "Market_Only", "News_Only", "Sentiment_Only", "All_Data"]
    colors = ["#4DBBD5", "#00A087", "#E64B35", "#F39B7F", "#3C5488"]

    x = np.arange(len(features_short))
    for i,data_agent in enumerate(data_agents):
        aapl_cr = {
            'Analysts': aapl_analysts[data_agent]['CR'] if data_agent in aapl_researcher else 0,
            'Researcher': aapl_researcher[data_agent]['CR'] if data_agent in aapl_researcher else 0,
            'Trader': aapl_trader[data_agent]['CR'] if data_agent in aapl_trader else 0,
            'Risk': aapl_data[data_agent]['CR'] if data_agent in aapl_data else 0,
        }
        aapl_cr = [aapl_cr[feat] for feat in aapl_cr.keys()]
        i_ = i - len(features_short)//2

        if data_agent == "All_Data":
            axes[0,0].bar(x + width * i_, aapl_cr, width, label=data_agent, alpha=0.7, color=colors[i],
                         edgecolor='black', linewidth=0.8, hatch='//')
        else:
            axes[0,0].bar(x + width * i_, aapl_cr, width, label=data_agent, alpha=0.8, color=colors[i],
                         edgecolor='black', linewidth=0.8)

    axes[0,0].set_ylabel('Cumulative Returns (%) $\\uparrow$', fontsize=16, fontweight='normal')
    axes[0,0].set_xticks(x)
    axes[0,0].set_xticklabels(features_short, fontsize=18, fontweight='bold')
    axes[0,0].set_xlabel('$\\bf{AAPL}$: Agent Thinking Deeper $\\rightarrow$', fontsize=16, fontweight='normal', style='italic', labelpad=10)
    axes[0,0].tick_params(axis='both', which='major', labelsize=13)
    axes[0,0].spines['right'].set_visible(False)
    axes[0,0].spines['top'].set_visible(False)
    axes[0,0].spines['left'].set_linewidth(1.2)
    axes[0,0].spines['bottom'].set_linewidth(1.2)
    axes[0,0].grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)

    for i,data_agent in enumerate(data_agents):
        goog_cr = {
            'Analysts': goog_analysts[data_agent]['CR'] if data_agent in goog_researcher else 0,
            'Researcher': goog_researcher[data_agent]['CR'] if data_agent in goog_researcher else 0,
            'Trader': goog_trader[data_agent]['CR'] if data_agent in goog_trader else 0,
            'Risk': goog_data[data_agent]['CR'] if data_agent in goog_data else 0,
        }
        goog_cr = [goog_cr[feat] for feat in goog_cr.keys()]
        i_ = i - len(features_short)//2

        if data_agent == "All_Data":
            axes[1,0].bar(x + width * i_, goog_cr, width, label=data_agent, alpha=0.7, color=colors[i],
                         edgecolor='black', linewidth=0.8, hatch='//')
        else:
            axes[1,0].bar(x + width * i_, goog_cr, width, label=data_agent, alpha=0.8, color=colors[i],
                         edgecolor='black', linewidth=0.8)

    axes[1,0].set_ylabel('Cumulative Returns (%) $\\uparrow$', fontsize=16, fontweight='normal')
    axes[1,0].set_xticks(x)
    axes[1,0].set_xticklabels(features_short, fontsize=18, fontweight='bold')
    axes[1,0].set_xlabel('$\\bf{GOOG}$: Agent Thinking Deeper $\\rightarrow$', fontsize=16, fontweight='normal', style='italic', labelpad=10)
    axes[1,0].tick_params(axis='both', which='major', labelsize=13)
    axes[1,0].spines['right'].set_visible(False)
    axes[1,0].spines['top'].set_visible(False)
    axes[1,0].spines['left'].set_linewidth(1.2)
    axes[1,0].spines['bottom'].set_linewidth(1.2)
    axes[1,0].grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)

    for i, data_agent in enumerate(data_agents):
        nvda_cr = {
            'Analysts': nvda_analysts[data_agent]['CR'] if data_agent in nvda_researcher else 0,
            'Researcher': nvda_researcher[data_agent]['CR'] if data_agent in nvda_researcher else 0,
            'Trader': nvda_trader[data_agent]['CR'] if data_agent in nvda_trader else 0,
            'Risk': nvda_data[data_agent]['CR'] if data_agent in nvda_data else 0,
        }

        nvda_cr = [nvda_cr[feat] for feat in nvda_cr.keys()]
        i_ = i - len(features_short)//2

        if data_agent == "All_Data":
            axes[2,0].bar(x + width * i_, nvda_cr, width, label=data_agent, alpha=0.7, color=colors[i],
                         edgecolor='black', linewidth=0.8, hatch='//')
        else:
            axes[2,0].bar(x + width * i_, nvda_cr, width, label=data_agent, alpha=0.8, color=colors[i],
                         edgecolor='black', linewidth=0.8)

    axes[2,0].set_ylabel('Cumulative Returns (%) $\\uparrow$', fontsize=16, fontweight='normal')
    axes[2,0].set_xticks(x)
    axes[2,0].set_xticklabels(features_short, fontsize=18, fontweight='bold')
    axes[2,0].set_xlabel('$\\bf{NVDA}$: Agent Thinking Deeper $\\rightarrow$', fontsize=16, fontweight='normal', style='italic', labelpad=10)
    axes[2,0].tick_params(axis='both', which='major', labelsize=13)
    axes[2,0].spines['right'].set_visible(False)
    axes[2,0].spines['top'].set_visible(False)
    axes[2,0].spines['left'].set_linewidth(1.2)
    axes[2,0].spines['bottom'].set_linewidth(1.2)
    axes[2,0].grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)


    # Plot 6: Fund, Market, News, Sentiment, Researcher , Trader, Final on Sharpe Ratio (%) (SR)
    for i,data_agent in enumerate(data_agents):
        aapl_sr = {
            'Analysts': aapl_analysts[data_agent]['SR'] if data_agent in aapl_researcher else 0,
            'Researcher': aapl_researcher[data_agent]['SR'] if data_agent in aapl_researcher else 0,
            'Trader': aapl_trader[data_agent]['SR'] if data_agent in aapl_trader else 0,
            'Risk': aapl_data[data_agent]['SR'] if data_agent in aapl_data else 0,
        }
        aapl_sr = [aapl_sr[feat] for feat in aapl_sr.keys()]
        i_ = i - len(features_short)//2

        if data_agent == "All_Data":
            axes[0,1].bar(x + width * i_, aapl_sr, width, label=data_agent, alpha=0.7, color=colors[i],
                         edgecolor='black', linewidth=0.8, hatch='//')
        else:
            axes[0,1].bar(x + width * i_, aapl_sr, width, label=data_agent, alpha=0.8, color=colors[i],
                         edgecolor='black', linewidth=0.8)

    axes[0,1].set_ylabel('Sharpe Ratio (%) $\\uparrow$', fontsize=16, fontweight='normal')
    axes[0,1].set_xticks(x)
    axes[0,1].set_xticklabels(features_short, fontsize=18, fontweight='bold')
    axes[0,1].set_xlabel('$\\bf{AAPL}$: Agent Thinking Deeper $\\rightarrow$', fontsize=16, fontweight='normal', style='italic', labelpad=10)
    axes[0,1].tick_params(axis='both', which='major', labelsize=13)
    axes[0,1].spines['right'].set_visible(False)
    axes[0,1].spines['top'].set_visible(False)
    axes[0,1].spines['left'].set_linewidth(1.2)
    axes[0,1].spines['bottom'].set_linewidth(1.2)
    axes[0,1].grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)

    for i,data_agent in enumerate(data_agents):
        goog_sr = {
            'Analysts': goog_analysts[data_agent]['SR'] if data_agent in goog_researcher else 0,
            'Researcher': goog_researcher[data_agent]['SR'] if data_agent in goog_researcher else 0,
            'Trader': goog_trader[data_agent]['SR'] if data_agent in goog_trader else 0,
            'Risk': goog_data[data_agent]['SR'] if data_agent in goog_data else 0,
        }
        goog_sr = [goog_sr[feat] for feat in goog_sr.keys()]
        i_ = i - len(features_short)//2

        if data_agent == "All_Data":
            axes[1,1].bar(x + width * i_, goog_sr, width, label=data_agent, alpha=0.7, color=colors[i],
                         edgecolor='black', linewidth=0.8, hatch='//')
        else:
            axes[1,1].bar(x + width * i_, goog_sr, width, label=data_agent, alpha=0.8, color=colors[i],
                         edgecolor='black', linewidth=0.8)

    axes[1,1].set_ylabel('Sharpe Ratio (%) $\\uparrow$', fontsize=16, fontweight='normal')
    axes[1,1].set_xticks(x)
    axes[1,1].set_xticklabels(features_short, fontsize=18, fontweight='bold')
    axes[1,1].set_xlabel('$\\bf{GOOG}$: Agent Thinking Deeper $\\rightarrow$', fontsize=16, fontweight='normal', style='italic', labelpad=10)
    axes[1,1].tick_params(axis='both', which='major', labelsize=13)
    axes[1,1].spines['right'].set_visible(False)
    axes[1,1].spines['top'].set_visible(False)
    axes[1,1].spines['left'].set_linewidth(1.2)
    axes[1,1].spines['bottom'].set_linewidth(1.2)
    axes[1,1].grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)

    for i,data_agent in enumerate(data_agents):
        nvda_sr = {
            'Analysts': nvda_analysts[data_agent]['SR'] if data_agent in nvda_researcher else 0,
            'Researcher': nvda_researcher[data_agent]['SR'] if data_agent in nvda_researcher else 0,
            'Trader': nvda_trader[data_agent]['SR'] if data_agent in nvda_trader else 0,
            'Risk': nvda_data[data_agent]['SR'] if data_agent in nvda_data else 0,
        }
        nvda_sr = [nvda_sr[feat] for feat in nvda_sr.keys()]
        i_ = i - len(features_short)//2

        if data_agent == "All_Data":
            axes[2,1].bar(x + width * i_, nvda_sr, width, label=data_agent, alpha=0.7, color=colors[i],
                         edgecolor='black', linewidth=0.8, hatch='//')
        else:
            axes[2,1].bar(x + width * i_, nvda_sr, width, label=data_agent, alpha=0.8, color=colors[i],
                         edgecolor='black', linewidth=0.8)

    axes[2,1].set_title('Average Sharpe Ratio (SR) % $\\uparrow$', fontsize=16, fontweight='normal')
    axes[2,1].set_ylabel('Sharpe Ratio (%) $\\uparrow$', fontsize=16, fontweight='normal')
    axes[2,1].set_xticks(x)
    axes[2,1].set_xticklabels(features_short, fontsize=18, fontweight='bold')
    axes[2,1].set_xlabel('$\\bf{NVDA}$: Agent Thinking Deeper $\\rightarrow$', fontsize=16, fontweight='normal', style='italic', labelpad=10)
    axes[2,1].tick_params(axis='both', which='major', labelsize=13)
    axes[2,1].spines['right'].set_visible(False)
    axes[2,1].spines['top'].set_visible(False)
    axes[2,1].spines['left'].set_linewidth(1.2)
    axes[2,1].spines['bottom'].set_linewidth(1.2)
    axes[2,1].grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)

    # Plot 7: Fund, Market, News, Sentiment, Researcher , Trader, Final on Max Drawdown (MDD)
    for i,data_agent in enumerate(data_agents):
        aapl_mdd = {
            'Analysts': aapl_analysts[data_agent]['MDD'] if data_agent in aapl_researcher else 0,
            'Researcher': aapl_researcher[data_agent]['MDD'] if data_agent in aapl_researcher else 0,
            'Trader': aapl_trader[data_agent]['MDD'] if data_agent in aapl_trader else 0,
            'Risk': aapl_data[data_agent]['MDD'] if data_agent in aapl_data else 0,
        }
        aapl_mdd = [aapl_mdd[feat] for feat in aapl_mdd.keys()]
        i_ = i - len(features_short)//2

        if data_agent == "All_Data":
            axes[0,2].bar(x + width * i_, aapl_mdd, width, label=data_agent, alpha=0.7, color=colors[i],
                         edgecolor='black', linewidth=0.8, hatch='//')
        else:
            axes[0,2].bar(x + width * i_, aapl_mdd, width, label=data_agent, alpha=0.8, color=colors[i],
                         edgecolor='black', linewidth=0.8)

    axes[0,2].set_title('Max Drawdown (MDD) % $\\downarrow$', fontsize=16, fontweight='normal')
    axes[0,2].set_ylabel('Max Drawdown (%) $\\downarrow$', fontsize=16, fontweight='normal')
    axes[0,2].set_xticks(x)
    axes[0,2].set_xticklabels(features_short, fontsize=18, fontweight='bold')
    axes[0,2].set_xlabel('$\\bf{AAPL}$: Agent Thinking Deeper $\\rightarrow$', fontsize=16, fontweight='normal', style='italic', labelpad=10)
    axes[0,2].tick_params(axis='both', which='major', labelsize=13)
    axes[0,2].spines['right'].set_visible(False)
    axes[0,2].spines['top'].set_visible(False)
    axes[0,2].spines['left'].set_linewidth(1.2)
    axes[0,2].spines['bottom'].set_linewidth(1.2)
    axes[0,2].grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)

    for i,data_agent in enumerate(data_agents):
        goog_mdd = {
            'Analysts': goog_analysts[data_agent]['MDD'] if data_agent in goog_researcher else 0,
            'Researcher': goog_researcher[data_agent]['MDD'] if data_agent in goog_researcher else 0,
            'Trader': goog_trader[data_agent]['MDD'] if data_agent in goog_trader else 0,
            'Risk': goog_data[data_agent]['MDD'] if data_agent in goog_data else 0,
        }
        goog_mdd = [goog_mdd[feat] for feat in goog_mdd.keys()]
        i_ = i - len(features_short)//2

        if data_agent == "All_Data":
            axes[1,2].bar(x + width * i_, goog_mdd, width, label=data_agent, alpha=0.7, color=colors[i],
                         edgecolor='black', linewidth=0.8, hatch='//')
        else:
            axes[1,2].bar(x + width * i_, goog_mdd, width, label=data_agent, alpha=0.8, color=colors[i],
                         edgecolor='black', linewidth=0.8)

    axes[1,2].set_ylabel('Max Drawdown (%) $\\downarrow$', fontsize=16, fontweight='normal')
    axes[1,2].set_xticks(x)
    axes[1,2].set_xticklabels(features_short, fontsize=18, fontweight='bold')
    axes[1,2].set_xlabel('$\\bf{GOOG}$: Agent Thinking Deeper $\\rightarrow$', fontsize=16, fontweight='normal', style='italic', labelpad=10)
    axes[1,2].tick_params(axis='both', which='major', labelsize=13)
    axes[1,2].spines['right'].set_visible(False)
    axes[1,2].spines['top'].set_visible(False)
    axes[1,2].spines['left'].set_linewidth(1.2)
    axes[1,2].spines['bottom'].set_linewidth(1.2)
    axes[1,2].grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)

    for i,data_agent in enumerate(data_agents):
        nvda_mdd = {
            'Analysts': nvda_analysts[data_agent]['MDD'] if data_agent in nvda_researcher else 0,
            'Researcher': nvda_researcher[data_agent]['MDD'] if data_agent in nvda_researcher else 0,
            'Trader': nvda_trader[data_agent]['MDD'] if data_agent in nvda_trader else 0,
            'Risk': nvda_data[data_agent]['MDD'] if data_agent in nvda_data else 0,
        }
        nvda_mdd = [nvda_mdd[feat] for feat in nvda_mdd.keys()]
        i_ = i - len(features_short)//2

        if data_agent == "All_Data":
            axes[2,2].bar(x + width * i_, nvda_mdd, width, label=data_agent, alpha=0.7, color=colors[i],
                         edgecolor='black', linewidth=0.8, hatch='//')
        else:
            axes[2,2].bar(x + width * i_, nvda_mdd, width, label=data_agent, alpha=0.8, color=colors[i],
                         edgecolor='black', linewidth=0.8)

    axes[2,2].set_ylabel('Max Drawdown (%) $\\downarrow$', fontsize=16, fontweight='normal')
    axes[2,2].set_xticks(x)
    axes[2,2].set_xticklabels(features_short, fontsize=18, fontweight='bold')
    axes[2,2].set_xlabel('$\\bf{NVDA}$: Agent Thinking Deeper $\\rightarrow$', fontsize=16, fontweight='normal', style='italic', labelpad=10)
    axes[2,2].tick_params(axis='both', which='major', labelsize=13)
    axes[2,2].spines['right'].set_visible(False)
    axes[2,2].spines['top'].set_visible(False)
    axes[2,2].spines['left'].set_linewidth(1.2)
    axes[2,2].spines['bottom'].set_linewidth(1.2)
    axes[2,2].grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)

    # Add shared legend at the top of the figure
    handles, labels = axes[0,0].get_legend_handles_labels()
    clean_labels = [label.replace('_', ' ') for label in labels]
    fig.legend(handles, clean_labels, loc='upper center', bbox_to_anchor=(0.5, 1.03), ncol=5, frameon=False, fontsize=16)

    plt.tight_layout()
    plt.savefig('ablation_study_plots_research_trader_risk.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_ablation_study_analyst_researcher():
    """Plot the ablation study results"""

    # Set font style
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

    # Create figure with subplots - 3 rows, 3 columns
    fig, axes = plt.subplots(3, 3, figsize=(22, 12.5))

    width = 0.16

    data_agents = ["Fundamental_Only", "Market_Only", "News_Only", "Sentiment_Only", "All_Data"]
    colors = ["#4DBBD5", "#00A087", "#E64B35", "#F39B7F", "#3C5488"]
    labels = ["LLM Analysts", "Human Analysts (pseudo)"]

    features_short = ['Fund. Only', 'Market Only', 'News Only', 'Sent. Only', 'All Data']
    x = np.arange(len(features_short))

    # ===== ROW 0: AAPL =====
    # Column 0: AAPL CR
    for i, data_agent in enumerate(data_agents):
        aapl_cr = {
            'Analysts': aapl_analysts[data_agent]['CR'] if data_agent in aapl_analysts else 0,
            'Researcher': aapl_researcher[data_agent]['CR'] if data_agent in aapl_researcher else 0,
        }
        aapl_cr = [aapl_cr[feat] for feat in aapl_cr.keys()]
        x_i = [
            x[i] + width * (j - len(labels)//2)
            for j, feat in enumerate(aapl_cr)
        ]

        if data_agent == "All_Data":
            axes[0,0].bar(x_i[0], aapl_cr[0], width, label=labels[0], alpha=0.7, color=colors[i],
                         edgecolor='black', linewidth=0.8, hatch='//')
            axes[0,0].bar(x_i[1], aapl_cr[1], width, label=labels[1], alpha=0.7, color=colors[i],
                         edgecolor='black', linewidth=0.8, hatch='xx')
        else:
            axes[0,0].bar(x_i[0], aapl_cr[0], width, label=labels[0], alpha=0.8, color=colors[i],
                         edgecolor='black', linewidth=0.8)
            axes[0,0].bar(x_i[1], aapl_cr[1], width, label=labels[1], alpha=0.8, color=colors[i],
                         edgecolor='black', linewidth=0.8, hatch='xx')

    axes[0,0].set_ylabel('Cumulative Returns (%) $\\uparrow$', fontsize=16)
    axes[0,0].set_xticks(x)
    axes[0,0].set_xticklabels(features_short, fontsize=18, fontweight='bold')
    # axes[0,0].set_xlabel('$\\bf{AAPL}$: Agent Thinking Deeper $\\rightarrow$', fontsize=16, style='italic', labelpad=10)
    axes[0,0].tick_params(axis='both', which='major', labelsize=13)
    axes[0,0].spines['right'].set_visible(False)
    axes[0,0].spines['top'].set_visible(False)
    axes[0,0].spines['left'].set_linewidth(1.2)
    axes[0,0].spines['bottom'].set_linewidth(1.2)
    axes[0,0].grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)

    # Column 1: AAPL SR
    nonzero_count = [0, 0, 0, 0, 0]
    for i, data_agent in enumerate(data_agents):
        aapl_sr = {
            'Analysts': aapl_analysts[data_agent]['SR'] if data_agent in aapl_analysts else 0,
            'Researcher': aapl_researcher[data_agent]['SR'] if data_agent in aapl_researcher else 0,
        }
        aapl_sr = [aapl_sr[feat] for feat in aapl_sr.keys()]
        x_i = [
            x[i] + width * (j - len(labels)//2)
            for j, feat in enumerate(aapl_sr)
        ]

        if data_agent == "All_Data":
            axes[0,1].bar(x_i[0], aapl_sr[0], width, label=labels[0], alpha=0.7, color=colors[i],
                         edgecolor='black', linewidth=0.8, hatch='//')
            axes[0,1].bar(x_i[1], aapl_sr[1], width, label=labels[1], alpha=0.7, color=colors[i],
                         edgecolor='black', linewidth=0.8, hatch='xx')
        else:
            axes[0,1].bar(x_i[0], aapl_sr[0], width, label=labels[0], alpha=0.8, color=colors[i],
                         edgecolor='black', linewidth=0.8)
            axes[0,1].bar(x_i[1], aapl_sr[1], width, label=labels[1], alpha=0.8, color=colors[i],
                         edgecolor='black', linewidth=0.8, hatch='xx')

    axes[0,1].set_ylabel('Sharpe Ratio (%) $\\uparrow$', fontsize=16)
    axes[0,1].set_xticks(x)
    axes[0,1].set_xticklabels(features_short, fontsize=18, fontweight='bold')
    # axes[0,1].set_xlabel('$\\bf{AAPL}$: Agent Thinking Deeper $\\rightarrow$', fontsize=16, style='italic', labelpad=10)
    axes[0,1].tick_params(axis='both', which='major', labelsize=13)
    axes[0,1].spines['right'].set_visible(False)
    axes[0,1].spines['top'].set_visible(False)
    axes[0,1].spines['left'].set_linewidth(1.2)
    axes[0,1].spines['bottom'].set_linewidth(1.2)
    axes[0,1].grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)

    # Column 2: AAPL MDD
    for i, data_agent in enumerate(data_agents):
        aapl_mdd = {
            'Analysts': aapl_analysts[data_agent]['MDD'] if data_agent in aapl_analysts else 0,
            'Researcher': aapl_researcher[data_agent]['MDD'] if data_agent in aapl_researcher else 0,
        }
        aapl_mdd = [aapl_mdd[feat] for feat in aapl_mdd.keys()]
        x_i = [
            x[i] + width * (j - len(labels)//2)
            for j, feat in enumerate(aapl_mdd)
        ]

        if data_agent == "All_Data":
            axes[0,2].bar(x_i[0], aapl_mdd[0], width, label=labels[0], alpha=0.7, color=colors[i],
                         edgecolor='black', linewidth=0.8, hatch='//')
            axes[0,2].bar(x_i[1], aapl_mdd[1], width, label=labels[1], alpha=0.7, color=colors[i],
                         edgecolor='black', linewidth=0.8, hatch='xx')
        else:
            axes[0,2].bar(x_i[0], aapl_mdd[0], width, label=labels[0], alpha=0.8, color=colors[i],
                         edgecolor='black', linewidth=0.8)
            axes[0,2].bar(x_i[1], aapl_mdd[1], width, label=labels[1], alpha=0.8, color=colors[i],
                         edgecolor='black', linewidth=0.8, hatch='xx')

    axes[0,2].set_ylabel('Max Drawdown (%) $\\downarrow$', fontsize=16)
    axes[0,2].set_xticks(x)
    axes[0,2].set_xticklabels(features_short, fontsize=18, fontweight='bold')
    # axes[0,2].set_xlabel('$\\bf{AAPL}$: Agent Thinking Deeper $\\rightarrow$', fontsize=16, style='italic', labelpad=10)
    axes[0,2].tick_params(axis='both', which='major', labelsize=13)
    axes[0,2].spines['right'].set_visible(False)
    axes[0,2].spines['top'].set_visible(False)
    axes[0,2].spines['left'].set_linewidth(1.2)
    axes[0,2].spines['bottom'].set_linewidth(1.2)
    axes[0,2].grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)

    # ===== ROW 1: GOOG =====
    # Column 0: GOOG CR
    nonzero_count = [0, 0, 0, 0, 0]
    for i, data_agent in enumerate(data_agents):
        goog_cr = {
            'Analysts': goog_analysts[data_agent]['CR'] if data_agent in goog_analysts else 0,
            'Researcher': goog_researcher[data_agent]['CR'] if data_agent in goog_researcher else 0,
        }
        goog_cr = [goog_cr[feat] for feat in goog_cr.keys()]
        x_i = [
            x[i] + width * (j - len(labels)//2)
            for j, feat in enumerate(goog_cr)
        ]

        if data_agent == "All_Data":
            axes[1,0].bar(x_i[0], goog_cr[0], width, label=labels[0], alpha=0.7, color=colors[i],
                         edgecolor='black', linewidth=0.8, hatch='//')
            axes[1,0].bar(x_i[1], goog_cr[1], width, label=labels[1], alpha=0.7, color=colors[i],
                         edgecolor='black', linewidth=0.8, hatch='xx')
        else:
            axes[1,0].bar(x_i[0], goog_cr[0], width, label=labels[0], alpha=0.8, color=colors[i],
                         edgecolor='black', linewidth=0.8)
            axes[1,0].bar(x_i[1], goog_cr[1], width, label=labels[1], alpha=0.8, color=colors[i],
                         edgecolor='black', linewidth=0.8, hatch='xx')

    axes[1,0].set_ylabel('Cumulative Returns (%) $\\uparrow$', fontsize=16)
    axes[1,0].set_xticks(x)
    axes[1,0].set_xticklabels(features_short, fontsize=18, fontweight='bold')
    # axes[1,0].set_xlabel('$\\bf{GOOG}$: Agent Thinking Deeper $\\rightarrow$', fontsize=16, style='italic', labelpad=10)
    axes[1,0].tick_params(axis='both', which='major', labelsize=13)
    axes[1,0].spines['right'].set_visible(False)
    axes[1,0].spines['top'].set_visible(False)
    axes[1,0].spines['left'].set_linewidth(1.2)
    axes[1,0].spines['bottom'].set_linewidth(1.2)
    axes[1,0].grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)

    # Column 1: GOOG SR
    nonzero_count = [0, 0, 0, 0, 0]
    for i, data_agent in enumerate(data_agents):
        goog_sr = {
            'Analysts': goog_analysts[data_agent]['SR'] if data_agent in goog_analysts else 0,
            'Researcher': goog_researcher[data_agent]['SR'] if data_agent in goog_researcher else 0,
        }
        goog_sr = [goog_sr[feat] for feat in goog_sr.keys()]
        x_i = [
            x[i] + width * (j - len(labels)//2)
            for j, feat in enumerate(goog_sr)
        ]

        if data_agent == "All_Data":
            axes[1,1].bar(x_i[0], goog_sr[0], width, label=labels[0], alpha=0.7, color=colors[i],
                         edgecolor='black', linewidth=0.8, hatch='//')
            axes[1,1].bar(x_i[1], goog_sr[1], width, label=labels[1], alpha=0.7, color=colors[i],
                         edgecolor='black', linewidth=0.8, hatch='xx')
        else:
            axes[1,1].bar(x_i[0], goog_sr[0], width, label=labels[0], alpha=0.8, color=colors[i],
                         edgecolor='black', linewidth=0.8)
            axes[1,1].bar(x_i[1], goog_sr[1], width, label=labels[1], alpha=0.8, color=colors[i],
                         edgecolor='black', linewidth=0.8, hatch='xx')

    axes[1,1].set_ylabel('Sharpe Ratio (%) $\\uparrow$', fontsize=16)
    axes[1,1].set_xticks(x)
    axes[1,1].set_xticklabels(features_short, fontsize=18, fontweight='bold')
    # axes[1,1].set_xlabel('$\\bf{GOOG}$: Agent Thinking Deeper $\\rightarrow$', fontsize=16, style='italic', labelpad=10)
    axes[1,1].tick_params(axis='both', which='major', labelsize=13)
    axes[1,1].spines['right'].set_visible(False)
    axes[1,1].spines['top'].set_visible(False)
    axes[1,1].spines['left'].set_linewidth(1.2)
    axes[1,1].spines['bottom'].set_linewidth(1.2)
    axes[1,1].grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)

    # Column 2: GOOG MDD
    nonzero_count = [0, 0, 0, 0, 0]
    for i, data_agent in enumerate(data_agents):
        goog_mdd = {
            'Analysts': goog_analysts[data_agent]['MDD'] if data_agent in goog_analysts else 0,
            'Researcher': goog_researcher[data_agent]['MDD'] if data_agent in goog_researcher else 0,
        }
        goog_mdd = [goog_mdd[feat] for feat in goog_mdd.keys()]
        x_i = [
            x[i] + width * (j - len(labels)//2)
            for j, feat in enumerate(goog_mdd)
        ]

        if data_agent == "All_Data":
            axes[1,2].bar(x_i[0], goog_mdd[0], width, label=labels[0], alpha=0.7, color=colors[i],
                         edgecolor='black', linewidth=0.8, hatch='//')
            axes[1,2].bar(x_i[1], goog_mdd[1], width, label=labels[1], alpha=0.7, color=colors[i],
                         edgecolor='black', linewidth=0.8, hatch='xx')
        else:
            axes[1,2].bar(x_i[0], goog_mdd[0], width, label=labels[0], alpha=0.8, color=colors[i],
                         edgecolor='black', linewidth=0.8)
            axes[1,2].bar(x_i[1], goog_mdd[1], width, label=labels[1], alpha=0.8, color=colors[i],
                         edgecolor='black', linewidth=0.8, hatch='xx')

    axes[1,2].set_ylabel('Max Drawdown (%) $\\downarrow$', fontsize=16)
    axes[1,2].set_xticks(x)
    axes[1,2].set_xticklabels(features_short, fontsize=18, fontweight='bold')
    # axes[1,2].set_xlabel('$\\bf{GOOG}$: Agent Thinking Deeper $\\rightarrow$', fontsize=16, style='italic', labelpad=10)
    axes[1,2].tick_params(axis='both', which='major', labelsize=13)
    axes[1,2].spines['right'].set_visible(False)
    axes[1,2].spines['top'].set_visible(False)
    axes[1,2].spines['left'].set_linewidth(1.2)
    axes[1,2].spines['bottom'].set_linewidth(1.2)
    axes[1,2].grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)

    # ===== ROW 2: NVDA =====
    # Column 0: NVDA CR
    nonzero_count = [0, 0, 0, 0, 0]
    for i, data_agent in enumerate(data_agents):
        nvda_cr = {
            'Analysts': nvda_analysts[data_agent]['CR'] if data_agent in nvda_analysts else 0,
            'Researcher': nvda_researcher[data_agent]['CR'] if data_agent in nvda_researcher else 0,
        }
        nvda_cr = [nvda_cr[feat] for feat in nvda_cr.keys()]
        x_i = [
            x[i] + width * (j - len(labels)//2)
            for j, feat in enumerate(nvda_cr)
        ]

        if data_agent == "All_Data":
            axes[2,0].bar(x_i[0], nvda_cr[0], width, label=labels[0], alpha=0.7, color=colors[i],
                         edgecolor='black', linewidth=0.8, hatch='//')
            axes[2,0].bar(x_i[1], nvda_cr[1], width, label=labels[1], alpha=0.7, color=colors[i],
                         edgecolor='black', linewidth=0.8, hatch='xx')
        else:
            axes[2,0].bar(x_i[0], nvda_cr[0], width, label=labels[0], alpha=0.8, color=colors[i],
                         edgecolor='black', linewidth=0.8)
            axes[2,0].bar(x_i[1], nvda_cr[1], width, label=labels[1], alpha=0.8, color=colors[i],
                         edgecolor='black', linewidth=0.8, hatch='xx')

    axes[2,0].set_ylabel('Cumulative Returns (%) $\\uparrow$', fontsize=16)
    axes[2,0].set_xticks(x)
    axes[2,0].set_xticklabels(features_short, fontsize=18, fontweight='bold')
    # axes[2,0].set_xlabel('$\\bf{NVDA}$: Agent Thinking Deeper $\\rightarrow$', fontsize=16, style='italic', labelpad=10)
    axes[2,0].tick_params(axis='both', which='major', labelsize=13)
    axes[2,0].spines['right'].set_visible(False)
    axes[2,0].spines['top'].set_visible(False)
    axes[2,0].spines['left'].set_linewidth(1.2)
    axes[2,0].spines['bottom'].set_linewidth(1.2)
    axes[2,0].grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)

    # Column 1: NVDA SR
    nonzero_count = [0, 0, 0, 0, 0]
    for i, data_agent in enumerate(data_agents):
        nvda_sr = {
            'Analysts': nvda_analysts[data_agent]['SR'] if data_agent in nvda_analysts else 0,
            'Researcher': nvda_researcher[data_agent]['SR'] if data_agent in nvda_researcher else 0,
        }
        nvda_sr = [nvda_sr[feat] for feat in nvda_sr.keys()]
        x_i = [
            x[i] + width * (j - len(labels)//2)
            for j, feat in enumerate(nvda_sr)
        ]

        if data_agent == "All_Data":
            axes[2,1].bar(x_i[0], nvda_sr[0], width, label=labels[0], alpha=0.7, color=colors[i],
                         edgecolor='black', linewidth=0.8, hatch='//')
            axes[2,1].bar(x_i[1], nvda_sr[1], width, label=labels[1], alpha=0.7, color=colors[i],
                         edgecolor='black', linewidth=0.8, hatch='xx')
        else:
            axes[2,1].bar(x_i[0], nvda_sr[0], width, label=labels[0], alpha=0.8, color=colors[i],
                         edgecolor='black', linewidth=0.8)
            axes[2,1].bar(x_i[1], nvda_sr[1], width, label=labels[1], alpha=0.8, color=colors[i],
                         edgecolor='black', linewidth=0.8, hatch='xx')

    axes[2,1].set_ylabel('Sharpe Ratio (%) $\\uparrow$', fontsize=16)
    axes[2,1].set_xticks(x)
    axes[2,1].set_xticklabels(features_short, fontsize=18, fontweight='bold')
    # axes[2,1].set_xlabel('$\\bf{NVDA}$: Agent Thinking Deeper $\\rightarrow$', fontsize=16, style='italic', labelpad=10)
    axes[2,1].tick_params(axis='both', which='major', labelsize=13)
    axes[2,1].spines['right'].set_visible(False)
    axes[2,1].spines['top'].set_visible(False)
    axes[2,1].spines['left'].set_linewidth(1.2)
    axes[2,1].spines['bottom'].set_linewidth(1.2)
    axes[2,1].grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)

    # Column 2: NVDA MDD
    for i, data_agent in enumerate(data_agents):
        nvda_mdd = {
            'Analysts': nvda_analysts[data_agent]['MDD'] if data_agent in nvda_analysts else 0,
            'Researcher': nvda_researcher[data_agent]['MDD'] if data_agent in nvda_researcher else 0,
        }
        nvda_mdd = [nvda_mdd[feat] for feat in nvda_mdd.keys()]
        x_i = [
            x[i] + width * (j - len(labels)//2)
            for j, feat in enumerate(nvda_mdd)
        ]

        if data_agent == "All_Data":
            axes[2,2].bar(x_i[0], nvda_mdd[0], width, label=labels[0], alpha=0.7, color=colors[i],
                         edgecolor='black', linewidth=0.8, hatch='//')
            axes[2,2].bar(x_i[1], nvda_mdd[1], width, label=labels[1], alpha=0.7, color=colors[i],
                         edgecolor='black', linewidth=0.8, hatch='xx')
        else:
            axes[2,2].bar(x_i[0], nvda_mdd[0], width, label=labels[0], alpha=0.8, color=colors[i],
                         edgecolor='black', linewidth=0.8)
            axes[2,2].bar(x_i[1], nvda_mdd[1], width, label=labels[1], alpha=0.8, color=colors[i],
                         edgecolor='black', linewidth=0.8, hatch='xx')

    axes[2,2].set_ylabel('Max Drawdown (%) $\\downarrow$', fontsize=16)
    axes[2,2].set_xticks(x)
    axes[2,2].set_xticklabels(features_short, fontsize=18, fontweight='bold')
    # axes[2,2].set_xlabel('$\\bf{NVDA}$: Agent Thinking Deeper $\\rightarrow$', fontsize=16, style='italic', labelpad=10)
    axes[2,2].tick_params(axis='both', which='major', labelsize=13)
    axes[2,2].spines['right'].set_visible(False)
    axes[2,2].spines['top'].set_visible(False)
    axes[2,2].spines['left'].set_linewidth(1.2)
    axes[2,2].spines['bottom'].set_linewidth(1.2)
    axes[2,2].grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)

    # 1. 获取所有句柄和标签
    handles, labels = axes[0, 0].get_legend_handles_labels()

    # 2. 只保留前两个（对应 LLM 和 Human），并重命名为要求的名称
    display_handles = handles[:2]

    # 3. 创建图例，设置 ncol=2
    fig.legend(display_handles, labels,
               loc='upper center',
               bbox_to_anchor=(0.5, 1.03),
               ncol=2,           # 只保留两列
               frameon=False,
               fontsize=16,
               columnspacing=2.0)

    plt.tight_layout(rect=[0, 0, 1, 0.98]) # 稍微调低顶距给图例留空间
    plt.savefig('ablation_study_analyst.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_mean_research_trader_risk():
    """Plot the mean of researcher, trader, and risk across all companies"""

    # Set style
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(25, 5))

    width = 0.12

    features_short = ['Analysts', 'Researcher', 'Trader', 'Risk Manager']
    data_agents = ["Fundamental_Only", "Market_Only", "News_Only", "Sentiment_Only", "All_Data"]
    colors = ["#4DBBD5", "#00A087", "#E64B35", "#F39B7F", "#3C5488"]


    x = np.arange(len(features_short))

    # Plot 1: Mean CR across all companies
    for i, data_agent in enumerate(data_agents):
        # Calculate mean across NVDA, AAPL, GOOG
        mean_cr = {
            'Analysts': np.mean([
                nvda_analysts[data_agent]['CR'] if data_agent in nvda_analysts else 0,
                aapl_analysts[data_agent]['CR'] if data_agent in aapl_analysts else 0,
                goog_analysts[data_agent]['CR'] if data_agent in goog_analysts else 0
            ]),
            'Researcher': np.mean([
                nvda_researcher[data_agent]['CR'] if data_agent in nvda_researcher else 0,
                aapl_researcher[data_agent]['CR'] if data_agent in aapl_researcher else 0,
                goog_researcher[data_agent]['CR'] if data_agent in goog_researcher else 0
            ]),
            'Trader': np.mean([
                nvda_trader[data_agent]['CR'] if data_agent in nvda_trader else 0,
                aapl_trader[data_agent]['CR'] if data_agent in aapl_trader else 0,
                goog_trader[data_agent]['CR'] if data_agent in goog_trader else 0
            ]),
            'Risk': np.mean([
                nvda_data[data_agent]['CR'] if data_agent in nvda_data else 0,
                aapl_data[data_agent]['CR'] if data_agent in aapl_data else 0,
                goog_data[data_agent]['CR'] if data_agent in goog_data else 0
            ]),
        }

        mean_cr_values = [mean_cr[feat] for feat in mean_cr.keys()]
        i_ = i - len(features_short)//2

        # Use different styling for All_Data to show it contains all others
        if data_agent == "All_Data":
            # Hatched pattern with thicker edge to show it's the combination
            axes[0].bar(x + width * i_, mean_cr_values, width, label=data_agent, alpha=0.7, color=colors[i],
                       edgecolor='black', linewidth=0.8, hatch='//')
        else:
            axes[0].bar(x + width * i_, mean_cr_values, width, label=data_agent, alpha=0.8, color=colors[i],
                       edgecolor='black', linewidth=0.8)

    axes[0].set_title('Average Cumulative Returns (CR) % $\\uparrow$', fontsize=16, fontweight='normal')
    axes[0].set_ylabel('Cumulative Returns (%) $\\uparrow$', fontsize=16, fontweight='normal')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(features_short, fontsize=18, fontweight='bold')
    axes[0].set_xlabel('Agent Thinking Deeper $\\rightarrow$', fontsize=16, fontweight='normal', style='italic', labelpad=10)
    axes[0].tick_params(axis='both', which='major', labelsize=13)
    axes[0].spines['right'].set_visible(False)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['left'].set_linewidth(1.2)
    axes[0].spines['bottom'].set_linewidth(1.2)
    axes[0].grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)

    # Plot 2: Mean SR across all companies
    for i, data_agent in enumerate(data_agents):
        mean_sr = {
            'Analysts': np.mean([
                nvda_analysts[data_agent]['SR'] if data_agent in nvda_analysts else 0,
                aapl_analysts[data_agent]['SR'] if data_agent in aapl_analysts else 0,
                goog_analysts[data_agent]['SR'] if data_agent in goog_analysts else 0
            ]),
            'Researcher': np.mean([
                nvda_researcher[data_agent]['SR'] if data_agent in nvda_researcher else 0,
                aapl_researcher[data_agent]['SR'] if data_agent in aapl_researcher else 0,
                goog_researcher[data_agent]['SR'] if data_agent in goog_researcher else 0
            ]),
            'Trader': np.mean([
                nvda_trader[data_agent]['SR'] if data_agent in nvda_trader else 0,
                aapl_trader[data_agent]['SR'] if data_agent in aapl_trader else 0,
                goog_trader[data_agent]['SR'] if data_agent in goog_trader else 0
            ]),
            'Risk': np.mean([
                nvda_data[data_agent]['SR'] if data_agent in nvda_data else 0,
                aapl_data[data_agent]['SR'] if data_agent in aapl_data else 0,
                goog_data[data_agent]['SR'] if data_agent in goog_data else 0
            ]),
        }

        mean_sr_values = [mean_sr[feat] for feat in mean_sr.keys()]
        i_ = i - len(features_short)//2

        # Use different styling for All_Data to show it contains all others
        if data_agent == "All_Data":
            axes[1].bar(x + width * i_, mean_sr_values, width, label=data_agent, alpha=0.7, color=colors[i],
                       edgecolor='black', linewidth=0.8, hatch='//')
        else:
            axes[1].bar(x + width * i_, mean_sr_values, width, label=data_agent, alpha=0.8, color=colors[i],
                       edgecolor='black', linewidth=0.8)

    axes[1].set_title('Average Sharpe Ratio (SR) % $\\uparrow$', fontsize=16, fontweight='normal')
    axes[1].set_ylabel('Sharpe Ratio (%) $\\uparrow$', fontsize=16, fontweight='normal')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(features_short, fontsize=18, fontweight='bold')
    axes[1].set_xlabel('Agent Thinking Deeper $\\rightarrow$', fontsize=16, fontweight='normal', style='italic', labelpad=10)
    axes[1].tick_params(axis='both', which='major', labelsize=13)
    axes[1].spines['right'].set_visible(False)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['left'].set_linewidth(1.2)
    axes[1].spines['bottom'].set_linewidth(1.2)
    axes[1].grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)

    # Plot 3: Mean MDD across all companies
    for i, data_agent in enumerate(data_agents):
        mean_mdd = {
            'Analysts': np.mean([
                nvda_analysts[data_agent]['MDD'] if data_agent in nvda_analysts else 0,
                aapl_analysts[data_agent]['MDD'] if data_agent in aapl_analysts else 0,
                goog_analysts[data_agent]['MDD'] if data_agent in goog_analysts else 0
            ]),
            'Researcher': np.mean([
                nvda_researcher[data_agent]['MDD'] if data_agent in nvda_researcher else 0,
                aapl_researcher[data_agent]['MDD'] if data_agent in aapl_researcher else 0,
                goog_researcher[data_agent]['MDD'] if data_agent in goog_researcher else 0
            ]),
            'Trader': np.mean([
                nvda_trader[data_agent]['MDD'] if data_agent in nvda_trader else 0,
                aapl_trader[data_agent]['MDD'] if data_agent in aapl_trader else 0,
                goog_trader[data_agent]['MDD'] if data_agent in goog_trader else 0
            ]),
            'Risk': np.mean([
                nvda_data[data_agent]['MDD'] if data_agent in nvda_data else 0,
                aapl_data[data_agent]['MDD'] if data_agent in aapl_data else 0,
                goog_data[data_agent]['MDD'] if data_agent in goog_data else 0
            ]),
        }

        mean_mdd_values = [mean_mdd[feat] for feat in mean_mdd.keys()]
        i_ = i - len(features_short)//2

        # Use different styling for All_Data to show it contains all others
        if data_agent == "All_Data":
            axes[2].bar(x + width * i_, mean_mdd_values, width, label=data_agent, alpha=0.7, color=colors[i],
                       edgecolor='black', linewidth=0.8, hatch='//')
        else:
            axes[2].bar(x + width * i_, mean_mdd_values, width, label=data_agent, alpha=0.8, color=colors[i],
                       edgecolor='black', linewidth=0.8)

    axes[2].set_title('Average Max Drawdown (MDD) % $\\downarrow$', fontsize=16, fontweight='normal')
    axes[2].set_ylabel('Max Drawdown (%) $\\downarrow$', fontsize=16, fontweight='normal')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(features_short, fontsize=18, fontweight='bold')
    axes[2].set_xlabel('Agent Thinking Deeper $\\rightarrow$', fontsize=16, fontweight='normal', style='italic', labelpad=10)
    axes[2].tick_params(axis='both', which='major', labelsize=13)
    axes[2].spines['right'].set_visible(False)
    axes[2].spines['top'].set_visible(False)
    axes[2].spines['left'].set_linewidth(1.2)
    axes[2].spines['bottom'].set_linewidth(1.2)
    axes[2].grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)

    # Create shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    # Clean up labels for better readability
    clean_labels = [label.replace('_', ' ') for label in labels]
    fig.legend(handles, clean_labels, loc='upper center', bbox_to_anchor=(0.5, 1.03),
               ncol=5, frameon=False, fontsize=16, columnspacing=1.5)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig('ablation_study_plots_research_trader_risk_mean.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_mean_analyst_researcher():
    """Plot the mean of analyst features across all companies"""

    # Set style
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(22, 4))

    width = 0.25

    features_short = ["Fund. Only", "Market Only", "News Only", "Senti. Only", "All Data"]
    data_agents = ["Fundamental_Only", "Market_Only", "News_Only", "Sentiment_Only", "All_Data"]
    colors = ["#4DBBD5", "#00A087", "#E64B35", "#F39B7F", "#3C5488"]
    labels = ["LLM Analysts", "Human Analysts (pseudo)"]

    x = np.arange(len(data_agents))

    # Plot 1: Mean CR across all companies
    for i, data_agent in enumerate(data_agents):
        # Calculate mean across NVDA, AAPL, GOOG for each feature
        mean_cr = {
            'Analysts': np.mean([
                nvda_analysts[data_agent]['CR'] if data_agent in nvda_analysts else 0,
                aapl_analysts[data_agent]['CR'] if data_agent in aapl_analysts else 0,
                goog_analysts[data_agent]['CR'] if data_agent in goog_analysts else 0
            ]),
            'Researcher': np.mean([
                nvda_researcher[data_agent]['CR'] if data_agent in nvda_researcher else 0,
                aapl_researcher[data_agent]['CR'] if data_agent in aapl_researcher else 0,
                goog_researcher[data_agent]['CR'] if data_agent in goog_researcher else 0
            ]),
        }

        mean_cr_values = [mean_cr[feat] for feat in mean_cr.keys()]
        x_i = [
            x[i] + width * (j - len(mean_cr_values) // 2 + 0.5)
            for j, feat in enumerate(mean_cr_values)
        ]

        if data_agent == "All_Data":
            axes[0].bar(x_i[0], mean_cr_values[0], width, label=labels[0], alpha=0.7, color=colors[i],
                       edgecolor='black', linewidth=0.8, hatch='//')
            axes[0].bar(x_i[1], mean_cr_values[1], width, label=labels[1], alpha=0.7, color=colors[i],
                       edgecolor='black', linewidth=0.8, hatch='xx')
        else:
            axes[0].bar(x_i[0], mean_cr_values[0], width, label=labels[0], alpha=0.8, color=colors[i],
                       edgecolor='black', linewidth=0.8)
            axes[0].bar(x_i[1], mean_cr_values[1], width, label=labels[1], alpha=0.8, color=colors[i],
                       edgecolor='black', linewidth=0.8, hatch='xx')

    axes[0].set_title('Average Cumulative Returns (CR) % $\\uparrow$', fontsize=16, fontweight='normal')
    axes[0].set_ylabel('Cumulative Returns (%) $\\uparrow$', fontsize=16, fontweight='normal')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(features_short, fontsize=16, fontweight='bold')
    # axes[0].set_xlabel('Agent Thinking Deeper $\\rightarrow$', fontsize=16, fontweight='normal', style='italic', labelpad=10)
    axes[0].tick_params(axis='both', which='major', labelsize=13)
    axes[0].spines['right'].set_visible(False)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['left'].set_linewidth(1.2)
    axes[0].spines['bottom'].set_linewidth(1.2)
    axes[0].grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)

    # Plot 2: Mean SR across all companies
    for i, data_agent in enumerate(data_agents):
        mean_sr = {
            'Analysts': np.mean([
                nvda_analysts[data_agent]['SR'] if data_agent in nvda_analysts else 0,
                aapl_analysts[data_agent]['SR'] if data_agent in aapl_analysts else 0,
                goog_analysts[data_agent]['SR'] if data_agent in goog_analysts else 0
            ]),
            'Researcher': np.mean([
                nvda_researcher[data_agent]['SR'] if data_agent in nvda_researcher else 0,
                aapl_researcher[data_agent]['SR'] if data_agent in aapl_researcher else 0,
                goog_researcher[data_agent]['SR'] if data_agent in goog_researcher else 0
            ]),
        }

        mean_sr_values = [mean_sr[feat] for feat in mean_sr.keys()]
        x_i = [
            x[i] + width * (j - len(mean_sr_values) // 2 + 0.5)
            for j, feat in enumerate(mean_sr_values)
        ]

        if data_agent == "All_Data":
            axes[1].bar(x_i[0], mean_sr_values[0], width, label=labels[0], alpha=0.7, color=colors[i],
                       edgecolor='black', linewidth=0.8, hatch='//')
            axes[1].bar(x_i[1], mean_sr_values[1], width, label=labels[1], alpha=0.7, color=colors[i],
                       edgecolor='black', linewidth=0.8, hatch='xx')

        else:
            axes[1].bar(x_i[0], mean_sr_values[0], width, label=labels[0], alpha=0.8, color=colors[i],
                       edgecolor='black', linewidth=0.8)
            axes[1].bar(x_i[1], mean_sr_values[1], width, label=labels[1], alpha=0.8, color=colors[i],
                       edgecolor='black', linewidth=0.8, hatch='xx')

    axes[1].set_title('Average Sharpe Ratio (SR) % $\\uparrow$', fontsize=16, fontweight='normal')
    axes[1].set_ylabel('Sharpe Ratio (%) $\\uparrow$', fontsize=16, fontweight='normal')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(features_short, fontsize=16, fontweight='bold')
    # axes[1].set_xlabel('Agent Thinking Deeper $\\rightarrow$', fontsize=16, fontweight='normal', style='italic', labelpad=10)
    axes[1].tick_params(axis='both', which='major', labelsize=13)
    axes[1].spines['right'].set_visible(False)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['left'].set_linewidth(1.2)
    axes[1].spines['bottom'].set_linewidth(1.2)
    axes[1].grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)

    # Plot 3: Mean MDD across all companies
    for i, data_agent in enumerate(data_agents):
        mean_mdd = {
            'Analysts': np.mean([
                nvda_analysts[data_agent]['MDD'] if data_agent in nvda_analysts else 0,
                aapl_analysts[data_agent]['MDD'] if data_agent in aapl_analysts else 0,
                goog_analysts[data_agent]['MDD'] if data_agent in goog_analysts else 0
            ]),
            'Researcher': np.mean([
                nvda_researcher[data_agent]['MDD'] if data_agent in nvda_researcher else 0,
                aapl_researcher[data_agent]['MDD'] if data_agent in aapl_researcher else 0,
                goog_researcher[data_agent]['MDD'] if data_agent in goog_researcher else 0
            ]),
        }

        mean_mdd_values = [mean_mdd[feat] for feat in mean_mdd.keys()]
        x_i = [
            x[i] + width * (j - len(mean_mdd_values) // 2 + 0.5)
            for j, feat in enumerate(mean_mdd_values)
        ]

        if data_agent == "All_Data":
            axes[2].bar(x_i[0], mean_mdd_values[0], width, label=labels[0], alpha=0.7, color=colors[i],
                       edgecolor='black', linewidth=0.8, hatch='//')
            axes[2].bar(x_i[1], mean_mdd_values[1], width, label=labels[1], alpha=0.7, color=colors[i],
                       edgecolor='black', linewidth=0.8, hatch='xx')
        else:
            axes[2].bar(x_i[0], mean_mdd_values[0], width, label=labels[0], alpha=0.8, color=colors[i],
                       edgecolor='black', linewidth=0.8)
            axes[2].bar(x_i[1], mean_mdd_values[1], width, label=labels[1], alpha=0.8, color=colors[i],
                       edgecolor='black', linewidth=0.8, hatch='xx')

    axes[2].set_title('Average Max Drawdown (MDD) % $\\downarrow$', fontsize=16, fontweight='normal')
    axes[2].set_ylabel('Max Drawdown (%) $\\downarrow$', fontsize=16, fontweight='normal')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(
        features_short,
        fontsize=16,
        fontweight='bold',
        # rotation=75,
        # ha='right'
    )
    # axes[2].set_xlabel('Agent Thinking Deeper $\\rightarrow$', fontsize=16, fontweight='normal', style='italic', labelpad=10)
    axes[2].tick_params(axis='both', which='major', labelsize=13)
    axes[2].spines['right'].set_visible(False)
    axes[2].spines['top'].set_visible(False)
    axes[2].spines['left'].set_linewidth(1.2)
    axes[2].spines['bottom'].set_linewidth(1.2)
    axes[2].grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)

    # 1. 获取所有句柄和标签
    handles, labels = axes[0].get_legend_handles_labels()

    # 2. 只保留前两个（对应 LLM 和 Human），并重命名为要求的名称
    display_handles = handles[:2]

    # 3. 创建图例，设置 ncol=2
    fig.legend(display_handles, labels,
               loc='upper center',
               bbox_to_anchor=(0.5, 1.03),
               ncol=2,           # 只保留两列
               frameon=False,
               fontsize=16,
               columnspacing=2.0)

    plt.tight_layout(rect=[0, 0, 1, 0.92]) # 稍微调低顶距给图例留空间
    plt.savefig('ablation_study_analyst_mean.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_performance_heatmap():
    """Create a heatmap of performance metrics"""

    # Prepare data for heatmap
    metrics = ['CR', 'ARR', 'SR', 'MDD']
    configurations = ['All Features', 'Fundamental', 'Market', 'News', 'Sentiment']

    # NVDA data matrix
    nvda_matrix = []
    for config in ['All_Data', 'Fundamental_Only', 'Market_Only', 'News_Only', 'Sentiment_Only']:
        row = [nvda_data[config][metric] for metric in metrics]
        nvda_matrix.append(row)

    # AAPL data matrix
    aapl_matrix = []
    for config in ['All_Data', 'Fundamental_Only', 'Market_Only', 'News_Only', 'Sentiment_Only']:
        row = [aapl_data[config][metric] for metric in metrics]
        aapl_matrix.append(row)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # NVDA heatmap
    sns.heatmap(nvda_matrix,
                xticklabels=metrics,
                yticklabels=configurations,
                annot=True,
                fmt='.1f',
                cmap='RdYlGn',
                center=0,
                ax=ax1)
    ax1.set_title('NVDA Performance Metrics', fontsize=14)

    # AAPL heatmap
    sns.heatmap(aapl_matrix,
                xticklabels=metrics,
                yticklabels=configurations,
                annot=True,
                fmt='.1f',
                cmap='RdYlGn',
                center=0,
                ax=ax2)
    ax2.set_title('AAPL Performance Metrics', fontsize=14)

    plt.tight_layout()
    # plt.show()
    plt.savefig('performance_heatmap.png')



if __name__ == "__main__":
    # Generate plots
    print("Generating ablation study plots...")

    plot_ablation_study_research_trader_risk()

    plot_ablation_study_analyst_researcher()

    # Mean research trader risk plot
    plot_mean_research_trader_risk()

    # Mean analyst researcher plot
    plot_mean_analyst_researcher()

    # Performance heatmap
    plot_performance_heatmap()

    print("Plots generated successfully!")

    # Figure 1:
    #   plot_ablation_study_research_trader_risk()
    #   Performance comparison of trading agents with
    #   progressive complexity across three stocks (AAPL,
    #    GOOG, NVDA). Each row shows results for one
    #   stock, with columns displaying Cumulative Returns
    #    (CR), Sharpe Ratio (SR), and Maximum Drawdown
    #   (MDD). Bars represent different agent
    #   configurations: All_Data (hatched, combining all
    #   data sources), Researcher (analyst integration),
    #   Trader (trading execution), and Risk Manager
    #   (risk management). The x-axis progression
    #   indicates increasing agent thinking depth from
    #   left to right.

    #   Figure 2: plot_mean_research_trader_risk()
    #   Averaged performance metrics across all three
    #   stocks for different agent configurations.
    #   Results are aggregated from NVDA, AAPL, and GOOG
    #   to show general trends. The hatched All_Data bars
    #    represent the baseline using all data sources,
    #   while subsequent bars show incremental addition
    #   of reasoning modules (Researcher, Trader, Risk
    #   Manager), demonstrating the impact of agent
    #   thinking depth on portfolio performance.

    #   Figure 3:
    #   plot_ablation_study_analyst_researcher()
    #   Ablation study of data source impact on trading
    #   agent performance across three stocks (AAPL,
    #   GOOG, NVDA). Each row shows one stock's results
    #   across CR, SR, and MDD metrics. Bars represent
    #   different data configurations: All_Data (hatched,
    #    full data), Fund (fundamental data only), Market
    #    (market data only), News (news data only), Sent
    #   (sentiment data only), and Researcher (integrated
    #    analysis). This ablation reveals the relative
    #   importance of each data source type.

    #   Figure 4: plot_mean_analyst_researcher()
    #   Averaged ablation study results across all three
    #   stocks showing the impact of different data
    #   sources on trading performance. Metrics are
    #   aggregated from NVDA, AAPL, and GOOG. The hatched
    #    All_Data configuration serves as the baseline,
    #   while individual data sources (Fund, Market,
    #   News, Sent) and the Researcher module demonstrate
    #    their isolated contributions to overall
    #   performance.