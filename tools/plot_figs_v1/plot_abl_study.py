import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data from Excel file
excel_file = 'TA_abl_analysts.xlsx'
df = pd.read_excel(excel_file, sheet_name='NVDA')

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

def plot_ablation_study_research_trader_risk():
    """Plot the ablation study results"""

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(20, 6.7))
    # fig.suptitle('Trading Agents Ablation Study Results', fontsize=16, fontweight='bold')

    width = 0.15

    # Plot 5: Fund, Market, News, Sentiment, Researcher , Trader, Final on CR
    features_short = ['Researcher', 'Trader', 'Risk']
    data_agents = ["All_Data", "Fundamental_Only", "Market_Only", "News_Only", "Sentiment_Only"]
    # colors = ["#326b77", "#568b87", "#80ae9a", "#122740", "#1b485e"]
    colors = ["#2066a8", "#0d7d87", "#f47a00", "#ae282c", "#800074"]

    x = np.arange(len(features_short))
    for i,data_agent in enumerate(data_agents):
        nvda_cr = {
            'Researcher': nvda_researcher[data_agent]['CR'] if data_agent in nvda_researcher else 0,
            'Trader': nvda_trader[data_agent]['CR'] if data_agent in nvda_trader else 0,
            'Risk': nvda_data[data_agent]['CR'] if data_agent in nvda_data else 0,
        }

        nvda_cr = [nvda_cr[feat] for feat in nvda_cr.keys()]
        i_ = i - len(features_short)//2 - 1
        axes[0,0].bar(x + width * i_, nvda_cr, width, label=data_agent, alpha=0.8, color=colors[i])

    axes[0,0].set_title('NVDA: Agent decisions on CR ')
    # axes[2,0].set_xlabel('Agent Name')
    axes[0,0].set_ylabel('Cumulative Returns ')
    axes[0,0].set_xticks(x)
    axes[0,0].set_xticklabels(features_short)
    axes[0,0].legend(
        loc='upper center',
        # bbox_to_anchor=(0.5, 1.45),
        ncol=3,
        frameon=False
    )
    # axes[0,0].grid(True, alpha=0.3)
    axes[0,0].set_ylim(0, 70)

    for i,data_agent in enumerate(data_agents):
        aapl_cr = {
            'Researcher': aapl_researcher[data_agent]['CR'] if data_agent in aapl_researcher else 0,
            'Trader': aapl_trader[data_agent]['CR'] if data_agent in aapl_trader else 0,
            'Risk': aapl_data[data_agent]['CR'] if data_agent in aapl_data else 0,
        }
        aapl_cr = [aapl_cr[feat] for feat in aapl_cr.keys()]
        i_ = i - len(features_short)//2 - 1
        axes[1,0].bar(x + width * i_, aapl_cr, width, label=data_agent, alpha=0.8, color=colors[i])

    axes[1,0].set_title('AAPL: Agent decisions on CR ')
    # axes[3,0].set_xlabel('Agent Name')
    axes[1,0].set_ylabel('Cumulative Returns ')
    axes[1,0].set_xticks(x)
    axes[1,0].set_xticklabels(features_short)
    axes[1,0].legend(
        loc='upper center',
        # bbox_to_anchor=(0.5, 1.45),
        ncol=3,
        frameon=False
    )
    # axes[1,0].grid(True, alpha=0.3)
    axes[1,0].set_ylim(-8, 5)

    # Plot 6: Fund, Market, News, Sentiment, Researcher , Trader, Final on Sharpe Ratio
    for i,data_agent in enumerate(data_agents):
        nvda_sr = {
            'Researcher': nvda_researcher[data_agent]['SR'] if data_agent in nvda_researcher else 0,
            'Trader': nvda_trader[data_agent]['SR'] if data_agent in nvda_trader else 0,
            'Risk': nvda_data[data_agent]['SR'] if data_agent in nvda_data else 0,
        }
        nvda_sr = [nvda_sr[feat] for feat in nvda_sr.keys()]
        i_ = i - len(features_short)//2 - 1
        axes[0,1].bar(x + width * i_, nvda_sr, width, label=data_agent, alpha=0.8, color=colors[i])

    axes[0,1].set_title('NVDA: Agent decisions on SR ')
    # axes[2,1].set_xlabel('Agent Name')
    axes[0,1].set_ylabel('Sharpe Ratio')
    axes[0,1].set_xticks(x)
    axes[0,1].set_xticklabels(features_short)
    axes[0,1].legend(
        loc='upper center',
        # bbox_to_anchor=(0.5, 1.45),
        ncol=3,
        frameon=False
    )
    # axes[0,1].grid(True, alpha=0.3)
    axes[0,1].set_ylim(0, 6)

    for i,data_agent in enumerate(data_agents):
        aapl_sr = {
            'Researcher': aapl_researcher[data_agent]['SR'] if data_agent in aapl_researcher else 0,
            'Trader': aapl_trader[data_agent]['SR'] if data_agent in aapl_trader else 0,
            'Risk': aapl_data[data_agent]['SR'] if data_agent in aapl_data else 0,
        }
        aapl_sr = [aapl_sr[feat] for feat in aapl_sr.keys()]
        i_ = i - len(features_short)//2 - 1
        axes[1,1].bar(x + width * i_, aapl_sr, width, label=data_agent, alpha=0.8, color=colors[i])

    axes[1,1].set_title('AAPL: Agent decisions on SR ')
    # axes[3,1].set_xlabel('Agent Name')
    axes[1,1].set_ylabel('Sharpe Ratio')
    axes[1,1].set_xticks(x)
    axes[1,1].set_xticklabels(features_short)
    axes[1,1].legend(
        loc='upper center',
        # bbox_to_anchor=(0.5, 1.45),
        ncol=3,
        frameon=False
    )
    # axes[1,1].grid(True, alpha=0.3)
    axes[1,1].set_ylim(-2, 3)

    # Plot 7: Fund, Market, News, Sentiment, Researcher , Trader, Final on Max Drawdown
    for i,data_agent in enumerate(data_agents):
        nvda_mdd = {
            'Researcher': nvda_researcher[data_agent]['MDD'] if data_agent in nvda_researcher else 0,
            'Trader': nvda_trader[data_agent]['MDD'] if data_agent in nvda_trader else 0,
            'Risk': nvda_data[data_agent]['MDD'] if data_agent in nvda_data else 0,
        }
        nvda_mdd = [nvda_mdd[feat] for feat in nvda_mdd.keys()]
        i_ = i - len(features_short)//2 - 1
        axes[0,2].bar(x + width * i_, nvda_mdd, width, label=data_agent, alpha=0.8, color=colors[i])

    axes[0,2].set_title('NVDA: Agent decisions on MDD ')
    # axes[2,2].set_xlabel('Agent Name')
    axes[0,2].set_ylabel('Max Drawdown ')
    axes[0,2].set_xticks(x)
    axes[0,2].set_xticklabels(features_short)
    axes[0,2].legend(
        loc='upper center',
        # bbox_to_anchor=(0.5, 1.45),
        ncol=3,
        frameon=False
    )
    # axes[0,2].grid(True, alpha=0.3)
    axes[0,2].set_ylim(0, 11)

    for i,data_agent in enumerate(data_agents):
        aapl_mdd = {
            'Researcher': aapl_researcher[data_agent]['MDD'] if data_agent in aapl_researcher else 0,
            'Trader': aapl_trader[data_agent]['MDD'] if data_agent in aapl_trader else 0,
            'Risk': aapl_data[data_agent]['MDD'] if data_agent in aapl_data else 0,
        }
        aapl_mdd = [aapl_mdd[feat] for feat in aapl_mdd.keys()]
        i_ = i - len(features_short)//2 - 1
        axes[1,2].bar(x + width * i_, aapl_mdd, width, label=data_agent, alpha=0.8, color=colors[i])

    axes[1,2].set_title('AAPL: Agent decisions on MDD ')
    # axes[2,2].set_xlabel('Agent Name')
    axes[1,2].set_ylabel('Max Drawdown ')
    axes[1,2].set_xticks(x)
    axes[1,2].set_xticklabels(features_short)
    axes[1,2].legend(
        loc='upper center',
        # bbox_to_anchor=(0.5, 1.45),
        ncol=3,
        frameon=False
    )
    # axes[1,2].grid(True, alpha=0.3)
    axes[1,2].set_ylim(0, 17)

    plt.tight_layout()
    # plt.show()
    plt.savefig('ablation_study_plots_research_trader_risk.png')
    plt.close()


def plot_ablation_study_analyst_researcher():
    """Plot the ablation study results"""

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(20, 6.7))
    # fig.suptitle('Trading Agents Ablation Study Results', fontsize=16, fontweight='bold')

    width = 0.15

    data_agents = ["All_Data", "Fundamental_Only", "Market_Only", "News_Only", "Sentiment_Only"]
    colors = ["#2066a8", "#0d7d87", "#f47a00", "#ae282c", "#800074"]

    # Plot 5: Fund, Market, News, Sentiment, Researcher , Trader, Final on CR
    features_short = ['Fund', 'Market', 'News', 'Sent', 'Researcher']
    x = np.arange(len(features_short))

    nonzero_count = [0, 0, 0, 0, 0]
    for i,data_agent in enumerate(["All_Data", "Fundamental_Only", "Market_Only", "News_Only", "Sentiment_Only"]):
        nvda_cr = {

            'Fund': nvda_fundamentals[data_agent]['CR'] if data_agent in nvda_fundamentals else 0,
            'Market': nvda_market[data_agent]['CR'] if data_agent in nvda_market else 0,
            'News': nvda_news[data_agent]['CR'] if data_agent in nvda_news else 0,
            'Sentiment': nvda_sentiment[data_agent]['CR'] if data_agent in nvda_sentiment else 0,
            'Researcher': nvda_researcher[data_agent]['CR'] if data_agent in nvda_researcher else 0,
        }

        nvda_cr = [nvda_cr[feat] for feat in nvda_cr.keys()]
        x_i = [
            x[j] + width * (nonzero_count[j] - len(features_short)//2) if j == 4
            else x[j] + width * (nonzero_count[j] - 0.5)
            for j,feat in enumerate(nvda_cr)
        ]
        nonzero_count = [nonzero_count[j] + 1 if feat != 0 else nonzero_count[j] for j,feat in enumerate(nvda_cr)]
        axes[0,0].bar(x_i, nvda_cr, width, label=data_agent, alpha=0.8, color=colors[i])

    axes[0,0].set_title('NVDA: Agent decisions on CR ')
    # axes[2,0].set_xlabel('Agent Name')
    axes[0,0].set_ylabel('Cumulative Returns ')
    axes[0,0].set_xticks(x)
    axes[0,0].set_xticklabels(features_short)
    axes[0,0].legend(
        loc='upper center',
        # bbox_to_anchor=(0.5, 1.55),
        ncol=3,
        frameon=False
    )
    # axes[0,0].grid(True, alpha=0.3)
    axes[0,0].set_ylim(0, 140)

    nonzero_count = [0, 0, 0, 0, 0]
    for i,data_agent in enumerate(["All_Data", "Fundamental_Only", "Market_Only", "News_Only", "Sentiment_Only"]):
        aapl_cr = {
            'Fund': aapl_fundamentals[data_agent]['CR'] if data_agent in aapl_fundamentals else 0,
            'Market': aapl_market[data_agent]['CR'] if data_agent in aapl_market else 0,
            'News': aapl_news[data_agent]['CR'] if data_agent in aapl_news else 0,
            'Sentiment': aapl_sentiment[data_agent]['CR'] if data_agent in aapl_sentiment else 0,
            'Researcher': aapl_researcher[data_agent]['CR'] if data_agent in aapl_researcher else 0,
        }
        aapl_cr = [aapl_cr[feat] for feat in aapl_cr.keys()]
        x_i = [
            x[j] + width * (nonzero_count[j] - len(features_short)//2) if j == 4
            else x[j] + width * (nonzero_count[j] - 0.5)
            for j,feat in enumerate(aapl_cr)
        ]
        nonzero_count = [nonzero_count[j] + 1 if feat != 0 else nonzero_count[j] for j,feat in enumerate(aapl_cr)]
        axes[1,0].bar(x_i, aapl_cr, width, label=data_agent, alpha=0.8, color=colors[i])

    axes[1,0].set_title('AAPL: Agent decisions on CR ')
    # axes[3,0].set_xlabel('Agent Name')
    axes[1,0].set_ylabel('Cumulative Returns ')
    axes[1,0].set_xticks(x)
    axes[1,0].set_xticklabels(features_short)
    axes[1,0].legend(
        loc='upper center',
        # bbox_to_anchor=(0.5, 1.55),
        ncol=3,
        frameon=False
    )
    # axes[1,0].grid(True, alpha=0.3)
    axes[1,0].set_ylim(-13, 7)

    # Plot 6: Fund, Market, News, Sentiment, Researcher , Trader, Final on Sharpe Ratio
    nonzero_count = [0, 0, 0, 0, 0]
    for i,data_agent in enumerate(["All_Data", "Fundamental_Only", "Market_Only", "News_Only", "Sentiment_Only"]):
        nvda_sr = {

            'Fund': nvda_fundamentals[data_agent]['SR'] if data_agent in nvda_fundamentals else 0,
            'Market': nvda_market[data_agent]['SR'] if data_agent in nvda_market else 0,
            'News': nvda_news[data_agent]['SR'] if data_agent in nvda_news else 0,
            'Sentiment': nvda_sentiment[data_agent]['SR'] if data_agent in nvda_sentiment else 0,
            'Researcher': nvda_researcher[data_agent]['SR'] if data_agent in nvda_researcher else 0,
        }
        nvda_sr = [nvda_sr[feat] for feat in nvda_sr.keys()]
        x_i = [
            x[j] + width * (nonzero_count[j] - len(features_short)//2) if j == 4
            else x[j] + width * (nonzero_count[j] - 0.5)
            for j,feat in enumerate(nvda_sr)
        ]
        nonzero_count = [nonzero_count[j] + 1 if feat != 0 else nonzero_count[j] for j,feat in enumerate(nvda_sr)]
        axes[0,1].bar(x_i, nvda_sr, width, label=data_agent, alpha=0.8, color=colors[i])

    axes[0,1].set_title('NVDA: Agent decisions on SR ')
    # axes[2,1].set_xlabel('Agent Name')
    axes[0,1].set_ylabel('Sharpe Ratio')
    axes[0,1].set_xticks(x)
    axes[0,1].set_xticklabels(features_short)
    axes[0,1].legend(
        loc='upper center',
        # bbox_to_anchor=(0.5, 1.55),
        ncol=3,
        frameon=False
    )
    # axes[0,1].grid(True, alpha=0.3)
    axes[0,1].set_ylim(0, 11)

    nonzero_count = [0, 0, 0, 0, 0]
    for i,data_agent in enumerate(["All_Data", "Fundamental_Only", "Market_Only", "News_Only", "Sentiment_Only"]):
        aapl_sr = {
            'Fund': aapl_fundamentals[data_agent]['SR'] if data_agent in aapl_fundamentals else 0,
            'Market': aapl_market[data_agent]['SR'] if data_agent in aapl_market else 0,
            'News': aapl_news[data_agent]['SR'] if data_agent in aapl_news else 0,
            'Sentiment': aapl_sentiment[data_agent]['SR'] if data_agent in aapl_sentiment else 0,
            'Researcher': aapl_researcher[data_agent]['SR'] if data_agent in aapl_researcher else 0,
        }
        aapl_sr = [aapl_sr[feat] for feat in aapl_sr.keys()]
        x_i = [
            x[j] + width * (nonzero_count[j] - len(features_short)//2) if j == 4
            else x[j] + width * (nonzero_count[j] - 0.5)
            for j,feat in enumerate(aapl_sr)
        ]
        nonzero_count = [nonzero_count[j] + 1 if feat != 0 else nonzero_count[j] for j,feat in enumerate(aapl_sr)]
        axes[1,1].bar(x_i, aapl_sr, width, label=data_agent, alpha=0.8, color=colors[i])

    axes[1,1].set_title('AAPL: Agent decisions on SR ')
    # axes[3,1].set_xlabel('Agent Name')
    axes[1,1].set_ylabel('Sharpe Ratio')
    axes[1,1].set_xticks(x)
    axes[1,1].set_xticklabels(features_short)
    axes[1,1].legend(
        loc='upper center',
        # bbox_to_anchor=(0.5, 1.55),
        ncol=3,
        frameon=False
    )
    # axes[1,1].grid(True, alpha=0.3)
    axes[1,1].set_ylim(-5, 6)

    # Plot 7: Fund, Market, News, Sentiment, Researcher , Trader, Final on Max Drawdown
    nonzero_count = [0, 0, 0, 0, 0]
    for i,data_agent in enumerate(["All_Data", "Fundamental_Only", "Market_Only", "News_Only", "Sentiment_Only"]):
        nvda_mdd = {
            'Fund': nvda_fundamentals[data_agent]['MDD'] if data_agent in nvda_fundamentals else 0,
            'Market': nvda_market[data_agent]['MDD'] if data_agent in nvda_market else 0,
            'News': nvda_news[data_agent]['MDD'] if data_agent in nvda_news else 0,
            'Sentiment': nvda_sentiment[data_agent]['MDD'] if data_agent in nvda_sentiment else 0,
            'Researcher': nvda_researcher[data_agent]['MDD'] if data_agent in nvda_researcher else 0,
        }
        nvda_mdd = [nvda_mdd[feat] for feat in nvda_mdd.keys()]
        x_i = [
            x[j] + width * (nonzero_count[j] - len(features_short)//2) if j == 4
            else x[j] + width * (nonzero_count[j] - 0.5)
            for j,feat in enumerate(nvda_mdd)
        ]
        nonzero_count = [nonzero_count[j] + 1 if feat != 0 else nonzero_count[j] for j,feat in enumerate(nvda_mdd)]
        axes[0,2].bar(x_i, nvda_mdd, width, label=data_agent, alpha=0.8, color=colors[i])

    axes[0,2].set_title('NVDA: Agent decisions on MDD ')
    # axes[2,2].set_xlabel('Agent Name')
    axes[0,2].set_ylabel('Max Drawdown ')
    axes[0,2].set_xticks(x)
    axes[0,2].set_xticklabels(features_short)
    axes[0,2].legend(
        loc='upper center',
        # bbox_to_anchor=(0.5, 1.55),
        ncol=3,
        frameon=False
    )
    # axes[0,2].grid(True, alpha=0.3)
    axes[0,2].set_ylim(0, 12)

    nonzero_count = [0, 0, 0, 0, 0]
    for i,data_agent in enumerate(["All_Data", "Fundamental_Only", "Market_Only", "News_Only", "Sentiment_Only"]):
        aapl_mdd = {
            'Fund': aapl_fundamentals[data_agent]['MDD'] if data_agent in aapl_fundamentals else 0,
            'Market': aapl_market[data_agent]['MDD'] if data_agent in aapl_market else 0,
            'News': aapl_news[data_agent]['MDD'] if data_agent in aapl_news else 0,
            'Sentiment': aapl_sentiment[data_agent]['MDD'] if data_agent in aapl_sentiment else 0,
            'Researcher': aapl_researcher[data_agent]['MDD'] if data_agent in aapl_researcher else 0,
        }
        aapl_mdd = [aapl_mdd[feat] for feat in aapl_mdd.keys()]
        x_i = [
            x[j] + width * (nonzero_count[j] - len(features_short)//2) if j == 4
            else x[j] + width * (nonzero_count[j] - 0.5)
            for j,feat in enumerate(aapl_mdd)
        ]
        nonzero_count = [nonzero_count[j] + 1 if feat != 0 else nonzero_count[j] for j,feat in enumerate(aapl_mdd)]
        axes[1,2].bar(x_i, aapl_mdd, width, label=data_agent, alpha=0.8, color=colors[i])

    axes[1,2].set_title('AAPL: Agent decisions on MDD ')
    # axes[2,2].set_xlabel('Agent Name')
    axes[1,2].set_ylabel('Max Drawdown ')
    axes[1,2].set_xticks(x)
    axes[1,2].set_xticklabels(features_short)
    axes[1,2].legend(
        loc='upper center',
        # bbox_to_anchor=(0.5, 1.55),
        ncol=3,
        frameon=False
    )
    # axes[1,2].grid(True, alpha=0.3)
    axes[1,2].set_ylim(0, 18)

    plt.tight_layout()
    # plt.show()
    plt.savefig('ablation_study_analyst_researcher.png')
    plt.close()



def plot_ablation_study():
    """Plot the ablation study results"""

    # Create figure with subplots
    fig, axes = plt.subplots(4, 2, figsize=(15, 12))
    fig.suptitle('Trading Agents Ablation Study Results', fontsize=16, fontweight='bold')

    # Plot 1: Cumulative Returns comparison
    features = list(nvda_data.keys())
    nvda_cr = [nvda_data[feat]['CR'] for feat in features]
    aapl_cr = [aapl_data[feat]['CR'] for feat in features]

    x = np.arange(len(features))
    width = 0.15

    axes[0,0].bar(x - width/2, nvda_cr, width, label='NVDA', alpha=0.8)
    axes[0,0].bar(x + width/2, aapl_cr, width, label='AAPL', alpha=0.8)
    axes[0,0].set_title('Final Decision Comparison on Cumulative Returns ')
    axes[0,0].set_xlabel('Data Configuration')
    axes[0,0].set_ylabel('Cumulative Returns ')
    axes[0,0].set_xticks(x)
    axes[0,0].set_xticklabels([f.replace('_', '\n') for f in features], rotation=45)
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)

    # Plot 2: Sharpe Ratio comparison
    nvda_sr = [nvda_data[feat]['SR'] for feat in features]
    aapl_sr = [aapl_data[feat]['SR'] for feat in features]

    axes[0,1].bar(x - width/2, nvda_sr, width, label='NVDA', alpha=0.8)
    axes[0,1].bar(x + width/2, aapl_sr, width, label='AAPL', alpha=0.8)
    axes[0,1].set_title('Final Decision Comparison on Sharpe Ratio')
    axes[0,1].set_xlabel('Data Configuration')
    axes[0,1].set_ylabel('Sharpe Ratio')
    axes[0,1].set_xticks(x)
    axes[0,1].set_xticklabels([f.replace('_', '\n') for f in features], rotation=45)
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)

    # Plot 3: Researcher vs Trader (NVDA)
    features_short = ['All', 'Fund', 'Market', 'News', 'Sent']
    nvda_research_cr = [nvda_researcher[feat]['CR'] for feat in nvda_researcher.keys()]
    nvda_trade_cr = [nvda_trader[feat]['CR'] for feat in nvda_trader.keys()]

    axes[1,0].bar(x - width/2, nvda_research_cr, width, label='Researcher', alpha=0.8)
    axes[1,0].bar(x + width/2, nvda_trade_cr, width, label='Trader', alpha=0.8)
    axes[1,0].set_title('NVDA: Researcher vs Trader Comparison on CR ')
    axes[1,0].set_xlabel('Data Configuration')
    axes[1,0].set_ylabel('Cumulative Returns ')
    axes[1,0].set_xticks(x)
    axes[1,0].set_xticklabels(features_short)
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)

    # Plot 4: Researcher vs Trader (AAPL)
    aapl_research_cr = [aapl_researcher[feat]['CR'] for feat in aapl_researcher.keys()]
    aapl_trade_cr = [aapl_trader[feat]['CR'] for feat in aapl_trader.keys()]

    axes[1,1].bar(x - width/2, aapl_research_cr, width, label='Researcher', alpha=0.8)
    axes[1,1].bar(x + width/2, aapl_trade_cr, width, label='Trader', alpha=0.8)
    axes[1,1].set_title('AAPL: Researcher vs Trader Comparison on CR ')
    axes[1,1].set_xlabel('Data Configuration')
    axes[1,1].set_ylabel('Cumulative Returns ')
    axes[1,1].set_xticks(x)
    axes[1,1].set_xticklabels(features_short)
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)

    # Plot 5: Fund, Market, News, Sentiment, Researcher , Trader, Final on CR
    features_short = ['Fund', 'Market', 'News', 'Sent', 'Researcher', 'Trader', 'Risk']
    x = np.arange(len(features_short))
    for i,data_agent in enumerate(["All_Data", "Fundamental_Only", "Market_Only", "News_Only", "Sentiment_Only"]):
        nvda_cr = {

            'Fund': nvda_fundamentals[data_agent]['CR'] if data_agent in nvda_fundamentals else 0,
            'Market': nvda_market[data_agent]['CR'] if data_agent in nvda_market else 0,
            'News': nvda_news[data_agent]['CR'] if data_agent in nvda_news else 0,
            'Sentiment': nvda_sentiment[data_agent]['CR'] if data_agent in nvda_sentiment else 0,
            'Researcher': nvda_researcher[data_agent]['CR'] if data_agent in nvda_researcher else 0,
            'Trader': nvda_trader[data_agent]['CR'] if data_agent in nvda_trader else 0,
            'Risk': nvda_data[data_agent]['CR'] if data_agent in nvda_data else 0,
        }

        nvda_cr = [nvda_cr[feat] for feat in nvda_cr.keys()]
        i = i - len(features_short)//2 + 1
        axes[2,0].bar(x + width * i, nvda_cr, width, label=data_agent, alpha=0.8)

    axes[2,0].set_title('NVDA: Agent decisions on CR ')
    # axes[2,0].set_xlabel('Agent Name')
    axes[2,0].set_ylabel('Cumulative Returns ')
    axes[2,0].set_xticks(x)
    axes[2,0].set_xticklabels(features_short)
    axes[2,0].legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 1.55),
        ncol=3,
        frameon=False
    )
    axes[2,0].grid(True, alpha=0.3)

    for i,data_agent in enumerate(["All_Data", "Fundamental_Only", "Market_Only", "News_Only", "Sentiment_Only"]):
        aapl_cr = {
            'Fund': aapl_fundamentals[data_agent]['CR'] if data_agent in aapl_fundamentals else 0,
            'Market': aapl_market[data_agent]['CR'] if data_agent in aapl_market else 0,
            'News': aapl_news[data_agent]['CR'] if data_agent in aapl_news else 0,
            'Sentiment': aapl_sentiment[data_agent]['CR'] if data_agent in aapl_sentiment else 0,
            'Researcher': aapl_researcher[data_agent]['CR'] if data_agent in aapl_researcher else 0,
            'Trader': aapl_trader[data_agent]['CR'] if data_agent in aapl_trader else 0,
            'Risk': aapl_data[data_agent]['CR'] if data_agent in aapl_data else 0,
        }
        aapl_cr = [aapl_cr[feat] for feat in aapl_cr.keys()]
        i = i - len(features_short)//2 + 1
        axes[3,0].bar(x + width * i, aapl_cr, width, label=data_agent, alpha=0.8)

    axes[3,0].set_title('AAPL: Agent decisions on CR ')
    # axes[3,0].set_xlabel('Agent Name')
    axes[3,0].set_ylabel('Cumulative Returns ')
    axes[3,0].set_xticks(x)
    axes[3,0].set_xticklabels(features_short)
    axes[3,0].legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 1.55),
        ncol=3,
        frameon=False
    )
    axes[3,0].grid(True, alpha=0.3)

    # Plot 6: Fund, Market, News, Sentiment, Researcher , Trader, Final on Sharpe Ratio
    features_short = ['Fund', 'Market', 'News', 'Sent', 'Researcher', 'Trader', 'Risk']
    for i,data_agent in enumerate(["All_Data", "Fundamental_Only", "Market_Only", "News_Only", "Sentiment_Only"]):
        nvda_sr = {

            'Fund': nvda_fundamentals[data_agent]['SR'] if data_agent in nvda_fundamentals else 0,
            'Market': nvda_market[data_agent]['SR'] if data_agent in nvda_market else 0,
            'News': nvda_news[data_agent]['SR'] if data_agent in nvda_news else 0,
            'Sentiment': nvda_sentiment[data_agent]['SR'] if data_agent in nvda_sentiment else 0,
            'Researcher': nvda_researcher[data_agent]['SR'] if data_agent in nvda_researcher else 0,
            'Trader': nvda_trader[data_agent]['SR'] if data_agent in nvda_trader else 0,
            'Risk': nvda_data[data_agent]['SR'] if data_agent in nvda_data else 0,
        }
        nvda_sr = [nvda_sr[feat] for feat in nvda_sr.keys()]
        i = i - len(features_short)//2 + 1
        axes[2,1].bar(x + width * i, nvda_sr, width, label=data_agent, alpha=0.8)

    axes[2,1].set_title('NVDA: Agent decisions on SR ')
    # axes[2,1].set_xlabel('Agent Name')
    axes[2,1].set_ylabel('Sharpe Ratio')
    axes[2,1].set_xticks(x)
    axes[2,1].set_xticklabels(features_short)
    axes[2,1].legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 1.55),
        ncol=3,
        frameon=False
    )
    axes[2,1].grid(True, alpha=0.3)

    for i,data_agent in enumerate(["All_Data", "Fundamental_Only", "Market_Only", "News_Only", "Sentiment_Only"]):
        aapl_sr = {
            'Fund': aapl_fundamentals[data_agent]['SR'] if data_agent in aapl_fundamentals else 0,
            'Market': aapl_market[data_agent]['SR'] if data_agent in aapl_market else 0,
            'News': aapl_news[data_agent]['SR'] if data_agent in aapl_news else 0,
            'Sentiment': aapl_sentiment[data_agent]['SR'] if data_agent in aapl_sentiment else 0,
            'Researcher': aapl_researcher[data_agent]['SR'] if data_agent in aapl_researcher else 0,
            'Trader': aapl_trader[data_agent]['SR'] if data_agent in aapl_trader else 0,
            'Risk': aapl_data[data_agent]['SR'] if data_agent in aapl_data else 0,
        }
        aapl_sr = [aapl_sr[feat] for feat in aapl_sr.keys()]
        i = i - len(features_short)//2 + 1
        axes[3,1].bar(x + width * i, aapl_sr, width, label=data_agent, alpha=0.8)

    axes[3,1].set_title('AAPL: Agent decisions on SR ')
    # axes[3,1].set_xlabel('Agent Name')
    axes[3,1].set_ylabel('Sharpe Ratio')
    axes[3,1].set_xticks(x)
    axes[3,1].set_xticklabels(features_short)
    axes[3,1].legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 1.55),
        ncol=3,
        frameon=False
    )
    axes[3,1].grid(True, alpha=0.3)

    plt.tight_layout()
    # plt.show()
    plt.savefig('ablation_study_plots.png')
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # NVDA heatmap
    sns.heatmap(nvda_matrix,
                xticklabels=metrics,
                yticklabels=configurations,
                annot=True,
                fmt='.1f',
                cmap='RdYlGn',
                center=0,
                ax=ax1)
    ax1.set_title('NVDA Performance Metrics')

    # AAPL heatmap
    sns.heatmap(aapl_matrix,
                xticklabels=metrics,
                yticklabels=configurations,
                annot=True,
                fmt='.1f',
                cmap='RdYlGn',
                center=0,
                ax=ax2)
    ax2.set_title('AAPL Performance Metrics')

    plt.tight_layout()
    # plt.show()
    plt.savefig('performance_heatmap.png')



if __name__ == "__main__":
    # Generate plots
    print("Generating ablation study plots...")

    # Main ablation study plot
    plot_ablation_study()

    plot_ablation_study_research_trader_risk()

    plot_ablation_study_analyst_researcher()

    # Performance heatmap
    plot_performance_heatmap()

    print("Plots generated successfully!")