#!/usr/bin/env python3
"""
Runner script for all backtesting strategies
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from buy_hold import run_buy_hold_backtest
from macd_strategy import run_macd_backtest
from rsi_strategy import run_rsi_backtest
from kdj_strategy import run_kdj_backtest
from zmr_strategy import run_zmr_backtest
from sma_strategy import run_sma_backtest
from tradingagents_strategy import run_tradingagents_backtest


def run_all_strategies(symbol='AAPL', start_date='2024-01-01', end_date='2024-01-29',
                      cash=10000, commission=0.002):
    """
    Run all baseline strategies and compare results

    Args:
        symbol: Stock symbol to trade
        start_date: Start date for backtest
        end_date: End date for backtest
        cash: Initial cash amount
        commission: Commission rate (0.002 = 0.2%)
    """

    print("🚀 Running ALL Baseline Strategies with backtesting package")
    print("=" * 80)
    print(f"Symbol: {symbol}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Initial Cash: ${cash:,}")
    print(f"Commission: {commission*100}%")
    print("=" * 80)

    strategies = {}
    results_summary = []

    # run tradingagents
    print("\n📊 7. Running TradingAgents Strategy...")
    print("-" * 50)
    try:
        bt_tradingagents, results_tradingagents = run_tradingagents_backtest(symbol, start_date, end_date, cash, commission)
        strategies['TradingAgents'] = (bt_tradingagents, results_tradingagents)
        results_summary.append({
            'Strategy': 'TradingAgents',
            'Total Return (%)': results_tradingagents['Return [%]'],
            'Annualized Return (%)': results_tradingagents.get('Return (Ann.) [%]', float('nan')),
            'Final Equity ($)': results_tradingagents['Equity Final [$]'],
            'Max Drawdown (%)': results_tradingagents['Max. Drawdown [%]'],
            'Sharpe Ratio': results_tradingagents['Sharpe Ratio'],
            'Number of Trades': results_tradingagents['# Trades'],
            'Win Rate (%)': results_tradingagents['Win Rate [%]']
        })
    except Exception as e:
        print(f"❌ TradingAgents failed: {e}")

    # Run Buy & Hold
    print("\n📊 1. Running Buy & Hold Strategy...")
    print("-" * 50)
    try:
        bt_bh, results_bh = run_buy_hold_backtest(symbol, start_date, end_date, cash, commission)
        if bt_bh:
            strategies['Buy & Hold'] = (bt_bh, results_bh)
            results_summary.append({
                'Strategy': 'Buy & Hold',
                'Total Return (%)': results_bh['Return [%]'],
                'Annualized Return (%)': results_bh.get('Return (Ann.) [%]', float('nan')),
                'Final Equity ($)': results_bh['Equity Final [$]'],
                'Max Drawdown (%)': results_bh['Max. Drawdown [%]'],
                'Sharpe Ratio': results_bh['Sharpe Ratio'],
                'Number of Trades': results_bh['# Trades'],
                'Win Rate (%)': results_bh['Win Rate [%]']
            })
    except Exception as e:
        print(f"❌ Buy & Hold failed: {e}")

    # Run MACD
    print("\n📊 2. Running MACD Strategy...")
    print("-" * 50)
    try:
        bt_macd, results_macd = run_macd_backtest(symbol, start_date, end_date, cash, commission)
        if bt_macd:
            strategies['MACD'] = (bt_macd, results_macd)
            results_summary.append({
                'Strategy': 'MACD',
                'Total Return (%)': results_macd['Return [%]'],
                'Annualized Return (%)': results_macd.get('Return (Ann.) [%]', float('nan')),
                'Final Equity ($)': results_macd['Equity Final [$]'],
                'Max Drawdown (%)': results_macd['Max. Drawdown [%]'],
                'Sharpe Ratio': results_macd['Sharpe Ratio'],
                'Number of Trades': results_macd['# Trades'],
                'Win Rate (%)': results_macd['Win Rate [%]']
            })
    except Exception as e:
        print(f"❌ MACD failed: {e}")

    # # Run RSI
    # print("\n📊 3. Running RSI Strategy...")
    # print("-" * 50)
    # try:
    #     bt_rsi, results_rsi = run_rsi_backtest(symbol, start_date, end_date, cash, commission)
    #     if bt_rsi:
    #         strategies['RSI'] = (bt_rsi, results_rsi)
    #         results_summary.append({
    #             'Strategy': 'RSI',
    #             'Total Return (%)': results_rsi['Return [%]'],
    #             'Annualized Return (%)': results_rsi.get('Return (Ann.) [%]', float('nan')),
    #             'Final Equity ($)': results_rsi['Equity Final [$]'],
    #             'Max Drawdown (%)': results_rsi['Max. Drawdown [%]'],
    #             'Sharpe Ratio': results_rsi['Sharpe Ratio'],
    #             'Number of Trades': results_rsi['# Trades'],
    #             'Win Rate (%)': results_rsi['Win Rate [%]']
    #         })
    # except Exception as e:
    #     print(f"❌ RSI failed: {e}")

    # Run KDJ
    print("\n📊 4. Running KDJ&RSI Strategy...")
    print("-" * 50)
    try:
        bt_kdj, results_kdj = run_kdj_backtest(symbol, start_date, end_date, cash, commission)
        if bt_kdj:
            strategies['KDJ&RSI'] = (bt_kdj, results_kdj)
            results_summary.append({
                'Strategy': 'KDJ&RSI',
                'Total Return (%)': results_kdj['Return [%]'],
                'Annualized Return (%)': results_kdj.get('Return (Ann.) [%]', float('nan')),
                'Final Equity ($)': results_kdj['Equity Final [$]'],
                'Max Drawdown (%)': results_kdj['Max. Drawdown [%]'],
                'Sharpe Ratio': results_kdj['Sharpe Ratio'],
                'Number of Trades': results_kdj['# Trades'],
                'Win Rate (%)': results_kdj['Win Rate [%]']
            })
    except Exception as e:
        print(f"❌ KDJ&RSI failed: {e}")

    # Run ZMR
    print("\n📊 5. Running ZMR Strategy...")
    print("-" * 50)
    try:
        bt_zmr, results_zmr = run_zmr_backtest(symbol, start_date, end_date, cash, commission)
        if bt_zmr:
            strategies['ZMR'] = (bt_zmr, results_zmr)
            results_summary.append({
                'Strategy': 'ZMR',
                'Total Return (%)': results_zmr['Return [%]'],
                'Annualized Return (%)': results_zmr.get('Return (Ann.) [%]', float('nan')),
                'Final Equity ($)': results_zmr['Equity Final [$]'],
                'Max Drawdown (%)': results_zmr['Max. Drawdown [%]'],
                'Sharpe Ratio': results_zmr['Sharpe Ratio'],
                'Number of Trades': results_zmr['# Trades'],
                'Win Rate (%)': results_zmr['Win Rate [%]']
            })
    except Exception as e:
        print(f"❌ ZMR failed: {e}")

    # Run SMA
    print("\n📊 6. Running SMA Strategy...")
    print("-" * 50)
    try:
        bt_sma, results_sma = run_sma_backtest(symbol, start_date, end_date, cash, commission)
        if bt_sma:
            strategies['SMA'] = (bt_sma, results_sma)
            results_summary.append({
                'Strategy': 'SMA',
                'Total Return (%)': results_sma['Return [%]'],
                'Annualized Return (%)': results_sma.get('Return (Ann.) [%]', float('nan')),
                'Final Equity ($)': results_sma['Equity Final [$]'],
                'Max Drawdown (%)': results_sma['Max. Drawdown [%]'],
                'Sharpe Ratio': results_sma['Sharpe Ratio'],
                'Number of Trades': results_sma['# Trades'],
                'Win Rate (%)': results_sma['Win Rate [%]']
            })
    except Exception as e:
        print(f"❌ SMA failed: {e}")

    # # run tradingagents
    # print("\n📊 7. Running TradingAgents Strategy...")
    # print("-" * 50)
    # try:
    #     bt_tradingagents, results_tradingagents = run_tradingagents_backtest(symbol, start_date, end_date, cash, commission)
    #     strategies['TradingAgents'] = (bt_tradingagents, results_tradingagents)
    #     results_summary.append({
    #         'Strategy': 'TradingAgents',
    #         'Total Return (%)': results_tradingagents['Return [%]'],
    #         'Annualized Return (%)': results_tradingagents.get('Return (Ann.) [%]', float('nan')),
    #         'Final Equity ($)': results_tradingagents['Equity Final [$]'],
    #         'Max Drawdown (%)': results_tradingagents['Max. Drawdown [%]'],
    #         'Sharpe Ratio': results_tradingagents['Sharpe Ratio'],
    #         'Number of Trades': results_tradingagents['# Trades'],
    #         'Win Rate (%)': results_tradingagents['Win Rate [%]']
    #     })
    # except Exception as e:
    #     print(f"❌ TradingAgents failed: {e}")

    # Generate comparison report
    if results_summary:
        generate_comparison_report(results_summary, symbol, start_date, end_date)
        generate_comparison_chart(strategies, symbol)

    return strategies, results_summary


def generate_comparison_report(results_summary, symbol, start_date, end_date):
    """Generate a comprehensive comparison report"""

    df = pd.DataFrame(results_summary)
    df = df.sort_values('Total Return (%)', ascending=False)

    print("\n📊 STRATEGY COMPARISON SUMMARY")
    print("=" * 80)
    print(f"Symbol: {symbol}")
    print(f"Period: {start_date} to {end_date}")
    print("=" * 80)

    # Print formatted table
    print(f"{'Strategy':<12} {'Return%':<8} {'AR%':<8} {'Final$':<10} {'MaxDD%':<8} {'Sharpe':<7} {'Trades':<7} {'WinRate%':<9}")
    print("-" * 88)

    for _, row in df.iterrows():
        ar_value = row['Annualized Return (%)'] if not pd.isna(row['Annualized Return (%)']) else float('nan')
        print(f"{row['Strategy']:<12} "
              f"{row['Total Return (%)']:<8.2f} "
              f"{ar_value:<8.2f} "
              f"{row['Final Equity ($)']:<10.2f} "
              f"{row['Max Drawdown (%)']:<8.2f} "
              f"{row['Sharpe Ratio']:<7.3f} "
              f"{row['Number of Trades']:<7} "
              f"{row['Win Rate (%)']:<9.2f}")

    print("=" * 88)

    # Find best performing strategy
    best_strategy = df.iloc[0]
    print(f"\n🏆 BEST PERFORMING STRATEGY: {best_strategy['Strategy']}")
    print(f"   📈 Total Return: {best_strategy['Total Return (%)']:.2f}%")
    print(f"   💰 Final Equity: ${best_strategy['Final Equity ($)']:,.2f}")
    print(f"   📉 Max Drawdown: {best_strategy['Max Drawdown (%)']:.2f}%")
    print(f"   📊 Sharpe Ratio: {best_strategy['Sharpe Ratio']:.3f}")

    # Save to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"results/{symbol}/backtesting_strategies_comparison_{timestamp}.txt"

    with open(report_file, 'w') as f:
        f.write(f"Backtesting Strategies Comparison Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n")
        f.write(f"Symbol: {symbol}\n")
        f.write(f"Period: {start_date} to {end_date}\n")
        f.write("=" * 80 + "\n\n")

        f.write("Strategy Performance Summary:\n")
        f.write("-" * 40 + "\n")
        for _, row in df.iterrows():
            ar_value = row['Annualized Return (%)'] if not pd.isna(row['Annualized Return (%)']) else float('nan')
            f.write(f"\n{row['Strategy']}:\n")
            f.write(f"  Total Return: {row['Total Return (%)']:.2f}%\n")
            f.write(f"  Annualized Return: {ar_value:.2f}%\n")
            f.write(f"  Final Equity: ${row['Final Equity ($)']:,.2f}\n")
            f.write(f"  Max Drawdown: {row['Max Drawdown (%)']:.2f}%\n")
            f.write(f"  Sharpe Ratio: {row['Sharpe Ratio']:.3f}\n")
            f.write(f"  Number of Trades: {row['Number of Trades']}\n")
            f.write(f"  Win Rate: {row['Win Rate (%)']:.2f}%\n")

        f.write(f"\nBest Performing Strategy: {best_strategy['Strategy']}\n")
        f.write(f"Return: {best_strategy['Total Return (%)']:.2f}%\n")

    print(f"\n📋 Detailed report saved to: {report_file}")


def generate_comparison_chart(strategies, symbol):
    """Generate comparison charts for all strategies"""

    if not strategies:
        return

    fig, ((ax1, ax2, ax5), (ax3, ax4, ax6)) = plt.subplots(2, 3, figsize=(20, 12))

    # Plot 1: Equity curves
    for name, (bt, results) in strategies.items():
        equity_curve = results['_equity_curve']['Equity']
        dates = results['_equity_curve'].index
        ax1.plot(dates, equity_curve, label=name, linewidth=2)

    ax1.set_title(f'{symbol} - Strategy Equity Curves', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Returns comparison
    strategy_names = []
    returns = []
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold', 'orange', 'purple', 'pink', 'cyan']

    for i, (name, (bt, results)) in enumerate(strategies.items()):
        strategy_names.append(name)
        returns.append(results['Return [%]'])

    bars = ax2.bar(strategy_names, returns, color=colors[:len(strategy_names)])
    ax2.set_title('Cumulative Returns Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Cumulative Return (%)')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, ret in zip(bars, returns):
        height = bar.get_height()
        ax2.annotate(f'{ret:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontweight='bold')

    # Plot 3: Max Drawdown comparison
    drawdowns = []
    for name, (bt, results) in strategies.items():
        drawdowns.append(abs(results['Max. Drawdown [%]']))  # Make positive for visualization

    bars = ax3.bar(strategy_names, drawdowns, color=colors[:len(strategy_names)])
    ax3.set_title('Max Drawdown Comparison', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Max Drawdown (%)')
    ax3.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, dd in zip(bars, drawdowns):
        height = bar.get_height()
        ax3.annotate(f'{dd:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontweight='bold')

    # Plot 4: Sharpe Ratio comparison
    sharpe_ratios = []
    for name, (bt, results) in strategies.items():
        sharpe_ratios.append(results['Sharpe Ratio'])

    bars = ax4.bar(strategy_names, sharpe_ratios, color=colors[:len(strategy_names)])
    ax4.set_title('Sharpe Ratio Comparison', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Sharpe Ratio')
    ax4.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, sr in zip(bars, sharpe_ratios):
        height = bar.get_height()
        ax4.annotate(f'{sr:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontweight='bold')

    # Plot 5: Annualized Return comparison
    annualized_returns = []
    for name, (bt, results) in strategies.items():
        ar_value = results.get('Return (Ann.) [%]', 0.0)
        # Handle nan values
        if pd.isna(ar_value):
            ar_value = 0.0
        annualized_returns.append(ar_value)

    bars = ax5.bar(strategy_names, annualized_returns, color=colors[:len(strategy_names)])
    ax5.set_title('Annualized Return Comparison', fontsize=14, fontweight='bold')
    ax5.set_ylabel('Annualized Return (%)')
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.tick_params(axis='x', rotation=45)

    # Add value labels on bars
    for bar, ar in zip(bars, annualized_returns):
        height = bar.get_height()
        ax5.annotate(f'{ar:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontweight='bold')

    # Plot 6: Leave empty or add another metric if needed
    ax6.axis('off')  # Turn off the 6th subplot for now

    plt.tight_layout()

    # Save chart
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    chart_file = f"results/{symbol}/backtesting_strategies_comparison_{timestamp}.png"
    plt.savefig(chart_file, dpi=300, bbox_inches='tight')
    print(f"📊 Comparison chart saved to: {chart_file}")

    plt.show()


if __name__ == "__main__":
    # Default parameters
    symbol = 'AAPL'
    start_date = '2024-01-01'
    end_date = '2024-01-29'
    cash = 10000
    commission = 0.002

    os.makedirs("results", exist_ok=True)

    # Run all strategies
    strategies, results = run_all_strategies(symbol, start_date, end_date, cash, commission)

    print(f"\n✅ All {len(strategies)} strategies completed successfully!")
    print(f"📊 Results saved in current directory")
    print(f"📈 Individual HTML plots generated for each strategy")
    print(f"📋 Comparison report and chart generated")