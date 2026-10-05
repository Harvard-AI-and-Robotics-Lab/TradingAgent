#!/usr/bin/env python3
"""
Verification pipeline to compare standard backtest strategies with
single-day decisions from TradingAgentsGraph
"""

import os
import re
import sys
import pandas as pd
import numpy as np
import yfinance as yf
import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
import argparse
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import TradingAgentsGraph
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.agents.utils.portfolio_utils import create_sample_portfolio
from tradingagents.default_config import DEFAULT_CONFIG


class TradingAgentsBacktestVerifier:
    """Verify TradingAgents decisions by comparing with standard backtests"""

    def __init__(self, selected_analysts=["fundamentals", "market", "news", "social"],
                 config=None, save_dir="", enable_reflection=False, skip_analysts=False,
                 portfolio_summary=None, initial_cash=10000.0):
        """Initialize the verifier with TradingAgentsGraph"""
        self.config = config or DEFAULT_CONFIG.copy()
        self.save_dir = save_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.portfolio_summary = portfolio_summary
        self.initial_cash = initial_cash
        self.selected_analysts = selected_analysts

        # Initialize TradingAgentsGraph
        try:
            print(f"skip_analysts: {skip_analysts}")
            self.ta_graph = TradingAgentsGraph(
                selected_analysts=selected_analysts,
                debug=True,
                config=self.config,
                save_dir=save_dir,
                enable_reflection=enable_reflection,
                skip_analysts=skip_analysts
            )

            png_bytes = self.ta_graph.graph.get_graph().draw_mermaid_png()
            print(f"Saving TradingAgentsGraph to {os.path.join(self.save_dir, 'ta_graph.png')}")
            with open(os.path.join(self.save_dir, "ta_graph.png"), "wb") as f:
                f.write(png_bytes)

        except Exception as e:
            print(f"Warning: Could not initialize TradingAgentsGraph: {e}")
            self.ta_graph = None

        # Create save directory
        os.makedirs(self.save_dir, exist_ok=True)

    def get_trading_days(self, symbol: str, start_date: str, end_date: str) -> List[str]:
        """Get list of trading days for the given period"""
        try:
            data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            return [date.strftime('%Y-%m-%d') for date in data.index]
        except Exception as e:
            print(f"Error getting trading days: {e}")
            return []

    def process_signal(self, signal: str) -> dict:
        """Process the signal from the TradingAgentsGraph to extract action and percentage"""
        decision = self.ta_graph.process_signal(signal)

        return decision

    def get_single_day_decisions(self, symbol: str, trading_days: List[str]) -> Dict[str, dict]:
        """Get TradingAgents decisions for each trading day with percentage"""
        decisions = {}
        intermediate_decisions = {}

        if not self.ta_graph:
            raise ValueError("TradingAgentsGraph not available")

        print(f"Getting TradingAgents decisions for {len(trading_days)} trading days...")

        # Initialize portfolio with no holdings
        if self.portfolio_summary is None:
            portfolio_summary = create_sample_portfolio(symbol, trading_days[0], days_back=0, initial_cash=self.initial_cash)
        else:
            portfolio_summary = self.portfolio_summary

        for i, date in enumerate(trading_days):
            try:
                print("="*100)
                print(f" \n\nProcessing day {i+1}/{len(trading_days)}: {date}")
                print("="*100)
                # Get decision from TradingAgentsGraph
                if self.ta_graph.portfolio_summary is None:
                    print( "  Portfolio_summary:", portfolio_summary)
                    final_state, decision = self.ta_graph.propagate(symbol, date, portfolio_summary=portfolio_summary)
                else:
                    print( "  Portfolio_summary:", self.ta_graph.portfolio_summary)
                    final_state, decision = self.ta_graph.propagate(symbol, date)
                decisions[date] = decision

                # Process intermediate decisions from various agents
                for agent in self.selected_analysts:
                    print(f"  Processing agent: {agent}")
                    if agent == "social":
                        agent = "sentiment"
                    agent_key = agent + '_report'
                    if agent_key not in final_state:
                        continue

                    agent_report = final_state[agent_key]
                    # Handle both string and dict formats
                    if isinstance(agent_report, str):
                        if len(agent_report) == 0:
                            continue
                    elif isinstance(agent_report, dict):
                        if not agent_report:
                            continue
                    else:
                        continue

                    if agent not in intermediate_decisions:
                        intermediate_decisions[agent] = {}
                    intermediate_decisions[agent][date] = self.process_signal(agent_report)
                    print(f"  {agent} decision: {intermediate_decisions[agent][date]}")

                for agent in ["trader_investment_plan", "investment_plan", "final_trade_decision"]:
                    print(f"  Processing agent: {agent}")
                    if agent not in final_state:
                        continue

                    agent_data = final_state[agent]

                    # Handle both string and dict formats
                    if isinstance(agent_data, str) or isinstance(agent_data, dict):
                        if len(agent_data) == 0:
                            continue
                    elif agent_data is None:
                        continue
                    elif hasattr(agent_data, '__len__'):
                        # Handle pandas Series or other sequence types
                        if len(agent_data) == 0:
                            continue
                    else:
                        continue

                    if agent not in intermediate_decisions:
                        intermediate_decisions[agent] = {}
                    intermediate_decisions[agent][date] = self.process_signal(agent_data)
                    print(f"  {agent} decision: {intermediate_decisions[agent][date]}")

            except Exception as e:
                print(f"    Error getting decision for {date}: {e}")
                if date not in decisions:
                    decisions[date] = {'action': 'HOLD', 'percentage': 0}  # Default to HOLD on error

        return decisions, intermediate_decisions

    def simulate_trading_agents_backtest(self, symbol: str, decisions: Dict[str, dict],
                                         start_date: str, end_date: str,
                                         initial_cash: float = 10000, commission: float = 0.002) -> Dict[str, Any]:
        """Simulate backtest based on TradingAgents daily decisions with percentage support"""

        # Get price data
        today_dt = datetime.now()
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        if start_dt.date() < today_dt.date() and end_dt.date() < today_dt.date():
            data = yf.download(symbol, start=start_date, end=end_date, progress=False)
        else:
            period_days = max((today_dt - start_dt).days, 1)
            data = yf.download(symbol, period=f"{period_days}d", interval="1d", progress=False)

        if data.empty:
            print(f"No stock price data found for {symbol} from {start_date} to {end_date}")
            return None

        # Initialize portfolio
        cash = initial_cash
        shares = 0  # need first BUY to buy shares
        portfolio_values = []
        trades = []

        for date, row in data.iterrows():
            date_str = date.strftime('%Y-%m-%d')
            decision_info = decisions.get(date_str, {'action': 'HOLD', 'percentage': 0})

            # Handle both dict format and legacy string format
            if isinstance(decision_info, str):
                action = decision_info
                percentage = 100.0
            else:
                action = decision_info.get('action', 'HOLD')
                percentage = decision_info.get('percentage', 0.0)

            price_close = row['Close'].iloc[0]
            price_avg = 0.5 * (row['High'].iloc[0] + row['Low'].iloc[0])

            portfolio_value_before = cash + shares * price_close

            if action == 'BUY' and cash > price_avg * (1 + commission):
                # Buy shares based on percentage of available cash
                available_cash = cash * (percentage / 100.0)
                shares_to_buy = int(available_cash / (price_avg * (1 + commission)))
                if shares_to_buy > 0:
                    cost = shares_to_buy * price_avg * (1 + commission)
                    cash -= cost
                    shares += shares_to_buy
                    trades.append({
                        'date': date_str,
                        'action': 'BUY',
                        'shares': shares_to_buy,
                        'price': price_avg,
                        'cost': cost,
                        'percentage': percentage
                    })

            elif action == 'SELL' and shares > 0:
                # Sell shares based on percentage of holdings
                shares_to_sell = int(shares * (percentage / 100.0))
                if shares_to_sell > 0:
                    proceeds = shares_to_sell * price_avg * (1 - commission)
                    cash += proceeds
                    trades.append({
                        'date': date_str,
                        'action': 'SELL',
                        'shares': shares_to_sell,
                        'price': price_avg,
                        'proceeds': proceeds,
                        'percentage': percentage
                    })
                    shares -= shares_to_sell

            else:
                print(f"No action taken for {date_str}, decision: {action} {percentage}%")

            # Calculate portfolio value
            portfolio_value = cash + shares * price_close
            portfolio_values.append({
                'date': date_str,
                'price': price_avg,
                'price_close': price_close,
                'cash': cash,
                'shares': shares,
                'portfolio_value': portfolio_value,
                'decision': action,
                'percentage': percentage
            })

        # Calculate final metrics
        portfolio_df = pd.DataFrame(portfolio_values)
        final_value = portfolio_df['portfolio_value'].iloc[-1]

        total_return = ((final_value - initial_cash) / initial_cash) * 100

        # Annualized return: 252 trading days in a year
        annualized_return = ((final_value / initial_cash) ** (252 / len(portfolio_df)) - 1) * 100

        # Calculate max drawdown
        running_max = portfolio_df['portfolio_value'].expanding().max()
        drawdown = (running_max - portfolio_df['portfolio_value']) / running_max * 100
        max_drawdown = drawdown.max()

        # Calculate other metrics
        returns = portfolio_df['portfolio_value'].pct_change().dropna()
        annual_return = returns.mean() * 252
        annual_volatility = returns.std() * np.sqrt(252)
        volatility = annual_volatility * 100
        # assign non-free = 0
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0

        # Win rate
        if len(trades) > 0:
            profitable_trades = 0
            for i in range(1, len(trades), 2):  # Every sell trade
                if i < len(trades) and trades[i]['action'] == 'SELL':
                    buy_price = trades[i-1]['price'] if i > 0 and trades[i-1]['action'] == 'BUY' else 0
                    sell_price = trades[i]['price']
                    if sell_price > buy_price:
                        profitable_trades += 1

            win_rate = (profitable_trades / (len(trades) // 2)) * 100 if len(trades) >= 2 else 0
        else:
            win_rate = 0

        return {
            'Equity Final [$]': final_value,
            'Return [%]': total_return,
            'Return (Ann.) [%]': annualized_return,
            'Max. Drawdown [%]': max_drawdown,
            'Sharpe Ratio': sharpe_ratio,
            'Volatility (Ann.) [%]': annual_volatility * 100,
            '# Trades': len(trades),
            'Win Rate [%]': win_rate,
            'Start': portfolio_df['date'].iloc[0],
            'End': portfolio_df['date'].iloc[-1],
            '_portfolio_values': portfolio_df,
            '_trades': trades,
            '_decisions': decisions
        }

    def generate_verification_report(self, results: Dict[str, Any],
                                     symbol: str, start_date: str, end_date: str,
                                     ta_decisions: Dict[str, str], decision_type: str,
                                     save_dir: str) -> str:
        """Generate a detailed verification report for TradingAgents only"""

        report = []
        report.append("🤖 TRADINGAGENTS VERIFICATION REPORT")
        report.append("=" * 60)
        report.append(f"Symbol: {symbol}")
        report.append(f"Period: {start_date} to {end_date}")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 60)

        # TradingAgents Performance
        if results:
            report.append("\n📊 TRADINGAGENTS PERFORMANCE")
            report.append("-" * 35)
            report.append(f"Total Return: {results['Return [%]']:.2f}%")
            report.append(f"Annualized Return: {results['Return (Ann.) [%]']:.2f}%")
            report.append(f"Final Equity: ${results['Equity Final [$]']:,.2f}")
            report.append(f"Max Drawdown: {results['Max. Drawdown [%]']:.2f}%")
            report.append(f"Sharpe Ratio: {results['Sharpe Ratio']:.3f}")
            report.append(f"Volatility: {results['Volatility (Ann.) [%]']:.2f}%")
            report.append(f"Number of Trades: {results['# Trades']}")
            report.append(f"Win Rate: {results['Win Rate [%]']:.2f}%")

            # Decision distribution
            decision_counts = {}
            for decision_info in ta_decisions.values():
                # Handle both dict format and legacy string format
                if isinstance(decision_info, dict):
                    action = decision_info.get('action', 'HOLD')
                else:
                    action = decision_info
                decision_counts[action] = decision_counts.get(action, 0) + 1

            total_days = len(ta_decisions)
            report.append(f"\nDECISION ANALYSIS ({total_days} trading days)")
            report.append("-" * 25)
            for decision, count in decision_counts.items():
                percentage = (count / total_days) * 100
                report.append(f"{decision}: {count} days ({percentage:.1f}%)")

            # Trading activity
            report.append(f"\nTRADING ACTIVITY")
            report.append("-" * 20)
            report.append(f"Trading Frequency: {results['# Trades']/total_days:.3f} trades/day")
            report.append(f"Average Days Between Trades: {total_days/max(1, results['# Trades']):.1f} days")

        report.append(f"\nVERIFICATION SUMMARY")
        report.append("-" * 25)
        report.append(f"✓ Processed {len(ta_decisions)} trading days")
        report.append(f"✓ TradingAgents pipeline: {'SUCCESS' if results else '❌ FAILED'}")

        report = "\n".join(report)
        os.makedirs(f"{save_dir}/{symbol}/ta_{self.timestamp}", exist_ok=True)
        with open(f"{save_dir}/{symbol}/ta_{self.timestamp}/{decision_type}_report.txt", "w") as f:
            f.write(report)
        return report

    def save_verification_results(self, results: Dict[str, Any], ta_decisions: Dict[str, str], decision_type: str,
                                  symbol: str, output_dir: str = "verification_results"):
        """Save verification results to files"""

        # Create output directory
        os.makedirs(f"{output_dir}/{symbol}", exist_ok=True)

        # Save TradingAgents decisions
        base_dir = f"{output_dir}/{symbol}/ta_{self.timestamp}"
        os.makedirs(base_dir, exist_ok=True)
        base_file = base_dir + f"/{decision_type}"
        decisions_file = base_file + "_decisions.json"
        with open(decisions_file, 'w') as f:
            json.dump(ta_decisions, f, indent=2)
        print(f"TradingAgents decisions saved to: {decisions_file}")

        # Save detailed results
        results_file = base_file + "_results.json"

        # Convert results for JSON serialization
        json_result = {}
        for key, value in results.items():
            if key.startswith('_'):  # Skip internal data
                continue
            if isinstance(value, (pd.Timestamp, np.datetime64)):
                json_result[key] = str(value)
            elif isinstance(value, np.ndarray):
                json_result[key] = value.tolist()
            else:
                json_result[key] = value

        with open(results_file, 'w') as f:
            json.dump(json_result, f, indent=2)
        print(f"📊 TradingAgents results saved to: {results_file}")

        # Save portfolio values for further analysis
        if '_portfolio_values' in results:
            portfolio_file = base_file + "_portfolio.csv"
            results['_portfolio_values'].to_csv(portfolio_file, index=False)
            print(f"📈 Portfolio values saved to: {portfolio_file}")

    def plot_tradingagents_analysis(self, results: Dict[str, Any], ta_decisions: Dict[str, str], decision_type: str,
                                   symbol: str, output_dir: str = "verification_results"):
        """Generate TradingAgents analysis charts"""

        if not results or '_portfolio_values' not in results:
            print("⚠️ No portfolio data available for plotting")
            return

        portfolio_df = results['_portfolio_values']
        portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Portfolio Value Over Time
        ax1.plot(portfolio_df['date'], portfolio_df['portfolio_value'],
                linewidth=2, color='#4ECDC4', label='Portfolio Value')
        ax1.plot(portfolio_df['date'], portfolio_df['price'] * (results['Equity Final [$]'] / portfolio_df['price'].iloc[0]),
                linewidth=2, color='#FF6B6B', alpha=0.7, label=f'{symbol} Price (normalized)')
        ax1.set_title(f'{symbol} - TradingAgents Portfolio Performance', fontweight='bold')
        ax1.set_ylabel('Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Decision Distribution
        decision_counts = {}
        for decision_info in ta_decisions.values():
            # Handle both dict format and legacy string format
            if isinstance(decision_info, dict):
                action = decision_info.get('action', 'HOLD')
            else:
                action = decision_info
            decision_counts[action] = decision_counts.get(action, 0) + 1

        colors = {'BUY': '#2ECC71', 'SELL': '#E74C3C', 'HOLD': '#F39C12'}
        decision_colors = [colors.get(decision, '#95A5A6') for decision in decision_counts.keys()]

        wedges, texts, autotexts = ax2.pie(decision_counts.values(),
                                          labels=decision_counts.keys(),
                                          colors=decision_colors,
                                          autopct='%1.1f%%',
                                          startangle=90)
        ax2.set_title('Decision Distribution', fontweight='bold')

        # Daily Returns
        returns = portfolio_df['portfolio_value'].pct_change().dropna() * 100
        ax3.hist(returns, bins=20, alpha=0.7, color='#3498DB', edgecolor='black')
        ax3.set_title('Daily Returns Distribution', fontweight='bold')
        ax3.set_xlabel('Daily Return (%)')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3)
        ax3.axvline(returns.mean(), color='red', linestyle='--',
                   label=f'Mean: {returns.mean():.3f}%')
        ax3.legend()

        # Drawdown Over Time
        running_max = portfolio_df['portfolio_value'].expanding().max()
        drawdown = (running_max - portfolio_df['portfolio_value']) / running_max * 100

        ax4.fill_between(portfolio_df['date'], drawdown, 0,
                        alpha=0.7, color='#E74C3C', label='Drawdown')
        ax4.set_title('Drawdown Over Time', fontweight='bold')
        ax4.set_ylabel('Drawdown (%)')
        ax4.grid(True, alpha=0.3)
        ax4.legend()

        plt.tight_layout()

        # Save chart
        chart_file = f"{output_dir}/{symbol}/ta_{self.timestamp}/{decision_type}_analysis.png"
        os.makedirs(f"{output_dir}/{symbol}/ta_{self.timestamp}", exist_ok=True)
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        print(f"📊 TradingAgents analysis chart saved to: {chart_file}")

        # plt.show()  # Disabled to prevent popup blocking execution


def run_tradingagents_backtest(
        config,
        selected_analysts=["fundamentals", "market", "social", "news"],
        symbol='AAPL', start_date='2024-01-01', end_date='2024-04-01',
        initial_cash=10000, commission=0.002, save_dir="",
        enable_reflection=False, skip_analysts=False, portfolio_summary=None,
        return_verifier=False
    ):
    """Main function to run TradingAgents verification pipeline"""
    # Initialize verifier
    verifier = TradingAgentsBacktestVerifier(
        selected_analysts, config, save_dir, enable_reflection, skip_analysts,
        portfolio_summary=portfolio_summary, initial_cash=initial_cash
    )

    # Get trading days
    print(f"Getting trading days for {symbol} from {start_date} to {end_date}...")
    trading_days = verifier.get_trading_days(symbol, start_date, end_date)
    print(f"   Found {len(trading_days)} trading days")

    # Get TradingAgents decisions
    print(f"\nGetting TradingAgents decisions...")
    ta_decisions, intermediate_decisions = verifier.get_single_day_decisions(symbol, trading_days)
    intermediate_decisions['final'] = ta_decisions

    print(f"Intermediate decisions: {intermediate_decisions.keys()}")

    for decision_type, decisions in intermediate_decisions.items():
        print(f"Decision type: {decision_type}")
        print(f"Decisions: {decisions}")

        try:

            # Simulate backtest
            print(f"\nSimulating TradingAgents backtest...")
            results = verifier.simulate_trading_agents_backtest(
                symbol, decisions, start_date, end_date, initial_cash, commission
            )

            if results and len(decisions) > 0:
                # Generate report
                report = verifier.generate_verification_report(
                    results, symbol, start_date, end_date, decisions, decision_type, save_dir=save_dir
                )

                # Save results
                verifier.save_verification_results(results, decisions, decision_type, symbol, output_dir=save_dir)

                # Generate analysis charts
                verifier.plot_tradingagents_analysis(results, decisions, decision_type, symbol, output_dir=save_dir)

                print(f"\n{decision_type}: TradingAgents verification completed successfully!")
                print(f"Check {save_dir}/{symbol}/ for detailed outputs")

            else:
                print(f"\n{decision_type}: TradingAgents verification failed!")

        except Exception as e:
            print(f"\n{decision_type}: TradingAgents verification failed: {e}")

    if return_verifier:
        return verifier


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run TradingAgents backtest with configurable parameters')

    # Config parameters
    parser.add_argument('--llm_provider', type=str, default='openai',
                        help='LLM provider (default: openai)')
    parser.add_argument('--backend_url', type=str, default='https://api.openai.com/v1',
                        help='Backend URL for LLM API (default: https://api.openai.com/v1)')
    parser.add_argument('--deep_think_llm', type=str, default='gpt-4o-mini',
                        help='Deep thinking LLM model (default: gpt-4o-mini)')
    parser.add_argument('--quick_think_llm', type=str, default='gpt-4o-mini',
                        help='Quick thinking LLM model (default: gpt-4o-mini)')
    parser.add_argument('--analyst_llms', nargs='+', default=[],
                        help='Analyst LLMs (default: [])')
    parser.add_argument('--analyst_llm_providers', nargs='+', default=[],
                        help='Analyst LLM providers (default: [])')
    parser.add_argument('--credibility_scorer_llm', type=str, default='',
                        help='Credibility scorer LLM model (default: gpt-4o-mini)')
    parser.add_argument('--credibility_scorer_llm_provider', type=str, default='',
                        help='Credibility scorer LLM provider (default: openai)')
    parser.add_argument('--max_debate_rounds', type=int, default=1,
                        help='Maximum debate rounds (default: 1)')
    parser.add_argument('--online_tools', action=argparse.BooleanOptionalAction, default=True,
                        help='Enable or disable online tools (default: enabled)')
    parser.add_argument('--enable_reflection', action='store_true', default=False,
                        help='Enable reflection (default: False)')
    parser.add_argument('--skip_analysts', action='store_true', default=False,
                        help='Skip analysts (default: False)')

    # Directory and file parameters
    parser.add_argument('--save_dir', type=str, default='baseline_analysis/results/tradingagents_strategy',
                        help='Directory to save results (default: baseline_analysis/results/tradingagents_strategy)')

    # Trading parameters
    parser.add_argument('--symbol', type=str, default='NVDA',
                        help='Stock symbol to trade (default: NVDA)')
    parser.add_argument("--selected_analysts", nargs='+',
                        default=["price", "fundamentals", "market", "social", "news"],
                        help='Selected analysts (default: market, social, news, fundamentals)')
    parser.add_argument('--start_date', type=str, default='2024-01-01',
                        help='Start date for backtest (YYYY-MM-DD) (default: 2024-01-01)')
    parser.add_argument('--end_date', type=str, default='2024-04-01',
                        help='End date for backtest (YYYY-MM-DD) (default: 2024-04-01)')
    parser.add_argument('--cash', type=float, default=10000.0,
                        help='Initial cash amount (default: 10000.0)')
    parser.add_argument('--commission', type=float, default=0.000,
                        help='Commission rate (default: 0.000)')

    # load data from cache
    parser.add_argument('--resume_from_cache', action='store_true', default=False,
                        help='Resume from cache (default: False)')
    parser.add_argument('--load_data_from_cache', action='store_true', default=False,
                        help='Load data from cache (default: False)')
    parser.add_argument('--load_data_from_cache_file', type=str, default='',
                        help='Load data from cache file (default: )')

    args = parser.parse_args()

    # Setup config with parsed arguments
    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = args.llm_provider
    config["backend_url"] = args.backend_url
    config["deep_think_llm"] = args.deep_think_llm
    config["quick_think_llm"] = args.quick_think_llm
    config["analyst_llms"] = args.analyst_llms
    config["analyst_llm_providers"] = args.analyst_llm_providers
    config["credibility_scorer_llm"] = args.credibility_scorer_llm
    config["credibility_scorer_llm_provider"] = args.credibility_scorer_llm_provider
    config["max_debate_rounds"] = args.max_debate_rounds
    config["online_tools"] = args.online_tools
    config["enable_reflection"] = args.enable_reflection
    config["skip_analysts"] = args.skip_analysts
    config["resume_from_cache"] = args.resume_from_cache
    config["load_data_from_cache"] = args.load_data_from_cache
    config["load_data_from_cache_file"] = args.load_data_from_cache_file

    print("🚀 TradingAgents Backtest Configuration")
    print("=" * 50)
    print(f"LLM Provider: {config['llm_provider']}")
    print(f"Backend URL: {config['backend_url']}")
    print(f"Deep Think LLM: {config['deep_think_llm']}")
    print(f"Quick Think LLM: {config['quick_think_llm']}")
    print(f"Analyst LLMs: {config['analyst_llms']}")
    print(f"Analyst LLM Providers: {config['analyst_llm_providers']}")
    print(f"Credibility Scorer LLM: {config['credibility_scorer_llm']}")
    print(f"Credibility Scorer LLM Provider: {config['credibility_scorer_llm_provider']}")
    print(f"Max Debate Rounds: {config['max_debate_rounds']}")
    print(f"Online Tools: {config['online_tools']}")
    print(f"Enable Reflection: {config['enable_reflection']}")
    print(f"Skip Analysts: {config['skip_analysts']}")
    print(f"Save Directory: {args.save_dir}")
    print(f"Symbol: {args.symbol}")
    print(f"Start Date: {args.start_date}")
    print(f"End Date: {args.end_date}")
    print(f"Initial Cash: ${args.cash:,.2f}")
    print(f"Commission: {args.commission:.3f}")
    print(f"Resume from Cache: {config['resume_from_cache']}")
    print(f"Load Data from Cache: {config['load_data_from_cache']}")
    print(f"Load Data from Cache File: {config['load_data_from_cache_file']}")
    print("=" * 50)


    # Run the backtest
    run_tradingagents_backtest(config, args.selected_analysts, args.symbol, args.start_date, args.end_date,
                               args.cash, args.commission, args.save_dir, args.enable_reflection, args.skip_analysts)
