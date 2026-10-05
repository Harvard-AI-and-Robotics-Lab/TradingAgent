#!/usr/bin/env python3
"""
Single day backtest script that reloads existing results, reflection memory, and logs
"""

import os
import sys
import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the backtest infrastructure
from tradingagents_strategy import run_tradingagents_backtest
from tradingagents.default_config import DEFAULT_CONFIG


def historical_decisions(save_dir: str, symbol: str, verifier: any):
    """
    Aggregate historical decisions from ta_2026* directories/files.

    For each decision type (final, fundamentals, market, news, sentiment, etc.):
    1. Collect all decisions by date
    2. Calculate portfolio performance
    3. Save aggregated decisions and portfolio
    """
    print(f"\n📊 Aggregating historical decisions for {symbol}...")

    save_path = os.path.join(save_dir, symbol)

    if not os.path.exists(save_path):
        print(f"⚠️  Symbol directory not found: {save_path}")
        return {}

    # Find all ta_2026* decision files or directories
    all_files = []
    for item in os.listdir(save_path):
        item_path = os.path.join(save_path, item)

        # Check if it's a directory with ta_2026* pattern
        if os.path.isdir(item_path) and item.startswith("ta_2026"):
            # Look for decision files inside the directory
            for file in os.listdir(item_path):
                if file.endswith("_decisions.json"):
                    all_files.append(os.path.join(item_path, file))

        # Check if it's a file with ta_2026* pattern
        elif os.path.isfile(item_path) and item.startswith("ta_2026") and item.endswith("_decisions.json"):
            all_files.append(item_path)

    if not all_files:
        print(f"⚠️  No ta_2026* decision files found in {save_path}")
        return {}

    print(f"✓ Found {len(all_files)} decision files")

    # Organize decisions by type
    decisions_by_type = {}

    for file_path in all_files:
        # Extract decision type from filename
        # Format: ta_20260118_123456_TYPE_decisions.json or TYPE_decisions.json
        filename = os.path.basename(file_path)

        # Remove _decisions.json suffix
        if filename.endswith("_decisions.json"):
            name_part = filename[:-len("_decisions.json")]

            # Extract decision type
            # Could be: ta_20260118_123456_final or ta_20260118_123456 (implies final)
            parts = name_part.split("_")

            # Find the decision type (everything after the timestamp)
            # Pattern: ta_YYYYMMDD_HHMMSS_TYPE
            decision_type = "_".join(parts[3:]) if name_part.startswith("ta_") else "_".join(parts)

            if not decision_type:
                decision_type = "final"
        else:
            continue

        # Load decisions
        try:
            with open(file_path, 'r') as f:
                decisions = json.load(f)

            if decision_type not in decisions_by_type:
                decisions_by_type[decision_type] = {}

            # Merge decisions (later timestamps override earlier ones for same date)
            for date, decision in decisions.items():
                decisions_by_type[decision_type][date] = decision

        except Exception as e:
            print(f"⚠️  Error loading {file_path}: {e}")
            continue

    print(f"✓ Found {len(decisions_by_type)} decision types: {list(decisions_by_type.keys())}")

    results_summary = {}

    for decision_type, decisions in decisions_by_type.items():
        if not decisions:
            continue

        print(f"\n📈 Processing {decision_type} decisions ({len(decisions)} days)...")

        try:
            # Sort decisions by date
            sorted_dates = sorted(decisions.keys())
            start_date = sorted_dates[0]
            end_date = (
                datetime.strptime(sorted_dates[-1], "%Y-%m-%d")
                + timedelta(days=1)
            ).strftime("%Y-%m-%d")
            print(f"\n\n\nDecision type: {decision_type}")
            print(f"Decisions: {decisions}", decisions.keys())
            print(f"Start date: {start_date}, End date: {end_date}")

            # Simulate backtest
            results = verifier.simulate_trading_agents_backtest(
                symbol, decisions, start_date, end_date,
                initial_cash=10000.0, commission=0.000
            )

            if results and len(decisions) > 0:
                # Save aggregated decisions
                aggregated_dir = os.path.join(save_path, "aggregated_historical")
                os.makedirs(aggregated_dir, exist_ok=True)

                # Generate report
                report = verifier.generate_verification_report(
                    results, symbol, start_date, end_date, decisions, decision_type, save_dir=aggregated_dir
                )

                # Save results
                verifier.save_verification_results(results, decisions, decision_type, symbol, output_dir=aggregated_dir)

                # Generate analysis charts
                verifier.plot_tradingagents_analysis(results, decisions, decision_type, symbol, output_dir=aggregated_dir)

                print(f"\n{decision_type}: TradingAgents verification completed successfully!")
                print(f"Check {aggregated_dir}/ for detailed outputs")

            else:
                print(f"\n{decision_type}: TradingAgents verification failed!")

        except Exception as e:
            print(f"  ⚠️  Error processing {decision_type}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n✅ Historical decisions aggregation complete!")


def reload_portfolio_summary_from_state_log(save_dir: str, symbol: str, test_date: str, online_tools: bool = True) -> dict:
    """Reload portfolio_summary from the most recent full_states_log file before test_date"""
    print(f"\n🔄 Attempting to reload portfolio_summary from state logs...")

    # Construct the log directory path
    symbol_dir = os.path.join(save_dir, symbol)
    log_dir = os.path.join(symbol_dir, "TradingAgentsStrategy_logs")

    if not os.path.exists(log_dir):
        print(f"⚠️  Log directory not found: {log_dir}")
        return None

    # Find state log files before test_date
    online_str = str(online_tools)
    log_files = []

    for file in os.listdir(log_dir):
        if file.startswith("full_states_log_") and file.endswith(f"_online_{online_str}.json"):
            # Extract date from filename: full_states_log_2024-01-03_online_True.json
            parts = file.split('_')
            if len(parts) >= 4:
                file_date = parts[3]  # 2024-01-03
                log_files.append((file_date, file))

    if not log_files:
        print(f"⚠️  No state log files found before {test_date}")
        return None

    # Sort by date and get the most recent
    log_files.sort(reverse=True)
    latest_date, latest_file = log_files[0]
    log_path = os.path.join(log_dir, latest_file)

    print(f"✓ Found state log: {latest_file}")

    try:
        with open(log_path, 'r') as f:
            state_data = json.load(f)[latest_date]

        # Extract portfolio_summary from various possible locations
        portfolio_summary = None

        if 'portfolio_summary' in state_data:
            portfolio_summary = state_data['portfolio_summary']
        else:
            print(f"⚠️  portfolio_summary not found in state log")

        if portfolio_summary:
            print(f"✓ Successfully loaded portfolio_summary from {latest_date}")
            print(f"  Portfolio state:")
            if isinstance(portfolio_summary, dict):
                for key, value in portfolio_summary.items():
                    if isinstance(value, (int, float)) and key in ['cash', 'total_value', 'equity']:
                        print(f"    {key}: ${value:,.2f}")
                    elif not isinstance(value, (dict, list)):
                        print(f"    {key}: {value}")
            return portfolio_summary
        else:
            print(f"⚠️  portfolio_summary not found in state log")
            return None

    except Exception as e:
        print(f"⚠️  Error loading state log: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Run single day backtest with data reloading')

    # Config parameters
    parser.add_argument('--llm_provider', type=str, default='openai',
                        help='LLM provider (default: openai)')
    parser.add_argument('--backend_url', type=str, default='https://api.openai.com/v1',
                        help='Backend URL for LLM API')
    parser.add_argument('--deep_think_llm', type=str, default='gpt-4o-mini',
                        help='Deep thinking LLM model')
    parser.add_argument('--quick_think_llm', type=str, default='gpt-4o-mini',
                        help='Quick thinking LLM model')
    parser.add_argument('--credibility_scorer_llm', type=str, default='',
                        help='Credibility scorer LLM model')
    parser.add_argument('--credibility_scorer_llm_provider', type=str, default='',
                        help='Credibility scorer LLM provider')
    parser.add_argument('--max_debate_rounds', type=int, default=1,
                        help='Maximum debate rounds')
    parser.add_argument('--online_tools', action='store_true', default=True,
                        help='Enable online tools')
    parser.add_argument('--enable_reflection', action='store_true', default=False,
                        help='Enable reflection')
    parser.add_argument('--skip_analysts', action='store_true', default=False,
                        help='Skip analysts')

    # Directory parameters
    parser.add_argument('--save_dir', type=str,
                        default='baseline_analysis/results/tradingagents_strategy',
                        help='Directory to save/load results')

    # Trading parameters
    parser.add_argument('--symbol', type=str, default='NVDA',
                        help='Stock symbol to trade')
    parser.add_argument('--selected_analysts', nargs='+',
                        default=["price", "fundamentals", "market", "social", "news"],
                        help='Selected analysts')
    parser.add_argument('--test_date', type=str,
                        default=datetime.now().strftime('%Y-%m-%d'),
                        help='Test date (YYYY-MM-DD), defaults to today')
    parser.add_argument('--cash', type=float, default=10000.0,
                        help='Initial cash amount')
    parser.add_argument('--commission', type=float, default=0.000,
                        help='Commission rate')
    parser.add_argument('--resume_portfolio', action='store_true', default=True,
                        help='Resume portfolio from previous day state log')
    parser.add_argument('--no_resume_portfolio', dest='resume_portfolio', action='store_false',
                        help='Do not resume portfolio (start fresh)')

    args = parser.parse_args()

    # Setup config
    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = args.llm_provider
    config["backend_url"] = args.backend_url
    config["deep_think_llm"] = args.deep_think_llm
    config["quick_think_llm"] = args.quick_think_llm
    config["credibility_scorer_llm"] = args.credibility_scorer_llm
    config["credibility_scorer_llm_provider"] = args.credibility_scorer_llm_provider
    config["max_debate_rounds"] = args.max_debate_rounds
    config["online_tools"] = args.online_tools
    config["enable_reflection"] = args.enable_reflection
    config["skip_analysts"] = args.skip_analysts

    print("🚀 Single Day Backtest Configuration")
    print("=" * 60)
    print(f"LLM Provider: {config['llm_provider']}")
    print(f"Backend URL: {config['backend_url']}")
    print(f"Deep Think LLM: {config['deep_think_llm']}")
    print(f"Quick Think LLM: {config['quick_think_llm']}")
    print(f"Credibility Scorer LLM: {config['credibility_scorer_llm']}")
    print(f"Max Debate Rounds: {config['max_debate_rounds']}")
    print(f"Online Tools: {config['online_tools']}")
    print(f"Enable Reflection: {config['enable_reflection']}")
    print(f"Skip Analysts: {config['skip_analysts']}")
    print(f"Save Directory: {args.save_dir}")
    print(f"Symbol: {args.symbol}")
    print(f"Test Date: {args.test_date}")
    print(f"Initial Cash: ${args.cash:,.2f}")
    print(f"Commission: {args.commission:.3f}")
    print(f"Resume Portfolio: {args.resume_portfolio}")
    print("=" * 60)

    # Run the backtest
    portfolio_summary = reload_portfolio_summary_from_state_log(
        args.save_dir, args.symbol, args.test_date, args.online_tools)

    start_date = args.test_date
    end_date = (datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
    verifier = run_tradingagents_backtest(
        config, args.selected_analysts, args.symbol, start_date, end_date,
        args.cash, args.commission, args.save_dir, args.enable_reflection, args.skip_analysts,
        portfolio_summary=portfolio_summary, return_verifier=True)

    # historical decisions
    historical_decisions(args.save_dir, args.symbol, verifier)


if __name__ == "__main__":
    main()