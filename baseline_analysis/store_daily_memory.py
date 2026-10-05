#!/usr/bin/env python3
"""
Standalone script to store daily trading information into memory
Use this when running trading once a day to maintain historical memory
"""

import os
import sys
import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tradingagents.agents.utils.memory import FinancialSituationMemory
from tradingagents.default_config import DEFAULT_CONFIG


class DailyMemoryStore:
    """Store daily trading decisions and outcomes into memory"""

    def __init__(self, config: dict, memory_name: str = "trading_memory"):
        """
        Initialize the memory store

        Parameters:
        - config: TradingAgents config dict
        - memory_name: Name for the ChromaDB collection
        """
        self.config = config
        self.memory_name = memory_name

        # Initialize memory instances for different components
        self.bull_memory = FinancialSituationMemory(
            f"{memory_name}_bull", config, time_decay_factor=0.95
        )
        self.bear_memory = FinancialSituationMemory(
            f"{memory_name}_bear", config, time_decay_factor=0.95
        )
        self.trader_memory = FinancialSituationMemory(
            f"{memory_name}_trader", config, time_decay_factor=0.95
        )
        self.invest_judge_memory = FinancialSituationMemory(
            f"{memory_name}_invest_judge", config, time_decay_factor=0.95
        )
        self.risk_manager_memory = FinancialSituationMemory(
            f"{memory_name}_risk_manager", config, time_decay_factor=0.95
        )

    def load_daily_results(self, results_dir: str, symbol: str, trade_date: str = None) -> dict:
        """
        Load the results from a specific trading day

        Parameters:
        - results_dir: Directory containing saved results
        - symbol: Stock symbol
        - trade_date: Specific date to load (YYYY-MM-DD), or None for latest

        Returns:
        - dict with loaded data
        """
        symbol_dir = os.path.join(results_dir, symbol)
        if not os.path.exists(symbol_dir):
            print(f"❌ No results found at {symbol_dir}")
            return None

        files = os.listdir(symbol_dir)

        # If trade_date specified, filter by date
        if trade_date:
            date_str = trade_date.replace('-', '')
            decision_files = [f for f in files if date_str in f and f.endswith('_decisions.json')]
        else:
            decision_files = [f for f in files if f.endswith('_decisions.json')]

        if not decision_files:
            print(f"❌ No decision files found")
            return None

        # Get the latest or specified file
        latest_decision_file = sorted(decision_files)[-1]
        decision_path = os.path.join(symbol_dir, latest_decision_file)

        with open(decision_path, 'r') as f:
            decisions = json.load(f)

        print(f"✓ Loaded decisions from: {latest_decision_file}")
        return {
            'decisions': decisions,
            'file_path': decision_path
        }

    def create_situation_from_decision(self, decision_data: dict, symbol: str) -> str:
        """
        Create a situation description from decision data

        Parameters:
        - decision_data: The decision dict for a specific date
        - symbol: Stock symbol

        Returns:
        - situation string
        """
        # Extract key information from decision
        action = decision_data.get('action', 'HOLD')
        percentage = decision_data.get('percentage', 0)
        price = decision_data.get('price', 'unknown')

        # Build situation description
        situation = f"Symbol: {symbol}\n"
        situation += f"Action: {action}\n"
        situation += f"Position Size: {percentage}%\n"
        situation += f"Price: {price}\n"

        # Add any additional context if available
        if 'fundamentals_report' in decision_data:
            situation += f"\nFundamentals: {decision_data['fundamentals_report'][:500]}\n"
        if 'market_report' in decision_data:
            situation += f"\nMarket: {decision_data['market_report'][:500]}\n"
        if 'news_report' in decision_data:
            situation += f"\nNews: {decision_data['news_report'][:500]}\n"
        if 'sentiment_report' in decision_data:
            situation += f"\nSentiment: {decision_data['sentiment_report'][:500]}\n"

        return situation

    def store_daily_memory(
        self,
        symbol: str,
        trade_date: str,
        situation: str,
        recommendation: str,
        improvement: str = "",
        component: str = "trader"
    ):
        """
        Store a single day's trading information into memory

        Parameters:
        - symbol: Stock symbol
        - trade_date: Trading date (YYYY-MM-DD)
        - situation: Description of market situation
        - recommendation: Summary/recommendation for this situation
        - improvement: Improvement suggestions (optional)
        - component: Which component to store to (bull/bear/trader/invest_judge/risk_manager)
        """
        print(f"\n📝 Storing memory for {symbol} on {trade_date} ({component})")

        # Select appropriate memory based on component
        memory_map = {
            'bull': self.bull_memory,
            'bear': self.bear_memory,
            'trader': self.trader_memory,
            'invest_judge': self.invest_judge_memory,
            'risk_manager': self.risk_manager_memory
        }

        memory = memory_map.get(component.lower(), self.trader_memory)

        # Add to memory
        memory.add_situations(
            [(situation, recommendation, improvement)],
            trade_date=trade_date
        )

        print(f"✓ Memory stored successfully")

    def store_from_results_file(
        self,
        results_dir: str,
        symbol: str,
        trade_date: str = None,
        manual_recommendation: str = None,
        manual_improvement: str = None
    ):
        """
        Load results from file and store into memory

        Parameters:
        - results_dir: Directory containing saved results
        - symbol: Stock symbol
        - trade_date: Specific date (YYYY-MM-DD), or None for latest
        - manual_recommendation: Override recommendation (optional)
        - manual_improvement: Override improvement (optional)
        """
        # Load results
        data = self.load_daily_results(results_dir, symbol, trade_date)
        if not data:
            return False

        decisions = data['decisions']

        # Process each date's decision
        for date_str, decision_info in decisions.items():
            situation = self.create_situation_from_decision(decision_info, symbol)

            # Use manual or extract from decision
            recommendation = manual_recommendation or decision_info.get('summary', f"Action: {decision_info.get('action', 'HOLD')}")
            improvement = manual_improvement or decision_info.get('improvement', "")

            # Store in trader memory by default
            self.store_daily_memory(
                symbol=symbol,
                trade_date=date_str,
                situation=situation,
                recommendation=recommendation,
                improvement=improvement,
                component='trader'
            )

        return True

    def bulk_store_historical_data(
        self,
        results_dir: str,
        symbol: str,
        start_date: str,
        end_date: str
    ):
        """
        Store multiple days of historical data into memory

        Parameters:
        - results_dir: Directory containing saved results
        - symbol: Stock symbol
        - start_date: Start date (YYYY-MM-DD)
        - end_date: End date (YYYY-MM-DD)
        """
        print(f"\n🔄 Bulk storing historical data from {start_date} to {end_date}")

        # Convert to datetime
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')

        # Iterate through each day
        current = start
        stored_count = 0

        while current <= end:
            date_str = current.strftime('%Y-%m-%d')
            success = self.store_from_results_file(results_dir, symbol, date_str)
            if success:
                stored_count += 1
            current += timedelta(days=1)

        print(f"\n✅ Stored {stored_count} days of historical data")

    def get_memory_stats(self, symbol: str, current_date: str = None):
        """
        Get statistics about stored memories

        Parameters:
        - symbol: Stock symbol
        - current_date: Current date (YYYY-MM-DD), defaults to today
        """
        if current_date is None:
            current_date = datetime.now().strftime('%Y-%m-%d')

        print(f"\n📊 Memory Statistics for {symbol}")
        print("=" * 60)

        # Simple situation for querying stats
        query_situation = f"Market analysis for {symbol}"

        components = {
            'Bull': self.bull_memory,
            'Bear': self.bear_memory,
            'Trader': self.trader_memory,
            'Investment Judge': self.invest_judge_memory,
            'Risk Manager': self.risk_manager_memory
        }

        for name, memory in components.items():
            stats = memory.get_memory_stats(query_situation, current_date)
            print(f"\n{name} Memory:")
            print(f"  Total Memories: {stats.get('total_memories', 0)}")
            if stats.get('date_range'):
                print(f"  Date Range: {stats['date_range']}")
                print(f"  Average Age: {stats.get('average_age_days', 0)} days")


def main():
    parser = argparse.ArgumentParser(
        description='Store daily trading information into memory'
    )

    # Mode selection
    parser.add_argument('--mode', type=str, default='single',
                        choices=['single', 'bulk', 'stats'],
                        help='Operation mode: single (one day), bulk (date range), stats (view memory)')

    # Directory and symbol
    parser.add_argument('--save_dir', type=str,
                        default='baseline_analysis/results/tradingagents_strategy',
                        help='Directory containing saved results')
    parser.add_argument('--symbol', type=str, default='NVDA',
                        help='Stock symbol')

    # Date parameters
    parser.add_argument('--trade_date', type=str,
                        default=None,
                        help='Trade date (YYYY-MM-DD) for single mode, or None for latest')
    parser.add_argument('--start_date', type=str,
                        help='Start date (YYYY-MM-DD) for bulk mode')
    parser.add_argument('--end_date', type=str,
                        help='End date (YYYY-MM-DD) for bulk mode')

    # Manual inputs (optional)
    parser.add_argument('--recommendation', type=str, default=None,
                        help='Manual recommendation/summary')
    parser.add_argument('--improvement', type=str, default=None,
                        help='Manual improvement suggestion')

    # Config parameters
    parser.add_argument('--memory_name', type=str, default='trading_memory',
                        help='Name for the memory collection')
    parser.add_argument('--online_tools', type=str, default='True',
                        help='Online tools setting (True/False)')
    parser.add_argument('--deep_think_llm', type=str, default='gpt-4o-mini',
                        help='Deep think LLM model used')
    parser.add_argument('--enable_reflection', type=str, default='False',
                        help='Reflection setting (True/False)')

    args = parser.parse_args()

    # Build final directory path
    final_dir = os.path.join(
        args.save_dir,
        args.online_tools,
        args.deep_think_llm,
        f"reflection_{args.enable_reflection}"
    )

    # Setup config
    config = DEFAULT_CONFIG.copy()

    print("💾 Daily Memory Store Tool")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Results Directory: {final_dir}")
    print(f"Symbol: {args.symbol}")
    print(f"Memory Name: {args.memory_name}")
    print("=" * 60)

    # Initialize memory store
    store = DailyMemoryStore(config=config, memory_name=args.memory_name)

    # Execute based on mode
    if args.mode == 'single':
        # Store single day
        success = store.store_from_results_file(
            results_dir=final_dir,
            symbol=args.symbol,
            trade_date=args.trade_date,
            manual_recommendation=args.recommendation,
            manual_improvement=args.improvement
        )
        if success:
            print("\n✅ Single day memory stored successfully!")
        else:
            print("\n❌ Failed to store memory")

    elif args.mode == 'bulk':
        # Store date range
        if not args.start_date or not args.end_date:
            print("❌ Error: --start_date and --end_date required for bulk mode")
            return

        store.bulk_store_historical_data(
            results_dir=final_dir,
            symbol=args.symbol,
            start_date=args.start_date,
            end_date=args.end_date
        )

    elif args.mode == 'stats':
        # Show memory statistics
        store.get_memory_stats(
            symbol=args.symbol,
            current_date=args.trade_date or datetime.now().strftime('%Y-%m-%d')
        )

    print("\n✅ Operation completed!")


if __name__ == "__main__":
    main()
