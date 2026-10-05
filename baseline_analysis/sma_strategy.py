#!/usr/bin/env python3
"""
SMA (Simple Moving Average) Strategy using backtesting package
"""
import os
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA


class SMAStrategy(Strategy):
    """
    SMA (Simple Moving Average) Strategy
    - Buy when short-term SMA crosses above long-term SMA (golden cross)
    - Sell when short-term SMA crosses below long-term SMA (death cross)
    """

    # Strategy parameters
    short_window = 10    # Short-term moving average period
    long_window = 20     # Long-term moving average period

    def init(self):
        """Initialize strategy indicators"""

        # Use built-in SMA; ensures proper dtype/shape
        self.sma_short = self.I(SMA, self.data.Close, self.short_window)
        self.sma_long  = self.I(SMA, self.data.Close, self.long_window)

    def next(self):
        # Skip warm-up (need valid SMA values)
        if np.isnan(self.sma_long[-1]) or np.isnan(self.sma_short[-1]):
            return

        # Buy: short crosses above long
        if (not self.position) and crossover(self.sma_short, self.sma_long):
            self.buy()

        # Sell/Close: short crosses below long
        elif self.position and crossover(self.sma_long, self.sma_short):
            self.position.close()

def get_stock_data(symbol, start_date, end_date):
    """Download stock data from Yahoo Finance"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date)

        if data.empty:
            raise ValueError(f"No data found for {symbol}")

        # Rename columns to match backtesting package requirements
        data.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']

        # Keep only OHLCV columns
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

        print(f"Successfully downloaded {len(data)} trading days of data")
        return data

    except Exception as e:
        print(f"Error downloading data: {e}")
        return None


def run_sma_backtest(symbol='AAPL', start_date='2024-01-01', end_date='2024-03-29',
                     cash=10000, commission=0.002):
    """
    Run SMA backtest

    Args:
        symbol: Stock symbol to trade
        start_date: Start date for backtest
        end_date: End date for backtest
        cash: Initial cash amount
        commission: Commission rate (0.002 = 0.2%)
    """
    start_date = datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=int(SMAStrategy.long_window * 365/252 + 2))

    print(f"📊 Running SMA Strategy with backtesting package...")
    print("-" * 60)
    print(f"Symbol: {symbol}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Initial Cash: ${cash:,}")
    print(f"Commission: {commission*100}%")
    print(f"SMA Parameters: Short={SMAStrategy.short_window}, Long={SMAStrategy.long_window}")
    print("-" * 60)

    # Download data
    print(f"Downloading {symbol} data from {start_date} to {end_date}...")
    data = get_stock_data(symbol, start_date, end_date)

    if data is None or data.empty:
        print("❌ Failed to download data")
        return None

    # Run backtest
    bt = Backtest(data, SMAStrategy, cash=cash, commission=commission)
    results = bt.run()

    # Print results
    print("\n📈 BACKTEST RESULTS")
    print("=" * 50)
    print(f"Start Date:           {results['Start']}")
    print(f"End Date:             {results['End']}")
    print(f"Duration:             {results['Duration']}")
    print(f"Initial Equity:       ${cash:,.2f}")
    print(f"Final Equity:         ${results['Equity Final [$]']:.2f}")
    print(f"Total Return:         {results['Return [%]']:.2f}%")
    print(f"Max Drawdown:         {results['Max. Drawdown [%]']:.2f}%")
    print(f"Annualized Return:    {results.get('Return (Ann.) [%]', 'N/A'):.2f}%" if isinstance(results.get('Return (Ann.) [%]'), (int, float)) else f"Annualized Return:    N/A")
    print(f"Volatility:           {results.get('Volatility (Ann.) [%]', 'N/A'):.2f}%" if isinstance(results.get('Volatility (Ann.) [%]'), (int, float)) else f"Volatility:           N/A")
    print(f"Sharpe Ratio:         {results.get('Sharpe Ratio', 'N/A'):.3f}" if isinstance(results.get('Sharpe Ratio'), (int, float)) else f"Sharpe Ratio:         N/A")
    print(f"Number of Trades:     {results['# Trades']}")
    print(f"Win Rate:             {results.get('Win Rate [%]', 'N/A'):.2f}%" if isinstance(results.get('Win Rate [%]'), (int, float)) else f"Win Rate:             N/A")
    print(f"Best Trade:           {results.get('Best Trade [%]', 'N/A'):.2f}%" if isinstance(results.get('Best Trade [%]'), (int, float)) else f"Best Trade:           N/A")
    print(f"Worst Trade:          {results.get('Worst Trade [%]', 'N/A'):.2f}%" if isinstance(results.get('Worst Trade [%]'), (int, float)) else f"Worst Trade:          N/A")
    print(f"Avg Trade:            {results.get('Avg. Trade [%]', 'N/A'):.2f}%" if isinstance(results.get('Avg. Trade [%]'), (int, float)) else f"Avg Trade:            N/A")
    print("=" * 50)

    return bt, results


def optimize_sma_parameters(data, cash=10000, commission=0.002):
    """
    Optimize SMA parameters using grid search
    """
    print("\n🔍 Optimizing SMA parameters...")

    bt = Backtest(data, SMAStrategy, cash=cash, commission=commission, finalize_trades=True)

    # Parameter optimization ranges
    optimization_results = bt.optimize(
        short_window=range(10, 31, 5),     # 10, 15, 20, 25, 30
        long_window=range(40, 101, 10),    # 40, 50, 60, 70, 80, 90, 100
        maximize='Return [%]'
    )

    print(f"📊 Best parameters found:")
    print(f"   Short Window: {optimization_results._strategy.short_window} days")
    print(f"   Long Window: {optimization_results._strategy.long_window} days")
    print(f"   Return: {optimization_results['Return [%]']:.2f}%")

    return optimization_results


if __name__ == "__main__":
    symbols = ['AAPL'] #, 'GOOG', 'NVDA']
    start_dates = ['2024-01-01', '2025-06-01', '2026-01-01']
    end_dates = ['2024-04-01', '2025-09-01', '2026-02-19']

    for symbol in symbols:
        outdir = f"baseline_analysis/results/baseline/sma/{symbol}"
        os.makedirs(outdir, exist_ok=True)
        for start_date, end_date in zip(start_dates, end_dates):
            # Run the backtest
            bt, results = run_sma_backtest(symbol, start_date=start_date, end_date=end_date)

            if bt is not None:
                # Generate HTML plot
                print("\n📊 Generating interactive plot...")
                plot_file = f"{outdir}/sma_backtest_{start_date}_{end_date}.html"
                bt.plot(filename=plot_file, open_browser=False)
                print(f"📈 Interactive plot saved to: {plot_file}")

                # Save results to file
                results_file = f"{outdir}/sma_results_{start_date}_{end_date}.txt"
                with open(results_file, 'w') as f:
                    f.write("SMA Strategy - Backtest Results\n")
                    f.write("=" * 50 + "\n")
                    for key, value in results.items():
                        f.write(f"{key}: {value}\n")

                print(f"📋 Results saved to: {results_file}")

                # Optional: Run parameter optimization
                print("\n🔧 Run parameter optimization? (y/n): ", end="")
                # For automated running, skip optimization
                # optimize_choice = input().lower()
                # if optimize_choice == 'y':
                #     data = get_stock_data('AAPL', '2024-01-01', '2024-03-29')
                #     optimized_results = optimize_sma_parameters(data)

                print("\n✅ SMA backtest completed successfully!")
