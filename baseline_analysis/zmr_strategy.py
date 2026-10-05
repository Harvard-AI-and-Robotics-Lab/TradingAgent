#!/usr/bin/env python3
"""
ZMR (Zero Mean Reversion) Strategy using backtesting package
"""
import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from backtesting import Backtest, Strategy
import numpy as np

class ZMRStrategy(Strategy):
    """
    ZMR (Zero Mean Reversion) Strategy
    - Buy when price is significantly below its rolling mean
    - Sell when price is significantly above its rolling mean
    """

    # Strategy parameters
    lookback_period = 20
    entry_threshold = 2.0  # Standard deviations from mean
    exit_threshold = 0.5   # Standard deviations from mean

    def init(self):
        """Initialize strategy indicators"""
        # Calculate rolling mean and standard deviation
        self.rolling_mean = self.I(
            lambda x: pd.Series(x).rolling(window=self.lookback_period).mean(),
            self.data.Close
        )

        self.rolling_std = self.I(
            lambda x: pd.Series(x).rolling(window=self.lookback_period).std(),
            self.data.Close
        )

        # Calculate Z-score (zero mean reversion indicator)
        self.z_score = self.I(
            lambda close, mean, std: (close - mean) / std,
            self.data.Close, self.rolling_mean, self.rolling_std
        )

    def next(self):
        """Execute strategy logic for each bar"""
        # Skip if we don't have enough data
        if len(self.z_score) < self.lookback_period:
            return

        current_z_score = self.z_score[-1]

        # Buy signal: Price is significantly below mean (oversold)
        if current_z_score < -self.entry_threshold:
            if not self.position:
                self.buy()

        # Sell signal: Price is significantly above mean (overbought)
        elif current_z_score > self.entry_threshold:
            if self.position:
                self.sell()

        # Exit signal: Price returns close to mean
        elif abs(current_z_score) < self.exit_threshold:
            if self.position:
                self.sell()


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


def run_zmr_backtest(symbol='AAPL', start_date='2024-01-01', end_date='2024-03-29',
                     cash=10000, commission=0.002):
    """
    Run ZMR backtest

    Args:
        symbol: Stock symbol to trade
        start_date: Start date for backtest
        end_date: End date for backtest
        cash: Initial cash amount
        commission: Commission rate (0.002 = 0.2%)
    """
    start_date = datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=int(ZMRStrategy.lookback_period * 365/252 + 2))

    print(f"📊 Running ZMR Strategy with backtesting package...")
    print("-" * 60)
    print(f"Symbol: {symbol}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Initial Cash: ${cash:,}")
    print(f"Commission: {commission*100}%")
    print(f"ZMR Parameters: Lookback={ZMRStrategy.lookback_period}, Entry={ZMRStrategy.entry_threshold}σ, Exit={ZMRStrategy.exit_threshold}σ")
    print("-" * 60)

    # Download data
    print(f"Downloading {symbol} data from {start_date} to {end_date}...")
    data = get_stock_data(symbol, start_date, end_date)

    if data is None or data.empty:
        print("❌ Failed to download data")
        return None

    # Run backtest
    bt = Backtest(data, ZMRStrategy, cash=cash, commission=commission, finalize_trades=True)
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


def optimize_zmr_parameters(data, cash=10000, commission=0.002):
    """
    Optimize ZMR parameters using grid search
    """
    print("\n🔍 Optimizing ZMR parameters...")

    bt = Backtest(data, ZMRStrategy, cash=cash, commission=commission, finalize_trades=True)

    # Parameter optimization ranges
    optimization_results = bt.optimize(
        lookback_period=range(10, 31, 5),      # 10, 15, 20, 25, 30
        entry_threshold=[1.5, 2.0, 2.5],      # Standard deviations
        exit_threshold=[0.2, 0.5, 0.8],       # Standard deviations
        maximize='Return [%]'
    )

    print(f"📊 Best parameters found:")
    print(f"   Lookback Period: {optimization_results._strategy.lookback_period}")
    print(f"   Entry Threshold: {optimization_results._strategy.entry_threshold}σ")
    print(f"   Exit Threshold: {optimization_results._strategy.exit_threshold}σ")
    print(f"   Return: {optimization_results['Return [%]']:.2f}%")

    return optimization_results


if __name__ == "__main__":
    symbols = ['AAPL', 'GOOG', 'NVDA']
    start_dates = ['2024-01-01', '2025-06-01', '2026-01-01']
    end_dates = ['2024-04-01', '2025-09-01', '2026-02-19']

    for symbol in symbols:
        outdir = f"baseline_analysis/results/baseline/zmr/{symbol}"
        os.makedirs(outdir, exist_ok=True)

        # Run the backtest
        for start_date, end_date in zip(start_dates, end_dates):
            bt, results = run_zmr_backtest(symbol, start_date=start_date, end_date=end_date)

            if bt is not None:
                # Generate HTML plot
                print("\n📊 Generating interactive plot...")
                plot_file = f"{outdir}/zmr_backtest_{start_date}_{end_date}.html"
                bt.plot(filename=plot_file, open_browser=False)
                print(f"📈 Interactive plot saved to: {plot_file}")

                # Save results to file
                results_file = f"{outdir}/zmr_results_{start_date}_{end_date}.txt"
                with open(results_file, 'w') as f:
                    f.write("ZMR Strategy - Backtest Results\n")
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
                #     optimized_results = optimize_zmr_parameters(data)

                print("\n✅ ZMR backtest completed successfully!")
