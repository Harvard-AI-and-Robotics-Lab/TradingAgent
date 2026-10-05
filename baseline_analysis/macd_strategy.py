#!/usr/bin/env python3
"""
MACD Strategy using backtesting package
"""
import os
import talib
import pandas as pd
import yfinance as yf
import numpy as np
from bokeh.io import export_png
from datetime import datetime, timedelta
from backtesting import Backtest, Strategy
from backtesting.lib import crossover


class MACDStrategy(Strategy):
    """
    MACD (Moving Average Convergence Divergence) Strategy
    - Buy when MACD line crosses above signal line
    - Sell when MACD line crosses below signal line
    """

    # Strategy parameters
    fast_period = 12
    slow_period = 26
    signal_period = 9

    def init(self):
        """Initialize strategy indicators"""
        # Calculate MACD using talib
        self.macd, self.macd_signal, self.macd_hist = self.I(
            talib.MACD,
            self.data.Close,
            fastperiod=self.fast_period,
            slowperiod=self.slow_period,
            signalperiod=self.signal_period
        )

    def next(self):
        # TA-Lib returns NaN during warm-up
        if np.isnan(self.macd[-1]) or np.isnan(self.macd_signal[-1]):
            return

        # # ✅ Force liquidation on the last bar (version-safe)
        # if len(self.data.Close) == len(self.data):
        #     if self.position:
        #         self.position.close()
        #     return

        # Buy: MACD crosses above Signal
        if (not self.position) and crossover(self.macd, self.macd_signal):
            self.buy()
            print(f"Bought {self.data.Close[-1]}")

        # Sell/Close: MACD crosses below Signal
        elif self.position and crossover(self.macd_signal, self.macd):
            self.position.close()
            print(f"Sold {self.data.Close[-1]}")


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


def run_macd_backtest(symbol='AAPL', start_date='2024-01-01', end_date='2024-04-01',
                      cash=10000, commission=0.000):
    """
    Run MACD backtest

    Args:
        symbol: Stock symbol to trade
        start_date: Start date for backtest
        end_date: End date for backtest
        cash: Initial cash amount
        commission: Commission rate (0.002 = 0.2%)
    """
    #

    start_date = datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=int(max(MACDStrategy.fast_period, MACDStrategy.slow_period, MACDStrategy.signal_period) * 365/252 +12))

    print(f"📊 Running MACD Strategy with backtesting package...")
    print("-" * 60)
    print(f"Symbol: {symbol}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Initial Cash: ${cash:,}")
    print(f"Commission: {commission*100}%")
    print(f"MACD Parameters: Fast={MACDStrategy.fast_period}, Slow={MACDStrategy.slow_period}, Signal={MACDStrategy.signal_period}")
    print("-" * 60)

    # Download data
    print(f"Downloading {symbol} data from {start_date} to {end_date}...")
    data = get_stock_data(symbol, start_date, end_date)

    if data is None or data.empty:
        print("❌ Failed to download data")
        return None

    # Run backtest
    bt = Backtest(data, MACDStrategy, cash=cash, commission=commission, finalize_trades=True)
    results = bt.run()
    print(results["_equity_curve"])


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


def optimize_macd_parameters(data, cash=10000, commission=0.002):
    """
    Optimize MACD parameters using grid search
    """
    print("\n🔍 Optimizing MACD parameters...")

    bt = Backtest(data, MACDStrategy, cash=cash, commission=commission)

    # Parameter optimization ranges
    optimization_results = bt.optimize(
        fast_period=range(8, 16, 2),        # 8, 10, 12, 14
        slow_period=range(20, 32, 4),       # 20, 24, 28
        signal_period=range(7, 12, 2),      # 7, 9, 11
        maximize='Return [%]'
    )

    print(f"📊 Best parameters found:")
    print(f"   Fast Period: {optimization_results._strategy.fast_period}")
    print(f"   Slow Period: {optimization_results._strategy.slow_period}")
    print(f"   Signal Period: {optimization_results._strategy.signal_period}")
    print(f"   Return: {optimization_results['Return [%]']:.2f}%")

    return optimization_results


if __name__ == "__main__":
    symbols = ['AAPL', 'GOOG', 'NVDA']
    start_dates = ['2024-01-01', '2025-06-01', '2026-01-01']
    end_dates = ['2024-04-01', '2025-09-01', '2026-02-19']

    for symbol in symbols:
        outdir = f"baseline_analysis/results/baseline/macd/{symbol}"
        os.makedirs(outdir, exist_ok=True)

        for start_date, end_date in zip(start_dates, end_dates):
            # Run the backtest
            bt, results = run_macd_backtest(symbol, start_date=start_date, end_date=end_date)

            if bt is not None:
                # Generate HTML plot
                print("\n📊 Generating interactive plot...")
                plot_file = f"{outdir}/macd_backtest_{start_date}_{end_date}.html"
                bt.plot(filename=plot_file, open_browser=False)
                print(f"📈 Interactive plot saved to: {plot_file}")

                # Save results to file
                results_file = f"{outdir}/macd_results_{start_date}_{end_date}.txt"
                with open(results_file, 'w') as f:
                    f.write("MACD Strategy - Backtest Results\n")
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
                #     optimized_results = optimize_macd_parameters(data)

                print("\n✅ MACD backtest completed successfully!")
