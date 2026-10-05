#!/usr/bin/env python3
"""
KDJ Strategy using backtesting package
"""
import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import talib


class KDJStrategy(Strategy):
    """
    KDJ (Stochastic Oscillator) Strategy
    - Buy when %K crosses above %D in oversold region (< 20)
    - Sell when %K crosses below %D in overbought region (> 80)
    """

    # Strategy parameters
    k_period = 9
    d_period = 3
    oversold_threshold = 20
    overbought_threshold = 80

    def init(self):
        """Initialize strategy indicators"""
        # Calculate Stochastic %K and %D using talib
        self.k_percent, self.d_percent = self.I(
            talib.STOCH,
            self.data.High,
            self.data.Low,
            self.data.Close,
            fastk_period=self.k_period,
            slowk_period=self.d_period,
            slowk_matype=0,
            slowd_period=self.d_period,
            slowd_matype=0
        )

        # Calculate J value: J = 3*K - 2*D
        self.j_percent = self.I(lambda k, d: 3*k - 2*d, self.k_percent, self.d_percent)

    def next(self):
        k = self.k_percent
        d = self.d_percent

        # Ensure valid values (TA-Lib outputs NaN for the warm-up period)
        if np.isnan(k[-1]) or np.isnan(d[-1]):
            return

        # Buy: K crosses above D AND K is in oversold region
        if (not self.position) and crossover(k, d) and (k[-1] < self.oversold_threshold):
            self.buy()

        # Sell/Close: K crosses below D AND K is in overbought region
        elif self.position and crossover(d, k) and (k[-1] > self.overbought_threshold):
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


def run_kdj_backtest(symbol='AAPL', start_date='2024-01-01', end_date='2024-04-01',
                     cash=10000, commission=0.002):
    """
    Run KDJ backtest

    Args:
        symbol: Stock symbol to trade
        start_date: Start date for backtest
        end_date: End date for backtest
        cash: Initial cash amount
        commission: Commission rate (0.002 = 0.2%)
    """
    start_date = datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=int(max(KDJStrategy.k_period, KDJStrategy.d_period) * 365/252 + 6))

    print(f"📊 Running KDJ Strategy with backtesting package...")
    print("-" * 60)
    print(f"Symbol: {symbol}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Initial Cash: ${cash:,}")
    print(f"Commission: {commission*100}%")
    print(f"KDJ Parameters: K={KDJStrategy.k_period}, D={KDJStrategy.d_period}, Oversold={KDJStrategy.oversold_threshold}, Overbought={KDJStrategy.overbought_threshold}")
    print("-" * 60)

    # Download data
    print(f"Downloading {symbol} data from {start_date} to {end_date}...")
    data = get_stock_data(symbol, start_date, end_date)

    if data is None or data.empty:
        print("❌ Failed to download data")
        return None

    # Run backtest
    bt = Backtest(data, KDJStrategy, cash=cash, commission=commission, finalize_trades=True)
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


def optimize_kdj_parameters(data, cash=10000, commission=0.002):
    """
    Optimize KDJ parameters using grid search
    """
    print("\n🔍 Optimizing KDJ parameters...")

    bt = Backtest(data, KDJStrategy, cash=cash, commission=commission, finalize_trades=True)

    # Parameter optimization ranges
    optimization_results = bt.optimize(
        k_period=range(10, 21, 2),             # 10, 12, 14, 16, 18, 20
        d_period=range(2, 6, 1),               # 2, 3, 4, 5
        oversold_threshold=range(15, 26, 5),    # 15, 20, 25
        overbought_threshold=range(75, 86, 5),  # 75, 80, 85
        maximize='Return [%]'
    )

    print(f"📊 Best parameters found:")
    print(f"   K Period: {optimization_results._strategy.k_period}")
    print(f"   D Period: {optimization_results._strategy.d_period}")
    print(f"   Oversold Threshold: {optimization_results._strategy.oversold_threshold}")
    print(f"   Overbought Threshold: {optimization_results._strategy.overbought_threshold}")
    print(f"   Return: {optimization_results['Return [%]']:.2f}%")

    return optimization_results


if __name__ == "__main__":
    start_dates = ['2024-01-01', '2025-06-01', '2026-01-01']
    end_dates = ['2024-04-01', '2025-09-01', '2026-02-19']
    symbols = ['AAPL', 'GOOG', 'NVDA']

    for symbol in symbols:
        plot_dir = f"baseline_analysis/results/baseline/kdj/{symbol}"
        os.makedirs(plot_dir, exist_ok=True)

        for start_date, end_date in zip(start_dates, end_dates):
            # Run the backtest
            bt, results = run_kdj_backtest(symbol, start_date=start_date, end_date=end_date)

            if bt is not None:
                # Generate HTML plot
                print("\n📊 Generating interactive plot...")
                bt.plot(filename=f"{plot_dir}/kdj_backtest_{start_date}_{end_date}.html", open_browser=False)
                print(f"📈 Interactive plot saved to: {plot_dir}/kdj_backtest_{start_date}_{end_date}.html")

                # Save results to file
                results_file = f"{plot_dir}/kdj_results_{start_date}_{end_date}.txt"
                with open(results_file, 'w') as f:
                    f.write("KDJ Strategy - Backtest Results\n")
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
                #     optimized_results = optimize_kdj_parameters(data)

                print("\n✅ KDJ backtest completed successfully!")
