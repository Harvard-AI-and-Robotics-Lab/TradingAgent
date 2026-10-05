#!/usr/bin/env python3
"""
RSI Strategy using backtesting package
"""
import os
import talib
import pandas as pd
import yfinance as yf
from backtesting import Backtest, Strategy


class RSIStrategy(Strategy):
    """
    RSI (Relative Strength Index) Strategy
    - Buy when RSI is oversold (< 30) and starts to rise
    - Sell when RSI is overbought (> 70) and starts to fall
    """

    # Strategy parameters
    rsi_period = 14
    oversold_threshold = 30
    overbought_threshold = 70

    def init(self):
        """Initialize strategy indicators"""
        # Calculate RSI using talib
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period)

    def next(self):
        """Execute strategy logic for each bar"""
        current_rsi = self.rsi[-1]
        previous_rsi = self.rsi[-2] if len(self.rsi) > 1 else current_rsi

        # Buy signal: RSI is oversold and starting to rise
        if (current_rsi < self.oversold_threshold or
            (previous_rsi < self.oversold_threshold and current_rsi > previous_rsi)):
            if not self.position:
                self.buy()

        # Sell signal: RSI is overbought and starting to fall
        elif (current_rsi > self.overbought_threshold or
              (previous_rsi > self.overbought_threshold and current_rsi < previous_rsi)):
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


def run_rsi_backtest(symbol='AAPL', start_date='2024-01-01', end_date='2024-03-29',
                     cash=10000, commission=0.002):
    """
    Run RSI backtest

    Args:
        symbol: Stock symbol to trade
        start_date: Start date for backtest
        end_date: End date for backtest
        cash: Initial cash amount
        commission: Commission rate (0.002 = 0.2%)
    """
    print(f"📊 Running RSI Strategy with backtesting package...")
    print("-" * 60)
    print(f"Symbol: {symbol}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Initial Cash: ${cash:,}")
    print(f"Commission: {commission*100}%")
    print(f"RSI Parameters: Period={RSIStrategy.rsi_period}, Oversold={RSIStrategy.oversold_threshold}, Overbought={RSIStrategy.overbought_threshold}")
    print("-" * 60)

    # Download data
    print(f"Downloading {symbol} data from {start_date} to {end_date}...")
    data = get_stock_data(symbol, start_date, end_date)

    if data is None or data.empty:
        print("❌ Failed to download data")
        return None

    # Run backtest
    bt = Backtest(data, RSIStrategy, cash=cash, commission=commission)
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


def optimize_rsi_parameters(data, cash=10000, commission=0.002):
    """
    Optimize RSI parameters using grid search
    """
    print("\n🔍 Optimizing RSI parameters...")

    bt = Backtest(data, RSIStrategy, cash=cash, commission=commission)

    # Parameter optimization ranges
    optimization_results = bt.optimize(
        rsi_period=range(10, 21, 2),           # 10, 12, 14, 16, 18, 20
        oversold_threshold=range(25, 36, 5),    # 25, 30, 35
        overbought_threshold=range(65, 76, 5),  # 65, 70, 75
        maximize='Return [%]'
    )

    print(f"📊 Best parameters found:")
    print(f"   RSI Period: {optimization_results._strategy.rsi_period}")
    print(f"   Oversold Threshold: {optimization_results._strategy.oversold_threshold}")
    print(f"   Overbought Threshold: {optimization_results._strategy.overbought_threshold}")
    print(f"   Return: {optimization_results['Return [%]']:.2f}%")

    return optimization_results


if __name__ == "__main__":
    symbol = 'AAPL'
    os.makedirs(f"results/{symbol}", exist_ok=True)

    # Run the backtest
    bt, results = run_rsi_backtest(symbol)

    if bt is not None:
        # Generate HTML plot
        print("\n📊 Generating interactive plot...")
        plot_file = f"results/{symbol}/rsi_backtest.html"
        bt.plot(filename=plot_file)
        print(f"📈 Interactive plot saved to: {plot_file}")

        # Save results to file
        results_file = f"results/{symbol}/rsi_results.txt"
        with open(results_file, 'w') as f:
            f.write("RSI Strategy - Backtest Results\n")
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
        #     optimized_results = optimize_rsi_parameters(data)

        print("\n✅ RSI backtest completed successfully!")