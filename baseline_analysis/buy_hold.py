#!/usr/bin/env python3
"""
Buy & Hold Strategy using backtesting package
"""
import os
import pandas as pd
import yfinance as yf
from backtesting import Backtest, Strategy


class BuyHold(Strategy):
    """
    Buy and Hold strategy - buy at the beginning and hold until the end
    """

    def init(self):
        """Initialize strategy"""
        self.bought = False

    def next(self):
        """Execute strategy logic for each bar"""
        # Buy on the first bar if we haven't bought yet
        if not self.bought and not self.position:
            self.buy()
            self.bought = True


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


def run_buy_hold_backtest(symbol='AAPL', start_date='2024-01-01', end_date='2024-03-29',
                         cash=10000, commission=0.002):
    """
    Run Buy & Hold backtest

    Args:
        symbol: Stock symbol to trade
        start_date: Start date for backtest
        end_date: End date for backtest
        cash: Initial cash amount
        commission: Commission rate (0.002 = 0.2%)
    """
    print(f"📊 Running Buy & Hold Strategy with backtesting package...")
    print("-" * 60)
    print(f"Symbol: {symbol}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Initial Cash: ${cash:,}")
    print(f"Commission: {commission*100}%")
    print("-" * 60)

    # Download data
    print(f"Downloading {symbol} data from {start_date} to {end_date}...")
    data = get_stock_data(symbol, start_date, end_date)

    if data is None or data.empty:
        print("❌ Failed to download data")
        return None

    # Run backtest
    bt = Backtest(data, BuyHold, cash=cash, commission=commission, finalize_trades=True)
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


if __name__ == "__main__":
    symbols = ['AAPL', 'GOOG', 'NVDA']
    start_dates = ['2024-01-01', '2025-06-01', '2026-01-01']
    end_dates = ['2024-04-01', '2025-09-01', '2026-02-19']

    for symbol in symbols:
        outdir = f"baseline_analysis/results/baseline/buy_hold/{symbol}"
        os.makedirs(outdir, exist_ok=True)

        for start_date, end_date in zip(start_dates, end_dates):
            # Run the backtest
            bt, results = run_buy_hold_backtest(symbol, start_date=start_date, end_date=end_date)

            if bt is not None:
                # Generate HTML plot
                print("\n📊 Generating interactive plot...")
                plot_file = f"{outdir}/buy_hold_backtest_{start_date}_{end_date}.html"
                bt.plot(filename=plot_file, open_browser=False)
                print(f"📈 Interactive plot saved to: {plot_file}")

                # Save results to file
                results_file = f"{outdir}/buy_hold_results_{start_date}_{end_date}.txt"
                with open(results_file, 'w') as f:
                    f.write("Buy & Hold Strategy - Backtest Results\n")
                    f.write("=" * 50 + "\n")
                    for key, value in results.items():
                        f.write(f"{key}: {value}\n")

                print(f"📋 Results saved to: {results_file}")
                print("\n✅ Buy & Hold backtest completed successfully!")
