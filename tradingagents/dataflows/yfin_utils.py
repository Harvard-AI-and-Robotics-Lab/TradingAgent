# gets data/stats

import yfinance as yf
from typing import Annotated, Callable, Any, Optional
from pandas import DataFrame
import pandas as pd
from functools import wraps
import os
import re
from datetime import datetime
from .config import DATA_DIR, DATA_DIR_PREFIX, DATA_CACHE_FILE_YFIN
from .utils import save_output, SavePathType, decorate_all_methods


def init_ticker(func: Callable) -> Callable:
    """Decorator to initialize yf.Ticker and pass it to the function."""

    @wraps(func)
    def wrapper(symbol: Annotated[str, "ticker symbol"], *args, **kwargs) -> Any:
        ticker = yf.Ticker(symbol)
        return func(ticker, *args, **kwargs)

    return wrapper

def get_data_path_yfin(symbol, curr_date=None, start_date=None, end_date=None, data_path=None):
    """
    Get the data path for the yfin data
    Args:
        symbol (str): the symbol of the company
        curr_date (str): the current date YYYY-MM-DD
        start_date (str): the start date YYYY-MM-DD
        end_date (str): the end date YYYY-MM-DD
    Returns:
        str: the data path
    """
    if data_path is None:
        data_path = os.path.join(
            DATA_DIR,
            DATA_DIR_PREFIX['market_data'],
            f"{symbol}-{DATA_CACHE_FILE_YFIN}"
        )

    if os.path.exists(data_path):
        return data_path
    else:
        print(f"\n\nWarning: Data path {data_path} does not exist\n\n")
        print(f"Searching for data path for {symbol} in {os.path.join(DATA_DIR, DATA_DIR_PREFIX['market_data'])}")
        candidate_data_paths = []
        pattern = re.compile(
            r"^" + re.escape(symbol) + r"-YFin-data-(\d{4}-\d{2}-\d{2})-(\d{4}-\d{2}-\d{2})\.csv$"
        )
        market_dir = os.path.join(DATA_DIR, DATA_DIR_PREFIX['market_data'])
        for filename in os.listdir(market_dir):
            match = pattern.match(filename)
            if not match:
                continue
            start_date_f, end_date_f = match.groups()
            start_date_f = datetime.strptime(start_date_f, "%Y-%m-%d")
            end_date_f = datetime.strptime(end_date_f, "%Y-%m-%d")

            if curr_date is not None:
                if isinstance(curr_date, str):
                    curr_date = datetime.strptime(curr_date, "%Y-%m-%d")
                if curr_date < start_date_f or curr_date > end_date_f:
                    continue

            if start_date is not None and end_date is not None:
                if isinstance(start_date, str):
                    start_date = datetime.strptime(start_date, "%Y-%m-%d")
                if isinstance(end_date, str):
                    end_date = datetime.strptime(end_date, "%Y-%m-%d")
                if start_date < start_date_f or end_date > end_date_f:
                    continue
            data_path = os.path.join(market_dir, filename)
            candidate_data_paths.append(data_path)

        if len(candidate_data_paths) == 0:
            raise Exception(f"No dcata path found for {symbol}")
        if len(candidate_data_paths) > 1:
            print(f"Multiple data paths found for {symbol}: {candidate_data_paths}")
            data_path = candidate_data_paths[0]

        print(f"\n\nUsing data path {data_path} for {symbol}\n\n")

    return data_path


@decorate_all_methods(init_ticker)
class YFinanceUtils:

    def get_stock_data(
        symbol: Annotated[str, "ticker symbol"],
        start_date: Annotated[
            str, "start date for retrieving stock price data, YYYY-mm-dd"
        ],
        end_date: Annotated[
            str, "end date for retrieving stock price data, YYYY-mm-dd"
        ],
        save_path: SavePathType = None,
    ) -> DataFrame:
        """retrieve stock price data for designated ticker symbol"""
        ticker = symbol
        # add one day to the end_date so that the data range is inclusive
        end_date = pd.to_datetime(end_date) + pd.DateOffset(days=1)
        end_date = end_date.strftime("%Y-%m-%d")
        stock_data = ticker.history(start=start_date, end=end_date)
        # save_output(stock_data, f"Stock data for {ticker.ticker}", save_path)
        return stock_data

    def get_stock_info(
        symbol: Annotated[str, "ticker symbol"],
    ) -> dict:
        """Fetches and returns latest stock information."""
        ticker = symbol
        stock_info = ticker.info
        return stock_info

    def get_company_info(
        symbol: Annotated[str, "ticker symbol"],
        save_path: Optional[str] = None,
    ) -> DataFrame:
        """Fetches and returns company information as a DataFrame."""
        ticker = symbol
        info = ticker.info
        company_info = {
            "Company Name": info.get("shortName", "N/A"),
            "Industry": info.get("industry", "N/A"),
            "Sector": info.get("sector", "N/A"),
            "Country": info.get("country", "N/A"),
            "Website": info.get("website", "N/A"),
        }
        company_info_df = DataFrame([company_info])
        if save_path:
            company_info_df.to_csv(save_path)
            print(f"Company info for {ticker.ticker} saved to {save_path}")
        return company_info_df

    def get_stock_dividends(
        symbol: Annotated[str, "ticker symbol"],
        save_path: Optional[str] = None,
    ) -> DataFrame:
        """Fetches and returns the latest dividends data as a DataFrame."""
        ticker = symbol
        dividends = ticker.dividends
        if save_path:
            dividends.to_csv(save_path)
            print(f"Dividends for {ticker.ticker} saved to {save_path}")
        return dividends

    def get_income_stmt(symbol: Annotated[str, "ticker symbol"]) -> DataFrame:
        """Fetches and returns the latest income statement of the company as a DataFrame."""
        ticker = symbol
        income_stmt = ticker.financials
        return income_stmt

    def get_balance_sheet(symbol: Annotated[str, "ticker symbol"]) -> DataFrame:
        """Fetches and returns the latest balance sheet of the company as a DataFrame."""
        ticker = symbol
        balance_sheet = ticker.balance_sheet
        return balance_sheet

    def get_cash_flow(symbol: Annotated[str, "ticker symbol"]) -> DataFrame:
        """Fetches and returns the latest cash flow statement of the company as a DataFrame."""
        ticker = symbol
        cash_flow = ticker.cashflow
        return cash_flow

    def get_analyst_recommendations(symbol: Annotated[str, "ticker symbol"]) -> tuple:
        """Fetches the latest analyst recommendations and returns the most common recommendation and its count."""
        ticker = symbol
        recommendations = ticker.recommendations
        if recommendations.empty:
            return None, 0  # No recommendations available

        # Assuming 'period' column exists and needs to be excluded
        row_0 = recommendations.iloc[0, 1:]  # Exclude 'period' column if necessary

        # Find the maximum voting result
        max_votes = row_0.max()
        majority_voting_result = row_0[row_0 == max_votes].index.tolist()

        return majority_voting_result[0], max_votes
