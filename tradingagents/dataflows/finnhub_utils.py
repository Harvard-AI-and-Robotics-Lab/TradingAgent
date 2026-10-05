import json
import os
from datetime import datetime

from .utils import get_fuzzy_matched_data_path


def get_data_in_range(ticker, start_date, end_date, data_type, data_dir, data_type_prefix="finnhub_data", period=None):
    """
    Gets finnhub data saved and processed on disk.
    Args:
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.
        data_type (str): Type of data from finnhub to fetch. Can be insider_trans, SEC_filings, news_data, insider_senti, or fin_as_reported.
        data_dir (str): Directory where the data is saved.
        period (str): Default to none, if there is a period specified, should be annual or quarterly.
    """

    if period:
        data_path = os.path.join(
            data_dir,
            data_type_prefix,
            data_type,
            f"{ticker}_{period}_data_formatted.json",
        )
    else:
        data_path = os.path.join(
            data_dir, data_type_prefix, data_type, f"{ticker}_data_formatted.json"
        )

    if not os.path.exists(data_path):
        print(f"\n\nWarning: Data path {data_path} does not exist\n\n")
        data_path = get_fuzzy_matched_data_path(ticker, data_path)

    data = open(data_path, "r")
    data = json.load(data)

    # filter keys (date, str in format YYYY-MM-DD) by the date range (str, str in format YYYY-MM-DD)
    filtered_data = {}
    if data_type in {"insider_trans", "news", "investor_relations"}:
        for value in data["data"]:
            if data_type == "insider_trans":
                if "filingDate" not in value:
                    continue
                if start_date <= value["filingDate"] <= end_date:
                    if value["filingDate"] not in filtered_data:
                        filtered_data[value["filingDate"]] = []
                    filtered_data[value["filingDate"]].append(value)
            elif data_type == "news":
                if "datetime" not in value:
                    continue
                # convert datetime to YYYY-MM-DD
                value["datetime"] = datetime.fromtimestamp(value["datetime"]).strftime("%Y-%m-%d")
                if start_date <= value["datetime"] <= end_date:
                    if value["datetime"] not in filtered_data:
                        filtered_data[value["datetime"]] = []
                    filtered_data[value["datetime"]].append(value)

            elif data_type == "investor_relations":
                if "datetime" not in value:
                    continue
                # here start_date = end_date = curr_date
                # convert datetime to YYYY-MM-DD
                value["datetime"] = datetime.fromtimestamp(value["datetime"]).strftime("%Y-%m-%d")
                if start_date >= value["datetime"]:
                    if value["datetime"] > filtered_data["datetime"]:
                        filtered_data = value.copy()
    else:
        for key, value in data.items():
            if start_date <= key <= end_date and len(value) > 0:
                filtered_data[key] = value

    return filtered_data