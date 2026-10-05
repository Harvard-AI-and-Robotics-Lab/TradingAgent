import os
import json
import pandas as pd
from datetime import date, timedelta, datetime
from typing import Annotated
import difflib

SavePathType = Annotated[str, "File path to save data. If None, data is not saved."]

def save_output(data: pd.DataFrame, tag: str, save_path: SavePathType = None) -> None:
    if save_path:
        data.to_csv(save_path)
        print(f"{tag} saved to {save_path}")


def get_current_date():
    return date.today().strftime("%Y-%m-%d")


def decorate_all_methods(decorator):
    def class_decorator(cls):
        for attr_name, attr_value in cls.__dict__.items():
            if callable(attr_value):
                setattr(cls, attr_name, decorator(attr_value))
        return cls

    return class_decorator


def get_next_weekday(date):

    if not isinstance(date, datetime):
        date = datetime.strptime(date, "%Y-%m-%d")

    if date.weekday() >= 5:
        days_to_add = 7 - date.weekday()
        next_weekday = date + timedelta(days=days_to_add)
        return next_weekday
    else:
        return date

def get_fuzzy_matched_data_path(symbol, data_path: str):
    """
    Get the fuzzy matched data path for a given symbol and data path.
    """
    filename = data_path.split("/")[-1]
    data_dir = '/'.join(data_path.split("/")[:-1])
    if not filename.startswith(symbol):
        return data_path

    candidates = [
        f for f in os.listdir(data_dir)
        if f.startswith(symbol) and f.split(".")[-1] == data_path.split(".")[-1]
    ]
    if not candidates:
        print(f"No file found in {data_dir} starting with {symbol}")
        return data_path

    if len(candidates) == 1:
        best_match = candidates[0]
    else:
        best_match = max(candidates, key=lambda f: difflib.SequenceMatcher(None, f, symbol).ratio())

    return os.path.join(data_dir, best_match)