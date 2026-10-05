import json
import os
import re
import yfinance as yf
import pandas as pd
from typing import Annotated, Dict
from dateutil.relativedelta import relativedelta
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from tqdm import tqdm
from google import genai
from google.genai import types
from openai import OpenAI
from anthropic import Anthropic
from xai_sdk import Client as XAI
from xai_sdk.chat import user as xai_user
from xai_sdk.tools import web_search as xai_web_search
from langchain.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage

from .reddit_utils import fetch_top_from_category
from .yfin_utils import *
from .stockstats_utils import *
from .googlenews_utils import *
from .finnhub_utils import get_data_in_range
from .config import get_config, set_config, DATA_DIR, DATA_DIR_PREFIX, DATA_CACHE_FILE_YFIN
from .yfin_utils import get_data_path_yfin
from .utils import get_fuzzy_matched_data_path


def gemini_web_search_chat(text, model=None):
    default_model = "gemini-2.5-flash-lite"
    model = default_model if model is None else model
    if not model.startswith("gemini-"):
        model = default_model
        print(f"gemini_web_search_chat: using default model: {model}")

    client = genai.Client()

    grounding_tool = types.Tool(
        google_search=types.GoogleSearch()
    )

    config = types.GenerateContentConfig(
        tools=[grounding_tool]
    )

    response = client.models.generate_content(
        model=model,
        contents=text,
        config=config,
    )

    return response.text


def openai_web_search_chat(text, model=None):
    default_model = "gpt-4o-mini"
    model = default_model if model is None else model
    if not model.startswith("gpt-"):
        model = default_model
        print(f"openai_web_search_chat: using default model: {model}")

    # GPT-5 and GPT-5-mini only support temperature=1, other models can use optimized values
    temperature = 1 if model in ["gpt-5", "gpt-5-mini"] else 0.3
    reasoning = {"effort": "low"} if model in ["gpt-5", "gpt-5-mini"] else {}
    max_output_tokens = 20000 if model in ["gpt-5", "gpt-5-mini"] else 8192

    client = OpenAI(base_url=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"))
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": text,
                    }
                ],
            }
        ],
        text={"format": {"type": "text"}},
        reasoning=reasoning,
        tools=[
            {
                "type": "web_search_preview",
                "user_location": {"type": "approximate"},
                #"search_context_size": "high",
            }
        ],
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        store=True,
    )

    # return response.output[-1].content[0].text
    return response.output_text

def xai_web_search_chat(text, model=None):
    XAI_client = XAI()
    default_model = "grok-4-fast-reasoning"
    model = default_model if model is None else model
    if not model.startswith("grok_"):
        model = default_model
        print(f"xai_web_search_chat: using default model: {model}")

    xai_chat = XAI_client.chat.create(
        model=model,
        include=["inline_citations"],  # Enable inline citations
        tools=[xai_web_search()],
    )

    xai_chat.append(xai_user(text))
    response = xai_chat.sample()

    return response.content

def anthropic_web_search_chat(text, model=None):
    client = Anthropic()
    default_llm = "claude-haiku-4-5-20251001"
    model = default_llm if model is None else model
    if not model.startswith("claude_"):
        model = default_llm
        print(f"anthropic_web_search_chat: using default model: {model}")

    messages = [
        {
            "role": "user",
            "content": text
        }
    ]

    response1 = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=messages,
        tools=[{
            "type": "web_search_20250305",
            "name": "web_search",
        }]
    )

    messages = [
        {
            "role": "user",
            "content": text
        },
        {
            "role": "assistant",
            "content": response1.content
        },
        {
            "role": "user",
            "content": "Please summarize a brief summary of the web search results, not redundant."
        }

    ]
    response2 = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=messages,
    )

    return response2.content


def get_finnhub_news(
    ticker: Annotated[
        str,
        "Search query of a company's, e.g. 'AAPL, TSM, etc.",
    ],
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "how many days to look back"],
):
    """
    Retrieve news about a company within a time frame

    Args
        ticker (str): ticker for the company you are interested in
        start_date (str): Start date in yyyy-mm-dd format
        end_date (str): End date in yyyy-mm-dd format
    Returns
        str: dataframe containing the news of the company in the time frame

    """

    start_date = datetime.strptime(curr_date, "%Y-%m-%d")
    before = start_date - relativedelta(days=look_back_days)
    before = before.strftime("%Y-%m-%d")

    result = get_data_in_range(ticker, before, curr_date, "news", DATA_DIR)

    if len(result) == 0:
        return ""

    combined_result = ""
    for day, data in result.items():
        if len(data) == 0:
            continue
        for entry in data:
            current_news = (
                "### " + entry["headline"] + f" ({day})" + "\n" + entry["summary"]
            )
            combined_result += current_news + "\n\n"

    return f"## {ticker} News, from {before} to {curr_date}:\n" + str(combined_result)


def get_finnhub_company_insider_sentiment(
    ticker: Annotated[str, "ticker symbol for the company"],
    curr_date: Annotated[
        str,
        "current date of you are trading at, yyyy-mm-dd",
    ],
    look_back_days: Annotated[int, "number of days to look back"],
):
    """
    Retrieve insider sentiment about a company (retrieved from public SEC information) for the past 15 days
    Args:
        ticker (str): ticker symbol of the company
        curr_date (str): current date you are trading on, yyyy-mm-dd
    Returns:
        str: a report of the sentiment in the past 15 days starting at curr_date
    """

    date_obj = datetime.strptime(curr_date, "%Y-%m-%d")
    before = date_obj - relativedelta(days=look_back_days)
    before = before.strftime("%Y-%m-%d")

    data = get_data_in_range(ticker, before, curr_date, "insider_senti", DATA_DIR)

    if len(data) == 0:
        return ""

    result_str = ""
    seen_dicts = []
    for date, senti_list in data.items():
        for entry in senti_list:
            if entry not in seen_dicts:
                result_str += f"### {entry['year']}-{entry['month']}:\nChange: {entry['change']}\nMonthly Share Purchase Ratio: {entry['mspr']}\n\n"
                seen_dicts.append(entry)

    return (
        f"## {ticker} Insider Sentiment Data for {before} to {curr_date}:\n"
        + result_str
        + "The change field refers to the net buying/selling from all insiders' transactions. The mspr field refers to monthly share purchase ratio."
    )


def get_finnhub_company_insider_transactions(
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[
        str,
        "current date you are trading at, yyyy-mm-dd",
    ],
    look_back_days: Annotated[int, "how many days to look back"],
):
    """
    Retrieve insider transcaction information about a company (retrieved from public SEC information) for the past 15 days
    Args:
        ticker (str): ticker symbol of the company
        curr_date (str): current date you are trading at, yyyy-mm-dd
    Returns:
        str: a report of the company's insider transaction/trading informtaion in the past 15 days
    """

    date_obj = datetime.strptime(curr_date, "%Y-%m-%d")
    before = date_obj - relativedelta(days=look_back_days)
    before = before.strftime("%Y-%m-%d")

    data = get_data_in_range(ticker, before, curr_date, "insider_trans", DATA_DIR)

    if len(data) == 0:
        return ""

    result_str = ""

    seen_dicts = []
    for date, trans_list in data.items():
        for entry in trans_list:
            if entry not in seen_dicts:
                try:
                    result_str += f"### Filing Date: {entry['filingDate']}, {entry['name']}:\nChange:{entry['change']}\nShares: {entry['share']}\nTransaction Price: {entry['transactionPrice']}\nTransaction Code: {entry['transactionCode']}\n\n"
                    seen_dicts.append(entry)
                except Exception as e:
                    print(f"Error adding entry to result_str: {e}")

    return (
        f"## {ticker} insider transactions from {before} to {curr_date}:\n"
        + result_str
        + "The change field reflects the variation in share count—here a negative number indicates a reduction in holdings—while share specifies the total number of shares involved. The transactionPrice denotes the per-share price at which the trade was executed, and transactionDate marks when the transaction occurred. The name field identifies the insider making the trade, and transactionCode (e.g., S for sale) clarifies the nature of the transaction. FilingDate records when the transaction was officially reported, and the unique id links to the specific SEC filing, as indicated by the source. Additionally, the symbol ties the transaction to a particular company, isDerivative flags whether the trade involves derivative securities, and currency notes the currency context of the transaction."
    )


def get_simfin_balance_sheet(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[
        str,
        "reporting frequency of the company's financial history: annual / quarterly",
    ],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
):
    data_path = os.path.join(
        DATA_DIR,
        DATA_DIR_PREFIX["simfin_balance_sheet"],
        f"us-balance-{freq}.csv",
    )
    if not os.path.exists(data_path):
        print(f"\n\nWarning: Data path {data_path} does not exist\n\n")
        data_path = get_fuzzy_matched_data_path(ticker, data_path)
    df = pd.read_csv(data_path, sep=";", engine="python", on_bad_lines="warn")

    # Convert date strings to datetime objects and remove any time components
    df["Report Date"] = pd.to_datetime(df["Report Date"], utc=True).dt.normalize()
    df["Publish Date"] = pd.to_datetime(df["Publish Date"], utc=True).dt.normalize()

    # Convert the current date to datetime and normalize
    curr_date_dt = pd.to_datetime(curr_date, utc=True).normalize()

    # Filter the DataFrame for the given ticker and for reports that were published on or before the current date
    filtered_df = df[(df["Ticker"] == ticker) & (df["Publish Date"] <= curr_date_dt)]

    # Check if there are any available reports; if not, return a notification
    if filtered_df.empty:
        print("No balance sheet available before the given current date.")
        return ""

    # Get the most recent balance sheet by selecting the row with the latest Publish Date
    latest_balance_sheet = filtered_df.loc[filtered_df["Publish Date"].idxmax()]

    # drop the SimFinID column
    latest_balance_sheet = latest_balance_sheet.drop("SimFinId")

    return (
        f"## {freq} balance sheet for {ticker} released on {str(latest_balance_sheet['Publish Date'])[0:10]}: \n"
        + str(latest_balance_sheet)
        + "\n\nThis includes metadata like reporting dates and currency, share details, and a breakdown of assets, liabilities, and equity. Assets are grouped as current (liquid items like cash and receivables) and noncurrent (long-term investments and property). Liabilities are split between short-term obligations and long-term debts, while equity reflects shareholder funds such as paid-in capital and retained earnings. Together, these components ensure that total assets equal the sum of liabilities and equity."
    )


def get_simfin_cashflow(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[
        str,
        "reporting frequency of the company's financial history: annual / quarterly",
    ],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
):
    data_path = os.path.join(
        DATA_DIR,
        DATA_DIR_PREFIX["simfin_cashflow"],
        f"us-cashflow-{freq}.csv",
    )
    if not os.path.exists(data_path):
        print(f"\n\nWarning: Data path {data_path} does not exist\n\n")
        data_path = get_fuzzy_matched_data_path(ticker, data_path)
    df = pd.read_csv(data_path, sep=";", engine="python", on_bad_lines="warn")

    # Convert date strings to datetime objects and remove any time components
    df["Report Date"] = pd.to_datetime(df["Report Date"], utc=True).dt.normalize()
    df["Publish Date"] = pd.to_datetime(df["Publish Date"], utc=True).dt.normalize()

    # Convert the current date to datetime and normalize
    curr_date_dt = pd.to_datetime(curr_date, utc=True).normalize()

    # Filter the DataFrame for the given ticker and for reports that were published on or before the current date
    filtered_df = df[(df["Ticker"] == ticker) & (df["Publish Date"] <= curr_date_dt)]

    # Check if there are any available reports; if not, return a notification
    if filtered_df.empty:
        print("No cash flow statement available before the given current date.")
        return ""

    # Get the most recent cash flow statement by selecting the row with the latest Publish Date
    latest_cash_flow = filtered_df.loc[filtered_df["Publish Date"].idxmax()]

    # drop the SimFinID column
    latest_cash_flow = latest_cash_flow.drop("SimFinId")

    return (
        f"## {freq} cash flow statement for {ticker} released on {str(latest_cash_flow['Publish Date'])[0:10]}: \n"
        + str(latest_cash_flow)
        + "\n\nThis includes metadata like reporting dates and currency, share details, and a breakdown of cash movements. Operating activities show cash generated from core business operations, including net income adjustments for non-cash items and working capital changes. Investing activities cover asset acquisitions/disposals and investments. Financing activities include debt transactions, equity issuances/repurchases, and dividend payments. The net change in cash represents the overall increase or decrease in the company's cash position during the reporting period."
    )


def get_simfin_income_statements(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[
        str,
        "reporting frequency of the company's financial history: annual / quarterly",
    ],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
):

    data_path = os.path.join(
        DATA_DIR,
        DATA_DIR_PREFIX["simfin_income_statements"],
        f"us-income-{freq}.csv",
    )

    if not os.path.exists(data_path):
        print(f"\n\nWarning: Data path {data_path} does not exist\n\n")
        data_path = get_fuzzy_matched_data_path(ticker, data_path)

    df = pd.read_csv(data_path, sep=";", engine="python", on_bad_lines="warn")

    # Convert date strings to datetime objects and remove any time components
    df["Report Date"] = pd.to_datetime(df["Report Date"], utc=True).dt.normalize()
    df["Publish Date"] = pd.to_datetime(df["Publish Date"], utc=True).dt.normalize()

    # Convert the current date to datetime and normalize
    curr_date_dt = pd.to_datetime(curr_date, utc=True).normalize()

    # Filter the DataFrame for the given ticker and for reports that were published on or before the current date
    filtered_df = df[(df["Ticker"] == ticker) & (df["Publish Date"] <= curr_date_dt)]

    # Check if there are any available reports; if not, return a notification
    if filtered_df.empty:
        print("No income statement available before the given current date.")
        return ""

    # Get the most recent income statement by selecting the row with the latest Publish Date
    latest_income = filtered_df.loc[filtered_df["Publish Date"].idxmax()]

    # drop the SimFinID column
    latest_income = latest_income.drop("SimFinId")

    return (
        f"## {freq} income statement for {ticker} released on {str(latest_income['Publish Date'])[0:10]}: \n"
        + str(latest_income)
        + "\n\nThis includes metadata like reporting dates and currency, share details, and a comprehensive breakdown of the company's financial performance. Starting with Revenue, it shows Cost of Revenue and resulting Gross Profit. Operating Expenses are detailed, including SG&A, R&D, and Depreciation. The statement then shows Operating Income, followed by non-operating items and Interest Expense, leading to Pretax Income. After accounting for Income Tax and any Extraordinary items, it concludes with Net Income, representing the company's bottom-line profit or loss for the period."
    )


def get_google_news(
    query: Annotated[str, "Query to search with"],
    curr_date: Annotated[str, "Curr date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "how many days to look back"],
) -> str:
    query = query.replace(" ", "+")

    start_date = datetime.strptime(curr_date, "%Y-%m-%d")
    before = start_date - relativedelta(days=look_back_days)
    before = before.strftime("%Y-%m-%d")

    news_results = getNewsData(query, before, curr_date)

    news_str = ""

    for news in news_results:
        news_str += (
            f"### {news['title']} (source: {news['source']}) \n\n{news['snippet']}\n\n"
        )

    if len(news_results) == 0:
        return ""

    return f"## {query} Google News, from {before} to {curr_date}:\n\n{news_str}"


def get_reddit_global_news(
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "how many days to look back"],
    max_limit_per_day: Annotated[int, "Maximum number of news per day"],
) -> str:
    """
    Retrieve the latest top reddit news
    Args:
        start_date: Start date in yyyy-mm-dd format
        end_date: End date in yyyy-mm-dd format
    Returns:
        str: A formatted dataframe containing the latest news articles posts on reddit and meta information in these columns: "created_utc", "id", "title", "selftext", "score", "num_comments", "url"
    """

    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    before = start_date - relativedelta(days=look_back_days)
    before = before.strftime("%Y-%m-%d")

    posts = []
    # iterate from start_date to end_date
    curr_date = datetime.strptime(before, "%Y-%m-%d")

    total_iterations = (start_date - curr_date).days + 1
    pbar = tqdm(desc=f"Getting Global News on {start_date}", total=total_iterations)

    while curr_date <= start_date:
        curr_date_str = curr_date.strftime("%Y-%m-%d")
        fetch_result = fetch_top_from_category(
            "global_news",
            curr_date_str,
            max_limit_per_day,
            data_path=os.path.join(DATA_DIR, "reddit_data"),
        )
        posts.extend(fetch_result)
        curr_date += relativedelta(days=1)
        pbar.update(1)

    pbar.close()

    if len(posts) == 0:
        return ""

    news_str = ""
    for post in posts:
        if post["content"] == "":
            news_str += f"### {post['title']}\n\n"
        else:
            news_str += f"### {post['title']}\n\n{post['content']}\n\n"

    return f"## Global News Reddit, from {before} to {curr_date}:\n{news_str}"


def get_reddit_company_news(
    ticker: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "how many days to look back"],
    max_limit_per_day: Annotated[int, "Maximum number of news per day"],
) -> str:
    """
    Retrieve the latest top reddit news
    Args:
        ticker: ticker symbol of the company
        start_date: Start date in yyyy-mm-dd format
        end_date: End date in yyyy-mm-dd format
    Returns:
        str: A formatted dataframe containing the latest news articles posts on reddit and meta information in these columns: "created_utc", "id", "title", "selftext", "score", "num_comments", "url"
    """

    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    before = start_date - relativedelta(days=look_back_days)
    before = before.strftime("%Y-%m-%d")

    posts = []
    # iterate from start_date to end_date
    curr_date = datetime.strptime(before, "%Y-%m-%d")

    total_iterations = (start_date - curr_date).days + 1
    pbar = tqdm(
        desc=f"Getting Company News for {ticker} on {start_date}",
        total=total_iterations,
    )

    while curr_date <= start_date:
        curr_date_str = curr_date.strftime("%Y-%m-%d")
        fetch_result = fetch_top_from_category(
            "company_news",
            curr_date_str,
            max_limit_per_day,
            ticker,
            data_path=os.path.join(DATA_DIR, "reddit_data"),
        )
        posts.extend(fetch_result)
        curr_date += relativedelta(days=1)

        pbar.update(1)

    pbar.close()

    if len(posts) == 0:
        return ""

    news_str = ""
    for post in posts:
        if post["content"] == "":
            news_str += f"### {post['title']}\n\n"
        else:
            news_str += f"### {post['title']}\n\n{post['content']}\n\n"

    return f"##{ticker} News Reddit, from {before} to {curr_date}:\n\n{news_str}"


def get_stock_stats_indicators_window(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to get the analysis and report of"],
    curr_date: Annotated[
        str, "The current trading date you are trading on, YYYY-mm-dd"
    ],
    look_back_days: Annotated[int, "how many days to look back"],
    online: Annotated[bool, "to fetch data online or offline"],
) -> str:

    best_ind_params = {
        # Moving Averages
        "close_10_ema": (
            "10 EMA: A responsive short-term average. "
            "Usage: Capture quick shifts in momentum and potential entry points. "
            "Tips: Prone to noise in choppy markets; use alongside longer averages for filtering false signals."
        ),
        "close_50_sma": (
            "50 SMA: A medium-term trend indicator. "
            "Usage: Identify trend direction and serve as dynamic support/resistance. "
            "Tips: It lags price; combine with faster indicators for timely signals."
        ),
        "close_200_sma": (
            "200 SMA: A long-term trend benchmark. "
            "Usage: Confirm overall market trend and identify golden/death cross setups. "
            "Tips: It reacts slowly; best for strategic trend confirmation rather than frequent trading entries."
        ),
        # MACD Related
        "macd": (
            "MACD: Computes momentum via differences of EMAs. "
            "Usage: Look for crossovers and divergence as signals of trend changes. "
            "Tips: Confirm with other indicators in low-volatility or sideways markets."
        ),
        "macds": (
            "MACD Signal: An EMA smoothing of the MACD line. "
            "Usage: Use crossovers with the MACD line to trigger trades. "
            "Tips: Should be part of a broader strategy to avoid false positives."
        ),
        "macdh": (
            "MACD Histogram: Shows the gap between the MACD line and its signal. "
            "Usage: Visualize momentum strength and spot divergence early. "
            "Tips: Can be volatile; complement with additional filters in fast-moving markets."
        ),
        # Momentum Indicators
        "rsi": (
            "RSI: Measures momentum to flag overbought/oversold conditions. "
            "Usage: Apply 70/30 thresholds and watch for divergence to signal reversals. "
            "Tips: In strong trends, RSI may remain extreme; always cross-check with trend analysis."
        ),
        # Volatility Indicators
        "boll": (
            "Bollinger Middle: A 20 SMA serving as the basis for Bollinger Bands. "
            "Usage: Acts as a dynamic benchmark for price movement. "
            "Tips: Combine with the upper and lower bands to effectively spot breakouts or reversals."
        ),
        "boll_ub": (
            "Bollinger Upper Band: Typically 2 standard deviations above the middle line. "
            "Usage: Signals potential overbought conditions and breakout zones. "
            "Tips: Confirm signals with other tools; prices may ride the band in strong trends."
        ),
        "boll_lb": (
            "Bollinger Lower Band: Typically 2 standard deviations below the middle line. "
            "Usage: Indicates potential oversold conditions. "
            "Tips: Use additional analysis to avoid false reversal signals."
        ),
        "atr": (
            "ATR: Averages true range to measure volatility. "
            "Usage: Set stop-loss levels and adjust position sizes based on current market volatility. "
            "Tips: It's a reactive measure, so use it as part of a broader risk management strategy."
        ),
        # Volume-Based Indicators
        "vwma": (
            "VWMA: A moving average weighted by volume. "
            "Usage: Confirm trends by integrating price action with volume data. "
            "Tips: Watch for skewed results from volume spikes; use in combination with other volume analyses."
        ),
        "mfi": (
            "MFI: The Money Flow Index is a momentum indicator that uses both price and volume to measure buying and selling pressure. "
            "Usage: Identify overbought (>80) or oversold (<20) conditions and confirm the strength of trends or reversals. "
            "Tips: Use alongside RSI or MACD to confirm signals; divergence between price and MFI can indicate potential reversals."
        ),
    }

    if indicator not in best_ind_params:
        raise ValueError(
            f"Indicator {indicator} is not supported. Please choose from: {list(best_ind_params.keys())}"
        )

    end_date = datetime.strptime(curr_date, "%Y-%m-%d")
    curr_date = datetime.strptime(curr_date, "%Y-%m-%d")
    before = curr_date - relativedelta(days=look_back_days)

    if not online:
        # read from YFin data
        data_path = os.path.join(
            DATA_DIR,
            DATA_DIR_PREFIX['market_data'],
            f"{symbol}-{DATA_CACHE_FILE_YFIN}",
        )
        if not os.path.exists(data_path):
            print(f"\n\nWarning: Data path {data_path} does not exist\n\n")
            data_path = get_data_path_yfin(symbol, curr_date, data_path=data_path)

        data = pd.read_csv(
            data_path,
            engine="python",
            on_bad_lines="warn"
        )

        data["Date"] = pd.to_datetime(data["Date"], utc=True)
        dates_in_df = data["Date"].astype(str).str[:10]

        ind_string = ""
        while curr_date >= before and curr_date <= end_date:
            # only do the trading dates
            if curr_date.strftime("%Y-%m-%d") in dates_in_df.values:
                indicator_value = get_stockstats_indicator(
                    symbol, indicator, curr_date.strftime("%Y-%m-%d"), online
                )

                ind_string += f"{curr_date.strftime('%Y-%m-%d')}: {indicator_value}\n"

            curr_date = curr_date - relativedelta(days=1)
    else:
        # online gathering
        ind_string = ""
        while curr_date >= before and curr_date <= end_date:
            indicator_value = get_stockstats_indicator(
                symbol, indicator, curr_date.strftime("%Y-%m-%d"), online
            )

            ind_string += f"{curr_date.strftime('%Y-%m-%d')}: {indicator_value}\n"

            curr_date = curr_date - relativedelta(days=1)

    result_str = (
        f"## {indicator} values from {before.strftime('%Y-%m-%d')} to {end_date}:\n\n"
        + ind_string
        + "\n\n"
        + best_ind_params.get(indicator, "No description available.")
    )

    return result_str


def get_stockstats_indicator(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to get the analysis and report of"],
    curr_date: Annotated[
        str, "The current trading date you are trading on, YYYY-mm-dd"
    ],
    online: Annotated[bool, "to fetch data online or offline"],
) -> str:

    curr_date = datetime.strptime(curr_date, "%Y-%m-%d")

    try:
        data_path = os.path.join(DATA_DIR, DATA_DIR_PREFIX["market_data"])
        if not os.path.exists(data_path):
            print(f"\n\nWarning: Data path {data_path} does not exist\n\n")
        # else:
        #     print(f"Getting stockstats indicator data for indicator {indicator} from {data_path}")
        indicator_value = StockstatsUtils.get_stock_stats(
            symbol,
            indicator,
            curr_date,
            data_path,
            online=online,
        )
    except Exception as e:
        print(
            f"\n\nError getting stockstats indicator data for indicator {indicator} on {curr_date}: {e}\n\n"
        )
        return ""

    return str(indicator_value)


def get_YFin_data_window(
    symbol: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "how many days to look back"],
) -> str:
    # calculate past days
    date_obj = datetime.strptime(curr_date, "%Y-%m-%d")
    before = date_obj - relativedelta(days=look_back_days)
    start_date = before.strftime("%Y-%m-%d")

    # read in data
    data_path = os.path.join(
        DATA_DIR,
        DATA_DIR_PREFIX['market_data'],
        f"{symbol}-{DATA_CACHE_FILE_YFIN}",
    )
    if not os.path.exists(data_path):
        print(f"\n\nWarning: Data path {data_path} does not exist\n\n")
        data_path = get_data_path_yfin(symbol, date_obj, data_path=data_path)

    data = pd.read_csv(
        data_path,
        engine="python",
        on_bad_lines="warn",
        parse_dates=["Date"],
        infer_datetime_format=True
    )

    # Extract just the date part for comparison
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce").dt.tz_localize(None)

    filtered_data = data[
        (data["Date"] >= start_date) & (data["Date"] <= date_obj)
    ].reset_index(drop=True)

    # Set pandas display options to show the full DataFrame
    with pd.option_context(
        "display.max_rows", None, "display.max_columns", None, "display.width", None
    ):
        df_string = filtered_data.to_string()

    return (
        f"## Raw Market Data for {symbol} from {start_date} to {curr_date}:\n\n"
        + df_string
    )


def get_YFin_data_online(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
    output_csv_string=True,
):

    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")

    # Create ticker object
    ticker = yf.Ticker(symbol.upper())

    # Fetch historical data for the specified date range
    data = ticker.history(start=start_date, end=end_date)

    # Check if data is empty
    if data.empty:
        return (
            f"No data found for symbol '{symbol}' between {start_date} and {end_date}"
        )

    # Remove timezone info from index for cleaner output
    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)

    # Round numerical values to 2 decimal places for cleaner display
    numeric_columns = ["Open", "High", "Low", "Close", "Adj Close"]
    for col in numeric_columns:
        if col in data.columns:
            data[col] = data[col].round(2)

    if not output_csv_string:
        return data
    # Convert DataFrame to CSV string
    csv_string = data.to_csv()

    # Add header information
    header = f"# Stock data for {symbol.upper()} from {start_date} to {end_date}\n"
    header += f"# Total records: {len(data)}\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    return header + csv_string


def get_YFin_data(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d")

    # read in data
    try:
        data_path = os.path.join(
            DATA_DIR,
            DATA_DIR_PREFIX['market_data'],
            f"{symbol}-{DATA_CACHE_FILE_YFIN}"
        )
        if not os.path.exists(data_path):
            print(f"\n\nWarning: Data path {data_path} does not exist\n\n")
            data_path = get_data_path_yfin(symbol, start_date=start_date, end_date=end_date, data_path=data_path)

        data = pd.read_csv(
            data_path,
            engine="python",           # more tolerant parser
            on_bad_lines="warn",       # skip or warn about malformed rows
            parse_dates=["Date"],      # datetime64
            infer_datetime_format=True
        )
    except Exception as e:
        raise Exception(
            f"Get_YFin_Data: {e}"
        )

    basename = data_path.split("/")[-1].replace(symbol+'-', "").removeprefix("YFin-data-").removesuffix(".csv")
    start_date_data, end_date_data = basename.split("-", 3)[:3], basename.split("-", 3)[3:]
    start_date_data = datetime.strptime("-".join(start_date_data), "%Y-%m-%d")
    end_date_data   = datetime.strptime("-".join(end_date_data), "%Y-%m-%d")

    if end_date > end_date_data:
        raise Exception(
            f"Get_YFin_Data: {end_date} is outside of the data range of {start_date_data} to {end_date_data}"
        )

    # Extract just the date part for comparison
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce").dt.tz_localize(None)

    filtered_data = data[
        (data["Date"] >= start_date) & (data["Date"] <= end_date)
    ].reset_index(drop=True)

    return filtered_data

## stock news
def get_stock_news_prompt(ticker, curr_date):
    start_date = datetime.strptime(curr_date, "%Y-%m-%d")
    before = (start_date - relativedelta(days=7)).strftime("%Y-%m-%d")
    text = f"You are a helpful financial trading assistant that can search Social Media from Twitter, Reddit, X, etc. for {ticker} from {before} to {curr_date}? Make sure you only get the data posted during that period."
    return text


def get_stock_news_openai(ticker, curr_date):
    config = get_config()
    start_date = datetime.strptime(curr_date, "%Y-%m-%d")
    before = (start_date - relativedelta(days=7)).strftime("%Y-%m-%d")

    text = get_stock_news_prompt(ticker, curr_date)
    if config.get("quick_think_llm") in ["gpt-5", "gpt-5-mini"]:
        text += f"""
    Do not ask clarifying questions, jsut answer the questions.
    """

    return openai_web_search_chat(text, config.get("quick_think_llm", "gpt-4o-mini"))

def get_stock_news_gemini(ticker, curr_date):
    config = get_config()
    start_date = datetime.strptime(curr_date, "%Y-%m-%d")
    before = (start_date - relativedelta(days=7)).strftime("%Y-%m-%d")

    text = get_stock_news_prompt(ticker, curr_date)

    return gemini_web_search_chat(text, config.get("quick_think_llm", None))

def get_stock_news_xai(ticker, curr_date):
    config = get_config()
    start_date = datetime.strptime(curr_date, "%Y-%m-%d")
    before = (start_date - relativedelta(days=7)).strftime("%Y-%m-%d")

    text = get_stock_news_prompt(ticker, curr_date)

    return xai_web_search_chat(text, config.get("quick_think_llm", None))

def get_stock_news_anthropic(ticker, curr_date):
    config = get_config()
    start_date = datetime.strptime(curr_date, "%Y-%m-%d")
    before = (start_date - relativedelta(days=7)).strftime("%Y-%m-%d")

    text = get_stock_news_prompt(ticker, curr_date)

    return anthropic_web_search_chat(text, config.get("quick_think_llm", None))

## global news
def get_global_news_prompt(curr_date):
    start_date = datetime.strptime(curr_date, "%Y-%m-%d")
    before = (start_date - relativedelta(days=7)).strftime("%Y-%m-%d")
    text = f"You are a helpful financial trading assistant that can search global or macroeconomics news from {before} to {curr_date} that would be informative for trading purposes. Make sure you only get the data posted during that period and exclude social media posts."
    return text

def get_global_news_openai(curr_date):
    config = get_config()
    start_date = datetime.strptime(curr_date, "%Y-%m-%d")
    before = (start_date - relativedelta(days=7)).strftime("%Y-%m-%d")

    text = get_global_news_prompt(curr_date)
    if config["quick_think_llm"] in ["gpt-5", "gpt-5-mini"]:
        text += f"""
        Do not ask clarifying questions, jsut directly answer the questions.
        """

    return openai_web_search_chat(text, config.get("quick_think_llm", "gpt-4o-mini"))

def get_global_news_gemini(curr_date):
    config = get_config()
    start_date = datetime.strptime(curr_date, "%Y-%m-%d")
    before = (start_date - relativedelta(days=7)).strftime("%Y-%m-%d")

    text = get_global_news_prompt(curr_date)

    return gemini_web_search_chat(text, config.get("quick_think_llm", None))

def get_global_news_xai(curr_date):
    config = get_config()
    start_date = datetime.strptime(curr_date, "%Y-%m-%d")
    before = (start_date - relativedelta(days=7)).strftime("%Y-%m-%d")

    text = get_global_news_prompt(curr_date)

    return xai_web_search_chat(text, config.get("quick_think_llm", None))

def get_global_news_anthropic(curr_date):
    config = get_config()
    start_date = datetime.strptime(curr_date, "%Y-%m-%d")
    before = (start_date - relativedelta(days=7)).strftime("%Y-%m-%d")

    text = get_global_news_prompt(curr_date)

    return anthropic_web_search_chat(text, config.get("quick_think_llm", None))

## fundamentals
def get_fundamentals_prompt(ticker, curr_date):
    text = f"""
Collect public fundamentals for {ticker} from the 30 days BEFORE {curr_date}.
Sources: IR sites, SEC/SEDAR filings, financial data providers, industry reports
Financials: Revenue growth, margins, ROE/ROIC/ROA, leverage, liquidity
Valuation: P/E (TTM/FWD), P/S, P/B, EV/EBITDA, PEG, dividend, analyst targets
Cash Flow: OCF, FCF, sustainability, working capital, capital allocation
Business: Market position, moats, industry trends, management, risks
Thesis: Drivers, catalysts, upside/downside, risks, valuation conclusion
Output: Specific numbers, historical trends, peer comparisons, tables, actionable insights, last 12 months up to {curr_date}
Avoid statements like "trends are mixed" - provide detailed but not redundant, data-driven analysis.
"""
    return text

def get_fundamentals_openai(ticker, curr_date):
    """
    Comprehensive fundamental analysis combining multiple data sources.
    This is the main tool for fundamental analysis, integrating data from various sources.
    """
    config = get_config()
    start_date = datetime.strptime(curr_date, "%Y-%m-%d")
    before30 = (start_date - relativedelta(days=30)).strftime("%Y-%m-%d")

    text = get_fundamentals_prompt(ticker, curr_date)
    if config.get("quick_think_llm") in ["gpt-5", "gpt-5-mini"]:
        text += f"""
    Do not ask clarifying questions, jsut answer the questions.
    """

    return openai_web_search_chat(text, config.get("quick_think_llm", "gpt-4o-mini"))

def get_fundamentals_gemini(ticker, curr_date):
    config = get_config()
    start_date = datetime.strptime(curr_date, "%Y-%m-%d")
    before30 = (start_date - relativedelta(days=30)).strftime("%Y-%m-%d")

    text = get_fundamentals_prompt(ticker, curr_date)

    return gemini_web_search_chat(text, config.get("quick_think_llm", None))

def get_fundamentals_xai(ticker, curr_date):
    config = get_config()
    start_date = datetime.strptime(curr_date, "%Y-%m-%d")
    before30 = (start_date - relativedelta(days=30)).strftime("%Y-%m-%d")

    text = get_fundamentals_prompt(ticker, curr_date)

    return xai_web_search_chat(text, config.get("quick_think_llm", None))

def get_fundamentals_anthropic(ticker, curr_date):
    config = get_config()
    start_date = datetime.strptime(curr_date, "%Y-%m-%d")
    before30 = (start_date - relativedelta(days=30)).strftime("%Y-%m-%d")

    text = get_fundamentals_prompt(ticker, curr_date)

    return anthropic_web_search_chat(text, config.get("quick_think_llm", None))

## company investor relations
def get_company_investor_relations_prompt(ticker, curr_date):
    company_ir_urls = {
        "NVDA": "https://investor.nvidia.com/home/default.aspx",
        "GOOGL": "https://abc.xyz/investor/",
        "GOOG": "https://abc.xyz/investor/",
        "TSLA": "https://ir.tesla.com/#quarterly-disclosure",
        "AAPL": "https://investor.apple.com/",
        "MSFT": "https://www.microsoft.com/en-us/Investor/",
        "AMZN": "https://ir.aboutamazon.com/",
        "META": "https://investor.fb.com/",
    }
    ir_url = company_ir_urls.get(ticker.upper(), f"https://investor.{ticker.lower()}.com/")

    text = f"""
Please search for and analyze the latest fundamental data for {ticker} from their investor relations website: {ir_url}

Focus on the following key metrics and data points:
- Latest quarterly and annual earnings reports
- Revenue growth trends
- Gross margin and net margin
- Price to earnings (P/E) ratio
- Price to sales (P/S) ratio
- Debt to equity ratio
- Return on Equity (ROE)
- Return on Invested Capital (ROIC)
- EBITDA
- Dividend yield
- Cash flow statements
- Balance sheet highlights

**CRITICAL DATA LEAKAGE PREVENTION:**
- ONLY use information published BEFORE {curr_date} 00:00:00 (exclude anything published on {curr_date} at any time)
- DO NOT include any data, news, or events from {curr_date} or later
- If a source has no clear publish date, discard it completely
- Focus on data from the past 12 months up to {curr_date}

Format the results as a comprehensive but not redundant table with key financial metrics and trends.
"""
    return text

def get_company_investor_relations_data_openai(ticker, curr_date):
    """
    Retrieve fundamental data from company investor relations websites.
    Includes earnings reports, financial statements, and key metrics.
    """
    config = get_config()

    text = get_company_investor_relations_prompt(ticker, curr_date)
    if config.get("quick_think_llm") in ["gpt-5", "gpt-5-mini"]:
        text += f"""
    Do not ask clarifying questions, jsut answer the questions.
    """

    return openai_web_search_chat(text, config.get("quick_think_llm", "gpt-4o-mini"))

def get_company_investor_relations_data_gemini(ticker, curr_date):
    config = get_config()
    text = get_company_investor_relations_prompt(ticker, curr_date)

    return gemini_web_search_chat(text, config.get("quick_think_llm", None))

def get_company_investor_relations_data_xai(ticker, curr_date):
    """
    Retrieve fundamental data from company investor relations websites.
    Includes earnings reports, financial statements, and key metrics.
    """
    config = get_config()
    text = get_company_investor_relations_prompt(ticker, curr_date)

    return xai_web_search_chat(text, config.get("quick_think_llm", None))

def get_company_investor_relations_data_anthropic(ticker, curr_date):
    """
    Retrieve fundamental data from company investor relations websites.
    Includes earnings reports, financial statements, and key metrics.
    """
    config = get_config()
    text = get_company_investor_relations_prompt(ticker, curr_date)

    return anthropic_web_search_chat(text, config.get("quick_think_llm", None))

## SEC filings
def get_sec_filings_prompt(ticker, curr_date):
    text = f"""
Please search for and analyze the latest SEC filings for {ticker} from https://www.sec.gov/

**CRITICAL DATA LEAKAGE PREVENTION:**
- ONLY use information published BEFORE {curr_date} 00:00:00 (exclude anything published on {curr_date} at any time)
- DO NOT include any data, news, or events from {curr_date} or later
- If a source has no clear publish date, discard it completely
- Focus on filings from the past 12 months up to {curr_date}

Focus on the following SEC documents:
- 10-K Annual Reports (most recent)
- 10-Q Quarterly Reports (most recent 2-3 quarters)
- 8-K Current Reports (recent material events)

Provide a comprehensive but not redundant analysis with key insights and financial metrics in table format.
"""
    return text

def get_sec_filings_data_openai(ticker, curr_date):
    """
    Retrieve fundamental data from SEC filings (10-K, 10-Q, 8-K reports).
    """
    config = get_config()

    text = get_sec_filings_prompt(ticker, curr_date)
    if config.get("quick_think_llm") in ["gpt-5", "gpt-5-mini"]:
        text += f"""
    Do not ask clarifying questions, jsut answer the questions.
    """

    return openai_web_search_chat(text, config.get("quick_think_llm", "gpt-4o-mini"))

def get_sec_filings_data_gemini(ticker, curr_date):
    config = get_config()
    text = get_sec_filings_prompt(ticker, curr_date)

    return gemini_web_search_chat(text, config.get("quick_think_llm", None))

def get_sec_filings_data_xai(ticker, curr_date):
    config = get_config()
    text = get_sec_filings_prompt(ticker, curr_date)

    return xai_web_search_chat(text, config.get("quick_think_llm", None))

def get_sec_filings_data_anthropic(ticker, curr_date):
    config = get_config()
    text = get_sec_filings_prompt(ticker, curr_date)

    return anthropic_web_search_chat(text, config.get("quick_think_llm", None))

## SEDAR filings
def get_sedar_filings_prompt(ticker, curr_date):
    text = f"""
Please search for and analyze the latest SEDAR filings for {ticker} from https://www.sedarplus.ca/

Focus on the following SEDAR documents:
- Annual Information Forms (AIF)
- Annual and Quarterly Financial Statements
- Management Discussion and Analysis (MD&A)
- Material Change Reports
- Business Acquisition Reports

**CRITICAL DATA LEAKAGE PREVENTION:**
- ONLY use information published BEFORE {curr_date} 00:00:00 (exclude anything published on {curr_date} at any time)
- DO NOT include any data, news, or events from {curr_date} or later
- If a source has no clear publish date, discard it completely
- Focus on filings from the past 12 months up to {curr_date}

Provide a comprehensive but not redundant analysis with key insights and financial metrics in table format.
"""
    return text

def get_sedar_filings_data_openai(ticker, curr_date):
    """
    Retrieve fundamental data from SEDAR filings for Canadian companies.
    """
    config = get_config()

    text = get_sedar_filings_prompt(ticker, curr_date)
    if config.get("quick_think_llm") in ["gpt-5", "gpt-5-mini"]:
        text += f"""
    Do not ask clarifying questions, jsut answer the questions.
    """

    return openai_web_search_chat(text, config.get("quick_think_llm", "gpt-4o-mini"))

def get_sedar_filings_data_gemini(ticker, curr_date):
    config = get_config()
    text = get_sedar_filings_prompt(ticker, curr_date)

    return gemini_web_search_chat(text, config.get("quick_think_llm", None))

def get_sedar_filings_data_xai(ticker, curr_date):
    config = get_config()
    text = get_sedar_filings_prompt(ticker, curr_date)

    return xai_web_search_chat(text, config.get("quick_think_llm", None))

def get_sedar_filings_data_anthropic(ticker, curr_date):
    config = get_config()
    text = get_sedar_filings_prompt(ticker, curr_date)

    return anthropic_web_search_chat(text, config.get("quick_think_llm", None))


#!/usr/bin/env python3
"""
Offline Fundamentals Tools - Additional tools for offline mode
"""
def get_company_investor_relations_data_offline(ticker, curr_date):
    """
    Offline version: Retrieve fundamental data from cached company investor relations data.
    """
    config = get_config()
    data_dir = config.get("data_dir", "./tradingagents/dataflows/data_cache")

    # Try to read cached investor relations data
    # ir_file_path = os.path.join(data_dir, "company_data", "investor_relations", f"{ticker}_ir_data.json")
    ir_data = get_data_in_range(
        ticker, curr_date, curr_date,
        data_type="investor_relations",
        data_dir=data_dir,
        data_type_prefix="company_data"
    )

    if len(ir_data) == 0:
        print(f"\n\nWarning: Data path {ir_file_path} does not exist\n\n")
        return f"""
## {ticker} Investor Relations Data (Offline)

**Status:** No cached investor relations data available for {ticker}

**Recommendation:**
- Use online tools for real-time investor relations data
- Or download and cache investor relations data for offline use

*Data as of {curr_date}*
"""
    else:
        try:
            with open(ir_file_path, 'r') as f:
                ir_data = json.load(f)

            return f"""
## {ticker} Investor Relations Data (Offline)

**Latest Financial Metrics:**
- Revenue: ${ir_data.get('revenue', 'N/A')}
- Net Income: ${ir_data.get('net_income', 'N/A')}
- EPS: ${ir_data.get('eps', 'N/A')}
- P/E Ratio: {ir_data.get('pe_ratio', 'N/A')}
- Market Cap: ${ir_data.get('market_cap', 'N/A')}

**Key Highlights:**
{ir_data.get('highlights', 'No highlights available')}

**Recent Updates:**
{ir_data.get('recent_updates', 'No recent updates available')}

*Data retrieved from cached investor relations data as of {curr_date}*
"""
        except Exception as e:
            return f"Error reading investor relations data for {ticker}: {str(e)}"


def get_sec_filings_data_offline(ticker, curr_date):
    """
    Offline version: Retrieve fundamental data from cached SEC filings.
    """
    config = get_config()
    data_dir = config.get("data_dir", "./tradingagents/dataflows/data_cache")

    # Try to read cached SEC filings data
    sec_file_path = os.path.join(data_dir, "company_data", "sec_filings", f"{ticker}_sec_data.json")

    if not os.path.exists(sec_file_path):
        sec_file_path = get_fuzzy_matched_data_path(ticker, sec_file_path)

    if not os.path.exists(sec_file_path):
        print(f"\n\nWarning: Data path {sec_file_path} does not exist\n\n")
        return f"""
## {ticker} SEC Filings Analysis (Offline)

**Status:** No cached SEC filings data available for {ticker}

**Recommendation:**
- Use online tools for real-time SEC filings
- Or download and cache SEC filings data for offline use

*Data as of {curr_date}*
"""
    else:
        try:
            with open(sec_file_path, 'r') as f:
                sec_data = json.load(f)

            return f"""
## {ticker} SEC Filings Analysis (Offline)

**Latest 10-K Report:**
- Filing Date: {sec_data.get('latest_10k_date', 'N/A')}
- Revenue: ${sec_data.get('revenue', 'N/A')}
- Net Income: ${sec_data.get('net_income', 'N/A')}
- Total Assets: ${sec_data.get('total_assets', 'N/A')}
- Total Liabilities: ${sec_data.get('total_liabilities', 'N/A')}

**Key Financial Ratios:**
- ROE: {sec_data.get('roe', 'N/A')}
- ROA: {sec_data.get('roa', 'N/A')}
- Debt-to-Equity: {sec_data.get('debt_to_equity', 'N/A')}
- Current Ratio: {sec_data.get('current_ratio', 'N/A')}

**Risk Factors:**
{sec_data.get('risk_factors', 'No risk factors data available')}

**Management Discussion:**
{sec_data.get('md_a', 'No MD&A data available')}

*Data retrieved from cached SEC filings as of {curr_date}*
"""
        except Exception as e:
            return f"Error reading SEC filings data for {ticker}: {str(e)}"


def get_sedar_filings_data_offline(ticker, curr_date):
    """
    Offline version: Retrieve fundamental data from cached SEDAR filings (Canadian companies).
    """
    config = get_config()
    data_dir = config.get("data_dir", "./tradingagents/dataflows/data_cache")

    # Try to read cached SEDAR filings data
    sedar_file_path = os.path.join(data_dir, "company_data", "sedar_filings", f"{ticker}_sedar_data.json")

    if not os.path.exists(sedar_file_path):
        sedar_file_path = get_fuzzy_matched_data_path(ticker, sedar_file_path)

    if not os.path.exists(sedar_file_path):
        print(f"\n\nWarning: Data path {sedar_file_path} does not exist\n\n")
        return f"""
## {ticker} SEDAR Filings Analysis (Offline)

**Status:** No cached SEDAR filings data available for {ticker}

**Note:** SEDAR filings are for Canadian companies. If {ticker} is a US company, use SEC filings instead.

**Recommendation:**
- Use online tools for real-time SEDAR filings
- Or download and cache SEDAR filings data for offline use

*Data as of {curr_date}*
"""
    else:
        try:
            with open(sedar_file_path, 'r') as f:
                sedar_data = json.load(f)

            return f"""
## {ticker} SEDAR Filings Analysis (Offline)

**Latest AIF Report:**
- Filing Date: {sedar_data.get('latest_aif_date', 'N/A')}
- Revenue: ${sedar_data.get('revenue', 'N/A')}
- Net Income: ${sedar_data.get('net_income', 'N/A')}
- Total Assets: ${sedar_data.get('total_assets', 'N/A')}

**Key Financial Metrics:**
- Revenue Growth: {sedar_data.get('revenue_growth', 'N/A')}
- Net Margin: {sedar_data.get('net_margin', 'N/A')}
- ROE: {sedar_data.get('roe', 'N/A')}
- Debt-to-Equity: {sedar_data.get('debt_to_equity', 'N/A')}

**Business Overview:**
{sedar_data.get('business_overview', 'No business overview available')}

**Risk Factors:**
{sedar_data.get('risk_factors', 'No risk factors data available')}

*Data retrieved from cached SEDAR filings as of {curr_date}*
"""
        except Exception as e:
            return f"Error reading SEDAR filings data for {ticker}: {str(e)}"
