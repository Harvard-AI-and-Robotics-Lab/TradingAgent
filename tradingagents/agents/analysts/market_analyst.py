import os
import re
import time
import json
from copy import deepcopy
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from ..utils.portfolio_utils import format_portfolio_context
from .tool_utils import get_tool_messages, get_model_name


def create_market_analyst(llm, toolkit):
    model_name = get_model_name(llm)
    llm_provider = model_name.split("-")[0]
    if llm_provider == "gpt":
        llm_provider = "openai"
    elif llm_provider == "gemini":
        llm_provider = "gemini"
    elif llm_provider == "grok":
        llm_provider = "xai"
    elif llm_provider == "claude":
        llm_provider = "anthropic"
    else:
        raise ValueError(f"Unsupported LLM model: {llm}")

    def market_analyst_node(state):
        trade_date = state["trade_date"]
        company_name = state["company_of_interest"]

        portfolio_context = format_portfolio_context(state.get("portfolio_summary"), company_name)

        if "fundamentals_analyst_state" in state:
            market_analyst_state = state["market_analyst_state"]
            market_reports_str = market_analyst_state.get("market_reports_str", "")
            count = market_analyst_state.get("count", 0)

        if toolkit.config["online_tools"]:
            tools = [
                toolkit.get_YFin_data_online,
                toolkit.get_stockstats_indicators_report_online,
            ]
        else:
            tools = [
                toolkit.get_YFin_data,
                toolkit.get_stockstats_indicators_report,
            ]

        system_message = (
            f"""You are a trading assistant tasked with analyzing financial markets. Your role is to select the **most relevant indicators** for a given market condition or trading strategy from the following list. The goal is to choose up to **8 indicators** that provide complementary insights without redundancy. Categories and each category's indicators are:

# Indicators
- **close_10_ema**: Short-term momentum, quick entries; noisy in chop → confirm with longer MAs.
- **close_50_sma**: Medium-term trend; acts as support/resistance.
- **close_200_sma**: Long-term benchmark; golden/death cross signals.
- **macd**: Momentum via EMA gaps; crossovers/divergence = trend change.
- **macds**: Smoothed MACD; cross with MACD line = triggers.
- **macdh**: MACD vs signal gap; strength/reversal cue but volatile.
- **rsi**: Overbought/oversold (>70/<30) + divergence; extremes persist in strong trends.
- **boll**: 20-SMA mid-band; with UB/LB spots breakouts/reversals.
- **boll_ub / boll_lb**: ±2sigma bands; overbought/oversold zones, confirm with other signals.
- **atr**: Volatility; use for stops and position sizing.
- **vwma**: Volume-weighted MA; confirms moves, but sensitive to spikes.

# Usage Rules
- Always call **`get_YFin_data` first** to load CSV.
- Use the **exact indicator names** as defined.
- Pick **complementary indicators** (trend + momentum + volatility + volume); avoid duplicates.
- Provide **clear, detailed analysis** (levels, conflicts, scenarios), not vague phrases like mixed trends.

All the indicators should be generated based on the trade date {trade_date} and the company of interest {company_name}.
DONOT involve collect information from the future, only use information before the trade date {trade_date}.

Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read.

Based on your analysis, provide a specific recommendation and always conclude your response with 'FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** X%' where X is the percentage (0-100) of the position to trade. If no specific percentage is mentioned, default to 100%.

Portfolio Context: {portfolio_context}
Decide the action based on the current Portfolio position: if no position, only Buy (open position) or Hold (stay in cash); if holding, choose Buy (increase), Sell (reduce), or Hold (maintain). Partial trading is allowed, but diversification is not considered.
"""
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " Execute what you can to make progress."
                    " Please provide the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** X%,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** X% so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    " For your reference, the trade date is {trade_date}. The company we want to look at is {company_name}. "
                    " DONOT involve collect information from the future, only use information before the trade date {trade_date}.",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        system_message = system_message.format(portfolio_context=portfolio_context)
        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(trade_date=trade_date)
        prompt = prompt.partial(company_name=company_name)

        chain = prompt | llm.bind_tools(tools)

        result = chain.invoke(state["messages"])

        report = ""
        if len(result.tool_calls) == 0:
            report = result.content

        if "market_analyst_state" in state:
            if report is not None and isinstance(report, str) and len(report) > 0:
                count = count + 1
                market_reports_str = market_reports_str + f"## Report {count} from {llm_provider}:\n{report}\n\n"


            return {
                "messages": [result],
                "market_report": report,
                "market_analyst_state": {
                    "market_reports_str": market_reports_str,
                    "count": count,
                },
            }
        else:
            return {
                "messages": [result],
                "market_report": report,
            }

    def market_analyst_node_load_data_from_cache(state):
        trade_date = state["trade_date"]
        company_name = state["company_of_interest"]

        if "market_analyst_state" in state:
            market_analyst_state = state["market_analyst_state"]
            market_reports_str = market_analyst_state.get("market_reports_str", "")
            count = market_analyst_state.get("count", 0)

        # check the format of the date yyyy-mm-dd
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", trade_date):
            raise ValueError(f"Invalid date format: {trade_date}")


        cache_file = deepcopy(toolkit.config["load_data_from_cache_file"])
        print(f"market_analyst cache_file original: {cache_file}")
        if toolkit.config["credibility_scorer_llm"] is not None and len(toolkit.config["credibility_scorer_llm"]) > 0:
            cache_file = cache_file.replace("/model_name/", f"/{model_name}/")
            cache_file = cache_file.replace("/company_name/", f"/{company_name}/")
            cache_file = cache_file.replace("_trade_date_", f"_{trade_date}_")
            cache_file = os.path.join(cache_file, "market_report.md")

            # load the cache file
            if os.path.exists(cache_file):
                with open(cache_file, "r", encoding="utf-8") as f:
                    report = f.read()
            else:
                print(f"Cache file {cache_file} does not exist!")
                return market_analyst_node(state)

        else:

            with open(cache_file, "r") as f:
                data_cache = json.load(f)

            dict_cache = data_cache[trade_date]
            assert company_name.upper() == dict_cache["company_of_interest"].upper(), \
                f"Company {company_name} is not the same as the company of interest in the cache {dict_cache['company_of_interest']}"
            report = dict_cache["market_report"]

        print("Loaded market report from cache file:", cache_file)
        if "market_analyst_state" in state:
            return {
                "messages": [],
                "market_report": report,
                "market_analyst_state": {
                    "market_reports_str": market_reports_str + f"## Report {count+1} from {llm_provider}:\n{report}\n\n",
                    "count": count + 1,
                },
            }
        else:
            return {
                "messages": [],
                "market_report": report,
            }

    if toolkit.config["load_data_from_cache"]:
        return market_analyst_node_load_data_from_cache
    else:
        return market_analyst_node
