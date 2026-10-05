import os
import re
import time
import json
from copy import deepcopy
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from ..utils.portfolio_utils import format_portfolio_context
from .tool_utils import get_tool_messages, get_model_name

def create_news_analyst(llm, toolkit):
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

    def news_analyst_node(state):
        trade_date = state["trade_date"]
        company_name = state["company_of_interest"]

        if "news_analyst_state" in state:
            news_analyst_state = state["news_analyst_state"]
            news_reports_str = news_analyst_state.get("news_reports_str", "")
            count = news_analyst_state.get("count", 0)

        portfolio_context = format_portfolio_context(state.get("portfolio_summary"), company_name)

        if toolkit.config["online_tools"]:
            tools = [
                getattr(toolkit, f"get_global_news_{llm_provider}"),
                toolkit.get_google_news
            ]
        else:
            tools = [
                toolkit.get_finnhub_news,
                # toolkit.get_reddit_news,
                toolkit.get_google_news,
            ]

        system_message = (
            f"You are a **News Researcher** tasked with analyzing news and trends at the company {company_name} between the period of {trade_date} and the prior week."
            " Please write a comprehensive report of the state of the world on the trade date that is relevant for trading and macroeconomics."
            " **Data Sources:** Look at news from EODHD, and finnhub to be comprehensive."
            " **Analysis Rules:** Do not simply state the trends are mixed, provide detailed and finegrained analysis and insights that may help traders make decisions."
            " **Output:** Make sure to append a Makrdown table at the end of the report to organize key points in the report, organized and easy to read.\n"
            f" **Portfolio Context:** {portfolio_context}"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question. Execute what you can to make progress."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    " For your reference, the trade date is {trade_date}. The company we want to look at is {company_name}. "
                    " DONOT involve collect information from the future, only use information before the trade date {trade_date}."
                    " Based on your analysis, provide a specific recommendation and always conclude your response with 'FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** X%', where X is the percentage (0-100) of the position to trade."
                    " If no specific percentage is mentioned, default to 100%.",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(trade_date=trade_date)
        prompt = prompt.partial(company_name=company_name)

        chain = prompt | llm.bind_tools(tools)
        result = chain.invoke(state["messages"])

        report = ""
        if len(result.tool_calls) == 0:
            report = result.content

        if "news_analyst_state" in state:
            if report is not None and isinstance(report, str) and len(report) > 0:
                count = count + 1
                news_reports_str = news_reports_str + f"## Report {count} from {llm_provider}:\n{report}\n\n"

            return {
                "messages": [result],
                "news_report": report,
                "news_analyst_state": {
                    "news_reports_str": news_reports_str,
                    "count": count,
                },
            }
        else:
            return {
                "messages": [result],
                "news_report": report,
            }

    def news_analyst_node_load_data_from_cache(state):
        trade_date = state["trade_date"]
        company_name = state["company_of_interest"]

        if "news_analyst_state" in state:
            news_analyst_state = state["news_analyst_state"]
            news_reports_str = news_analyst_state.get("news_reports_str", "")
            count = news_analyst_state.get("count", 0)

        # check the format of the date yyyy-mm-dd
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", trade_date):
            raise ValueError(f"Invalid date format: {trade_date}")

        cache_file = deepcopy(toolkit.config["load_data_from_cache_file"])
        print(f"news_analyst cache_file original: {cache_file}")
        if toolkit.config["credibility_scorer_llm"] is not None and len(toolkit.config["credibility_scorer_llm"]) > 0:
            cache_file = cache_file.replace("/model_name/", f"/{model_name}/")
            cache_file = cache_file.replace("/company_name/", f"/{company_name}/")
            cache_file = cache_file.replace("_trade_date_", f"_{trade_date}_")
            cache_file = os.path.join(cache_file, "news_report.md")

            # load the cache file
            if os.path.exists(cache_file):
                with open(cache_file, "r", encoding="utf-8") as f:
                    report = f.read()
            else:
                print(f"Cache file {cache_file} does not exist!")
                return news_analyst_node(state)

        else:
            with open(cache_file, "r") as f:
                data_cache = json.load(f)

            dict_cache = data_cache[trade_date]
            assert company_name.upper() == dict_cache["company_of_interest"].upper(), \
                f"Company {company_name} is not the same as the company of interest in the cache {dict_cache['company_of_interest']}"
            report = dict_cache["news_report"]

        print("Loaded news report from cache file:", cache_file)
        if "news_analyst_state" in state:
            return {
                "messages": [],
                "news_report": report,
                "news_analyst_state": {
                    "news_reports_str": news_reports_str + f"## Report {count+1} from {llm_provider}:\n{report}\n\n",
                    "count": count + 1,
                },
            }
        else:
            return {
                "messages": [],
                "news_report": report,
            }

    if toolkit.config["load_data_from_cache"]:
        return news_analyst_node_load_data_from_cache
    else:
        return news_analyst_node