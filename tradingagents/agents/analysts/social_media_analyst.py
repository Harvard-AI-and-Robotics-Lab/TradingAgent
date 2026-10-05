import os
import re
import time
import json
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from ..utils.portfolio_utils import format_portfolio_context
from .tool_utils import get_tool_messages, get_model_name

def create_social_media_analyst(llm, toolkit):
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

    def social_media_analyst_node(state):
        trade_date = state["trade_date"]
        company_name = state["company_of_interest"]

        if "sentiment_analyst_state" in state:
            sentiment_analyst_state = state["sentiment_analyst_state"]
            sentiment_reports_str = sentiment_analyst_state.get("sentiment_reports_str", "")
            count = sentiment_analyst_state.get("count", 0)

        portfolio_context = format_portfolio_context(state.get("portfolio_summary"), company_name)

        if toolkit.config["online_tools"]:
            tools = [
                getattr(toolkit, f"get_stock_news_{llm_provider}")
            ]
        else:
            tools = [
                toolkit.get_reddit_stock_info,
            ]

        system_message = (
    "You are a Social Media Sentiment Analyst producing **investment-grade sentiment intelligence** "
    f"for **{company_name}** using ONLY public discourse available on or before **{trade_date}**.\n\n"

    f"**Time-Lock:** No post-{trade_date} knowledge, price action, or outcomes. No hindsight.\n\n"

    "**Analysis Rules:**\n"
    "- Avoid generic labels (e.g., 'mixed'). Explain *why* sentiment differs.\n"
    "- Compare retail (X/Reddit) vs. professional/investor discourse.\n"
    "- Extract concrete drivers (product, execution, valuation, macro, leadership).\n"
    "- Weigh conviction and credibility, not just volume.\n\n"

    f"**Decision:** Evaluate sentiment within **{portfolio_context}** and recommend:\n"
    "Buy / Sell / Hold (partial scaling allowed).\n\n"

    "**Output:** Narrative analysis + Markdown table summarizing drivers, evidence strength, "
    "sentiment direction, and final action."
)


        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question. Execute what you can to make progress."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    " For your reference, the trade date is {trade_date}. The company we want to analyze is {company_name}. "
                    " DONOT involve collect information from the future, only use information before the trade date {trade_date}."
                    " Based on your analysis, provide a specific recommendation and always conclude your response with 'FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** X%', where X is the percentage (0-100) of the position to trade. "
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

        if "sentiment_analyst_state" in state:
            if report is not None and isinstance(report, str) and len(report) > 0:
                count = count + 1
                sentiment_reports_str = sentiment_reports_str + f"## Report {count} from {llm_provider}:\n{report}\n\n"

            return {
                "messages": [result],
                "sentiment_report": report,
                "sentiment_analyst_state": {
                    "sentiment_reports_str": sentiment_reports_str,
                    "count": count,
                },
            }
        else:
            return {
                "messages": [result],
                "sentiment_report": report,
            }

    def social_media_analyst_node_load_data_from_cache(state):
        trade_date = state["trade_date"]
        company_name= state["company_of_interest"]

        if "sentiment_analyst_state" in state:
            sentiment_analyst_state = state["sentiment_analyst_state"]
            sentiment_reports_str = sentiment_analyst_state.get("sentiment_reports_str", "")
            count = sentiment_analyst_state.get("count", 0)

        # check the format of the date yyyy-mm-dd
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", trade_date):
            raise ValueError(f"Invalid date format: {trade_date}")


        cache_file = toolkit.config["load_data_from_cache_file"]
        if toolkit.config["credibility_scorer_llm"] is not None and len(toolkit.config["credibility_scorer_llm"]) > 0:
            cache_file = cache_file.replace("/model_name/", f"/{model_name}/")
            cache_file = cache_file.replace("/company_name/", f"/{company_name}/")
            cache_file = cache_file.replace("_trade_date_", f"_{trade_date}_")
            cache_file = os.path.join(cache_file, "sentiment_report.md")

            # load the cache file
            if os.path.exists(cache_file):
                with open(cache_file, "r", encoding="utf-8") as f:
                    report = f.read()
            else:
                print(f"Cache file {cache_file} does not exist!")
                return social_media_analyst_node(state)

        else:

            with open(toolkit.config["load_data_from_cache_file"], "r") as f:
                data_cache = json.load(f)

            dict_cache = data_cache[trade_date]
            assert company_name.upper() == dict_cache["company_of_interest"].upper(), \
                f"Company {company_name} is not the same as the company of interest in the cache {dict_cache['company_of_interest']}"
            report = dict_cache["sentiment_report"]

        print("Loaded sentiment report from cache file:", cache_file)
        if "sentiment_analyst_state" in state:
            return {
                "messages": [],
                "sentiment_report": report,
                "sentiment_analyst_state": {
                    "sentiment_reports_str": sentiment_reports_str + f"## Report {count+1} from {llm_provider}:\n{report}\n\n",
                    "count": count + 1,
                },
            }
        else:
            return {
                "messages": [],
                "sentiment_report": report,
            }

    if toolkit.config["load_data_from_cache"]:
        return social_media_analyst_node_load_data_from_cache
    else:
        return social_media_analyst_node
