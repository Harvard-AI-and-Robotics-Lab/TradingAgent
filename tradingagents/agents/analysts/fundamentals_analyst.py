import os
import re
import time
import json
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from copy import deepcopy
from .tool_utils import get_tool_messages, get_model_name


def create_fundamentals_analyst(llm, toolkit):
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

    def fundamentals_analyst_node(state):
        trade_date = state["trade_date"]
        company_name = state["company_of_interest"]
        portfolio_summary = state["portfolio_summary"]

        if "fundamentals_analyst_state" in state:
            fundamentals_analyst_state = state["fundamentals_analyst_state"]
            fundamentals_reports_str = fundamentals_analyst_state.get("fundamentals_reports_str", "")
            count = fundamentals_analyst_state.get("count", 0)

        if toolkit.config["online_tools"]:
            tools = [
                getattr(toolkit, f"get_fundamentals_{llm_provider}"),
                getattr(toolkit, f"get_company_investor_relations_data_{llm_provider}"),
                getattr(toolkit, f"get_sec_filings_data_{llm_provider}"),
                getattr(toolkit, f"get_sedar_filings_data_{llm_provider}")
            ]
        else:
            tools = [
                # Original offline tools
                toolkit.get_finnhub_company_insider_sentiment,
                toolkit.get_finnhub_company_insider_transactions,
                toolkit.get_simfin_balance_sheet,
                toolkit.get_simfin_cashflow,
                toolkit.get_simfin_income_stmt,
                # Additional offline tools based on online functionality
                toolkit.get_company_investor_relations_data_offline,
                toolkit.get_sec_filings_data_offline,
                toolkit.get_sedar_filings_data_offline,
            ]

        system_message = (
                    f"""You are a senior fundamental analyst tasked with conducting a comprehensive analysis of company {company_name} on {trade_date} financial health and investment potential.

Your analysis should cover the following key areas:

**1. Financial Health & Performance:**
- Revenue growth trends (YoY, QoQ)
- Profitability metrics (Gross margin, Operating margin, Net margin)
- Efficiency ratios (ROE, ROIC, ROA)
- Leverage ratios (Debt-to-Equity, Interest coverage)
- Liquidity ratios (Current ratio, Quick ratio)

**2. Valuation Metrics:**
- P/E ratio (trailing and forward)
- P/S ratio, P/B ratio
- EV/EBITDA
- PEG ratio
- Dividend yield

**3. Data Sources to Utilize:**
- Company investor relations websites (earnings reports, financial statements)
- SEC filings (10-K, 10-Q, 8-K) for US companies
- SEDAR filings for Canadian companies
- Insider sentiment and transactions
- Financial statements (balance sheet, income statement, cash flow)

Provide actionable insights that help traders make informed decisions. Avoid generic statements like "trends are mixed" - instead provide specific, data-driven analysis with concrete numbers and trends.

Format your report with clear sections and include a comprehensive Markdown table summarizing key financial metrics and your analysis conclusions.

Based on your analysis, provide a specific recommendation and always conclude your response with 'FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** X%' where X is the percentage (0-100) of the position to trade. If no specific percentage is mentioned, default to 100%.
"""
        )

        if portfolio_summary:
            system_message += f"""
        Decide the action based on the current Portfolio position: if no position, only Buy (open position) or Hold (stay in cash); if holding, choose Buy (increase), Sell (reduce), or Hold (maintain). Partial trading is allowed, but diversification is not considered.
        Portfolio Context: {portfolio_summary}
        """

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " Execute what you can to make progress."
                    " You must always provide the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** X% as part of your response,"
                    " and prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** X% so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    " For your reference, the trade date is {trade_date}. The company we want to look at is {company_name}. "
                    " DONOT involve collect information from the future, only use information BEFORE the trade date {trade_date}.",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(trade_date=trade_date)
        prompt = prompt.partial(company_name=company_name)

        chain = prompt | llm.bind_tools(tools)  # The output of prompt (sequence of messages) is automatically passed to llm as input

        result = chain.invoke(state["messages"])

        # if len(result.tool_calls) > 0:
        #     tool_messages = get_tool_messages(result, tools)
        #     result = chain.invoke([state["messages"], *tool_messages])

        # assert len(result.tool_calls) == 0, "Tool calls are not allowed"
        # report = result.content

        report = ""
        if len(result.tool_calls) == 0:
            report = result.content

        if "fundamentals_analyst_state" in state:
            if report is not None and isinstance(report, str) and len(report) > 0:
                count = count + 1
                fundamentals_reports_str = fundamentals_reports_str + f"## Report {count} from {llm_provider}:\n{report}\n\n"

            return {
                "messages": [result],
                "fundamentals_report": report,
                "fundamentals_analyst_state": {
                    "fundamentals_reports_str": fundamentals_reports_str,
                    "count": count,
                },
            }
        else:
            return {
                "messages": [result],
                "fundamentals_report": report,
            }

    def fundamentals_analyst_node_load_data_from_cache(state):
        trade_date = state["trade_date"]
        company_name = state["company_of_interest"]

        if "fundamentals_analyst_state" in state:
            fundamentals_analyst_state = state["fundamentals_analyst_state"]
            fundamentals_reports_str = fundamentals_analyst_state.get("fundamentals_reports_str", "")
            count = fundamentals_analyst_state.get("count", 0)

        # check the format of the date yyyy-mm-dd
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", trade_date):
            raise ValueError(f"Invalid date format: {trade_date}")


        cache_file = deepcopy(toolkit.config["load_data_from_cache_file"])
        print(f"fundamentals_analyst cache_file original: {cache_file}")
        if toolkit.config["credibility_scorer_llm"] is not None and len(toolkit.config["credibility_scorer_llm"]) > 0:
            cache_file = cache_file.replace("/model_name/", f"/{model_name}/")
            cache_file = cache_file.replace("/company_name/", f"/{company_name}/")
            cache_file = cache_file.replace("_trade_date_", f"_{trade_date}_")
            cache_file = os.path.join(cache_file, "fundamentals_report.md")

            # load the cache file
            if os.path.exists(cache_file):
                with open(cache_file, "r", encoding="utf-8") as f:
                    report = f.read()
            else:
                print(f"Cache file {cache_file} does not exist! use fundamentals_analyst_node to generate the report")
                return fundamentals_analyst_node(state)

        else:
            with open(cache_file, "r") as f:
                data_cache = json.load(f)

            dict_cache = data_cache[trade_date]
            assert company_name.upper() == dict_cache["company_of_interest"].upper(), \
                f"Company {company_name} is not the same as the company of interest in the cache {dict_cache['company_of_interest']}"
            report = dict_cache["fundamentals_report"]

        print("Loaded fundamentals report from cache file:", cache_file)
        if "fundamentals_analyst_state" in state:
            return {
                "messages": [],
                "fundamentals_report": report,
                "fundamentals_analyst_state": {
                    "fundamentals_reports_str": fundamentals_reports_str + f"## Report {count+1} from {llm_provider}:\n{report}\n\n",
                    "count": count + 1,
                },
            }
        else:
            return {
                "messages": [],
                "fundamentals_report": report,
            }

    if toolkit.config["load_data_from_cache"]:
        return fundamentals_analyst_node_load_data_from_cache
    else:
        return fundamentals_analyst_node