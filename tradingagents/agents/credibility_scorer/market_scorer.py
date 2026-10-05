import functools
import time
import json
import os

from .utils import extract_report_to_dict, extract_decision_from_report, get_model_name


def create_market_credibility_scorer(llm, toolkit):
    model_name = get_model_name(llm)
    def market_credibility_scorer_node(state):
        trade_date = state["trade_date"]
        ticker = state["company_of_interest"]
        company_name = state["company_of_interest"]

        market_analyst_state = state["market_analyst_state"]
        market_reports_str = market_analyst_state["market_reports_str"]

        company_info = f"Based only on the provided set of market analysis reports for {company_name} on {trade_date}, perform the following tasks."
        report_info = f"Here is the report: {market_reports_str}"

        context = {
            "role": "user",
            "content": company_info + "\n" + """

## Consider the following market / technical factors:
- **Trend indicators**: moving averages (e.g., 10-day EMA, 50-day SMA, 200-day SMA) and their relative positioning.
- **Momentum indicators**: MACD (line, signal, histogram) and RSI, including signal direction and threshold states.
- **Volatility indicators**: ATR or equivalent measures derived from historical price ranges.
- Other market / technical factors: price-to-earnings ratio, price-to-book ratio, price-to-sales ratio, etc.

## Tasks:

1. High-Confidence Consensus Signals
- Identify indicator values or signal states that are consistently reported across multiple market reports and are numerically or semantically aligned (e.g., “price above 10-day SMA”, “MACD histogram negative”).
- collect as many high-confidence signals as possible, but don't include any low-confidence signals.

2. Internal Conflicts / Low-Confidence Signals
- Flag any numerical discrepancies, contradictory indicator interpretations, or inconsistent signal descriptions across reports.

3. Evidence Weighting and Ranking
- Select the **high-confidence** and **low-confidence** fundamental facts
- Return a concise Markdown table for the above 10 facts with columns:
Market Signal | Indicator Source | Cross-Report Consistency | Confidence Level (High / Medium / Low).

4. Generate **three separate summary reports**:
- **High-Confidence Report**: based only on the high-confidence facts, detail a comprehensive analysis of the facts and keep the original numerical values.
- **Low-Confidence Report**: based only on the low-confidence facts, detail a comprehensive analysis of the facts and clearly explaining sources of uncertainty.
- **Data Leakage Audit Report**: State whether data leakage is detected, if yes, list and briefly explain any facts flagged for leakage. If none are found, explicitly state that no data leakage is detected.

5. Decision Making:
- Based on your analysis, provide a specific recommendation and always conclude your response with 'FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** X%' where X is the percentage (0-100) of the position to trade. If no specific percentage is mentioned, default to 100%.
- **Output the final decision in the format**: {"action": "BUY|SELL|HOLD", "percentage": number}

""" + report_info
        }

        messages = [
            {
                "role": "system",
                "content": f"""You are an experienced Market Credibility Scorer. Evaluate only technical and market indicators derived from historical price and volume data. Focus on indicator values, signal states, and cross-indicator consistency.
Consistent signals across reports indicate higher credibility, while discrepancies suggest lower reliability.""",
            },
            context,
        ]

        result = llm.invoke(messages)
        report_str = result.content

        report_dict = extract_report_to_dict(llm, report_str)

        if "high_confidence_report" in report_dict:
            if isinstance(report_dict["high_confidence_report"], str):
                high_confidence_report = report_dict["high_confidence_report"]
            else:
                high_confidence_report = report_str
        elif "high_confidence_facts" in report_dict:
            if isinstance(report_dict["high_confidence_facts"], str):
                high_confidence_report = report_dict["high_confidence_facts"]
            elif isinstance(report_dict["high_confidence_facts"], list):
                high_confidence_report = "\n".join(report_dict["high_confidence_facts"])
            else:
                high_confidence_report = report_str
        else:
            high_confidence_report = report_str

        low_confidence_report = None
        if "low_confidence_report" in report_dict and isinstance(report_dict["low_confidence_report"], str):
            low_confidence_report = report_dict["low_confidence_report"]

        decision_high_confidence, decision_low_confidence = extract_decision_from_report(
            llm, high_confidence_report, low_confidence_report
        )
        report_dict["decision_high_confidence"] = decision_high_confidence
        report_dict["decision_low_confidence"] = decision_low_confidence
        report_dict["decision"] = decision_high_confidence
        high_confidence_report = high_confidence_report + f"\n\nFINAL TRANSACTION PROPOSAL: {decision_high_confidence}"
        report_dict["high_confidence_report"] = high_confidence_report
        if low_confidence_report is not None:
            low_confidence_report = low_confidence_report + f"\n\nFINAL TRANSACTION PROPOSAL: {decision_low_confidence}"
            report_dict["low_confidence_report"] = low_confidence_report

        return {
            "market_report": high_confidence_report,
            "market_analyst_state": {
                "market_reports_str": market_reports_str,
                "market_credibility_scorer": report_dict,
                "count": market_analyst_state["count"],
            },
        }

    def market_credibility_scorer_node_load_data_from_cache(state):
        trade_date = state["trade_date"]
        ticker = state["company_of_interest"]
        company_name = state["company_of_interest"]


        cache_file = toolkit.config["load_data_from_cache_file"]
        cache_file = cache_file.replace("/model_name/", f"/{model_name}/")
        cache_file = cache_file.replace("/company_name/", f"/{company_name}/")
        cache_file = cache_file.replace("_trade_date_", f"_{trade_date}_")
        # cache_file = os.path.join(cache_file, "fundamentals_report.md")
        cache_file = cache_file + '.json'
        print(f"market_credibility_scorer cache_file: {cache_file}")

        # load the cache file
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                data_cache = json.load(f)
        else:
            print(f"Cache file {cache_file} does not exist!")
            return market_credibility_scorer_node(state)

        dict_cache = data_cache[trade_date]
        assert company_name.upper() == dict_cache["company_of_interest"].upper(), \
            f"Company {company_name} is not the same as the company of interest in the cache {dict_cache['company_of_interest']}"

        if "market_analyst_state" in dict_cache and dict_cache["market_analyst_state"] is not None:
            market_analyst_state = dict_cache["market_analyst_state"]
            market_reports_str = market_analyst_state["market_reports_str"]
            report_dict = market_analyst_state["market_credibility_scorer"]
            count = market_analyst_state.get("count", 0)

            # market_report = dict_cache["market_report"]
            test_low_confidence_report = False
            if test_low_confidence_report and "low_confidence_report" in report_dict and report_dict["low_confidence_report"] is not None:
                market_report = report_dict["low_confidence_report"]  # ablation study
                if "low_confidence_signals" in report_dict and report_dict["low_confidence_signals"] is not None:
                    market_report = market_report + "\n\n" + "\n".join(report_dict["low_confidence_signals"])
                print("market_credibility_scorer: using low_confidence_report market")
                messages = [
                    {
                        "role": "system",
                        "content": f"""
                        In the report, some terms and phrases are used to explicitly indicate that certain information is low confidence, speculative, anecdotal, or uncertain (e.g., wording related to low confidence, rumors, speculation, or unverified signals).
                        # Task: Remove all such words and phrases that convey low confidence or uncertainty, like low_confidence_signals, while preserving all remaining factual content exactly as written. Return the modified report.

                        # Requirements:
                        - Do not introduce any new facts, interpretations, assumptions, or rewording beyond the removal of low-confidence indicators.
                        - Do not add, infer, or correct information.
                        - Maintain the original structure, data, and wording of the remaining content as much as possible.
                        - The final output should read as a standard report, without any explicit signals of uncertainty or confidence qualifiers.
                        """},
                    {
                        "role": "user",
                        "content": market_report,
                    },
                ]
                result = llm.invoke(messages)
                market_report = result.content
            else:
                market_report = dict_cache["market_report"]

                if "price_report" in state and state["price_report"] is not None:
                    messages = [
                        {
                            "role": "system",
                            "content": f"""
                            You are given a Price Report and a Market Report for the same stock. Please revise the Market Report and Final Market Decision based on the Price Report.
                            # Guidelines:
                            - Treat the **price report as the dominant signal** for market direction and timing.
                            - Reinterpret and reweight existing **market factors** to align with observed price trends.
                            - If market factors and price diverge, explicitly explain the divergence (e.g., expectations already priced in, lagging market factors, narrative shift).
                            - Do **not** introduce new market facts; only adjust emphasis and interpretation.
                            - Maintain a cautious, analyst-style tone.

                            # Output Requirements:
                            - Return **only the revised Market Report and Final Market Decision**.
                            """},
                        {
                            "role": "user",
                            "content": f"Price Report: {state['price_report']}\nMarket Report: {market_report}",
                        },
                    ]

                    result = llm.invoke(messages)
                    market_report = result.content
                    report_dict["market_report"] = market_report
                    report_dict["high_confidence_report"] = market_report
                    decision_high_confidence, decision_low_confidence = extract_decision_from_report(
                        llm, market_report, None
                    )
                    report_dict["decision_high_confidence"] = decision_high_confidence
                    report_dict["decision"] = decision_high_confidence

            return {
                "market_report": market_report,
                "market_analyst_state": {
                    "market_reports_str": market_reports_str,
                    "market_credibility_scorer": report_dict,
                    "count": count,
                },
            }
        else:
            print(f"market_credibility_scorer cache_file does not contain market_analyst_state! use market_credibility_scorer_node to generate the report")
            return market_credibility_scorer_node(state)

    if toolkit.config["load_data_from_cache"]:
        return market_credibility_scorer_node_load_data_from_cache
    else:
        return market_credibility_scorer_node