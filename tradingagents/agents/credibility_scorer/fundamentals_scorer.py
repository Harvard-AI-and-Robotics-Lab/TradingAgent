import functools
import time
import json
import os
from copy import deepcopy

from .utils import extract_report_to_dict, extract_decision_from_report, get_model_name

from ..analysts.fundamentals_analyst import create_fundamentals_analyst


def create_fundamentals_credibility_scorer(llm, toolkit):
    model_name = get_model_name(llm)

    fundamentals_analyst = create_fundamentals_analyst(llm, toolkit)

    def fundamentals_credibility_scorer_node(state):
        trade_date = state["trade_date"]
        ticker = state["company_of_interest"]
        company_name = state["company_of_interest"]

        fundamentals_analyst_state = state["fundamentals_analyst_state"]
        fundamentals_reports_str = fundamentals_analyst_state["fundamentals_reports_str"]

        company_info = f"Based only on the provided set of fundamental analysis reports for {company_name} on {trade_date}, perform the following tasks."
        report_info = f"Here is the report: {fundamentals_reports_str}"


        context = {
            "role": "user",
            "content": company_info + "\n" + """
## Consider the following fundamental factors:
- Realized financial results: annual and quarterly revenue, net income, gross/operating/net margins.
- Cash flow metrics: operating cash flow, free cash flow.
- Capital structure & liquidity: total assets, total liabilities, shareholders equity, debt-to-equity, current ratio, quick ratio.
- Efficiency & return metrics: ROE, ROA, ROIC (note definition or calculation differences if present).
- Other fundamental factors: earnings per share, price-to-earnings ratio, price-to-book ratio, price-to-sales ratio, etc.

## Tasks:
1. High-Confidence Consensus Facts
- Identify factual statements that are consistently cited across multiple reports and numerically or semantically identical. These represent high-confidence invariant facts.
- collect as many high-confidence signals as possible, but don't include any low-confidence signals.

2. Internal Conflicts / Low-Confidence Signals
- Detect any numerical inconsistencies, definition mismatches, or contradictory claims across the reports. Flag these as low-confidence information and briefly explain the conflict.

3. Evidence Weighting and Ranking
- Select the **high-confidence** and **low-confidence** fundamental facts
- Return a concise Markdown table for the above facts with columns:
Fact, Indicator Source, Cross-Report Consistency, Confidence Level (High / Medium / Low).

4. Generate **three separate summary reports**:
- **High-Confidence Report**: based on the high-confidence facts, detail a comprehensive analysis of the facts and keep the original numerical values.
- **Low-Confidence Report**: based on the low-confidence facts, detail a comprehensive analysis of the facts and clearly explaining sources of uncertainty.
- **Data Leakage Audit Report**: State whether data leakage is detected, if yes, list and briefly explain any facts flagged for leakage. If none are found, explicitly state that no data leakage is detected.

5. Decision Making:
- Based on your analysis, provide a specific recommendation and always conclude your response with 'FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** X%' where X is the percentage (0-100) of the position to trade. If no specific percentage is mentioned, default to 100%.
- **Output the final decision in the format**: {"action": "BUY|SELL|HOLD", "percentage": number}

""" + report_info
        }

        messages = [
            {
                "role": "system",
                "content": f"""You are an experienced Fundamental Credibility Scorer. Evaluate only fundamental information in the provided reports.
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
            "fundamentals_report": high_confidence_report,
            "fundamentals_analyst_state": {
                "fundamentals_reports_str": fundamentals_reports_str,
                "fundamentals_credibility_scorer": report_dict,
                "count": fundamentals_analyst_state["count"],
            },
        }

    def fundamentals_credibility_scorer_node_load_data_from_cache(state):
        trade_date = state["trade_date"]
        ticker = state["company_of_interest"]
        company_name = state["company_of_interest"]

        cache_file = deepcopy(toolkit.config["load_data_from_cache_file"])
        cache_file = cache_file.replace("/model_name/", f"/{model_name}/")
        cache_file = cache_file.replace("/company_name/", f"/{company_name}/")
        cache_file = cache_file.replace("_trade_date_", f"_{trade_date}_")
        # cache_file = os.path.join(cache_file, "fundamentals_report.md")
        cache_file = cache_file + '.json'
        print(f"fundamentals_credibility_scorer cache_file updated: {cache_file}")

        # load the cache file
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                data_cache = json.load(f)
        else:
            print(f"Cache file {cache_file} does not exist!")
            return fundamentals_credibility_scorer_node(state)

        dict_cache = data_cache[trade_date]
        assert company_name.upper() == dict_cache["company_of_interest"].upper(), \
            f"Company {company_name} is not the same as the company of interest in the cache {dict_cache['company_of_interest']}"

        if "fundamentals_analyst_state" in dict_cache and dict_cache["fundamentals_analyst_state"] is not None:
            fundamentals_analyst_state = dict_cache["fundamentals_analyst_state"]
            fundamentals_reports_str = fundamentals_analyst_state["fundamentals_reports_str"]
            report_dict = fundamentals_analyst_state["fundamentals_credibility_scorer"]
            count = fundamentals_analyst_state.get("count", 0)

            # fundamentals_report = dict_cache["fundamentals_report"]
            test_low_confidence_report = False
            if test_low_confidence_report and "low_confidence_report" in report_dict and report_dict["low_confidence_report"] is not None:
                fundamentals_report = report_dict["low_confidence_report"]  # ablation study
                if "low_confidence_signals" in report_dict and report_dict["low_confidence_signals"] is not None:
                    fundamentals_report = fundamentals_report + "\n\n" + "\n".join(report_dict["low_confidence_signals"])

                print("fundamentals_credibility_scorer: using low_confidence_report")
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
                        "content": fundamentals_report,
                    },
                ]
                result = llm.invoke(messages)
                fundamentals_report = result.content
            else:
                fundamentals_report = dict_cache["fundamentals_report"]

                if "price_report" in state and state["price_report"] is not None:
                    messages = [
                        {
                            "role": "system",
                            "content": f"""
                            You are given a Price Report and a Fundamentals Report for the same stock. Please revise the Fundamentals Report and Final Fundamentals Decision based on the Price Report.

                            # Guidelines:
                            - Treat the **price report as the dominant signal** for market direction and timing.
                            - Reinterpret and reweight existing **fundamental factors** to align with observed price trends.
                            - If fundamentals and price diverge, explicitly explain the divergence (e.g., expectations already priced in, lagging fundamentals, narrative shift).
                            - Do **not** introduce new fundamental facts; only adjust emphasis and interpretation.
                            - Maintain a cautious, analyst-style tone.

                            # Output Requirements:
                            - Return **only the revised Fundamentals Report and Final Fundamentals Decision**.
                            """,
                        },
                        {
                            "role": "user",
                            "content": f"Price Report: {state['price_report']}\nFundamentals Report: {fundamentals_report}",
                        },
                    ]
                    result = llm.invoke(messages)
                    fundamentals_report = result.content
                    report_dict["fundamentals_report"] = fundamentals_report
                    report_dict["high_confidence_report"] = fundamentals_report
                    decision_high_confidence, decision_low_confidence = extract_decision_from_report(
                        llm, fundamentals_report, None
                    )
                    report_dict["decision_high_confidence"] = decision_high_confidence
                    report_dict["decision"] = decision_high_confidence

            return {
                "fundamentals_report": fundamentals_report,
                "fundamentals_analyst_state": {
                    "fundamentals_reports_str": fundamentals_reports_str,
                    "fundamentals_credibility_scorer": report_dict,
                    "count": count,
                },
            }
        else:
            print(f"fundamentals_credibility_scorer cache_file does not contain fundamentals_analyst_state! use fundamentals_credibility_scorer_node to generate the report")
            return fundamentals_credibility_scorer_node(state)

    if toolkit.config["load_data_from_cache"]:
        return fundamentals_credibility_scorer_node_load_data_from_cache
    else:
        return fundamentals_credibility_scorer_node