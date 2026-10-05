import functools
import time
import json
import os

from .utils import extract_report_to_dict, extract_decision_from_report, get_model_name


def create_news_credibility_scorer(llm, toolkit):
    model_name = get_model_name(llm)
    def news_credibility_scorer_node(state):
        trade_date = state["trade_date"]
        ticker = state["company_of_interest"]
        company_name = state["company_of_interest"]

        news_analyst_state = state["news_analyst_state"]
        news_reports_str = news_analyst_state["news_reports_str"]

        company_info = f"Based only on the provided set of news analysis reports for {company_name} on {trade_date}, perform the following tasks."
        report_info = f"Here is the report: {news_reports_str}"

        context = {
            "role": "user",
            "content": company_info + "\n" + """

## Consider only the following categories of news information:
- **Macro and policy events**: officially released economic data, central bank decisions, regulatory actions.
- **Company-specific events**: reported developments involving products, supply chain, regulation, competition, legal matters, or corporate actions.
- **Industry or geopolitical events**: externally reported events that directly affect the company’s operations or markets.
- Other news information: news about the company's products, supply chain, regulation, competition, legal matters, or corporate actions.

## Tasks:
1. High-Confidence Consensus News Facts
- Identify clearly reported, time-stamped events or data points that are consistent across multiple reports or sources. These represent high-confidence news facts.
- collect as many high-confidence signals as possible, but don't include any low-confidence signals.

2. Internal Conflicts / Low-Confidence Signals
- Flag any conflicting figures, unclear sourcing, rumors, or narrative claims presented as facts. Briefly explain why reliability is low.

3. Evidence Weighting and Ranking
- Select the **high-confidence** and **low-confidence** fundamental facts
- Return a concise Markdown table for the above 10 facts with columns:
News Fact / Event | Source Type | Cross-Report Consistency | Confidence Level (High / Medium / Low).

4. Generate **three separate summary reports**:
- **High-Confidence Report**: based only on the high-confidence facts, detail a comprehensive analysis of the facts and keep the numerical values.
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
                "content": f"""You are an experienced News Credibility Scorer. Evaluate only verifiable news events and factual developments available up to the specified trade date.
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
            "news_report": high_confidence_report,
            "news_analyst_state": {
                "news_reports_str": news_reports_str,
                "news_credibility_scorer": report_dict,
                "count": news_analyst_state["count"],
            },
        }

    def news_credibility_scorer_node_load_data_from_cache(state):
        trade_date = state["trade_date"]
        ticker = state["company_of_interest"]
        company_name = state["company_of_interest"]

        cache_file = toolkit.config["load_data_from_cache_file"]
        cache_file = cache_file.replace("/model_name/", f"/{model_name}/")
        cache_file = cache_file.replace("/company_name/", f"/{company_name}/")
        cache_file = cache_file.replace("_trade_date_", f"_{trade_date}_")
        cache_file = cache_file + '.json'
        print(f"news_credibility_scorer cache_file: {cache_file}")

        # load the cache file
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                data_cache = json.load(f)
        else:
            print(f"Cache file {cache_file} does not exist!")
            return news_credibility_scorer_node(state)

        dict_cache = data_cache[trade_date]

        assert company_name.upper() == dict_cache["company_of_interest"].upper(), \
            f"Company {company_name} is not the same as the company of interest in the cache {dict_cache['company_of_interest']}"
        report = dict_cache["news_report"]

        if "news_analyst_state" in dict_cache and dict_cache["news_analyst_state"] is not None:
            news_analyst_state = dict_cache["news_analyst_state"]
            news_reports_str = news_analyst_state["news_reports_str"]
            report_dict = news_analyst_state["news_credibility_scorer"]
            count = news_analyst_state.get("count", 0)

            # news_report = dict_cache["news_report"]
            test_low_confidence_report = False
            if test_low_confidence_report and "low_confidence_report" in report_dict and report_dict["low_confidence_report"] is not None:
                news_report = report_dict["low_confidence_report"]  # ablation study
                if "low_confidence_signals" in report_dict and report_dict["low_confidence_signals"] is not None:
                    news_report = news_report + "\n\n" + "\n".join(report_dict["low_confidence_signals"])
                print("news_credibility_scorer: using low_confidence_report news")
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
                        "content": news_report,
                    },
                ]
                result = llm.invoke(messages)
                news_report = result.content
            else:
                news_report = state["news_report"]

                if "price_report" in state and state["price_report"] is not None:
                    messages = [
                        {
                            "role": "system",
                            "content": f"""
                            You are given a Price Report and a News Report for the same stock. Please revise the News Report and Final News Decision based on the Price Report.
                            # Guidelines:
                            - Treat the **price report as the dominant signal** for market direction and timing.
                            - Reinterpret and reweight existing **news factors** to align with observed price trends.
                            - If news factors and price diverge, explicitly explain the divergence (e.g., expectations already priced in, lagging news factors, narrative shift).
                            - Do **not** introduce new news facts; only adjust emphasis and interpretation.
                            - Maintain a cautious, analyst-style tone.

                            # Output Requirements:
                            - Return **only the revised News Report and Final News Decision**.
                            """},
                        {
                            "role": "user",
                            "content": f"Price Report: {state['price_report']}\nNews Report: {news_report}",
                        },
                    ]
                    result = llm.invoke(messages)
                    news_report = result.content
                    report_dict["news_report"] = news_report
                    report_dict["high_confidence_report"] = news_report
                    decision_high_confidence, decision_low_confidence = extract_decision_from_report(
                        llm, news_report, None
                    )
                    report_dict["decision_high_confidence"] = decision_high_confidence
                    report_dict["decision"] = decision_high_confidence

            return {
                "news_report": news_report,
                "news_analyst_state": {
                    "news_reports_str": news_reports_str,
                    "news_credibility_scorer": report_dict,
                    "count": count,
                },
            }
        else:
            print(f"news_credibility_scorer cache_file does not contain news_analyst_state! use news_credibility_scorer_node to generate the report")
            return news_credibility_scorer_node(state)

    if toolkit.config["load_data_from_cache"]:
        return news_credibility_scorer_node_load_data_from_cache
    else:
        return news_credibility_scorer_node