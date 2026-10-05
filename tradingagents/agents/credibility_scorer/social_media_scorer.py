import functools
import time
import json
import os

from .utils import extract_report_to_dict, extract_decision_from_report, get_model_name


def create_social_media_credibility_scorer(llm, toolkit):
    model_name = get_model_name(llm)
    def social_media_credibility_scorer_node(state):
        trade_date = state["trade_date"]
        ticker = state["company_of_interest"]
        company_name = state["company_of_interest"]

        sentiment_analyst_state = state["sentiment_analyst_state"]
        sentiment_reports_str = sentiment_analyst_state["sentiment_reports_str"]

        company_info = f"Based only on the provided set of sentiment analysis reports for {company_name} on {trade_date}, perform the following tasks."
        report_info = f"Here is the report: {sentiment_reports_str}"

        context = {
            "role": "user",
            "content": company_info + "\n" + """

## Consider only the following sentiment factors:
- **Activity metrics**: volume of mentions, posting frequency, engagement counts, and their temporal changes.
- **Sentiment polarity**: aggregated tone (positive / neutral / negative) if clearly defined by the source methodology.
- **Temporal patterns**: short-term spikes, drops, or persistence in sentiment signals relative to the reporting window.
- Other sentiment factors: sentiment analysis of the company's products, supply chain, regulation, competition, legal matters, or corporate actions.

## Tasks:

1. High-Confidence Sentiment Signals
- Identify sentiment observations that are clearly measured, numerically defined, and consistently reported across multiple sentiment reports (e.g., “mention volume increased sharply post-holiday”).
- collect as many high-confidence signals as possible, but don't include any low-confidence signals.

2. Internal Conflicts / Low-Confidence Signals
Flag any unsupported sentiment interpretations, including:
- sentiment conclusions without corresponding metrics,
- speculative explanations for sentiment changes.

3. Evidence Weighting and Ranking
- Select the **high-confidence** and **low-confidence** facts
- Return a concise Markdown table for the above 10 facts with columns:
Sentiment Signal | Source Type | Cross-Report Consistency | Confidence Level (High / Medium / Low).

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
                "content": f"""You are an experienced Scoial Media Credibility Scorer. Across multiple reports, evaluate only observable sentiment signals derived from public discourse.
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
            "sentiment_report": high_confidence_report,
            "sentiment_analyst_state": {
                "sentiment_reports_str": sentiment_reports_str,
                "sentiment_credibility_scorer": report_dict,
                "count": sentiment_analyst_state["count"],
            },
        }

    def social_media_credibility_scorer_node_load_data_from_cache(state):
        trade_date = state["trade_date"]
        ticker = state["company_of_interest"]
        company_name = state["company_of_interest"]

        cache_file = toolkit.config["load_data_from_cache_file"]
        cache_file = cache_file.replace("/model_name/", f"/{model_name}/")
        cache_file = cache_file.replace("/company_name/", f"/{company_name}/")
        cache_file = cache_file.replace("_trade_date_", f"_{trade_date}_")
        cache_file = cache_file + '.json'
        print(f"social_media_credibility_scorer cache_file: {cache_file}")

        # load the cache file
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                data_cache = json.load(f)
        else:
            print(f"Cache file {cache_file} does not exist!")
            return social_media_credibility_scorer_node(state)

        dict_cache = data_cache[trade_date]
        assert company_name.upper() == dict_cache["company_of_interest"].upper(), \
            f"Company {company_name} is not the same as the company of interest in the cache {dict_cache['company_of_interest']}"

        if "sentiment_analyst_state" in dict_cache and dict_cache["sentiment_analyst_state"] is not None:
            sentiment_analyst_state = dict_cache["sentiment_analyst_state"]
            sentiment_reports_str = sentiment_analyst_state["sentiment_reports_str"]
            report_dict = sentiment_analyst_state["sentiment_credibility_scorer"]
            count = sentiment_analyst_state.get("count", 0)

            # sentiment_report = dict_cache["sentiment_report"]
            test_low_confidence_report = False
            if test_low_confidence_report and "low_confidence_report" in report_dict and report_dict["low_confidence_report"] is not None:
                sentiment_report = report_dict["low_confidence_report"]  # ablation study
                if "low_confidence_signals" in report_dict and report_dict["low_confidence_signals"] is not None:
                    sentiment_report = sentiment_report + "\n\n" + "\n".join(report_dict["low_confidence_signals"])
                print("social_media_credibility_scorer: using low_confidence_report social_media")
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
                        "content": sentiment_report,
                    },
                ]
                result = llm.invoke(messages)
                sentiment_report = result.content
            else:
                sentiment_report = state["sentiment_report"]

                if "price_report" in state and state["price_report"] is not None:
                    messages = [
                        {
                            "role": "system",
                            "content": f"""
                            You are given a Price Report and a Sentiment Report for the same stock. Please revise the Sentiment Report and Final Sentiment Decision based on the Price Report.
                            # Guidelines:
                            - Treat the **price report as the dominant signal** for market direction and timing.
                            - Reinterpret and reweight existing **sentiment factors** to align with observed price trends.
                            - If sentiment factors and price diverge, explicitly explain the divergence (e.g., expectations already priced in, lagging sentiment factors, narrative shift).
                            - Do **not** introduce new sentiment facts; only adjust emphasis and interpretation.
                            - Maintain a cautious, analyst-style tone.

                            # Output Requirements:
                            - Return **only the revised Sentiment Report and Final Sentiment Decision**.
                            """},
                        {
                            "role": "user",
                            "content": f"Price Report: {state['price_report']}\nSentiment Report: {sentiment_report}",
                        },
                    ]
                    result = llm.invoke(messages)
                    sentiment_report = result.content
                    report_dict["sentiment_report"] = sentiment_report
                    report_dict["high_confidence_report"] = sentiment_report
                    decision_high_confidence, decision_low_confidence = extract_decision_from_report(
                        llm, sentiment_report, None
                    )
                    report_dict["decision_high_confidence"] = decision_high_confidence
                    report_dict["decision"] = decision_high_confidence

            return {
                "sentiment_report": sentiment_report,
                "sentiment_analyst_state": {
                    "sentiment_reports_str": sentiment_reports_str,
                    "sentiment_credibility_scorer": report_dict,
                    "count": count,
                },
            }

        else:
            print(f"sentiment_credibility_scorer cache_file does not contain sentiment_analyst_state! use sentiment_credibility_scorer_node to generate the report")
            return social_media_credibility_scorer_node(state)

    if toolkit.config["load_data_from_cache"]:
        return social_media_credibility_scorer_node_load_data_from_cache
    else:
        return social_media_credibility_scorer_node