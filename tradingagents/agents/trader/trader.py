import functools
import time
import json
from ..utils.portfolio_utils import format_portfolio_context, get_portfolio_analysis_questions
from tradingagents.graph.reflection import get_prompt_for_compressed_situation


def create_trader(llm, memory):
    def trader_node(state, name):
        trade_date = state["trade_date"]
        company_name = state["company_of_interest"]
        investment_plan = state["investment_plan"]
        market_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        latest_long_short_memory = state.get("latest_long_short_memory", "")

        portfolio_context = format_portfolio_context(state.get("portfolio_summary"), company_name)
        portfolio_questions = get_portfolio_analysis_questions(
            state.get("portfolio_summary"), company_name, "trade"
        )

        if memory is not None:
            if "compressed_situation" in state and len(state["compressed_situation"]) > 0:
                compressed_situation = state["compressed_situation"]
            else:
                curr_situation = f"Market: {market_report}\n\nSentiment: {sentiment_report}\n\nNews: {news_report}\n\nFundamentals: {fundamentals_report}"
                compressed_situation = llm.invoke(get_prompt_for_compressed_situation(curr_situation)).content
                state["compressed_situation"] = compressed_situation

            past_memories = memory.get_memories(compressed_situation, trade_date=state.get("trade_date"), n_matches=2)
            past_memory_str = ""
            for i, rec in enumerate(past_memories, 1):
                past_memory_str += f"Date: {rec['date']} ({rec['days_ago']} days ago)\n" + \
                f"Situation Similarity Score: {rec['similarity_score']:.3f}\n" + \
                f"Time Weight: {rec['time_weight']:.3f}\n" + \
                f"Relevance Score: {rec['relevance_score']:.3f}\n" + \
                f"Recommendation: {rec['recommendation']}\n\n"
        else:
            past_memory_str = "No past memories found."
        print(f"Trader: past_memory_str: {past_memory_str}")

        portfolio_questions_str = "\n".join([f"• {q}" for q in portfolio_questions])

        context = {
            "role": "user",
            "content": f"""
You are given an analyst team’s proposed investment plan for {company_name}. Use it as an input, but you must still form your own final decision.

All the information should be generated based on the trade date {trade_date} and the company of interest {company_name}.
DONOT involve collect information from the future, only use information before the trade date {trade_date}.

# Portfolio context: {portfolio_context}
Decide the action based on the current Portfolio position.

# Proposed Investment Plan: {investment_plan}
Leverage these insights to make an informed and strategic decision that considers both the company's prospects.

# Historical Trading Performance: {latest_long_short_memory}
Short-term and long-term strategy reports summarizing your historical trading performance (retrospective evaluation).
## Important constraint:
    They may ONLY be used to:
    - adjust confidence level
    - adjust position sizing
    - adjust risk tolerance
    - decide whether to be conservative or aggressive

Now produce your decision and explanation following the required structure.
""",
        }

        system_message = """
You are **a trading agent** analyzing market data to make investment decisions, only for single-asset trading.

# Task:
- You must produce ONE actionable decision for the specified asset: BUY / HOLD / SELL with a trade percentage X (0-100).
- Partial trading is allowed. Diversification is NOT considered.

# Decision explanation requirement (must follow exactly):
- Restate the proposed investment plan’s recommended action and key rationale in 2-4 sentences.
- Restate the current portfolio position in 1-2 sentences.
- Your independent judgment: state whether you AGREE or DISAGREE with the plan’s action.
- Output format: end with exactly: FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** X%

"""
        if memory is not None:
            system_message += f"Learn from past decisions and apply lessons from similar situations: {past_memory_str}"

        messages = [
            {
                "role": "system",
                "content": system_message
            },
            context,
        ]

        result = llm.invoke(messages)

        # # with fundamental, news reports, the result is the final transaction proposal
        # context["content"] = context["content"] + "\n\n" + f"""
        # # Resources available:
        # Fundamental Analyst: {fundamentals_report}
        # News Analyst: {news_report}
        # Market Analyst: {market_report}
        # Social Media Analyst: {sentiment_report}
        # """

        # messages = [
        #     {
        #         "role": "system",
        #         "content": system_message
        #     },
        #     context,
        # ]

        # result_with_reports = llm.invoke(messages)


        return {
            "sender": name,
            "messages": [result],
            "trader_investment_plan": result.content,
            "final_trade_decision": result.content,
            # "trader_investment_plan_with_data_reports": result_with_reports.content,
            # "final_trade_decision_with_data_reports": result_with_reports.content,
        }

    return functools.partial(trader_node, name="Trader")
