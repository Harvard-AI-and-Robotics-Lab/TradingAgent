import time
import json
from ..utils.portfolio_utils import format_portfolio_context


def create_neutral_debator(llm):
    def neutral_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        neutral_history = risk_debate_state.get("neutral_history", "")

        current_risky_response = risk_debate_state.get("current_risky_response", "")
        current_safe_response = risk_debate_state.get("current_safe_response", "")

        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        trader_decision = state["trader_investment_plan"]
        company_symbol = state["company_of_interest"]
        portfolio_context = format_portfolio_context(state.get("portfolio_summary"), company_symbol)

        prompt = f"""As the Neutral Risk Analyst, only for single-asset trading, your role is to provide a balanced perspective, weighing both the potential benefits and risks of the trader's decision or plan. You prioritize a well-rounded approach, evaluating the upsides and downsides while factoring in broader market trends, potential economic shifts, and diversification strategies.

portfolio_context: {portfolio_context} As the Balanced Risk Analyst, weigh risk and return trade-offs, recommend moderate sizing and prudent cash use, and assess diversification, timing, and correlations to keep the portfolio aligned with its risk budget.

Here is the trader's decision: {trader_decision} Your task is to challenge both the Risky and Safe Analysts, pointing out where each perspective may be overly optimistic or overly cautious. Use insights from the following data sources to support a moderate, sustainable strategy to adjust the trader's decision:

Market Research Report: {market_research_report}
Social Media Sentiment Report: {sentiment_report}
Latest World Affairs Report: {news_report}
Company Fundamentals Report: {fundamentals_report}

Current conversation history: {history}
Last response from risky analyst: {current_risky_response}
Last response from safe analyst: {current_safe_response}

If there are no responses from other viewpoints, do not hallucinate and just present your point.

Engage actively by analyzing both sides critically, addressing weaknesses in the risky and conservative arguments to advocate for a more balanced portfolio approach. Use the portfolio context to show why moderate positioning often leads to better long-term portfolio outcomes than extreme approaches. Challenge each of their points to illustrate why a moderate risk strategy might offer the best balance of growth potential and portfolio protection. Focus on debating rather than simply presenting data, using specific portfolio metrics and context to support your balanced perspective. Output conversationally as if you are speaking without any special formatting."""

        response = llm.invoke(prompt)

        argument = f"Neutral Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "risky_history": risk_debate_state.get("risky_history", ""),
            "safe_history": risk_debate_state.get("safe_history", ""),
            "neutral_history": neutral_history + "\n" + argument,
            "latest_speaker": "Neutral",
            "current_risky_response": risk_debate_state.get(
                "current_risky_response", ""
            ),
            "current_safe_response": risk_debate_state.get("current_safe_response", ""),
            "current_neutral_response": argument,
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return neutral_node
