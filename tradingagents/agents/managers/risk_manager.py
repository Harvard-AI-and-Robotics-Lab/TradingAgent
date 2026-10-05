import time
import json
from ..utils.portfolio_utils import format_portfolio_context, calculate_position_impact


def create_risk_manager(llm, memory):
    def risk_manager_node(state) -> dict:

        company_name = state["company_of_interest"]

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        market_research_report = state["market_report"]
        news_report = state["news_report"]
        fundamentals_report = state["news_report"]
        sentiment_report = state["sentiment_report"]
        trader_plan = state["investment_plan"]

        portfolio_context = format_portfolio_context(state.get("portfolio_summary"), company_name)

        if memory is not None:
            curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
            past_memories = memory.get_memories(curr_situation, trade_date=state.get("trade_date"), n_matches=2)
            past_memory_str = ""
            for i, rec in enumerate(past_memories, 1):
                past_memory_str += rec["recommendation"] + "\n\n"
        else:
            past_memory_str = "No past memories found."

        prompt = f"""As the Risk Management Judge and Debate Facilitator, only for single-asset trading, your goal is to evaluate the debate between three risk analysts—Risky, Neutral, and Safe/Conservative—and determine the best course of action for the trader. Your decision must result in a clear recommendation: Buy, Sell, or Hold. Choose Hold only if strongly justified by specific arguments, not as a fallback when all sides seem valid. Strive for clarity and decisiveness.

Guidelines for Decision-Making:
1. **Summarize Key Arguments**: Extract the strongest points from each analyst, focusing on relevance to both company prospects and portfolio implications.
2. **Portfolio Risk Assessment**: Evaluate how this decision affects overall portfolio risk, diversification, and performance.
3. **Position Sizing**: Consider appropriate position size relative to available capital and existing holdings.
4. **Provide Rationale**: Support your recommendation with direct quotes from the debate, portfolio considerations, and risk management principles.
5. **Refine the Trader's Plan**: Start with the trader's original plan, **{trader_plan}**, and adjust it based on both analyst insights and portfolio context.
6. **Learn from Past Mistakes**: Use lessons from **{past_memory_str}** to address prior misjudgments and improve the decision.

portfolio_context: {portfolio_context}

Deliverables:
- A clear and actionable recommendation: Buy, Sell, or Hold.
- Specific position sizing guidance if recommending Buy.
- Decide the action based on the current Portfolio position: if no position, only Buy (open position) or Hold (stay in cash); if holding, choose Buy (increase), Sell (reduce), or Hold (maintain).

---

**Analysts Debate History:**
{history}

---

Focus on actionable insights that optimize both individual security selection AND portfolio-level risk management. Build on past lessons, critically evaluate all perspectives through a portfolio lens, and ensure each decision advances better portfolio outcomes.
Please provide the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** X% where X is the percentage (0-100) of the position to trade. If no specific percentage is mentioned, default to 100%.
"""

        response = llm.invoke(prompt)

        new_risk_debate_state = {
            "judge_decision": response.content,
            "history": risk_debate_state["history"],
            "risky_history": risk_debate_state["risky_history"],
            "safe_history": risk_debate_state["safe_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "latest_speaker": "Judge",
            "current_risky_response": risk_debate_state["current_risky_response"],
            "current_safe_response": risk_debate_state["current_safe_response"],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": response.content,
        }

    return risk_manager_node
