from langchain_core.messages import AIMessage
import time
import json
from ..utils.portfolio_utils import format_portfolio_context
from tradingagents.graph.reflection import get_prompt_for_compressed_situation


def create_bear_researcher(llm, memory):
    def bear_node(state) -> dict:
        trade_date = state["trade_date"]
        company_name = state["company_of_interest"]
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bear_history = investment_debate_state.get("bear_history", "")

        current_response = investment_debate_state.get("current_response", "")
        market_research_report = state.get("market_report", "")
        sentiment_report = state.get("sentiment_report", "")
        news_report = state.get("news_report", "")
        fundamentals_report = state.get("fundamentals_report", "")
        price_report = state.get("price_report", "")

        company_symbol = state["company_of_interest"]
        portfolio_context = format_portfolio_context(state.get("portfolio_summary"), company_symbol)

        if memory is not None:
            if "compressed_situation" in state and len(state["compressed_situation"]) > 0:
                compressed_situation = state["compressed_situation"]
            else:
                curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}\n\n{price_report}"
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
        print(f"Bear Researcher: past_memory_str: {past_memory_str}")

        prompt = f"""You are a **Bear Analyst** evaluating a **single-asset investment decision**. Do NOT consider portfolio diversification, asset allocation, or cross-asset risk mitigation.

Your goal is to present a well-reasoned argument emphasizing risks, challenges, and negative indicators **based on the below resources only**. Leverage the provided research and data to highlight potential downsides and counter bullish arguments effectively.

# Key points to focus on:
- Risks and Challenges: Highlight factors like market saturation, financial instability, or macroeconomic threats that could hinder the stock's performance.
- Competitive Weaknesses: Emphasize vulnerabilities such as weaker market positioning, declining innovation, or threats from competitors.
- Negative Indicators: Use evidence from financial data, market trends, or recent adverse news to support your position.
- Bull Counterpoints: Critically analyze the bull argument with specific data and sound reasoning, exposing weaknesses or over-optimistic assumptions.
- Engagement: Present your argument in a conversational style, directly engaging with the bull analyst's points and debating effectively rather than simply listing facts.

# Resources available:
- **Market research report:** {market_research_report}
- **Social media sentiment report:** {sentiment_report}
- **Latest world affairs news:** {news_report}
- **Company fundamentals report:** {fundamentals_report}
- **Price analysis report:** {price_report}
- **Conversation history of the debate:** {history}
- **Last bull argument:** {current_response}
- **Reflections from similar situations and lessons learned:** {past_memory_str}

# Portfolio context: {portfolio_context}
Decide the action based on the current Portfolio position: if no position, only Buy (open position) or Hold (stay in cash); if holding, choose Buy (increase), Sell (reduce), or Hold (maintain). Partial trading is allowed, but diversification is not considered.

Use this information to deliver a compelling bear argument that highlights not just the company's risks, but also the portfolio-level risks of this investment. Show how this position could undermine portfolio performance and risk management objectives. Refute the bull's claims with both company-specific and portfolio-aware reasoning.

All the information should be generated based on the trade date {trade_date} and the company of interest {company_name}.
DONOT involve collect information from the future, only use information before the trade date {trade_date}.
"""

        response = llm.invoke(prompt)

        argument = f"Bear Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bear_history": bear_history + "\n" + argument,
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bear_node
