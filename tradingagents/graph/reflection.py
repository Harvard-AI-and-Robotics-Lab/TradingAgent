# TradingAgents/graph/reflection.py

import re
import json
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI


def get_prompt_for_compressed_situation(situation_string: str) -> str:
    system_prompt = f"""You are an expert financial analyst. Your task is to extract and synthesize the most important information from a provided situation string, which describes the current market environment.

The situation string may include information on company fundamentals, market/price data, news events, and investor sentiment. Your goal is to identify the most salient and decision-relevant points across these categories and produce a concise, coherent summary of the current market situation.

# Output requirements:
- Return a single structured narrative describing the market situation.
- Cover fundamentals, market/price conditions, news, and sentiment where available.
- Prioritize high-confidence, material information; omit redundancy and minor details.
- Do not introduce new information or speculation.
- The output must not exceed 10,000 tokens.

"""
    return [("system", system_prompt), ("human", f"Input situation string: {situation_string}")]

class Reflector:
    """Handles reflection on decisions and updating memory."""

    def __init__(self, quick_thinking_llm: ChatOpenAI):
        """Initialize the reflector with an LLM."""
        self.quick_thinking_llm = quick_thinking_llm
        self.reflection_system_prompt = self._get_reflection_prompt()

    def _get_reflection_prompt(self) -> str:
        """Get the system prompt for reflection."""
        schema_hint = {
            "Reasoning": "string",
            "Improvement": "string",
            "Summary": "string",
        }
        return """
You are an expert financial analyst tasked with reviewing trading decisions/analysis and providing a comprehensive, step-by-step analysis.
Your goal is to deliver detailed insights into investment decisions and highlight opportunities for improvement, adhering strictly to the following guidelines:

1. Reasoning: (No more than 1000 tokens)
   - For each trading decision, determine whether it was correct or incorrect. A correct decision results in an increase in returns, while an incorrect decision does the opposite.
   - Analyze the contributing factors to each success or mistake. Consider:
     - Fundamental data analysis.
     - Market data analysis.
     - News analysis.
     - Social media and sentiment analysis.
     - Technical indicators/signals.
     - Price movement analysis.
     - Weight the importance of each factor in the decision-making process.

2. Improvement: (No more than 1000 tokens)
   - Returns represent the realized cumulative return at different future holding horizons, where positive values indicate profitable decisions and negative values indicate losses. Sharpe ratios are computed for longer horizons (≥7 days) to assess risk-adjusted performance and ensure statistical stability.
   - For any incorrect decisions, propose revisions to maximize returns.
   - Provide a detailed list of corrective actions or improvements, including specific recommendations (e.g., changing a decision from HOLD to BUY on a particular date or from BUY to SELL on a particular date).

3. Summary: (No more than 500 tokens)
   - Summarize the lessons learned from the successes and mistakes.
   - Highlight how these lessons can be adapted for future trading scenarios and draw connections between similar situations to apply the knowledge gained.
   - DONOT include any specific/raw info in the returns/sharpe ratios (beacuse it is the future information, not the current information) to avoid data leakage.

Adhere strictly to these instructions, and ensure your output is detailed, accurate, and actionable. You will also be given objective descriptions of the market from a price movements, technical indicator, news, and sentiment perspective to provide more context for your analysis.
""" + f"**OUTPUT SCHEMA** (example types):\n{json.dumps(schema_hint, indent=2)}\n\n"


    def _extract_current_situation(self, current_state: Dict[str, Any]) -> str:
        """Extract the current market situation from the state."""
        curr_fundamentals_report = current_state.get("fundamentals_report", "")
        curr_market_report = current_state.get("market_report", "")
        curr_news_report = current_state.get("news_report", "")
        curr_sentiment_report = current_state.get("sentiment_report", "")
        curr_price_report = current_state.get("price_report", "")

        situation = f"Fundamentals: {curr_fundamentals_report}\n\nMarket: {curr_market_report}\n\nNews: {curr_news_report}\n\nSentiment: {curr_sentiment_report}\n\nPrice: {curr_price_report}"
        situation = self.quick_thinking_llm.invoke(get_prompt_for_compressed_situation(situation)).content
        return situation

    def _reflect_on_component(
        self, component_type: str, report: str, situation: str, return_losses
    ) -> str:
        """Generate reflection for a component."""
        messages = [
            ("system", self.reflection_system_prompt),
            (
                "human",
                f"Objective Fundamentals/Market/News/Sentiment/Price Reports for Reference: {situation}\n\nAnalysis/Decision: {report}\n\nReturns: {return_losses}",
            ),
        ]

        result = self.quick_thinking_llm.invoke(messages).content
        print(f"Component: {component_type} Reflection result: {result}")
        return self.extract_json_block(result)

    def extract_json_block(self, text: str) -> str:
        """
        Extract JSON from a response that might include code fences.
        """
        text = text.strip()
        # Remove ```json fences if present
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            print(f"Error parsing JSON: {text}")
            return text
        return json.loads(text)

    def reflect_bull_researcher(self, current_state, return_losses, bull_memory):
        """Reflect on bull researcher's analysis and update memory."""
        situation = self._extract_current_situation(current_state)  # too long
        if "investment_debate_state" in current_state and "bull_reflection" in current_state["investment_debate_state"]:
            result = current_state["investment_debate_state"]["bull_reflection"]
            if isinstance(result, dict) and "Summary" in result:
                recommendation = result.get("Summary", "")
            else:
                recommendation = str(result)
            if isinstance(result, dict) and "Improvement" in result:
                improvement = result.get("Improvement", "")
            else:
                improvement = str(result)
        else:
            bull_debate_history = current_state["investment_debate_state"]["bull_history"]

            result = self._reflect_on_component(
                "BULL", bull_debate_history, situation, return_losses
            )
            current_state["investment_debate_state"]["bull_reflection"] = result
            if isinstance(result, dict) and "Summary" in result:
                recommendation = result.get("Summary", "")
            else:
                recommendation = str(result)
            if isinstance(result, dict) and "Improvement" in result:
                improvement = result.get("Improvement", "")
            else:
                improvement = str(result)

        bull_memory.add_situations([(situation, recommendation, improvement)], trade_date=current_state.get("trade_date"))

    def reflect_bear_researcher(self, current_state, return_losses, bear_memory):
        """Reflect on bear researcher's analysis and update memory."""
        situation = self._extract_current_situation(current_state)
        if "investment_debate_state" in current_state and "bear_reflection" in current_state["investment_debate_state"]:
            result = current_state["investment_debate_state"]["bear_reflection"]
            if isinstance(result, dict) and "Summary" in result:
                recommendation = result.get("Summary", "")
            else:
                recommendation = str(result)
            if isinstance(result, dict) and "Improvement" in result:
                improvement = result.get("Improvement", "")
            else:
                improvement = str(result)
        else:
            bear_debate_history = current_state["investment_debate_state"]["bear_history"]

            result = self._reflect_on_component(
                "BEAR", bear_debate_history, situation, return_losses
            )
            current_state["investment_debate_state"]["bear_reflection"] = result
            if isinstance(result, dict) and "Summary" in result:
                recommendation = result.get("Summary", "")
            else:
                recommendation = str(result)
            if isinstance(result, dict) and "Improvement" in result:
                improvement = result.get("Improvement", "")
            else:
                improvement = str(result)

        bear_memory.add_situations([(situation, recommendation, improvement)], trade_date=current_state.get("trade_date"))

    def reflect_trader(self, current_state, return_losses, trader_memory):
        """Reflect on trader's decision and update memory."""
        situation = self._extract_current_situation(current_state)
        if "trader_reflection" in current_state:
            result = current_state["trader_reflection"]
            if isinstance(result, dict) and "Summary" in result:
                recommendation = result.get("Summary", "")
            else:
                recommendation = str(result)
            if isinstance(result, dict) and "Improvement" in result:
                improvement = result.get("Improvement", "")
            else:
                improvement = str(result)
        else:
            if "trader_investment_plan" in current_state:
                trader_decision = current_state["trader_investment_plan"]
            elif "trader_investment_decision" in current_state:
                trader_decision = current_state["trader_investment_decision"]
            else:
                raise ValueError("No trader decision found in current state")

            result = self._reflect_on_component(
                "TRADER", trader_decision, situation, return_losses
            )
            current_state["trader_reflection"] = result
            if isinstance(result, dict) and "Summary" in result:
                recommendation = result.get("Summary", "")
            else:
                recommendation = str(result)
            if isinstance(result, dict) and "Improvement" in result:
                improvement = result.get("Improvement", "")
            else:
                improvement = str(result)

        trader_memory.add_situations([(situation, recommendation, improvement)], trade_date=current_state.get("trade_date"))

    def reflect_invest_judge(self, current_state, return_losses, invest_judge_memory):
        """Reflect on investment judge's decision and update memory."""
        situation = self._extract_current_situation(current_state)
        if "investment_debate_state" in current_state and "judge_reflection" in current_state["investment_debate_state"]:
            result = current_state["investment_debate_state"]["judge_reflection"]
            if isinstance(result, dict) and "Summary" in result:
                recommendation = result.get("Summary", "")
            else:
                recommendation = str(result)
            if isinstance(result, dict) and "Improvement" in result:
                improvement = result.get("Improvement", "")
            else:
                improvement = str(result)
        else:
            judge_decision = current_state["investment_debate_state"]["judge_decision"]

            result = self._reflect_on_component(
                "INVEST JUDGE", judge_decision, situation, return_losses
            )
            current_state["investment_debate_state"]["judge_reflection"] = result
            if isinstance(result, dict) and "Summary" in result:
                recommendation = result.get("Summary", "")
            else:
                recommendation = str(result)
            if isinstance(result, dict) and "Improvement" in result:
                improvement = result.get("Improvement", "")
            else:
                improvement = str(result)

        invest_judge_memory.add_situations([(situation, recommendation, improvement)], trade_date=current_state.get("trade_date"))

    def reflect_risk_manager(self, current_state, return_losses, risk_manager_memory):
        """Reflect on risk manager's decision and update memory."""
        situation = self._extract_current_situation(current_state)

        if "risk_debate_state" in current_state and "judge_reflection" in current_state["risk_debate_state"]:
            result = current_state["risk_debate_state"]["judge_reflection"]
            if isinstance(result, dict) and "Summary" in result:
                recommendation = result.get("Summary", "")
            else:
                recommendation = str(result)
            if isinstance(result, dict) and "Improvement" in result:
                improvement = result.get("Improvement", "")
            else:
                improvement = str(result)
        else:
            judge_decision = current_state["risk_debate_state"]["judge_decision"]

            result = self._reflect_on_component(
                "RISK JUDGE", judge_decision, situation, return_losses
            )
            current_state["risk_debate_state"]["judge_reflection"] = result
            if isinstance(result, dict) and "Summary" in result:
                recommendation = result.get("Summary", "")
            else:
                improvement = str(result)

        risk_manager_memory.add_situations([(situation, recommendation, improvement)], trade_date=current_state.get("trade_date"))