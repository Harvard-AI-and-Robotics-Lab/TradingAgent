# TradingAgents/graph/propagation.py

from typing import Dict, Any
from tradingagents.agents.utils.agent_states import (
    FundamentalsAnalystState,
    MarketAnalystState,
    NewsAnalystState,
    SentimentAnalystState,
    PriceAnalystState,
    AgentState,
    InvestDebateState,
    RiskDebateState,
)


class Propagator:
    """Handles state initialization and propagation through the graph."""

    def __init__(self, max_recur_limit=100):
        """Initialize with configuration parameters."""
        self.max_recur_limit = max_recur_limit

    def create_initial_state(
        self, company_name: str, trade_date: str, portfolio_summary=None,
        enable_risk_manager=True, enable_multiple_analysts=False
    ) -> Dict[str, Any]:
        """Create the initial state for the agent graph with optional portfolio context.

        Args:
            company_name: Stock ticker symbol to analyze
            trade_date: Date for the trading analysis
            portfolio_summary: Optional portfolio context for portfolio-aware decisions
        """
        state = {
            "messages": [("human", company_name)],
            "company_of_interest": company_name,
            "trade_date": str(trade_date),
            "portfolio_summary": portfolio_summary,
            "investment_debate_state": InvestDebateState(
                {"history": "", "current_response": "", "count": 0}
            ),
            "market_report": "",
            "fundamentals_report": "",
            "sentiment_report": "",
            "news_report": "",
            "price_report": "",
            "compressed_situation": "",

            # memory bank
            "returns_losses": {},
            "latest_long_short_memory": "",
        }
        if enable_multiple_analysts:
            state["fundamentals_analyst_state"] = FundamentalsAnalystState(
                {
                    "fundamentals_reports_str": "",
                    "fundamentals_credibility_scorer": "",
                    "count": 0,
                }
            )
            state["market_analyst_state"] = MarketAnalystState(
                {
                    "market_reports_str": "",
                    "market_credibility_scorer": "",
                    "count": 0,
                }
            )
            state["sentiment_analyst_state"] = SentimentAnalystState(
                {
                    "sentiment_reports_str": "",
                    "sentiment_credibility_scorer": "",
                    "count": 0,
                }
            )
            state["news_analyst_state"] = NewsAnalystState(
                {
                    "news_reports_str": "",
                    "news_credibility_scorer": "",
                    "count": 0,
                }
            )
            state["price_analyst_state"] = PriceAnalystState(
                {
                    "price_reports_str": "",
                    "count": 0,
                }
            )
        if enable_risk_manager:
            risk_debate_state = RiskDebateState(
                {
                    "history": "",
                    "current_risky_response": "",
                    "current_safe_response": "",
                    "current_neutral_response": "",
                    "count": 0,
                }
            )
            state["risk_debate_state"] = risk_debate_state

        return state

    def get_graph_args(self) -> Dict[str, Any]:
        """Get arguments for the graph invocation."""
        return {
            "stream_mode": "values",
            "config": {"recursion_limit": self.max_recur_limit},
        }