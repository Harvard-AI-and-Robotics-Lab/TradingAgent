from typing import Annotated, Sequence, Dict, List, Any
from datetime import date, timedelta, datetime
from typing_extensions import TypedDict, Optional
from langchain_openai import ChatOpenAI
from tradingagents.agents import *
from langgraph.prebuilt import ToolNode
from langgraph.graph import END, StateGraph, START, MessagesState


# Analyst team state
class FundamentalsAnalystState(TypedDict):
    fundamentals_reports_str: Annotated[str, "Report from the Fundamentals Analyst"]
    fundamentals_credibility_scorer: Annotated[str, "Credibility scorer report from the Fundamentals Analyst"]
    count: Annotated[int, "Length of the current report"]

class MarketAnalystState(TypedDict):
    market_reports_str: Annotated[str, "Reports from the Market Analyst"]
    market_credibility_scorer: Annotated[str, "Credibility scorer report from the Market Analyst"]
    count: Annotated[int, "Length of the current report"]

class NewsAnalystState(TypedDict):
    news_reports_str: Annotated[str, "Report from the News Researcher of world affairs"]
    news_credibility_scorer: Annotated[str, "Credibility scorer report from the News Researcher"]
    count: Annotated[int, "Length of the current report"]

class SentimentAnalystState(TypedDict):
    sentiment_reports_str: Annotated[str, "Reports from the Social Media Analyst"]
    sentiment_credibility_scorer: Annotated[str, "Credibility scorer report from the Social Media Analyst"]
    count: Annotated[int, "Length of the current report"]

class PriceAnalystState(TypedDict):
    price_reports_str: Annotated[str, "Report from the Price Analyst"]
    price_credibility_scorer: Annotated[str, "Credibility scorer report from the Price Analyst"]
    count: Annotated[int, "Length of the current report"]

# Researcher team state
class InvestDebateState(TypedDict):
    bull_history: Annotated[
        str, "Bullish Conversation history"
    ]  # Bullish Conversation history
    bear_history: Annotated[
        str, "Bearish Conversation history"
    ]  # Bullish Conversation history
    history: Annotated[str, "Conversation history"]  # Conversation history
    current_response: Annotated[str, "Latest response"]  # Last response
    judge_decision: Annotated[str, "Final judge decision"]  # Last response
    count: Annotated[int, "Length of the current conversation"]  # Conversation length


# Risk management team state
class RiskDebateState(TypedDict):
    risky_history: Annotated[
        str, "Risky Agent's Conversation history"
    ]  # Conversation history
    safe_history: Annotated[
        str, "Safe Agent's Conversation history"
    ]  # Conversation history
    neutral_history: Annotated[
        str, "Neutral Agent's Conversation history"
    ]  # Conversation history
    history: Annotated[str, "Conversation history"]  # Conversation history
    latest_speaker: Annotated[str, "Analyst that spoke last"]
    current_risky_response: Annotated[
        str, "Latest response by the risky analyst"
    ]  # Last response
    current_safe_response: Annotated[
        str, "Latest response by the safe analyst"
    ]  # Last response
    current_neutral_response: Annotated[
        str, "Latest response by the neutral analyst"
    ]  # Last response
    judge_decision: Annotated[str, "Judge's decision"]
    count: Annotated[int, "Length of the current conversation"]  # Conversation length


# Portfolio holdings data structure
class PortfolioHolding(TypedDict):
    symbol: Annotated[str, "Stock ticker symbol"]
    quantity: Annotated[float, "Number of shares held"]
    avg_cost: Annotated[float, "Average cost basis per share"]
    current_price: Annotated[float, "Current market price per share"]
    market_value: Annotated[float, "Current market value of holding"]
    unrealized_pnl: Annotated[float, "Unrealized profit/loss"]
    weight: Annotated[float, "Portfolio weight as percentage"]
    sector: Annotated[str, "Sector classification"]
    acquired_date: Annotated[str, "Date when position was acquired"]


class PortfolioSummary(TypedDict):
    total_value: Annotated[float, "Total portfolio value"]
    cash_balance: Annotated[float, "Available cash balance"]
    total_unrealized_pnl: Annotated[float, "Total unrealized P&L"]
    total_invested: Annotated[float, "Total amount invested"]
    portfolio_return: Annotated[float, "Overall portfolio return percentage"]
    holdings: Annotated[List[PortfolioHolding], "List of individual holdings"]
    sector_allocation: Annotated[Dict[str, float], "Sector allocation percentages"]
    top_holdings: Annotated[List[str], "Top 5 holdings by weight"]
    risk_metrics: Annotated[Dict[str, float], "Risk metrics like beta, volatility"]


class AgentState(MessagesState):
    company_of_interest: Annotated[str, "Company that we are interested in trading"]
    trade_date: Annotated[str, "What date we are trading at"]

    # Portfolio context
    portfolio_summary: Annotated[Optional[PortfolioSummary], "Current portfolio holdings and metrics"]

    sender: Annotated[str, "Agent that sent this message"]

    # research step
    fundamentals_report: Annotated[str, "Report from the Fundamentals Researcher"]
    market_report: Annotated[str, "Report from the Market Analyst"]
    sentiment_report: Annotated[str, "Report from the Social Media Analyst"]
    news_report: Annotated[
        str, "Report from the News Researcher of world affairs"
    ]
    price_report: Annotated[str, "Report from the Price Analyst"]

    fundamentals_analyst_state: Annotated[FundamentalsAnalystState, "State of the Fundamentals Researcher"]
    market_analyst_state: Annotated[MarketAnalystState, "State of the Market Analyst"]
    sentiment_analyst_state: Annotated[SentimentAnalystState, "State of the Social Media Analyst"]
    news_analyst_state: Annotated[NewsAnalystState, "State of the News Researcher"]
    price_analyst_state: Annotated[PriceAnalystState, "State of the Price Analyst"]

    # researcher team discussion step
    investment_debate_state: Annotated[
        InvestDebateState, "Current state of the debate on if to invest or not"
    ]
    investment_plan: Annotated[str, "Plan generated by the Analyst"]

    trader_investment_plan: Annotated[str, "Plan generated by the Trader"]

    # risk management team discussion step
    risk_debate_state: Annotated[
        RiskDebateState, "Current state of the debate on evaluating risk"
    ]
    final_trade_decision: Annotated[str, "Final decision made by the Risk Analysts"]
    final_trade_decision_processed: Annotated[dict, "Final decision made by the Risk Analysts processed"]