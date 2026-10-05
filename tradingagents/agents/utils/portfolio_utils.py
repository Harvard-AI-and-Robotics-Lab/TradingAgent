import re
import os
import json
import random
import yfinance as yf
from datetime import datetime, timedelta
from openai import OpenAI

from typing import Optional, Dict, List
from .agent_states import PortfolioSummary, PortfolioHolding


def format_portfolio_context(portfolio_summary: Optional[PortfolioSummary], company_symbol: str) -> str:
    """Format portfolio holdings information for inclusion in agent prompts.

    Args:
        portfolio_summary: Current portfolio holdings and metrics
        company_symbol: Symbol of the company being analyzed

    Returns:
        Formatted string containing relevant portfolio context
    """
    if not portfolio_summary:
        return """Portfolio Context: No current portfolio information available. This analysis will be conducted without existing position context."""

    portfolio_context = f"""
=== CURRENT PORTFOLIO CONTEXT ===

Portfolio Overview:
• Total Value: ${portfolio_summary['total_value']:,.2f}
• Available Cash: ${portfolio_summary['cash_balance']:,.2f}
• Total Unrealized P&L: ${portfolio_summary['total_unrealized_pnl']:,.2f} ({portfolio_summary['portfolio_return']:.2f}%)
• Total Invested: ${portfolio_summary['total_invested']:,.2f}

"""

    # Check if we already hold the target company
    current_holding = None
    for holding in portfolio_summary['holdings']:
        if holding['symbol'].upper() == company_symbol.upper():
            current_holding = holding
            break

    if current_holding:
        portfolio_context += f"""
EXISTING POSITION IN {company_symbol.upper()}:
• Shares Owned: {current_holding['quantity']:,.0f}
• Average Cost: ${current_holding['avg_cost']:.2f}
• Current Price: ${current_holding['current_price']:.2f}
• Market Value: ${current_holding['market_value']:,.2f}
• Unrealized P&L: ${current_holding['unrealized_pnl']:,.2f}
• Portfolio Weight: {current_holding['weight']:.2f}%
• Acquired Date: {current_holding['acquired_date']}
• Position Status: {'PROFITABLE' if current_holding['unrealized_pnl'] > 0 else 'LOSS' if current_holding['unrealized_pnl'] < 0 else 'BREAK-EVEN'}

"""
    else:
        portfolio_context += f"""
POSITION STATUS: No current position in {company_symbol.upper()}
• Available cash for new positions: ${portfolio_summary['cash_balance']:,.2f}

"""

    # Top holdings for context
    if portfolio_summary['top_holdings']:
        portfolio_context += f"""
Current Top Holdings: {', '.join(portfolio_summary['top_holdings'])}
"""

    # Sector allocation
    if portfolio_summary['sector_allocation']:
        top_sectors = sorted(portfolio_summary['sector_allocation'].items(),
                           key=lambda x: x[1], reverse=True)[:3]
        portfolio_context += f"""
Top Sector Allocations: {', '.join([f"{sector} ({weight:.1f}%)" for sector, weight in top_sectors])}
"""

    # Risk metrics
    if portfolio_summary.get('risk_metrics'):
        risk_info = []
        for metric, value in portfolio_summary['risk_metrics'].items():
            risk_info.append(f"{metric}: {value:.2f}")
        if risk_info:
            portfolio_context += f"""
Portfolio Risk Metrics: {', '.join(risk_info)}
"""

    return portfolio_context


def get_portfolio_analysis_questions(portfolio_summary: Optional[PortfolioSummary],
                                   company_symbol: str,
                                   proposed_action: str = "trade") -> List[str]:
    """Generate portfolio-specific analysis questions for agents.

    Args:
        portfolio_summary: Current portfolio holdings
        company_symbol: Target company symbol
        proposed_action: Type of action being considered (buy/sell/hold)

    Returns:
        List of portfolio-relevant questions to guide analysis
    """
    questions = [
        "How does this trade align with current portfolio objectives?",
        "What are the portfolio concentration risks of this position?",
        "How would this trade affect overall portfolio diversification?"
    ]

    if not portfolio_summary:
        questions.extend([
            "What position size would be appropriate for initial portfolio allocation?",
            "How should this position fit into a broader portfolio strategy?"
        ])
        return questions

    # Check for existing position
    has_position = any(h['symbol'].upper() == company_symbol.upper()
                      for h in portfolio_summary['holdings'])

    if has_position:
        questions.extend([
            "Should we add to, reduce, or maintain the current position size?",
            "How has the existing position performed relative to portfolio?",
            "What would be the tax implications of changing this position?",
            "Does the current position size align with conviction level?"
        ])
    else:
        questions.extend([
            f"What would be an appropriate position size given ${portfolio_summary['cash_balance']:,.0f} available cash?",
            "How would this new position complement existing holdings?",
            "Does adding this position improve or worsen portfolio balance?"
        ])

    # Cash management questions
    if portfolio_summary['cash_balance'] < portfolio_summary['total_value'] * 0.05:  # Less than 5% cash
        questions.append("Given limited cash availability, should other positions be trimmed?")

    # Risk management questions
    if portfolio_summary.get('portfolio_return', 0) < -10:  # Portfolio down >10%
        questions.extend([
            "Given current portfolio drawdown, is this the right time for new risk?",
            "Should focus be on risk reduction rather than new positions?"
        ])

    return questions


def calculate_position_impact(portfolio_summary: Optional[PortfolioSummary],
                            company_symbol: str,
                            shares: float,
                            price: float) -> Dict[str, float]:
    """Calculate the impact of a proposed trade on portfolio metrics.

    Args:
        portfolio_summary: Current portfolio state
        company_symbol: Target company symbol
        shares: Number of shares (positive for buy, negative for sell)
        price: Price per share

    Returns:
        Dictionary with impact metrics
    """
    if not portfolio_summary:
        trade_value = abs(shares * price)
        return {
            "trade_value": trade_value,
            "new_portfolio_weight": 100.0 if shares > 0 else 0.0,
            "cash_required": trade_value if shares > 0 else -trade_value,
            "new_cash_balance": -trade_value if shares > 0 else trade_value
        }

    trade_value = abs(shares * price)
    current_total = portfolio_summary['total_value']
    current_cash = portfolio_summary['cash_balance']

    # Find existing position
    existing_position = None
    for holding in portfolio_summary['holdings']:
        if holding['symbol'].upper() == company_symbol.upper():
            existing_position = holding
            break

    # Calculate new values
    if shares > 0:  # Buying
        cash_required = trade_value
        new_cash_balance = current_cash - cash_required
        if existing_position:
            new_position_value = existing_position['market_value'] + trade_value
        else:
            new_position_value = trade_value
    else:  # Selling
        cash_generated = trade_value
        new_cash_balance = current_cash + cash_generated
        if existing_position:
            new_position_value = max(0, existing_position['market_value'] - trade_value)
        else:
            new_position_value = 0

    new_total_value = current_total + (cash_generated if shares < 0 else 0)
    new_portfolio_weight = (new_position_value / new_total_value) * 100 if new_total_value > 0 else 0

    return {
        "trade_value": trade_value,
        "new_portfolio_weight": new_portfolio_weight,
        "cash_required": cash_required if shares > 0 else -cash_generated,
        "new_cash_balance": new_cash_balance,
        "new_total_value": new_total_value,
        "position_change": "BUY" if shares > 0 else "SELL",
        "sufficient_cash": new_cash_balance >= 0
    }

def create_sample_portfolio(company_name, trade_date, days_back=0, web_search=False, initial_cash=10000.0):
    """Create a realistic sample portfolio using web search to get real market data.

    Portfolio scenario: Invested $10,000 total in the given ticker on a random date
    within 30 days before the trade_date, using all available cash.

    Args:
        company_name: Stock ticker symbol (e.g., 'AAPL')
        trade_date: Current trade date (e.g., '2024-01-25')

    Returns:
        Dictionary with realistic portfolio data based on actual market prices
    """

    if days_back <= 0:
        return {
            "total_value": initial_cash,
            "cash_balance": initial_cash,
            "total_unrealized_pnl": 0,
            "total_invested": 0,
            "portfolio_return": 0,
            "holdings": [],
            "sector_allocation": {},
            "top_holdings": [],
            "risk_metrics": {}
        }

    try:
        # Parse trade date
        if isinstance(trade_date, str):
            current_date = datetime.strptime(trade_date, "%Y-%m-%d")
        else:
            current_date = trade_date

        # Generate random purchase date (1-30 days before trade_date)
        days_back = random.randint(0, 30) if days_back < 0 else days_back
        purchase_date = current_date - timedelta(days=days_back)
        purchase_date_str = purchase_date.strftime("%Y-%m-%d")

        if not web_search:
            print(f"Getting price from data cache for {company_name.upper()} on {purchase_date_str} and {trade_date}")

            # Create ticker object
            ticker = yf.Ticker(company_name.upper())

            # Fetch historical data for the specified date range
            data = []
            while len(data) == 0:
                data = ticker.history(start=purchase_date_str, end=trade_date)
                if data.empty:
                    days_back = days_back + 1
                    purchase_date = current_date - timedelta(days=days_back)
                    purchase_date_str = purchase_date.strftime("%Y-%m-%d")

            # Check if data is empty
            if data.empty:
                return (
                    f"Warning: No data found for symbol '{company_name.upper()}' between {start_date} and {end_date}"
                )

            # Remove timezone info from index for cleaner output
            if data.index.tz is not None:
                data.index = data.index.tz_localize(None)

            # Round numerical values to 2 decimal places for cleaner display
            numeric_columns = ["Open", "High", "Low", "Close", "Adj Close"]
            for col in numeric_columns:
                if col in data.columns:
                    data[col] = data[col].round(2)

            #extract the purchase price and the trade date price
            purchase_price = data['Close'].iloc[0]
            current_price = data['Close'].iloc[-1]
            company_sector = ticker.info['sector']

        else:
            # Initialize OpenAI client for web search
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY is required when portfolio price lookup uses web search"
                )
            client = OpenAI(
                api_key=api_key,
                base_url=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
            )

            # Use web search to get historical and current stock price,
            search_query = f"Find the closing stock price of {company_name} on the purchase date {purchase_date_str} and the trade date {trade_date}. Normally, the two prices are different."
            print(f"Search query: {search_query}")

            response = client.responses.create(
                model="gpt-4o-mini",
                input=search_query + 'Return JSON format: {"purchase_price": xxx, "trade_date_price": xxx, "company_sector": xxx}',
                tools=[
                    {
                    "type": "web_search_preview",
                    "search_context_size": "low",
                    }
                ],
                store=True,
            )

            # Parse the response
            response_content = response.output_text
            print(f"Web search response for {company_name} on {purchase_date_str} and {trade_date}: {response_content[:500]}...")

            # Try to extract JSON from response
            try:
                # Look for JSON pattern in the response
                json_match = re.search(r'\{[^}]*"purchase_price"[^}]*\}', response_content, re.DOTALL)

                if json_match:
                    json_str = json_match.group()
                    price_data = json.loads(json_str)
                else:
                    # Try to extract prices with regex if JSON parsing fails
                    purchase_match = re.search(r'purchase_price["\s:]+([0-9.]+)', response_content, re.IGNORECASE)
                    current_match = re.search(r'trade_date_price["\s:]+([0-9.]+)', response_content, re.IGNORECASE)
                    sector_match = re.search(r'sector["\s:]+([A-Za-z\s]+)', response_content, re.IGNORECASE)

                    price_data = {
                        "purchase_price": float(purchase_match.group(1)) if purchase_match else None,
                        "trade_date_price": float(current_match.group(1)) if current_match else None,
                        "company_sector": sector_match.group(1).strip() if sector_match else "Technology"
                    }

            except (json.JSONDecodeError, ValueError, AttributeError) as e:
                print(f"Error parsing web search results for {company_name}: {e}")
                price_data = {"purchase_price": None, "current_price": None, "company_sector": "Technology"}

            purchase_price = price_data.get("purchase_price")
            current_price = price_data.get("trade_date_price")
            company_sector = price_data.get("company_sector", "Technology")
            # Fallback to realistic estimated prices if web search didn't work
            assert purchase_price is not None, "Purchase price is None"
            assert current_price is not None, "Current price is None"

        print(f"Purchase price: {purchase_price}, Current price: {current_price}, Company sector: {company_sector}")

        # Portfolio parameters
        shares_purchased = int(initial_cash / purchase_price)
        actual_investment = shares_purchased * purchase_price
        actual_remaining_cash = initial_cash - actual_investment

        # Calculate current values
        current_market_value = shares_purchased * current_price
        unrealized_pnl = current_market_value - actual_investment
        total_portfolio_value = current_market_value + actual_remaining_cash
        portfolio_return = (unrealized_pnl / actual_investment) * 100 if actual_investment > 0 else 0
        position_weight = (current_market_value / total_portfolio_value) * 100 if total_portfolio_value > 0 else 0
        cash_weight = (actual_remaining_cash / total_portfolio_value) * 100 if total_portfolio_value > 0 else 0

        # Create realistic portfolio
        portfolio = {
            "total_value": float(round(total_portfolio_value, 2)),
            "cash_balance": float(round(actual_remaining_cash, 2)),
            "total_unrealized_pnl": float(round(unrealized_pnl, 2)),
            "total_invested": float(round(actual_investment, 2)),
            "portfolio_return": float(round(portfolio_return, 2)),
            "holdings": [
                {
                    "symbol": company_name.upper(),
                    "quantity": float(shares_purchased),
                    "avg_cost": float(round(purchase_price, 2)),
                    "current_price": round(current_price, 2),
                    "market_value": float(round(current_market_value, 2)),
                    "unrealized_pnl": float(round(unrealized_pnl, 2)),
                    "weight": float(round(position_weight, 1)),
                    "sector": company_sector,
                    "acquired_date": purchase_date_str
                }
            ],
            "sector_allocation": {
                company_sector: float(round(position_weight, 1)),
                "Cash": float(round(cash_weight, 1))
            },
            "top_holdings": [company_name.upper()],
            "risk_metrics": {}
        }

        print(f"Created portfolio for {company_name}: {shares_purchased} shares @ ${purchase_price:.2f}, now ${current_price:.2f} ({portfolio_return:.1f}% return)")
        return portfolio

    except Exception as e:
        print(f"Blank portfolio created for {company_name}: {str(e)}")

        return {
            "total_value": initial_cash,
            "cash_balance": initial_cash,
            "total_unrealized_pnl": 0,
            "total_invested": 0,
            "portfolio_return": 0,
            "holdings": [],
            "sector_allocation": {},
            "top_holdings": [],
            "risk_metrics": {}
        }

def update_portfolio(portfolio_summary: dict, symbol: str, action: str, percentage: float, current_price: float, commission: float = 0.0) -> dict:
    """Update portfolio based on trading action"""
    cash = portfolio_summary['cash_balance']
    holdings = portfolio_summary['holdings'].copy()

    # Find existing holding
    holding_index = None
    for idx, holding in enumerate(holdings):
        if holding['symbol'].upper() == symbol.upper():
            holding_index = idx
            break

    if action == 'BUY':
        # Buy shares based on percentage of available cash
        available_cash = cash * (percentage / 100.0)
        shares_to_buy = int(available_cash / (current_price * (1 + commission)))

        if shares_to_buy <= 0:
            shares_to_buy = 0.0
        cost = shares_to_buy * current_price * (1 + commission)
        cash -= cost

        if holding_index is not None:
            # Update existing holding
            holding = holdings[holding_index]
            total_shares = holding['quantity'] + shares_to_buy
            total_cost = holding['avg_cost'] * holding['quantity'] + cost
            holdings[holding_index] = {
                'symbol': symbol.upper(),
                'quantity': float(round(total_shares, 2)),
                'avg_cost': float(round(total_cost / total_shares, 2)),
                'current_price': float(round(current_price, 2)),
                'market_value': float(round(total_shares * current_price, 2)),
                'unrealized_pnl': float(round(total_shares * current_price - total_cost, 2)),
                'sector': holding.get('sector', 'Unknown')
            }
        else:
            # Create new holding
            holdings.append({
                'symbol': symbol.upper(),
                'quantity': float(round(shares_to_buy, 2)),
                'avg_cost': float(round(current_price * (1 + commission), 2)),
                'current_price': float(round(current_price, 2)),
                'market_value': float(round(shares_to_buy * current_price, 2)),
                'unrealized_pnl': float(round(shares_to_buy * current_price - cost, 2)),
                'sector': 'Unknown'
            })

    elif action == 'SELL' and holding_index is not None:
        # Sell shares based on percentage of holdings
        holding = holdings[holding_index]
        shares_to_sell = int(holding['quantity'] * (percentage / 100.0))

        if shares_to_sell <= 0:
            shares_to_sell = 0.0
        proceeds = shares_to_sell * current_price * (1 - commission)
        cash += proceeds

        remaining_shares = holding['quantity'] - shares_to_sell
        if remaining_shares <= 0:
            remaining_shares = 0.0
        holdings[holding_index] = {
            'symbol': symbol.upper(),
            'quantity': float(round(remaining_shares, 2)),
            'avg_cost': holding['avg_cost'],
            'current_price': float(round(current_price, 2)),
            'market_value': float(round(remaining_shares * current_price, 2)),
            'unrealized_pnl': float(round(remaining_shares * (current_price - holding['avg_cost']), 2)),
            'sector': holding.get('sector', 'Unknown')
        }
    elif holding_index is not None:
        holding = holdings[holding_index]
        holdings[holding_index] = {
            'symbol': symbol.upper(),
            'quantity': holding['quantity'],
            'avg_cost': holding['avg_cost'],
            'current_price': float(round(current_price, 2)),
            'market_value': float(round(holding['quantity'] * current_price, 2)),
            'unrealized_pnl': float(round(holding['quantity'] * (current_price - holding['avg_cost']), 2)),
            'sector': holding.get('sector', 'Unknown')
        }

    # Recalculate portfolio metrics
    total_market_value = sum(h['market_value'] for h in holdings)
    total_invested = sum(h['avg_cost'] * h['quantity'] for h in holdings)
    total_unrealized_pnl = sum(h['unrealized_pnl'] for h in holdings)
    total_value = cash + total_market_value
    portfolio_return = (total_unrealized_pnl / total_invested * 100) if total_invested > 0 else 0

    # Calculate weights for each holding
    for holding in holdings:
        holding['weight'] = (holding['market_value'] / total_value * 100) if total_value > 0 else 0
        # Add acquired_date if not present (required by portfolio_utils)
        if 'acquired_date' not in holding:
            holding['acquired_date'] = 'N/A'

    return {
        'total_value': float(round(total_value, 2)),
        'cash_balance': float(round(cash, 2)),
        'total_unrealized_pnl': float(round(total_unrealized_pnl, 2)),
        'total_invested': float(round(total_invested, 2)),
        'portfolio_return': float(round(portfolio_return, 2)),
        'holdings': holdings,
        'sector_allocation': {},
        'top_holdings': [holding['symbol'] for holding in holdings[:5]],
        'risk_metrics': {}
    }
