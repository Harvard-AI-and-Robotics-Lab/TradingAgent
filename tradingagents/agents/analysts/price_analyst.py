"""
Price Analyst Module

This module implements a pure price-based trading analyst that:
1. Aggregates historical price data across multiple time horizons
2. Predicts next-day price movements using technical patterns
3. Generates trading recommendations based on price signals

Design: Uses only historical price data without external information sources.
"""

import os
import re
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import Dict, Tuple, List, Annotated
from copy import deepcopy

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

from ..utils.portfolio_utils import format_portfolio_context
from .tool_utils import get_tool_messages, get_model_name
import tradingagents.dataflows.interface as interface


class PriceAggregator:
    """Aggregates historical price data across multiple time horizons."""

    TIME_HORIZONS = {
        '1_week': 7,
        '1_month': int(30 * 252/356),
        '3_months': int(90 * 252/356),
        '6_months': int(180 * 252/356),
        '1_year': 252 ,  # Trading days
    }

    @staticmethod
    def determine_polyfit_order(data_points: int, horizon_name: str) -> int:
        """
        Dynamically determine polynomial order based on data characteristics.

        Args:
            data_points: Number of data points available
            horizon_name: Time horizon name

        Returns:
            Optimal polynomial order (1-3)
        """
        # Rule 1: Need at least (order + 1) * 2 points for stable fit
        if data_points < 8:
            return 1
        elif data_points < 15:
            return min(2, data_points // 5)
        elif data_points < 60:
            return 2
        else:
            # For longer horizons with more data, allow higher order
            if horizon_name in ['5_years', '1_year']:
                return 3
            elif horizon_name == '3_months':
                return 2
            else:
                return 1

    @staticmethod
    def calculate_horizon_stats(prices_df: pd.DataFrame, horizon_name: str) -> Dict:
        """
        Calculate statistics for a specific time horizon.

        Args:
            prices_df: DataFrame with Date, Open, High, Low, Close, Volume columns
            horizon_name: Name of the horizon (e.g., '1_week', '1_month')

        Returns:
            Dictionary containing price statistics for the horizon
        """
        if prices_df.empty or len(prices_df) < 2:
            return {
                'horizon': horizon_name,
                'data_points': 0,
                'return_pct': 0.0,
                'volatility': 0.0,
                'trend': 'insufficient_data',
                'high': 0.0,
                'low': 0.0,
                'current': 0.0,
                'avg_volume': 0.0
            }

        # Sort by date
        prices_df = prices_df.sort_values('Date')

        # Calculate returns
        start_price = prices_df.iloc[0]['Close']
        end_price = prices_df.iloc[-1]['Close']
        return_pct = ((end_price - start_price) / start_price) * 100 if start_price > 0 else 0.0

        # Calculate volatility (standard deviation of daily returns)
        prices_df['Daily_Return'] = prices_df['Close'].pct_change()
        volatility = prices_df['Daily_Return'].std() * 100  # As percentage

        # Determine trend using polynomial regression
        if len(prices_df) >= 3:
            x = np.arange(len(prices_df))
            y = prices_df['Close'].values

            # Dynamically determine polynomial order
            poly_order = PriceAggregator.determine_polyfit_order(len(prices_df), horizon_name)

            # Fit polynomial
            coefficients = np.polyfit(x, y, poly_order)
            poly_func = np.poly1d(coefficients)

            # Calculate fitted values
            fitted_values = poly_func(x)

            # Determine trend direction from fitted curve
            # Compare start and end of fitted curve
            fitted_start = fitted_values[0]
            fitted_end = fitted_values[-1]
            fitted_return = ((fitted_end - fitted_start) / fitted_start) * 100 if fitted_start > 0 else 0

            # For higher order polynomials, also check recent slope (derivative at end)
            if poly_order >= 2:
                # Calculate derivative at the last point
                derivative = np.polyder(poly_func)
                recent_slope = derivative(len(prices_df) - 1)

                # Normalize slope by price level for comparison
                normalized_slope = (recent_slope / fitted_end) * 100 if fitted_end > 0 else 0

                # Weight both overall fitted return and recent slope
                trend_signal = 0.6 * fitted_return + 0.4 * normalized_slope
            else:
                trend_signal = fitted_return

            # Classify trend based on trend signal
            if trend_signal > 5:
                trend = 'strong_uptrend'
            elif trend_signal > 1:
                trend = 'uptrend'
            elif trend_signal > -1:
                trend = 'sideways'
            elif trend_signal > -5:
                trend = 'downtrend'
            else:
                trend = 'strong_downtrend'
        else:
            trend = 'insufficient_data'

        return {
            'horizon': horizon_name,
            'data_points': len(prices_df),
            'return_pct': round(return_pct, 2),
            'volatility': round(volatility, 2) if not np.isnan(volatility) else 0.0,
            'trend': trend,
            'high': round(prices_df['High'].max(), 2),
            'low': round(prices_df['Low'].min(), 2),
            'current': round(end_price, 2),
            'avg_volume': int(prices_df['Volume'].mean()) if 'Volume' in prices_df.columns else 0
        }

    @staticmethod
    def aggregate_historical_prices(trade_date: str, prices_df: pd.DataFrame) -> Dict:
        """
        Aggregate historical price data across all time horizons.

        Args:
            symbol: Stock ticker symbol
            trade_date: Current trade date in yyyy-mm-dd format

        Returns:
            Dictionary with statistics for each time horizon
        """
        trade_date_obj = datetime.strptime(trade_date, "%Y-%m-%d")

        results = {}
        for horizon_name, days_back in PriceAggregator.TIME_HORIZONS.items():
            start_date = trade_date_obj - timedelta(days=int(days_back * 1.5))  # Buffer for weekends/holidays
            start_date_str = start_date.strftime("%Y-%m-%d")

            # Fetch price data
            try:
                # current date is the trade date
                prices_df_curr = prices_df[prices_df['Date'] >= start_date_str]
                results[horizon_name] = PriceAggregator.calculate_horizon_stats(
                    prices_df_curr, horizon_name
                )

            except Exception as e:
                print(f"Error fetching data for {horizon_name}: {str(e)}")

        return results


class PricePredictor:
    """Predicts next-day price movements based on historical patterns."""

    @staticmethod
    def calculate_momentum_indicators(prices_df: pd.DataFrame) -> Dict:
        """Calculate momentum-based indicators for prediction."""
        if len(prices_df) < 5:
            return {'insufficient_data': True}

        prices_df = prices_df.sort_values('Date')
        closes = prices_df['Close'].values

        # Calculate moving averages
        ma_5 = np.mean(closes[-5:]) if len(closes) >= 5 else closes[-1]
        ma_10 = np.mean(closes[-10:]) if len(closes) >= 10 else closes[-1]
        ma_20 = np.mean(closes[-20:]) if len(closes) >= 20 else closes[-1]

        current_price = closes[-1]

        # Rate of change
        roc_5 = ((current_price - closes[-5]) / closes[-5] * 100) if len(closes) >= 5 else 0
        roc_10 = ((current_price - closes[-10]) / closes[-10] * 100) if len(closes) >= 10 else 0

        # Relative position to moving averages
        above_ma_5 = current_price > ma_5
        above_ma_10 = current_price > ma_10
        above_ma_20 = current_price > ma_20

        # Recent volatility
        recent_returns = np.diff(closes[-10:]) / closes[-10:-1]
        volatility = np.std(recent_returns) if len(recent_returns) > 0 else 0

        return {
            'insufficient_data': False,
            'ma_5': ma_5,
            'ma_10': ma_10,
            'ma_20': ma_20,
            'current_price': current_price,
            'roc_5': roc_5,
            'roc_10': roc_10,
            'above_ma_5': above_ma_5,
            'above_ma_10': above_ma_10,
            'above_ma_20': above_ma_20,
            'volatility': volatility
        }

    @staticmethod
    def predict_next_day_movement(
        prices_df: pd.DataFrame,
        aggregated_stats: Dict
    ) -> Dict:
        """
        Predict next trading day's price movement.

        Args:
            prices_df: DataFrame with Date, Open, High, Low, Close, Volume columns

        Returns:
            Dictionary with direction (up/down) and estimated magnitude (%)
        """
        try:
            # Calculate momentum indicators
            indicators = PricePredictor.calculate_momentum_indicators(prices_df)

            if indicators.get('insufficient_data', False):
                return {
                    'direction': 'uncertain',
                    'magnitude_pct': 0.0,
                    'confidence': 0.0,
                    'reasoning': 'Insufficient data for momentum analysis'
                }

            # Prediction logic based on multiple signals
            bullish_signals = 0
            bearish_signals = 0

            # Signal 1: Moving average alignment
            if indicators['above_ma_5'] and indicators['above_ma_10'] and indicators['above_ma_20']:
                bullish_signals += 2
            elif not indicators['above_ma_5'] and not indicators['above_ma_10'] and not indicators['above_ma_20']:
                bearish_signals += 2

            # Signal 2: Rate of change
            if indicators['roc_5'] > 2:
                bullish_signals += 1
            elif indicators['roc_5'] < -2:
                bearish_signals += 1

            # Signal 3: Multi-timeframe trend alignment
            week_trend = aggregated_stats.get('1_week', {}).get('trend', 'sideways')
            month_trend = aggregated_stats.get('1_month', {}).get('trend', 'sideways')

            if 'uptrend' in week_trend and 'uptrend' in month_trend:
                bullish_signals += 2
            elif 'downtrend' in week_trend and 'downtrend' in month_trend:
                bearish_signals += 2

            # Determine direction and magnitude
            total_signals = bullish_signals + bearish_signals
            if total_signals == 0:
                direction = 'uncertain'
                confidence = 0.0
                magnitude = 0.0
            elif bullish_signals > bearish_signals:
                direction = 'up'
                confidence = bullish_signals / (bullish_signals + bearish_signals)
                # Base magnitude on recent volatility and momentum
                magnitude = min(abs(indicators['roc_5']) * 0.5, indicators['volatility'] * 100 * 2)
                magnitude = max(0.5, min(magnitude, 5.0))  # Cap between 0.5% and 5%
            else:
                direction = 'down'
                confidence = bearish_signals / (bullish_signals + bearish_signals)
                magnitude = min(abs(indicators['roc_5']) * 0.5, indicators['volatility'] * 100 * 2)
                magnitude = max(0.5, min(magnitude, 5.0))

            reasoning = f"Based on {bullish_signals} bullish and {bearish_signals} bearish signals. "
            reasoning += f"\nMA alignment: {'bullish' if indicators['above_ma_20'] else 'bearish'}, "
            reasoning += f"\nROC(5d): {indicators['roc_5']:.2f}%, "
            reasoning += f"\nTrend alignment: {week_trend}/{month_trend}"

            return {
                'direction': direction,
                'magnitude_pct': round(magnitude, 2),
                'confidence': round(confidence, 2),
                'reasoning': reasoning
            }

        except Exception as e:
            print(f"Error in price prediction: {str(e)}")
            return {
                'direction': 'uncertain',
                'magnitude_pct': 0.0,
                'confidence': 0.0,
                'reasoning': f'Error during prediction: {str(e)}'
            }


class TradingRecommender:
    """Generates trading recommendations based on price signals."""

    @staticmethod
    def generate_recommendation(
        aggregated_stats: Dict,
        prediction: Dict,
        portfolio_summary: Dict = None,
        company_name: str = None
    ) -> Dict:
        """
        Generate trading recommendation based on price analysis.

        Args:
            aggregated_stats: Statistics from all time horizons
            prediction: Next-day price movement prediction
            portfolio_summary: Current portfolio state
            company_name: Company of interest

        Returns:
            Dictionary with action (BUY/SELL/HOLD) and position_pct
        """
        # Extract key metrics
        week_return = aggregated_stats.get('1_week', {}).get('return_pct', 0)
        month_return = aggregated_stats.get('1_month', {}).get('return_pct', 0)
        quarter_return = aggregated_stats.get('3_months', {}).get('return_pct', 0)
        year_return = aggregated_stats.get('1_year', {}).get('return_pct', 0)

        week_trend = aggregated_stats.get('1_week', {}).get('trend', 'sideways')
        month_trend = aggregated_stats.get('1_month', {}).get('trend', 'sideways')
        quarter_trend = aggregated_stats.get('3_months', {}).get('trend', 'sideways')

        pred_direction = prediction.get('direction', 'uncertain')
        pred_confidence = prediction.get('confidence', 0.0)
        pred_magnitude = prediction.get('magnitude_pct', 0.0)

        # Calculate overall confidence score (0-100)
        confidence_score = 0

        # Factor 1: Trend consistency across timeframes (0-30 points)
        if 'uptrend' in week_trend and 'uptrend' in month_trend and 'uptrend' in quarter_trend:
            confidence_score += 30
        elif 'downtrend' in week_trend and 'downtrend' in month_trend and 'downtrend' in quarter_trend:
            confidence_score += 30
        elif 'uptrend' in week_trend and 'uptrend' in month_trend:
            confidence_score += 20
        elif 'downtrend' in week_trend and 'downtrend' in month_trend:
            confidence_score += 20
        else:
            confidence_score += 10

        # Factor 2: Prediction confidence (0-30 points)
        confidence_score += int(pred_confidence * 30)

        # Factor 3: Recent performance (0-20 points)
        if abs(week_return) > 5:
            confidence_score += 20
        elif abs(week_return) > 2:
            confidence_score += 15
        else:
            confidence_score += 5

        # Factor 4: Prediction magnitude (0-20 points)
        if pred_magnitude >= 3:
            confidence_score += 20
        elif pred_magnitude >= 1.5:
            confidence_score += 15
        else:
            confidence_score += 5

        confidence_score = min(confidence_score, 100)

        # Determine action and position size
        action = 'HOLD'
        position_pct = 0

        # Check portfolio position
        has_position = False
        if portfolio_summary and company_name:
            ticker = company_name.upper()
            holdings = portfolio_summary.get('holdings', {})
            if ticker in holdings and holdings[ticker].get('shares', 0) > 0:
                has_position = True

        # Decision logic
        if pred_direction == 'up':
            if not has_position:
                # Opening position - scale with confidence
                if confidence_score >= 75:
                    action = 'BUY'
                    position_pct = 100
                elif confidence_score >= 60:
                    action = 'BUY'
                    position_pct = 75
                elif confidence_score >= 45:
                    action = 'BUY'
                    position_pct = 50
                else:
                    action = 'HOLD'
                    position_pct = 0
            else:
                # Already holding - consider adding
                if confidence_score >= 80 and week_return > 3:
                    action = 'BUY'
                    position_pct = 50  # Add to position
                else:
                    action = 'HOLD'
                    position_pct = 0

        elif pred_direction == 'down':
            if has_position:
                # Exit position - scale with confidence
                if confidence_score >= 75:
                    action = 'SELL'
                    position_pct = 100
                elif confidence_score >= 60:
                    action = 'SELL'
                    position_pct = 75
                elif confidence_score >= 45:
                    action = 'SELL'
                    position_pct = 50
                else:
                    action = 'HOLD'
                    position_pct = 0
            else:
                # No position - stay in cash
                action = 'HOLD'
                position_pct = 0

        else:  # uncertain
            action = 'HOLD'
            position_pct = 0

        reasoning = f"Confidence Score: {confidence_score}/100. "
        reasoning += f"Trend Alignment: {week_trend}/{month_trend}/{quarter_trend}. "
        reasoning += f"Prediction: {pred_direction} {pred_magnitude}% (confidence: {pred_confidence:.0%}). "
        reasoning += f"Recent returns: 1W={week_return:.1f}%, 1M={month_return:.1f}%, 3M={quarter_return:.1f}%."

        return {
            'action': action,
            'position_pct': position_pct,
            'pred_direction': pred_direction,
            'confidence_score': confidence_score,
            'reasoning': reasoning
        }


# Define tools for price analysis
def get_historical_price_analysis_and_recommendation(
    symbol: Annotated[str, "Stock ticker symbol"],
    trade_date: Annotated[str, "Current trade date in yyyy-mm-dd format"]
) -> str:
    trade_date_obj = datetime.strptime(trade_date, "%Y-%m-%d")
    max_days_back = max(PriceAggregator.TIME_HORIZONS.values())
    max_start_date = trade_date_obj - timedelta(days=int(max_days_back * 1.5))  # Buffer for weekends/holidays
    max_start_date_str = max_start_date.strftime("%Y-%m-%d")
    prices_df = interface.get_YFin_data_online(symbol, max_start_date_str, trade_date, output_csv_string=False)
    prices_df = prices_df.reset_index()
    prices_df['Date'] = pd.to_datetime(prices_df['Date'])

    """
    Aggregate historical price data across multiple time horizons (1 week, 1 month, 3 months, 1 year, 5 years).
    Returns comprehensive statistics including returns, volatility, trends, and price levels for each horizon.
    """
    price_history_results = PriceAggregator.aggregate_historical_prices(trade_date, prices_df)

    # Format as readable string
    price_history = "# Historical Price Analysis\n\n"
    for horizon, stats in price_history_results.items():
        price_history += f"## {horizon.replace('_', ' ').title()}\n"
        price_history += f"- Data Points: {stats['data_points']}\n"
        price_history += f"- Return: {stats['return_pct']:.2f}%\n"
        price_history += f"- Volatility: {stats['volatility']:.2f}%\n"
        price_history += f"- Trend: {stats['trend']}\n"
        price_history += f"- Price Range: ${stats['low']:.2f} - ${stats['high']:.2f}\n"
        price_history += f"- Current Price: ${stats['current']:.2f}\n"
        price_history += f"- Avg Volume: {stats['avg_volume']:,}\n\n"


    """
    Predict next trading day's price movement (up/down) with estimated magnitude (%).
    Uses momentum indicators and multi-timeframe trend analysis.
    """
    # Generate prediction
    next_day_prediction = PricePredictor.predict_next_day_movement(prices_df, price_history_results)

    price_prediction = "# Next-Day Price Prediction\n\n"
    price_prediction += f"**Direction**: {next_day_prediction['direction'].upper()}\n"
    price_prediction += f"**Estimated Magnitude**: {next_day_prediction['magnitude_pct']:.2f}%\n"
    price_prediction += f"**Confidence**: {next_day_prediction['confidence']:.0%}\n"
    price_prediction += f"**Reasoning**: {next_day_prediction['reasoning']}\n"


    """
    Generate comprehensive trading recommendation (BUY/SELL/HOLD with position %)
    based on aggregated price signals and prediction.
    """
    # Get all analysis components
    recommendation = TradingRecommender.generate_recommendation(
        price_history_results,
        next_day_prediction
    )

    trading_recommendation = "# Price-Based Trading Recommendation\n\n"
    trading_recommendation += f"**Action**: {recommendation['action']}\n"
    trading_recommendation += f"**Position Size**: {recommendation['position_pct']}%\n"
    trading_recommendation += f"**Confidence Score**: {recommendation['confidence_score']}/100\n"
    trading_recommendation += f"**Reasoning**: {recommendation['reasoning']}\n\n"

    # Add summary table
    summary_table = "## Summary Table\n\n"
    summary_table += "| Metric | Value |\n"
    summary_table += "|--------|-------|\n"
    summary_table += f"| Recommendation | **{recommendation['action']} {recommendation['position_pct']}%** |\n"
    summary_table += f"| Confidence | {recommendation['confidence_score']}/100 |\n"
    summary_table += f"| Next-Day Prediction | {next_day_prediction['direction'].upper()} {next_day_prediction['magnitude_pct']:.2f}% |\n"
    summary_table += f"| 1-Week Return | {price_history_results.get('1_week', {}).get('return_pct', 0):.2f}% |\n"
    summary_table += f"| 1-Month Return | {price_history_results.get('1_month', {}).get('return_pct', 0):.2f}% |\n"
    summary_table += f"| 3-Month Return | {price_history_results.get('3_months', {}).get('return_pct', 0):.2f}% |\n"
    summary_table += f"| 6-Month Return | {price_history_results.get('6_months', {}).get('return_pct', 0):.2f}% |\n"
    summary_table += f"| 1-Year Return | {price_history_results.get('1_year', {}).get('return_pct', 0):.2f}% |\n"

    return price_history + price_prediction + trading_recommendation + summary_table


def create_price_analyst(llm, toolkit):
    """
    Create a price analyst node for the trading agent graph.

    This analyst focuses purely on price-based signals without fundamentals or news.
    """
    model_name = get_model_name(llm)
    llm_provider = model_name.split("-")[0] if model_name else "unknown"
    if llm_provider == "gpt":
        llm_provider = "openai"
    elif llm_provider == "gemini":
        llm_provider = "gemini"
    elif llm_provider == "grok":
        llm_provider = "xai"
    elif llm_provider == "claude":
        llm_provider = "anthropic"
    else:
        raise ValueError(f"Unsupported LLM model: {llm}")

    def price_analyst_node(state):
        """Main price analyst node that coordinates the analysis."""
        trade_date = state["trade_date"]
        company_name = state["company_of_interest"]

        portfolio_context = format_portfolio_context(state.get("portfolio_summary"), company_name)

        if "price_analyst_state" in state:
            price_analyst_state = state["price_analyst_state"]
            price_reports_str = price_analyst_state.get("price_reports_str", "")
            count = price_analyst_state.get("count", 0)

        tools = [
            toolkit.get_historical_price_analysis_and_recommendation
        ]

        system_message = f"""You are a price-focused trading analyst. Your role is to analyze pure price action and generate data-driven trading recommendations.

**Your Analysis Workflow:** call `historical_price_analysis_and_recommendation`

**Analysis Guidelines:**
- Focus ONLY on price patterns, trends, momentum, and volatility
- Do NOT incorporate fundamentals, news, or sentiment
- Look for trend consistency across timeframes
- Consider momentum indicators and recent price action
- Evaluate risk through volatility measures
- Be specific with numbers and percentages, not vague descriptions

**Output Requirements:**
- Provide clear, quantitative analysis with specific price levels and percentages
- Include a comprehensive markdown table summarizing key metrics
- Always conclude with: **FINAL TRANSACTION PROPOSAL: BUY/HOLD/SELL X%**

Trade Date: {trade_date}
Company: {company_name}
Portfolio Context: {portfolio_context}

**Portfolio Rules:**
- If no position: only BUY (open) or HOLD (stay in cash)
- If holding: BUY (increase), SELL (reduce), or HOLD (maintain)
- Partial trading is allowed (specify percentage)
"""

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a helpful AI assistant, collaborating with other assistants."
                " Use the provided tools to progress towards answering the question."
                " Execute what you can to make progress."
                " Always conclude with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** X%."
                " You have access to the following tools: {tool_names}.\n{system_message}"
                " Trade date: {trade_date}. Company: {company_name}."
                " Do NOT use data from after {trade_date}."
            ),
            MessagesPlaceholder(variable_name="messages"),
        ])

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(trade_date=trade_date)
        prompt = prompt.partial(company_name=company_name)

        chain = prompt | llm.bind_tools(tools)
        result = chain.invoke(state["messages"])

        report = ""
        if len(result.tool_calls) == 0:
            report = result.content

        if "price_analyst_state" in state:
            if report and isinstance(report, str) and len(report) > 0:
                count = count + 1
                price_reports_str = price_reports_str + f"## Report {count} from {llm_provider}:\n{report}\n\n"

            return {
                "messages": [result],
                "price_report": report,
                "price_analyst_state": {
                    "price_reports_str": price_reports_str,
                    "count": count,
                },
            }
        else:
            return {
                "messages": [result],
                "price_report": report,
            }

    def price_analyst_node_load_data_from_cache(state):
        """Load price analysis from cache if available."""
        trade_date = state["trade_date"]
        company_name = state["company_of_interest"]

        if "price_analyst_state" in state:
            price_analyst_state = state["price_analyst_state"]
            price_reports_str = price_analyst_state.get("price_reports_str", "")
            count = price_analyst_state.get("count", 0)

        if not re.match(r"^\d{4}-\d{2}-\d{2}$", trade_date):
            raise ValueError(f"Invalid date format: {trade_date}")

        cache_file = deepcopy(toolkit.config["load_data_from_cache_file"])
        print(f"price_analyst cache_file original: {cache_file}")

        if toolkit.config["credibility_scorer_llm"] is not None and len(toolkit.config["credibility_scorer_llm"]) > 0:
            cache_file = cache_file.replace("/model_name/", f"/{model_name}/")
            cache_file = cache_file.replace("/company_name/", f"/{company_name}/")
            cache_file = cache_file.replace("_trade_date_", f"_{trade_date}_")
            cache_file = os.path.join(cache_file, "price_report.md")

        if not os.path.exists(cache_file):
            print(f"Cache file {cache_file} does not exist! Running analyst instead.")
            return price_analyst_node(state)

        print("Loaded price report from cache file:", cache_file)
        with open(cache_file, "r", encoding="utf-8") as f:
            data_cache = json.load(f)

        dict_cache = data_cache[trade_date]
        assert company_name.upper() == dict_cache["company_of_interest"].upper(), \
            f"Company {company_name} != {dict_cache['company_of_interest']}"
        report = dict_cache["price_report"]

        if "price_analyst_state" in state:
            return {
                "messages": [],
                "price_report": report,
                "price_analyst_state": {
                    "price_reports_str": price_reports_str + f"## Report {count+1} from {llm_provider}:\n{report}\n\n",
                    "count": count + 1,
                },
            }
        else:
            return {
                "messages": [],
                "price_report": report,
            }

    if toolkit.config.get("load_data_from_cache", False):
        return price_analyst_node_load_data_from_cache
    else:
        return price_analyst_node
