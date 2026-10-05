import os
import json
import numpy as np
import pandas as pd
import yfinance as yf
import pandas_market_calendars as mcal
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from langchain_core.messages import HumanMessage


class MemoryBank:
    def __init__(self,
        log_dir: Optional[str] = None,
        llm_model = None,
        trade_date: str = None,
        ticker: str = None,
        rolling_window: int = 10,
        key_periods: List[int] = [1, 7, 14, 28, 90, 180, 360],
    ):
        """
        Initialize the memory system.

        Args:
            csv_path: Path to CSV file for persistent storage of trade history
            llm_model: LLM model for evaluating trading strategy
            rolling_window: Number of most recent trades to analyze (default: 30)
            trade_date: Current trading date
        """
        self.nyse = mcal.get_calendar("NYSE") # New York Stock Exchange

        self.trade_history = {}
        self.short_term_memory = {}
        self.long_term_memory = {}
        self.latest_long_short_memory = None

        # lookup table for return periods
        self.rolling_window = rolling_window
        self.key_periods = key_periods
        self.trade_date = trade_date  # current trading date

        # CSV persistence
        self.log_dir = Path(log_dir) if log_dir else None
        self.csv_path = self.log_dir / "memory_bank.csv" if self.log_dir else None
        self.llm_model = llm_model

        # Load existing history from CSV if available
        if self.csv_path and self.csv_path.exists():
            self._load_from_csv()
            self._load_historical_memory()

    def len_trade_history(self):
        return len(self.trade_history.keys())

    def set_trade_date(self, trade_date):
        self.trade_date = trade_date

    def set_ticker(self, ticker):
        self.ticker = ticker

    def add_trade_history(self, trade_date, trade_record):
        """Add trade record and save to CSV if configured."""
        try:
            self.trade_history[trade_date] = trade_record
            self._update_historical_returns()
            self.evaluate_and_reflect_trading_strategy()
            # Auto-save to CSV after each update
            if self.csv_path:
                self._save_to_csv()
        except Exception as e:
            print(f"Error adding trade history: {e}")

    def get_trade_history(self, trade_date):
        """Get trade history for a given date."""
        if trade_date not in self.trade_history:
            print(f"Warning: Trade date {trade_date} not found in trade history")
            return None
        return self.trade_history[trade_date]

    def _load_from_csv(self):
        """Load trade history from CSV file."""
        try:
            df = pd.read_csv(self.csv_path)
            for _, row in df.iterrows():
                trade_date = row['trade_date']
                trade_record = row.to_dict()

                # Parse JSON columns if needed
                for col in ['returns', 'sharpe_ratios', 'returns_slope', 'sharpe_ratios_slope']:
                    if col in trade_record and isinstance(trade_record[col], str):
                        trade_record[col] = json.loads(trade_record[col])

                self.trade_history[trade_date] = trade_record

            print(f"✓ Loaded {len(self.trade_history)} trades from {self.csv_path}")
        except Exception as e:
            print(f"Warning: Could not load CSV from {self.csv_path}: {e}")

    def _save_to_csv(self):
        """Save trade history to CSV file."""
        if not self.csv_path:
            return

        try:
            # Convert trade_history to DataFrame
            rows = []
            for trade_date, record in self.trade_history.items():
                row = record.copy()

                # Serialize complex objects to JSON strings
                for col in ['returns', 'sharpe_ratios', 'returns_slope', 'sharpe_ratios_slope']:
                    if col in row and isinstance(row[col], dict):
                        row[col] = json.dumps(row[col])

                rows.append(row)

            df = pd.DataFrame(rows)

            # Ensure directory exists
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)

            # Save to CSV
            df.to_csv(self.csv_path, index=False)
            print(f"✓ Saved {len(rows)} trades to {self.csv_path}")
        except Exception as e:
            print(f"Error saving to CSV: {e}")

    def _update_short_term_memory(self, short_term_strategy_report):
        """Update short-term memory based on short-term strategy report."""
        self.short_term_memory[self.trade_date] = short_term_strategy_report

    def _update_long_term_memory(self, long_term_strategy_report):
        """Update long-term memory based on long-term strategy report."""
        self.long_term_memory[self.trade_date] = long_term_strategy_report

    def _update_latest_long_short_memory(self, long_term_strategy_report, short_term_strategy_report):
        self.latest_long_short_memory = f"short-term strategy report: {short_term_strategy_report}\nlong-term strategy report: {long_term_strategy_report}"

    def get_long_term_memory(self, trade_date=None):
        """Get long-short memory for a given date."""
        if trade_date is None:
            trade_date = self.trade_date
        return self.long_term_memory[trade_date]

    def get_short_term_memory(self, trade_date=None):
        """Get short-term memory for a given date."""
        if trade_date is None:
            trade_date = self.trade_date
        return self.short_term_memory[trade_date]

    def get_latest_long_short_memory(self):
        """Get latest long-short memory."""
        return self.latest_long_short_memory

    def _load_historical_memory(self):
        """Load historical memory from JSON files."""
        json_files = [f for f in os.listdir(self.log_dir) if f.endswith(".json")]
        # file name format: full_states_log_2024-01-03_online_True.json
        json_files.sort(key=lambda x: datetime.strptime(x.split("_")[3], "%Y-%m-%d"))
        for file in json_files:
            with open(os.path.join(self.log_dir, file), "r") as f:
                log_states_dict = json.load(f)
            for trade_date_prev_day, state in log_states_dict.items():
                self.trade_date = trade_date_prev_day
                self.trade_history[trade_date_prev_day] = state["returns_losses"] if "returns_losses" in state else None
                if "long_term_memory" in self.trade_history[trade_date_prev_day]:
                    long_term_memory = self.trade_history[trade_date_prev_day]["long_term_memory"]
                    short_term_memory = self.trade_history[trade_date_prev_day]["short_term_memory"]
                    self._update_long_term_memory(long_term_memory)
                    self._update_short_term_memory(short_term_memory)
                    self._update_latest_long_short_memory(long_term_memory, short_term_memory)

    def _update_historical_returns(self):
        """
        Update returns for all past trades based on current date.
        Calculate returns retrospectively for configured periods.

        """
        current_date = self.trade_date
        current_date_obj = pd.to_datetime(current_date)

        # Iterate through all past trades
        for trade_date_prev, trade_record in self.trade_history.items():
            trade_date_prev_obj = pd.to_datetime(trade_date_prev)

            # Skip if trade is in the future
            if trade_date_prev_obj >= current_date_obj:
                continue

            # Calculate trading days elapsed since trade
            # days_elapsed = (current_date_obj - trade_date_prev_obj).days
            schedule = self.nyse.schedule(
                start_date=trade_date_prev_obj,
                end_date=current_date_obj
            )
            days_elapsed = len(schedule) - 1

            # Skip if no shares were traded
            if trade_record.get('shares_changed', 0) == 0:
                continue

            ticker = trade_record.get('ticker', self.ticker)

            # Get price data for this ticker
            try:
                trade_dt = datetime.strptime(trade_date_prev, "%Y-%m-%d").date()
                current_dt = datetime.strptime(current_date, "%Y-%m-%d").date()

                temp_price_data = yf.download(
                    ticker,
                    start=trade_dt.strftime("%Y-%m-%d"),
                    end=(current_dt + timedelta(days=1)).strftime("%Y-%m-%d"),
                    progress=False
                )
            except Exception as e:
                print(f"[update_returns] Error loading price data for {ticker}: {e}")
                continue

            # Check if trade date exists in price data
            if trade_date_prev_obj not in temp_price_data.index:
                print(f"[update_returns] Trade date {trade_date_prev} not in price data for {ticker}")
                continue

            # Get trade position in data
            try:
                trade_idx = temp_price_data.index.get_loc(trade_date_prev_obj)
            except KeyError:
                print(f"[update_returns] Trade date {trade_date_prev} not found in index for {ticker}")
                continue

            # Extract trade details
            entry_price = trade_record['entry_price']
            shares_changed = trade_record['shares_changed']
            trade_direction = trade_record['trade_direction']
            portfolio_value = trade_record['portfolio_value']

            # Update returns for each period that has elapsed
            returns_updated = {}
            sharpe_updated = {}

            for period_days in self.key_periods:
                key = f"{period_days}d"

                # Only calculate if enough days have elapsed
                if days_elapsed < period_days:
                    break

                # if days_elapsed > period_days:
                #     continue

                future_idx = trade_idx + period_days

                # Check if we have data for this period
                if future_idx >= len(temp_price_data):
                    continue

                # Get period data
                period_data = temp_price_data.iloc[trade_idx:future_idx + 1]
                if len(period_data) < 2:
                    continue

                future_price = period_data['Close'].iloc[-1]
                if isinstance(future_price, pd.Series):
                    future_price = future_price.iloc[0]

                # Calculate portfolio-weighted return
                price_change = (future_price - entry_price) * trade_direction
                position_pnl = price_change * shares_changed
                return_pct = (position_pnl / portfolio_value) * 100

                returns_updated[key] = round(return_pct, 3)

                # Calculate Sharpe ratio for periods >= 7 days
                if period_days >= 7 and len(period_data) > 1:
                    daily_prices = period_data['Close'].values
                    if isinstance(daily_prices[0], pd.Series):
                        daily_prices = np.array([p.iloc[0] if isinstance(p, pd.Series) else p for p in daily_prices])
                    else:
                        print(f"[update_sharpe_ratios] Daily prices are not a series for {ticker} on {daily_prices}")

                    daily_pnl = np.diff(daily_prices) * trade_direction * shares_changed
                    daily_returns_pct = (daily_pnl / portfolio_value) * 100

                    if len(daily_returns_pct) > 1:
                        mean_return = daily_returns_pct.mean()
                        std_return = daily_returns_pct.std()
                        if std_return > 0:
                            sharpe = (mean_return / std_return) * np.sqrt(252)
                            sharpe_updated[key] = round(sharpe, 3)
                        else:
                            print(f"[update_sharpe_ratios] Standard deviation is {std_return} for {ticker} on {daily_returns_pct}")
                    else:
                        print(f"[update_sharpe_ratios] Daily returns are not a series for {ticker} on {daily_returns_pct}")

            # Update trade history and returns_losses if this trade has new return data
            if returns_updated:
                print(self.trade_history[trade_date_prev]['returns'], returns_updated)
                self.trade_history[trade_date_prev]['returns'].update(returns_updated)
                self.trade_history[trade_date_prev]['sharpe_ratios'].update(sharpe_updated)

    def _compute_trade_history_analysis(self):
        """
        Compute share-weighted aggregated statistics over rolling window.

        Returns:
            dict with weighted returns and sharpe ratios per period
        """
        # Get most recent trades (rolling window)
        sorted_dates = sorted(self.trade_history.keys())

        # Initialize accumulators
        weighted_returns = {period: {'values': [], 'weights': []}
                           for period in self.key_periods}
        weighted_sharpe = {period: {'values': [], 'weights': []}
                          for period in self.key_periods}

        rolling_statistics = {
            "returns_slope": {},
            "sharpe_ratios_slope": {},
        }

        def get_trend_slope(values, min_points=3):
            if len(values) < min_points:
                return 0.0

            y = np.asarray(values, dtype=float)
            x = np.arange(len(y))
            slope = np.polyfit(x, y, 1)[0]
            return slope

        for period in self.key_periods:
            key = f"{period}d"
            for trade_date in sorted_dates[::-1][period:]:
                if len(weighted_returns[period]['values']) >= self.rolling_window:
                    break

                trade_record = self.trade_history[trade_date]

                # Use shares_changed as weight (absolute value for position size)
                weight = abs(trade_record.get('shares_changed', 0))

                # Aggregate returns in the rolling window
                if key in trade_record.get('returns', {}):
                    return_val = trade_record['returns'][key]
                    weighted_returns[period]['values'].append(return_val)
                    weighted_returns[period]['weights'].append(weight)

                if key in trade_record.get('sharpe_ratios', {}):
                    sharpe_val = trade_record['sharpe_ratios'][key]
                    weighted_sharpe[period]['values'].append(sharpe_val)
                    weighted_sharpe[period]['weights'].append(weight)

            # return statistics
            ret_data = weighted_returns[period]
            rolling_statistics["returns_slope"][key] = -1 * get_trend_slope(ret_data['values'])  # latest --> previous time point

            # sharpe statistics: the sharpe ratio within the period
            sharpe_data = weighted_sharpe[period]
            rolling_statistics["sharpe_ratios_slope"][key] = -1 * get_trend_slope(sharpe_data['values'])  # latest --> previous time point

        self.trade_history[self.trade_date].update(rolling_statistics)

    def _build_slope_prompt_text(self) -> str:
        """
        Build an LLM-facing prompt summarizing rolling-window trend slopes.
        Expects:
        self.trade_history[self.trade_date]["returns_slope"] -> {"1d": float, ...}
        self.trade_history[self.trade_date]["sharpe_ratios_slope"] -> {"1d": float, ...}
        """
        rec = self.trade_history.get(self.trade_date, {})

        rs = rec.get("returns_slope", {}) or {}
        ss = rec.get("sharpe_ratios_slope", {}) or {}
        rw = getattr(self, "rolling_window", None)

        # Be robust to the typo in existing code: key_periods vs key_periods
        key_periods = getattr(self, "key_periods", None) or getattr(self, "key_periods", []) or []

        def _describe_slope_with_value(slope, eps=1e-2):
            if slope > 5 * eps:
                return "strong upward trend"
            elif slope > eps:
                return "mild upward trend"
            elif slope < -5 * eps:
                return "strong downward trend"
            elif slope < -eps:
                return "mild downward trend"
            else:
                return "no clear trend / stable"


        lines = []
        lines.append("Rolling-window trend summary (slope-based).")
        lines.append(
            "Definition: For each horizon (e.g., 1d/7d/14d/28d), we maintain a time series of rolling-window "
            "performance estimates (e.g., return and Sharpe ratios). "
            "The values below are linear-trend slopes of those rolling series over recent trading dates "
            f"(rolling window size = {rw if rw is not None else 'N/A'})."
        )
        lines.append(
            "Interpretation: Positive slope = improving over time; negative slope = deteriorating; "
            "near-zero slope = no consistent trend. Raw slope values are provided for precise reasoning."
        )

        # Returns slope
        lines.append("")
        lines.append("Return trend slopes (raw values):")
        for p in self.key_periods:
            key = f"{p}d"
            v = float(rs.get(key, 0.0))
            label = _describe_slope_with_value(v)
            lines.append(f"- {key}: slope = {v:.6g} ({label})")

        # Sharpe slope
        lines.append("")
        lines.append("Sharpe-ratio trend slopes (raw values):")
        for p in self.key_periods:
            key = f"{p}d"
            v = float(ss.get(key, 0.0))
            label = _describe_slope_with_value(v)
            lines.append(f"- {key}: slope = {v:.6g} ({label})")

        # Combined guidance (what you wanted: slope > 0 => getting better)
        lines.append("")
        lines.append(
            "Combined rule-of-thumb: a horizon is 'getting better' when BOTH return slope and Sharpe slope are > 0. "
            "If returns slope > 0 but Sharpe slope < 0, performance may be improving by taking more risk."
        )

        lines.append("")
        lines.append(
            "Note: These slopes come from rolling-window estimates, so they reflect *trend in recent performance* "
            "rather than a single-day jump. Prefer using them to adjust risk/position sizing instead of chasing noise."
        )

        return "\n".join(lines)

    def evaluate_and_reflect_trading_strategy(self) -> Tuple[str, str]:
        """
        Evaluate and reflect trading strategy based on historical performance data.
        """
        self._compute_trade_history_analysis()

        short_term_strategy_report = self._evaluate_and_reflect_strategy_with_llm("short_term")
        self._update_short_term_memory(short_term_strategy_report)
        long_term_strategy_report = self._evaluate_and_reflect_strategy_with_llm("long_term")
        self._update_long_term_memory(long_term_strategy_report)

        self.latest_long_short_memory = self._update_latest_long_short_memory(
            long_term_strategy_report,
            short_term_strategy_report
        )

    def _evaluate_and_reflect_strategy_with_llm(self, analysis_type: str = "comprehensive") -> str:
        """
        Use LLM to evaluate trading strategy based on historical performance data.

        Args:
            analysis_type: Type of analysis
                - "short_term": Focus on recent performance (last 1-14 days)
                - "long_term": Focus on extended performance (28-360 days)
                - "comprehensive": Analyze both short and long term

        Returns:
            LLM evaluation report as string
        """
        if not self.llm_model:
            return "No LLM model configured for evaluation"

        if not self.trade_history:
            return "No trade history available for evaluation"

        # Prepare data summary
        df = self._get_dataframe()

        # Build evaluation prompt
        prompt = self._build_evaluation_prompt(df, analysis_type)

        # Get LLM evaluation
        try:
            messages = [HumanMessage(content=prompt)]
            response = self.llm_model.invoke(messages)
            return response.content
        except Exception as e:
            return f"Error during LLM evaluation: {e}"

    def _get_dataframe(self) -> pd.DataFrame:
        """Convert trade history to DataFrame for analysis."""

        rows = []
        for trade_date, record in self.trade_history.items():
            row = {
                'trade_date': trade_date,
                'ticker': record.get('ticker', self.ticker),
                'action': record.get('action', None),
                'percentage': record.get('percentage', 0),
                'shares_changed': record.get('shares_changed', 0),
                'entry_price': record.get('entry_price', 0),
            }

            # Add returns
            returns = record.get('returns', {})
            for period in self.key_periods:
                key = f"{period}d"
                row[f'return_{key}'] = returns.get(key, None)

            # Add sharpe ratios
            sharpe_ratios = record.get('sharpe_ratios', {})
            for period in self.key_periods:
                key = f"{period}d"
                row[f'sharpe_{key}'] = sharpe_ratios.get(key, None)

            # Add slopes
            returns_slope = record.get('returns_slope', {})
            for period in self.key_periods:
                key = f"{period}d"
                row[f'return_slope_{key}'] = returns_slope.get(key, None)

            sharpe_slope = record.get('sharpe_ratios_slope', {})
            for period in self.key_periods:
                key = f"{period}d"
                row[f'sharpe_slope_{key}'] = sharpe_slope.get(key, None)

            rows.append(row)

        return pd.DataFrame(rows)

    def _build_evaluation_prompt(self, df: pd.DataFrame, analysis_type: str) -> str:
        """Build LLM prompt for strategy evaluation."""

        # Define period groups
        short_term_periods = [1, 7]
        long_term_periods = [14, 28, 90, 180, 360]

        if analysis_type == "short_term":
            periods = short_term_periods
            focus = "short-term tactical performance (1-14 days)"
        elif analysis_type == "long_term":
            periods = long_term_periods
            focus = "long-term strategic performance (28-360 days)"
        else:
            periods = self.key_periods
            focus = "comprehensive multi-horizon performance"

        prompt = f"""You are an expert quantitative analyst evaluating a trading strategy.

# Task
Analyze the following trading performance data with a focus on {focus}.

# Data Summary
Total trades: {len(df)}
Date range: {df['trade_date'].min()} to {df['trade_date'].max()}
Unique tickers: {df['ticker'].nunique()}

# Performance Statistics
The slope is computed over the most recent {self.rolling_window} trades and is used to quantify the directional trend of strategy performance.

"""

        # Add statistics for each period
        for period in periods:
            period_key = f"{period}d"
            return_col = f'return_{period_key}'
            sharpe_col = f'sharpe_{period_key}'
            return_slope_col = f'return_slope_{period_key}'
            sharpe_slope_col = f'sharpe_slope_{period_key}'

            if return_col in df.columns:
                valid_returns = df[return_col].dropna()
                valid_sharpe = df[sharpe_col].dropna() if sharpe_col in df.columns else pd.Series()

                if len(valid_returns) > 0:
                    prompt += f"\n## {period}-Day Performance\n"
                    prompt += f"- Returns: mean={valid_returns.mean():.2f}%, std={valid_returns.std():.2f}%, "
                    prompt += f"min={valid_returns.min():.2f}%, max={valid_returns.max():.2f}%\n"
                    prompt += f"- Win rate: {(valid_returns > 0).sum() / len(valid_returns) * 100:.1f}%\n"

                    if len(valid_sharpe) > 0:
                        prompt += f"- Sharpe ratio: mean={valid_sharpe.mean():.3f}, std={valid_sharpe.std():.3f}\n"

                    # Add slope trends if available
                    if return_slope_col in df.columns:
                        latest_return_slope = df[return_slope_col].iloc[-1] if not pd.isna(df[return_slope_col].iloc[-1]) else None
                        latest_sharpe_slope = df[sharpe_slope_col].iloc[-1] if sharpe_slope_col in df.columns and not pd.isna(df[sharpe_slope_col].iloc[-1]) else None
                        latest_5_return_slopes = df[return_slope_col].iloc[-5:].mean() if return_slope_col in df.columns else None
                        latest_5_sharpe_slopes = df[sharpe_slope_col].iloc[-5:].mean() if sharpe_slope_col in df.columns else None

                        if latest_return_slope is not None:
                            trend = "improving" if latest_return_slope > 0 else "deteriorating"
                            prompt += f"- Latest return trend: {trend} (slope={latest_return_slope:.6f})\n"
                            prompt += f"- Latest 5 return mean trend: {trend} (slope={latest_5_return_slopes:.6f})\n"
                        if latest_sharpe_slope is not None:
                            trend = "improving" if latest_sharpe_slope > 0 else "deteriorating"
                            prompt += f"- Latest Sharpe trend: {trend} (slope={latest_sharpe_slope:.6f})\n"
                            prompt += f"- Latest 5 Sharpe mean trend: {trend} (slope={latest_5_sharpe_slopes:.6f})\n"

        # Add recent trades
        prompt += f"\n# Recent Trades (Last 10)\n\n"
        recent = df.tail(10)[['trade_date', 'ticker', 'action', 'percentage']].to_string(index=False)
        prompt += recent

        # Add evaluation instructions
        prompt += f"""

# Evaluation Instructions

Please provide a comprehensive evaluation addressing:

1. **Performance Assessment**
   - Overall strategy effectiveness across different time horizons
   - Risk-adjusted returns (Sharpe ratios) analysis
   - Win rate and consistency patterns

2. **{"Short-term" if analysis_type == "short_term" else "Long-term" if analysis_type == "long_term" else "Multi-horizon"} Trends**
   - Return trajectory (improving vs deteriorating)
   - Risk management effectiveness (Sharpe ratio trends)
   - Strategy stability over time

3. **Strengths and Weaknesses**
   - What is working well?
   - What needs improvement?
   - Any concerning patterns or risks?

4. **Actionable Recommendations**
   - Specific adjustments to improve performance
   - Risk management considerations
   - Position sizing or timing improvements

Please structure your response clearly with headers and bullet points.
"""

        return prompt

    def generate_performance_report(self) -> str:
        """Generate a comprehensive text report of current performance."""
        df = self._get_dataframe()

        if len(df) == 0:
            return "No trade history available"

        report = f"""
# Trading Strategy Performance Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview
- Total Trades: {len(df)}
- Date Range: {df['trade_date'].min()} to {df['trade_date'].max()}
- Unique Tickers: {df['ticker'].nunique()}
- Actions: {df['action'].value_counts().to_dict()}

## Performance by Period
"""

        for period in self.key_periods:
            period_key = f"{period}d"
            return_col = f'return_{period_key}'
            sharpe_col = f'sharpe_{period_key}'

            if return_col in df.columns:
                valid_returns = df[return_col].dropna()
                valid_sharpe = df[sharpe_col].dropna() if sharpe_col in df.columns else pd.Series()

                if len(valid_returns) > 0:
                    report += f"\n### {period}-Day Horizon\n"
                    report += f"- Trades with data: {len(valid_returns)}\n"
                    report += f"- Mean return: {valid_returns.mean():.2f}%\n"
                    report += f"- Std dev: {valid_returns.std():.2f}%\n"
                    report += f"- Win rate: {(valid_returns > 0).sum() / len(valid_returns) * 100:.1f}%\n"
                    report += f"- Best: {valid_returns.max():.2f}%, Worst: {valid_returns.min():.2f}%\n"

                    if len(valid_sharpe) > 0:
                        report += f"- Mean Sharpe: {valid_sharpe.mean():.3f}\n"

        return report
