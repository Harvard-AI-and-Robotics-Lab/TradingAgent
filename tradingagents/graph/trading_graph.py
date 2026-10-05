# TradingAgents/graph/trading_graph.py

import os
import re
import json
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import Dict, Any, Tuple, List, Optional

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_xai import ChatXAI

from langgraph.prebuilt import ToolNode

from tradingagents.agents import *
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.agents.utils.memory import FinancialSituationMemory
from tradingagents.agents.utils.agent_states import (
    AgentState,
    InvestDebateState,
    RiskDebateState,
)
from tradingagents.agents.utils.portfolio_utils import update_portfolio
from tradingagents.dataflows.interface import set_config

from .conditional_logic import ConditionalLogic
from .setup import GraphSetup
from .setup_multiagent import GraphSetupMultiAgent
from .propagation import Propagator
from .reflection import Reflector
from .signal_processing import SignalProcessor
from .memory_bank import MemoryBank


def create_llms(llm_provider, llm_name, backend_url=None):
    """Create LLMs based on the configuration."""

    backend_url_dict = {
        "openai": "https://api.openai.com/v1",
        "anthropic": "https://api.anthropic.com/v1",
    }

    if llm_provider.lower() == "openai" or llm_provider == "ollama" or llm_provider == "openrouter":
        temperature = 1 if llm_name in ["gpt-5", "gpt-5-mini"] else 0.7
        backend_url = backend_url if backend_url is not None else backend_url_dict[llm_provider]
        return ChatOpenAI(
            model=llm_name,
            base_url=backend_url,
            temperature=temperature
        )
    elif llm_provider.lower() == "google" or llm_provider.lower() == "gemini":
        return ChatGoogleGenerativeAI(
            model=llm_name
        )
    elif llm_provider.lower() == "xai":
        return ChatXAI(
            model=llm_name
        )
    elif llm_provider.lower() == "anthropic":
        return ChatAnthropic(
            model=llm_name,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {llm_provider}")

class TradingAgentsGraph:
    """Main class that orchestrates the trading agents framework."""

    def __init__(
        self,
        selected_analysts=["price", "fundamentals", "market", "social", "news"],
        debug=False,
        config: Dict[str, Any] = None,
        save_dir="results",
        enable_reflection=False,
        skip_analysts=False,
    ):
        """Initialize the trading agents graph and components.

        Args:
            selected_analysts: List of analyst types to include
            debug: Whether to run in debug mode
            config: Configuration dictionary. If None, uses default config
        """
        self.debug = debug
        self.config = DEFAULT_CONFIG.copy()
        self.config.update(config if config is not None else {})

        print(f"config: {self.config}")
        self.save_dir = save_dir
        self.enable_reflection = enable_reflection
        self.online_tools = self.config.get("online_tools", True)

        self.resume_from_cache = self.config.get("resume_from_cache", False)
        if "load_data_from_cache" not in self.config:
            self.config["load_data_from_cache"] = False
        if "load_data_from_cache_file" not in self.config:
            self.config["load_data_from_cache_file"] = ""

        # Update the interface's config
        set_config(self.config)

        # Create necessary directories
        os.makedirs(
            os.path.join(self.config["project_dir"], "dataflows/data_cache"),
            exist_ok=True,
        )

        # Initialize LLMs
        self.deep_thinking_llm = create_llms(
            self.config["llm_provider"],
            self.config["deep_think_llm"],
            self.config["backend_url"]
        )
        self.quick_thinking_llm = create_llms(
            self.config["llm_provider"],
            self.config["quick_think_llm"],
            self.config["backend_url"]
        )

        if len(self.config["analyst_llms"]) > 0 and len(self.config["analyst_llm_providers"]) > 0:
            assert len(self.config["analyst_llms"]) == len(self.config["analyst_llm_providers"]), "Number of analyst LLMs and providers must match"
            self.analyst_llms = [
                create_llms(llm_provider, llm_name)
                for llm_provider, llm_name in zip(self.config["analyst_llm_providers"], self.config["analyst_llms"])
            ]
        else:
            self.analyst_llms = []

        credibility_scorer_llm = self.config.get("credibility_scorer_llm", None)
        credibility_scorer_llm_provider = self.config.get("credibility_scorer_llm_provider", None)
        if isinstance(credibility_scorer_llm, str) and len(credibility_scorer_llm) == 0:
            credibility_scorer_llm = None
        if isinstance(credibility_scorer_llm_provider, str) and len(credibility_scorer_llm_provider) == 0:
            credibility_scorer_llm_provider = None
        if credibility_scorer_llm is not None and credibility_scorer_llm_provider is not None:
            self.credibility_scorer_llm = create_llms(
                credibility_scorer_llm_provider,
                credibility_scorer_llm,
            )
        else:
            self.credibility_scorer_llm = None

        self.toolkit = Toolkit(config=self.config)

        # Initialize memories
        if self.enable_reflection:
            self.bull_memory = FinancialSituationMemory("bull_memory", self.config)
            self.bear_memory = FinancialSituationMemory("bear_memory", self.config)
            self.trader_memory = FinancialSituationMemory("trader_memory", self.config)
            self.invest_judge_memory = FinancialSituationMemory("invest_judge_memory", self.config)
        else:
            self.bull_memory = None
            self.bear_memory = None
            self.trader_memory = None
            self.invest_judge_memory = None

        # Create tool nodes
        self.tool_nodes = self._create_tool_nodes()

        # Initialize components
        self.conditional_logic = ConditionalLogic(max_analyst_reports=len(self.analyst_llms))

        if len(self.analyst_llms) > 0 and self.credibility_scorer_llm is not None:
            self.graph_setup = GraphSetupMultiAgent(
                config=self.config,
                quick_thinking_llm=self.quick_thinking_llm,
                deep_thinking_llm=self.deep_thinking_llm,
                analyst_llms=self.analyst_llms,
                credibility_scorer_llm=self.credibility_scorer_llm,
                toolkit=self.toolkit,
                tool_nodes=self.tool_nodes,
                bull_memory=self.bull_memory,
                bear_memory=self.bear_memory,
                trader_memory=self.trader_memory,
                invest_judge_memory=self.invest_judge_memory,
                risk_manager_memory=None,  # ignore agents in risk management if not set
                conditional_logic=self.conditional_logic,
                skip_analysts=skip_analysts,
            )
        else:
            self.graph_setup = GraphSetup(
                quick_thinking_llm=self.quick_thinking_llm,
                deep_thinking_llm=self.deep_thinking_llm,
                toolkit=self.toolkit,
                tool_nodes=self.tool_nodes,
                bull_memory=self.bull_memory,
                bear_memory=self.bear_memory,
                trader_memory=self.trader_memory,
                invest_judge_memory=self.invest_judge_memory,
                risk_manager_memory=None,  # ignore agents in risk management if not set
                conditional_logic=self.conditional_logic,
            )

        self.propagator = Propagator()
        self.signal_processor = SignalProcessor(self.quick_thinking_llm)

        # Initialize memory system and reflector
        # Configure return calculation periods
        self.return_periods = self.config.get('return_periods', [1, 7, 14, 28, 90, 180, 360])
        assert isinstance(self.return_periods, list), f"return_periods must be a list: {self.return_periods}"
        self.return_periods = sorted(self.return_periods)
        if self.enable_reflection:
            self.reflector = Reflector(self.quick_thinking_llm)
        else:
            self.reflector = None

        # State tracking variables
        self.curr_state = None
        self.ticker = None
        self.trade_date = None
        self.log_dir = None
        self.log_file = None
        self.log_states_dict = {}  # date to full state dict
        self.stock_price_data = None
        self.portfolio_summary = None
        self.memory_bank = None

        # Set up the graph
        self.graph = self.graph_setup.setup_graph(selected_analysts)


    def _create_tool_nodes(self) -> Dict[str, ToolNode]:
        """Create tool nodes for different data sources."""
        return {
            "price": ToolNode(
                [
                    self.toolkit.get_historical_price_analysis_and_recommendation,
                ]
            ),
            "fundamentals": ToolNode(
                [
                    # online tools
                    self.toolkit.get_fundamentals_openai,
                    self.toolkit.get_fundamentals_gemini,
                    self.toolkit.get_fundamentals_xai,
                    self.toolkit.get_fundamentals_anthropic,
                    self.toolkit.get_company_investor_relations_data_openai,
                    self.toolkit.get_company_investor_relations_data_gemini,
                    self.toolkit.get_company_investor_relations_data_xai,
                    self.toolkit.get_company_investor_relations_data_anthropic,
                    self.toolkit.get_sec_filings_data_openai,
                    self.toolkit.get_sec_filings_data_gemini,
                    self.toolkit.get_sec_filings_data_xai,
                    self.toolkit.get_sec_filings_data_anthropic,
                    self.toolkit.get_sedar_filings_data_openai,
                    self.toolkit.get_sedar_filings_data_gemini,
                    self.toolkit.get_sedar_filings_data_xai,
                    self.toolkit.get_sedar_filings_data_anthropic,
                    # offline tools
                    self.toolkit.get_finnhub_company_insider_sentiment,
                    self.toolkit.get_finnhub_company_insider_transactions,
                    self.toolkit.get_simfin_balance_sheet,
                    self.toolkit.get_simfin_cashflow,
                    self.toolkit.get_simfin_income_stmt,
                    # additional offline tools based on online functionality
                    self.toolkit.get_company_investor_relations_data_offline,
                    self.toolkit.get_sec_filings_data_offline,
                    self.toolkit.get_sedar_filings_data_offline,
                ]
            ),
            "market": ToolNode(
                [
                    # online tools from yfinance
                    self.toolkit.get_YFin_data_online,
                    self.toolkit.get_stockstats_indicators_report_online,
                    # offline tools from yfinance
                    self.toolkit.get_YFin_data,
                    self.toolkit.get_stockstats_indicators_report,
                ]
            ),
            "social": ToolNode(
                [
                    # online tools
                    self.toolkit.get_stock_news_openai,
                    self.toolkit.get_stock_news_gemini,
                    self.toolkit.get_stock_news_xai,
                    self.toolkit.get_stock_news_anthropic,
                    # offline tools
                    self.toolkit.get_reddit_stock_info,
                ]
            ),
            "news": ToolNode(
                [
                    # online tools
                    self.toolkit.get_global_news_openai,
                    self.toolkit.get_global_news_gemini,
                    self.toolkit.get_global_news_xai,
                    self.toolkit.get_global_news_anthropic,
                    self.toolkit.get_google_news,
                    # offline tools
                    self.toolkit.get_finnhub_news,
                    self.toolkit.get_reddit_news,
                ]
            ),
        }

    def process_signal(self, full_signal):
        """Process a signal to extract the core decision."""
        return self.signal_processor.process_signal_with_percentage(full_signal)

    def propagate(self, company_name, trade_date, portfolio_summary=None):
        """Run the trading agents graph for a company on a specific date.

        Args:
            company_name: Stock ticker symbol (e.g., 'AAPL')
            trade_date: Trading date (e.g., '2024-01-25')
            portfolio_summary: Optional portfolio context for portfolio-aware decisions
                               Can be PortfolioSummary object or dictionary with portfolio data
        """

        self.ticker = company_name
        self.trade_date = trade_date

        # Save to file
        if self.log_dir is None:
            self.log_dir = Path(f"{self.save_dir}/{self.ticker}/TradingAgentsStrategy_logs")
            self.log_dir.mkdir(parents=True, exist_ok=True)
            memory_config = {
                'trade_date': trade_date,
                'ticker': company_name,
                'rolling_window': self.config.get('rolling_window', 10),
                'key_periods': self.config.get('key_periods', [1, 7, 14, 28, 90, 180, 360]),
            }
        if self.memory_bank is None:
            self.memory_bank = MemoryBank(
                log_dir=self.log_dir,
                llm_model=self.quick_thinking_llm,
                **memory_config
            )
        self.memory_bank.set_trade_date(trade_date)
        self.memory_bank.set_ticker(company_name)

        self.log_file = f"{self.log_dir}/full_states_log_{trade_date}_online_{self.online_tools}.json"
        if not os.path.exists(self.log_file):
            # read all files in self.log_dir and find the latest file
            files = os.listdir(self.log_dir)
            files.sort(key=lambda x: os.path.getmtime(os.path.join(self.log_dir, x)))
            files = files[::-1]
            for file in files:
                if file.endswith(".json"):
                    with open(os.path.join(self.log_dir, file), "r") as f:
                        log_states_dict = json.load(f)
                        if str(trade_date) in log_states_dict:
                            self.log_file = os.path.join(self.log_dir, file)
                            break

        if self.config["load_data_from_cache_file"] == "":
            self.config["load_data_from_cache_file"] = self.log_file
        self.toolkit.update_config({"load_data_from_cache_file": self.config["load_data_from_cache_file"]})
        print(f"data cache file: {self.config['load_data_from_cache_file']}")

        # log file exists, load dict to self.curr_state
        if self.resume_from_cache and isinstance(self.log_file, str) and os.path.exists(self.log_file):  # continue from previous state
            with open(self.log_file, "r") as f:
                self.log_states_dict = json.load(f)
            if str(trade_date) in self.log_states_dict:
                self.curr_state = self.log_states_dict[str(trade_date)]
                self.portfolio_summary = self.curr_state.get("portfolio_summary", None)
                decision = self.curr_state["final_trade_decision_processed"]

                trade_record = self.curr_state.get("returns_losses", {})
                # Update historical returns for all past trades based on current date
                self.memory_bank.add_trade_history(trade_date, trade_record)

                with open(self.log_file, "w") as f:
                    json.dump(self.log_states_dict, f, indent=4)

                return self.curr_state, decision

        if portfolio_summary is not None:
            self.portfolio_summary = portfolio_summary

        # Initialize state with portfolio context
        init_agent_state = self.propagator.create_initial_state(
            company_name, trade_date, self.portfolio_summary,
            enable_risk_manager=False,
            enable_multiple_analysts=len(self.analyst_llms) > 0 and self.credibility_scorer_llm is not None
        )
        args = self.propagator.get_graph_args()
        init_agent_state["latest_long_short_memory"] = self.memory_bank.get_latest_long_short_memory()
        print(f"Initial agent state: {init_agent_state}")

        if self.debug:
            # Debug mode with tracing
            trace = []
            for chunk in self.graph.stream(init_agent_state, **args):

                if len(chunk["messages"]) == 0:
                    pass
                else:
                    for i in range(len(chunk["messages"])):
                        chunk["messages"][i].pretty_print()
                    trace.append(chunk)

            self.curr_state = trace[-1]
        else:
            # Standard mode without tracing
            self.curr_state = self.graph.invoke(init_agent_state, **args)

        decision = self.process_signal(self.curr_state["final_trade_decision"])
        trade_record = self._backtest_decision(company_name, trade_date, decision)

        self.curr_state["portfolio_summary"] = self.portfolio_summary
        self.curr_state["final_trade_decision_processed"] = decision
        self.curr_state["returns_losses"] = trade_record

        # Update historical returns for all past trades based on current date
        self.memory_bank.add_trade_history(trade_date, self.curr_state["returns_losses"])
        # Store current state for reflection
        if self.enable_reflection:
            self.reflect_and_remember()

        # Log state
        self._log_state(trade_date)

        # Return decision and processed signal
        return self.curr_state, decision

    def _backtest_decision(self, company_name, trade_date, decision):
        """Store trade decision and calculate immediate portfolio changes."""
        if self.portfolio_summary is None:
            return  # no portfolio summary to backtest against

        if self.stock_price_data is None:
            # Get price data
            trade_dt = datetime.strptime(trade_date, "%Y-%m-%d").date()
            today = date.today()

            if trade_dt >= today:
                self.stock_price_data = yf.download(
                    company_name,
                    period="1mo",
                    interval="1d",
                    progress=False
                )
            else:
                self.stock_price_data = yf.download(
                    company_name,
                    start="2000-01-01",
                    end=(trade_dt + timedelta(days=90)).strftime("%Y-%m-%d"),
                    progress=False
                )

        trade_date_obj = pd.to_datetime(trade_date)
        if trade_date_obj not in self.stock_price_data.index:
            print(f"No price data for {company_name} on {trade_date} in stock_price_data")
            return  # no price data for this date

        stock_price = self.stock_price_data.loc[trade_date_obj]['Close']
        if isinstance(stock_price, pd.Series):
            stock_price = stock_price.iloc[0]

        assert isinstance(decision, dict), 'decision must be a dict: {"action": "BUY/HOLD/SELL", "percentage": 0-100}'
        assert 'action' in decision and 'percentage' in decision, 'decision must contain "action" key: {"action": "BUY/HOLD/SELL", "percentage": 0-100}'

        # Store portfolio state before trade (deep copy to preserve original state)
        portfolio_before = self.portfolio_summary.copy()

        # Calculate shares changed by this decision
        cash = portfolio_before['cash_balance']
        action = decision['action']
        percentage = decision['percentage']

        # Find existing holding
        current_shares = 0
        avg_cost = stock_price
        for holding in portfolio_before.get('holdings', []):
            if holding['symbol'].upper() == company_name.upper():
                current_shares = holding['quantity']
                avg_cost = holding['avg_cost']
                break

        # Calculate shares traded
        if action == 'BUY':
            available_cash = cash * (percentage / 100.0)
            shares_changed = int(available_cash / stock_price)
            trade_direction = 1  # positive for buy
        elif action == 'SELL':
            shares_changed = int(current_shares * (percentage / 100.0))
            trade_direction = -1  # negative for sell
        else:  # HOLD
            shares_changed = 0
            trade_direction = 0

        # Update portfolio
        self.portfolio_summary = update_portfolio(
            self.portfolio_summary, company_name, decision['action'], decision['percentage'], stock_price
        )

        # Store trade record for retrospective return calculation (will be updated retrospectively)
        portfolio_value_before = portfolio_before['total_value']
        portfolio_value_after = self.portfolio_summary['total_value']

        trade_record = {
            "trade_date": trade_date,
            'ticker': company_name,
            'action': action,
            'percentage': percentage,
            'shares_changed': shares_changed,
            'trade_direction': trade_direction,
            'entry_price': stock_price,
            'portfolio_value': portfolio_value_before,
            'current_shares_before': current_shares,
            # returns and sharpe ratios
            "decision": decision,
            "entry_price": stock_price,
            "returns": {f"{period}d": 0.0 for period in self.memory_bank.key_periods},  # Will be updated retrospectively
            "sharpe_ratios": {f"{period}d": 0.0 for period in self.memory_bank.key_periods},  # Will be updated retrospectively
            "portfolio_value_before": round(portfolio_value_before, 2),
            "portfolio_value_after": round(portfolio_value_after, 2),
            "portfolio_change_pct": round(((portfolio_value_after - portfolio_value_before) / portfolio_value_before) * 100, 2)
        }

        return trade_record

    def _log_state(self, trade_date):
        """Log the final state to a JSON file."""
        final_state = self.curr_state

        curr_log_state = {
            "company_of_interest": final_state["company_of_interest"],
            "trade_date": final_state["trade_date"],
            "portfolio_summary": final_state["portfolio_summary"],

            "price_analyst_state": {
                "price_reports_str": final_state["price_analyst_state"]["price_reports_str"],
                "price_credibility_scorer": final_state["price_analyst_state"]["price_credibility_scorer"],
                "count": final_state["price_analyst_state"]["count"],
            } if "price_analyst_state" in final_state else None,

            "fundamentals_analyst_state": {
                "fundamentals_reports_str": final_state["fundamentals_analyst_state"]["fundamentals_reports_str"],
                "fundamentals_credibility_scorer": final_state["fundamentals_analyst_state"]["fundamentals_credibility_scorer"],
                "count": final_state["fundamentals_analyst_state"]["count"],
            } if "fundamentals_analyst_state" in final_state else None,
            "market_analyst_state": {
                "market_reports_str": final_state["market_analyst_state"]["market_reports_str"],
                "market_credibility_scorer": final_state["market_analyst_state"]["market_credibility_scorer"],
                "count": final_state["market_analyst_state"]["count"],
            } if "market_analyst_state" in final_state else None,
            "news_analyst_state": {
                "news_reports_str": final_state["news_analyst_state"]["news_reports_str"],
                "news_credibility_scorer": final_state["news_analyst_state"]["news_credibility_scorer"],
                "count": final_state["news_analyst_state"]["count"],
            } if "news_analyst_state" in final_state else None,
            "sentiment_analyst_state": {
                "sentiment_reports_str": final_state["sentiment_analyst_state"]["sentiment_reports_str"],
                "sentiment_credibility_scorer": final_state["sentiment_analyst_state"]["sentiment_credibility_scorer"],
                "count": final_state["sentiment_analyst_state"]["count"],
            } if "sentiment_analyst_state" in final_state else None,

            "price_report": final_state.get("price_report", ""),
            "fundamentals_report": final_state["fundamentals_report"],
            "market_report": final_state["market_report"],
            "sentiment_report": final_state["sentiment_report"],
            "news_report": final_state["news_report"],
            "compressed_situation": final_state.get("compressed_situation", ""),

            "investment_debate_state": {
                "bull_history": final_state["investment_debate_state"]["bull_history"],
                "bear_history": final_state["investment_debate_state"]["bear_history"],
                "history": final_state["investment_debate_state"]["history"],
                "current_response": final_state["investment_debate_state"][
                    "current_response"
                ],
                "judge_decision": final_state["investment_debate_state"][
                    "judge_decision"
                ],
            },
            "trader_investment_plan": "" if len(final_state.get("trader_investment_plan", "")) == 0 else final_state.get("trader_investment_plan", ""),
            "final_trade_decision": final_state["final_trade_decision"],
            "final_trade_decision_processed": final_state["final_trade_decision_processed"],
        }
        if "trader_investment_plan_with_data_reports" in final_state:
            curr_log_state["trader_investment_plan_with_data_reports"] = final_state["trader_investment_plan_with_data_reports"]
        if "final_trade_decision_with_data_reports" in final_state:
            curr_log_state["final_trade_decision_with_data_reports"] = final_state["final_trade_decision_with_data_reports"]
        if self.enable_reflection:
            curr_log_state["investment_debate_state"]["bull_reflection"] = final_state["investment_debate_state"]["bull_reflection"] if "bull_reflection" in final_state["investment_debate_state"] else None
            curr_log_state["investment_debate_state"]["bear_reflection"] = final_state["investment_debate_state"]["bear_reflection"] if "bear_reflection" in final_state["investment_debate_state"] else None
            curr_log_state["investment_debate_state"]["judge_reflection"] = final_state["investment_debate_state"]["judge_reflection"] if "judge_reflection" in final_state["investment_debate_state"] else None
            curr_log_state["trader_reflection"] = final_state["trader_reflection"] if "trader_reflection" in final_state else None
            if "returns_losses" in final_state:
                curr_log_state["returns_losses"] = final_state["returns_losses"]

        for key, value in final_state.items():
            if "memory" in key and key not in curr_log_state:
                curr_log_state[key] = value

        self.log_states_dict[str(trade_date)] = curr_log_state

        with open(self.log_file, "w") as f:
            json.dump(self.log_states_dict, f, indent=4)
        self._dump_markdown_reports(curr_log_state)
        self.log_file = None

    def _dump_markdown_reports(self, curr_log_state):
        def slugify(s: str) -> str:
            s = str(s).strip()
            s = re.sub(r"[^\w\-\.]+", "_", s)          # keep letters/numbers/_-. ; replace others with _
            s = re.sub(r"_+", "_", s).strip("_")
            return s or "unknown"

        def save_markdown_report(report_key, report_content):
            md_path = base_dir / f"{slugify(report_key)}.md"
            md_text = f"# {report_key}\n\n- Trade date: `{trade_date}`\n- Ticker: `{ticker}`\n\n"
            assert isinstance(report_content, str), f"Report content must be a string: {report_content}"
            md_text += report_content.strip() + "\n"
            md_path.write_text(md_text, encoding="utf-8")

        trade_date = curr_log_state.get("trade_date", "UNKNOWN")
        ticker = curr_log_state.get("company_of_interest", "UNKNOWN")
        base_dir = Path(self.log_file.replace(".json", ""))
        base_dir.mkdir(parents=True, exist_ok=True)

        # Save each report field as its own markdown
        for report_key, report in curr_log_state.items():
            if isinstance(report, str):
                save_markdown_report(report_key, report)
            elif isinstance(report, dict):
                for k, v in report.items():
                    if isinstance(v, str):
                        save_markdown_report(report_key + "." + k, v)
                    elif isinstance(v, dict):
                        save_markdown_report(report_key + "." + k, json.dumps(v, indent=2))
                    else:
                        try:
                            save_markdown_report(report_key + "." + k, str(v))
                        except Exception as e:
                            print(f"Skipping non-string value for {report_key}.{k}: {v}, error: {e}")
            else:
                pass


    def reflect_and_remember(self):
        """Reflect on decisions and update memory based on returns.
        """

        # used for trader reflection
        self.curr_state["long_term_memory"] = self.memory_bank.get_long_term_memory()
        self.curr_state["short_term_memory"] = self.memory_bank.get_short_term_memory()

        if isinstance(self.reflector, Reflector):
            self.reflect_and_remember_rag()


    def reflect_and_remember_rag(self):
        """Reflect on decisions and update memory based on returns.

        Args:
            trade_record: Returns/losses in the trade record.
        """
        trade_record = self.memory_bank.get_trade_history(self.trade_date)

        if trade_record is None:
            print("Warning: No returns_losses / trade_record data available for reflection")
            return

        if self.curr_state is None:
            print("Warning: No current state available for reflection")
            return

        self.reflector.reflect_bull_researcher(
            self.curr_state, trade_record, self.bull_memory
        )
        self.reflector.reflect_bear_researcher(
            self.curr_state, trade_record, self.bear_memory
        )
        self.reflector.reflect_trader(
            self.curr_state, trade_record, self.trader_memory
        )
        self.reflector.reflect_invest_judge(
            self.curr_state, trade_record, self.invest_judge_memory
        )

        print("Reflection complete and memories updated")
