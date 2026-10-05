"""Minimal command-line entry point for a single TrustTrade decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a single TrustTrade or base TradingAgents decision."
    )
    parser.add_argument("--ticker", default="NVDA", help="Ticker symbol")
    parser.add_argument("--trade-date", required=True, help="Trading date (YYYY-MM-DD)")
    parser.add_argument("--mode", choices=("trusttrade", "base"), default="trusttrade")
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--backend-url", default=None)
    parser.add_argument(
        "--analysts",
        nargs="+",
        default=["price", "fundamentals", "market", "social", "news"],
    )
    parser.add_argument("--analyst-models", nargs="+", default=None)
    parser.add_argument("--analyst-providers", nargs="+", default=None)
    parser.add_argument("--scorer-model", default=None)
    parser.add_argument("--scorer-provider", default=None)
    parser.add_argument(
        "--online-tools",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--enable-reflection", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("results/single_run"))
    parser.add_argument("--debug", action="store_true")
    return parser


def build_config(args: argparse.Namespace) -> dict:
    config = DEFAULT_CONFIG.copy()
    config.update(
        {
            "llm_provider": args.provider,
            "deep_think_llm": args.model,
            "quick_think_llm": args.model,
            "online_tools": args.online_tools,
        }
    )
    if args.backend_url:
        config["backend_url"] = args.backend_url

    if args.mode == "trusttrade":
        analyst_models = args.analyst_models or [args.model, args.model]
        analyst_providers = args.analyst_providers or [args.provider, args.provider]
        if len(analyst_models) != len(analyst_providers):
            raise ValueError("--analyst-models and --analyst-providers must have equal lengths")
        config.update(
            {
                "analyst_llms": analyst_models,
                "analyst_llm_providers": analyst_providers,
                "credibility_scorer_llm": args.scorer_model or args.model,
                "credibility_scorer_llm_provider": args.scorer_provider or args.provider,
            }
        )
    else:
        config.update(
            {
                "analyst_llms": [],
                "analyst_llm_providers": [],
                "credibility_scorer_llm": None,
                "credibility_scorer_llm_provider": None,
            }
        )
    return config


def main() -> None:
    args = build_parser().parse_args()
    config = build_config(args)
    graph = TradingAgentsGraph(
        selected_analysts=args.analysts,
        debug=args.debug,
        config=config,
        save_dir=str(args.output_dir),
        enable_reflection=args.enable_reflection,
    )
    _, decision = graph.propagate(args.ticker.upper(), args.trade_date)
    print(json.dumps(decision, indent=2, default=str))


if __name__ == "__main__":
    main()
