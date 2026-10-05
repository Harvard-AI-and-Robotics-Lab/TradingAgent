#!/usr/bin/env python3
"""Launch a backtest from a checked-in JSON experiment manifest."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "period",
        "symbols",
        "portfolio",
        "selected_analysts",
        "online_tools",
        "llm",
        "output_dir",
    }
    missing = sorted(required - manifest.keys())
    if missing:
        raise ValueError(f"Manifest is missing required keys: {', '.join(missing)}")
    llm = manifest["llm"]
    if len(llm.get("analyst_models", [])) != len(llm.get("analyst_providers", [])):
        raise ValueError("analyst_models and analyst_providers must have equal lengths")
    return manifest


def build_command(manifest: dict, symbol: str) -> list[str]:
    if symbol not in manifest["symbols"]:
        raise ValueError(f"{symbol} is not listed in manifest symbols")

    llm = manifest["llm"]
    period = manifest["period"]
    portfolio = manifest["portfolio"]
    command = [
        sys.executable,
        str(ROOT / "baseline_analysis" / "tradingagents_strategy.py"),
        "--symbol",
        symbol,
        "--start_date",
        period["start"],
        "--end_date",
        period["end"],
        "--cash",
        str(portfolio["initial_cash"]),
        "--commission",
        str(portfolio["commission"]),
        "--save_dir",
        str(ROOT / manifest["output_dir"]),
        "--llm_provider",
        llm["provider"],
        "--backend_url",
        llm["backend_url"],
        "--quick_think_llm",
        llm["quick_model"],
        "--deep_think_llm",
        llm["deep_model"],
        "--max_debate_rounds",
        str(manifest.get("max_debate_rounds", 1)),
        "--selected_analysts",
        *manifest["selected_analysts"],
        "--analyst_llms",
        *llm.get("analyst_models", []),
        "--analyst_llm_providers",
        *llm.get("analyst_providers", []),
        "--credibility_scorer_llm",
        llm.get("credibility_scorer_model", ""),
        "--credibility_scorer_llm_provider",
        llm.get("credibility_scorer_provider", ""),
    ]
    command.append("--online_tools" if manifest["online_tools"] else "--no-online_tools")
    if manifest.get("enable_reflection", False):
        command.append("--enable_reflection")
    if manifest.get("skip_analysts", False):
        command.append("--skip_analysts")
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest(args.config.resolve())
    command = build_command(manifest, args.symbol.upper())
    print(shlex.join(command))
    if not args.dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
