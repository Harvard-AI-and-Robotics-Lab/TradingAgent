from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path

from tradingagents.default_config import DEFAULT_CONFIG


ROOT = Path(__file__).resolve().parents[1]


def load_runner_module():
    path = ROOT / "scripts" / "run_experiment.py"
    spec = importlib.util.spec_from_file_location("run_experiment", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ReleaseMetadataTests(unittest.TestCase):
    def test_core_agent_stack_is_present(self):
        required = {
            "tradingagents/agents/analysts/price_analyst.py",
            "tradingagents/agents/analysts/fundamentals_analyst.py",
            "tradingagents/agents/analysts/market_analyst.py",
            "tradingagents/agents/analysts/news_analyst.py",
            "tradingagents/agents/analysts/social_media_analyst.py",
            "tradingagents/agents/credibility_scorer/price_scorer.py",
            "tradingagents/agents/credibility_scorer/fundamentals_scorer.py",
            "tradingagents/agents/credibility_scorer/market_scorer.py",
            "tradingagents/agents/credibility_scorer/news_scorer.py",
            "tradingagents/agents/credibility_scorer/social_media_scorer.py",
            "tradingagents/agents/researchers/bull_researcher.py",
            "tradingagents/agents/researchers/bear_researcher.py",
            "tradingagents/agents/trader/trader.py",
            "tradingagents/graph/setup_multiagent.py",
            "tradingagents/graph/memory_bank.py",
            "tradingagents/graph/reflection.py",
            "tradingagents/graph/trading_graph.py",
        }
        missing = sorted(path for path in required if not (ROOT / path).is_file())
        self.assertEqual(missing, [])

    def test_default_config_has_release_keys(self):
        required = {
            "llm_provider",
            "deep_think_llm",
            "quick_think_llm",
            "analyst_llms",
            "analyst_llm_providers",
            "online_tools",
            "return_periods",
        }
        self.assertFalse(required - DEFAULT_CONFIG.keys())

    def test_example_manifests_are_well_formed(self):
        runner = load_runner_module()
        for path in sorted((ROOT / "configs" / "experiments").glob("*.json")):
            manifest = runner.load_manifest(path)
            self.assertEqual(manifest["schema_version"], 1)
            self.assertEqual(len(manifest["symbols"]), 3)
            for symbol in manifest["symbols"]:
                command = runner.build_command(manifest, symbol)
                self.assertIn(symbol, command)
                self.assertIn("--no-online_tools", command)

    def test_model_registry_stores_env_names_not_values(self):
        registry = json.loads(
            (ROOT / "configs" / "models.template.json").read_text(encoding="utf-8")
        )
        for provider in registry["providers"].values():
            self.assertTrue(provider["api_key_env"].endswith("_API_KEY"))
            self.assertNotIn("api_key", provider)


if __name__ == "__main__":
    unittest.main()
