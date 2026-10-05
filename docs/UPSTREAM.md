# Upstream provenance

TrustTrade is derived from the Apache-2.0 licensed
[`TauricResearch/TradingAgents`](https://github.com/TauricResearch/TradingAgents)
project and retains its `tradingagents` Python package name for compatibility.

The private research lineage shares upstream history through commit
`a438acdbbd622a4d3c112d23f7462651f8f9aeee`. This clean public snapshot was
prepared from the TrustTrade research branch at
`9e51a6793121f54dccaf09c4e475d0f840acbfe3`.

## Material TrustTrade additions

- parallel analyst-agent generation across providers;
- per-source credibility scorers and selective-consensus routing;
- deterministic price/temporal analysis;
- portfolio-aware decisions and backtesting;
- reflective short- and long-horizon memory; and
- human/LLM comparison and academic analysis utilities.

The public snapshot removes private credentials, generated notebook state,
person-level human-study artifacts and non-redistributable caches. Changes to
upstream files are documented by this notice and repository history.

## Upstream citation

```bibtex
@misc{xiao2025tradingagentsmultiagentsllmfinancial,
  title         = {TradingAgents: Multi-Agents LLM Financial Trading Framework},
  author        = {Yijia Xiao and Edward Sun and Di Luo and Wei Wang},
  year          = {2025},
  eprint        = {2412.20138},
  archivePrefix = {arXiv},
  primaryClass  = {q-fin.TR},
  url           = {https://arxiv.org/abs/2412.20138}
}
```
