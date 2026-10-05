# Plot Consensus Flowcharts

This folder contains a standalone script to draw flowcharts that explain how the multi-agent pipeline:

- uses multiple analysts per domain,
- computes cross-report consensus,
- keeps high-consensus signals,
- filters or isolates low-consensus signals,
- then synthesizes the final trading decision.

## Script

- `tools/plot_consensus/plot_multiagent_consensus_flow.py`

## Run

```bash
python tools/plot_consensus/plot_multiagent_consensus_flow.py
```

Optional output directory:

```bash
python tools/plot_consensus/plot_multiagent_consensus_flow.py --outdir tools/plot_consensus/outputs
```

## Outputs

- `multiagent_consensus_flow_overview.png`
- `multiagent_consensus_flow_overview.svg`
- `price_consensus_filter_flow.png`
- `price_consensus_filter_flow.svg`

## Code Grounding

The flowchart logic is aligned with:

- `tradingagents/graph/trading_graph.py`
- `tradingagents/graph/setup_multiagent.py`
- `tradingagents/graph/conditional_logic.py`
- `tradingagents/agents/credibility_scorer/price_scorer.py`
- `tradingagents/agents/credibility_scorer/fundamentals_scorer.py`
- `tradingagents/agents/credibility_scorer/market_scorer.py`
- `tradingagents/agents/credibility_scorer/news_scorer.py`
- `tradingagents/agents/credibility_scorer/social_media_scorer.py`
