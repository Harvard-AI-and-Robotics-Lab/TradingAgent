# Reproducibility guide

TrustTrade separates three validation levels so that a successful software
check is not confused with reproduction of the paper results.

## 1. Offline release validation

```bash
bash scripts/validate_release.sh
```

This checks repository hygiene, JSON schemas, Python syntax and offline unit
tests. It makes no network or paid API calls.

## 2. Functional model run

After installing dependencies and exporting provider credentials:

```bash
python main.py --ticker NVDA --trade-date 2024-05-10 --mode trusttrade
```

This verifies that the selected providers and current data sources work. Live
retrieval is not a reproduction of a historical experiment because results can
change after the decision date.

## 3. Paper-result reproduction

A defensible reproduction requires all of the following:

1. Immutable input caches restricted to information available on each decision
   date, with checksums and source manifests.
2. Exact model and provider snapshots, temperatures, prompts, tool settings and
   retry behavior.
3. Exact stock universes, trading calendars, initial portfolios, commission
   assumptions and allocation constraints.
4. A run manifest linking every output to its code commit and configuration.
5. Scripts that regenerate each reported table and figure from released or
   access-controlled inputs.
6. Multiple-run handling for stochastic LLM outputs and a documented random
   seed wherever deterministic components use randomness.

The checked-in `*.example.json` files are safe launch templates. They are not
claimed to be the final paper manifests.

## Author verification required before `v1.0`

- Confirm the exact 2024-Q1 and 2026-Q1 date endpoints.
- Confirm every model identifier and provider used in each paper figure.
- Record prompt revisions and decoding parameters.
- Decide whether sanitized model outputs can legally be redistributed.
- Provide checksums for the frozen market/news/social cache.
- Map every paper table and figure to one command and expected output digest.
- Obtain approval for any aggregate human-study artifact.

## Example manifest runner

Preview a command:

```bash
python scripts/run_experiment.py \
  --config configs/experiments/2024_q1.example.json \
  --symbol AAPL \
  --dry-run
```

Remove `--dry-run` only after reviewing provider cost and data provenance.
