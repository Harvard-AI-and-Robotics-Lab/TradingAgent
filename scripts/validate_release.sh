#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

python scripts/check_release.py
python -m unittest discover -s tests -v
python scripts/run_experiment.py \
  --config configs/experiments/2024_q1.example.json \
  --symbol AAPL \
  --dry-run >/dev/null

echo "TrustTrade release validation completed without API calls."
