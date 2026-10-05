# Public release checklist

Use this checklist before changing any GitHub repository to public.

## Security and privacy

- Revoke every credential that ever appeared in the private research or study-UI
  repositories. Removing it from the new snapshot does not revoke it.
- Run `bash scripts/validate_release.sh` and an independent secret scanner on
  the exact commit that will be published.
- Confirm that notebooks, `.env` files, provider caches, model transcripts and
  person-level human-study records are absent from both the tree and Git history.
- Keep this snapshot's fresh history; do not merge or force-push the private
  research history into it.

## Scientific verification

- Verify the title, author order, affiliations, DOI and preferred citation.
- Replace example manifests with author-approved frozen manifests before making
  a paper-reproduction claim.
- Record the exact code commit, model snapshots, prompts, decoding parameters,
  trading calendar, costs and portfolio assumptions for every reported result.
- Map every paper table and figure to a command, input manifest and expected
  output checksum.

## Data and licensing

- Obtain an author/PI decision on whether any aggregate human-study artifact can
  be shared under the approved consent and ethics protocol.
- Check redistribution terms for market, news, social-media and model-output
  data; publish access instructions when raw files cannot be redistributed.
- Retain `LICENSE`, `NOTICE` and the upstream TradingAgents attribution.

## GitHub publication

- Keep the code release in `Harvard-AI-and-Robotics-Lab/TradingAgent` so the
  existing Netlify study UI and its deployment history remain separate.
- Push only this clean repository, enable branch protection and secret scanning,
  and review all GitHub Actions permissions.
- Create a signed or annotated `v0.1.0` tag and a GitHub release that states the
  release scope and known reproducibility boundaries.
- Archive the release with Zenodo (or an institutional repository), then add the
  resulting software DOI to `CITATION.cff` and the README.

## Final smoke test

From a new clone and clean virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
bash scripts/validate_release.sh
python main.py --help
trusttrade --help
```

Do not run a paid model experiment as part of a public CI workflow.
