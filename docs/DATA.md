# Data, privacy and redistribution

## What is included

This repository releases source code, safe experiment templates and plotting
utilities. It does not include raw API responses, private caches, participant
records or credentials.

## Human-study data

The paper reports an online study with 19 human annotators. The research branch
contained person-level identifiers, demographic attributes, rationales and
performance summaries. Those records are deliberately excluded from this
public tree.

Any future human-data release must be independently reviewed for:

- informed-consent and IRB/ethics compatibility;
- de-identification and re-identification risk;
- removal of free-text disclosures and direct or indirect identifiers;
- a documented retention, withdrawal and access policy; and
- an explicit data license distinct from the software license.

Aggregate statistics should be published only when cell sizes and combinations
of attributes do not make participants identifiable.

## Market, news and social data

The code can retrieve or consume data from services such as Yahoo Finance,
Finnhub, Google News and Reddit. Their terms govern access and redistribution.
Apache-2.0 applies to the source code, not automatically to retrieved content.

For a reproducible historical run, create an immutable, time-bounded cache with
a manifest containing:

- source and retrieval timestamp;
- observation-time cutoff used for every trading date;
- file size and SHA-256 digest;
- applicable license or terms; and
- a leakage audit confirming no post-decision information is present.

Keep private or non-redistributable files under `data/private/` or another
ignored external path.

## Model outputs

Model outputs may contain copyrighted source summaries, personal information or
provider-specific restrictions. Review outputs before redistribution and record
the provider, model snapshot, request date and prompt revision.
