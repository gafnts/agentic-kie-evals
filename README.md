# agentic-kie-evals

[![CI](https://github.com/gafnts/agentic-kie-evals/actions/workflows/ci.yml/badge.svg)](https://github.com/gafnts/agentic-kie-evals/actions/workflows/ci.yml)
[![codecov](https://codecov.io/github/gafnts/agentic-kie-evals/graph/badge.svg)](https://codecov.io/github/gafnts/agentic-kie-evals)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

Benchmarking single-pass and agentic extraction strategies across LLM providers on the Kleister NDA dataset

## Installation

Requires Python 3.13 or later. Dependencies are managed with [uv](https://docs.astral.sh/uv/).


```bash
git clone https://github.com/gafnts/agentic-kie-evals.git
cd agentic-kie-evals
make install
```


## Dataset

This project uses the [Kleister NDA](https://github.com/applicaai/kleister-nda) dataset from Applica AI: 254 train / 83 dev / 203 test NDA documents sourced from SEC Edgar, annotated with four entity types: `effective_date`, `jurisdiction`, `party`, and `term`.

The dataset is hosted in [LangSmith](https://smith.langchain.com/) for evaluation. A `LANGCHAIN_API_KEY` environment variable is required to interact with it.

### Uploading the dataset

```bash
# Dry run (validates parquet files and PDF paths, no API calls)
uv run python -m agentic_kie_evals.upload_dataset --dry-run

# Upload all partitions
uv run python -m agentic_kie_evals.upload_dataset

# Upload specific partitions
uv run python -m agentic_kie_evals.upload_dataset --partitions train dev-0

# Delete and recreate the dataset from scratch
uv run python -m agentic_kie_evals.upload_dataset --recreate
```

The upload script is idempotent: re-running it will reuse an existing dataset and deterministic example IDs prevent duplicates.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, available `make` targets, and the CI/CD pipeline.
