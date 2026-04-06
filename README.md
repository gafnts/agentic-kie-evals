# Agentic KIE Evals

[![CI](https://github.com/gafnts/agentic-kie-evals/actions/workflows/ci.yml/badge.svg)](https://github.com/gafnts/agentic-kie-evals/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

Evaluation suite for [Agentic KIE](https://github.com/gafnts/agentic-kie), benchmarking its single-pass and agentic strategies across LLM providers on the Kleister NDA dataset.

## Installation

Requires Python 3.13 or later. Dependencies are managed with [uv](https://docs.astral.sh/uv/).


```bash
git clone https://github.com/gafnts/agentic-kie-evals.git
cd agentic-kie-evals
make install
```


## Dataset

This project uses the [Kleister NDA](https://github.com/applicaai/kleister-nda) dataset from Applica AI, which consists of NDA documents sourced from SEC Edgar, annotated with four entity types: `effective_date`, `jurisdiction`, `party`, and `term`.

Dataset preprocessing and delivery is handled by [kleister-nda-preparation](https://github.com/gafnts/kleister-nda-preparation). The preparation pipeline reads the original TSV partitions, transforms raw labels into structured records validated against a Pydantic schema, relocates the corresponding PDF documents, and writes the results as partitioned Parquet files. This step runs automatically as part of `make install`.

### Uploading the dataset to LangSmith

The dataset is hosted in [LangSmith](https://smith.langchain.com/) for evaluation. A `LANGCHAIN_API_KEY` environment variable is required to interact with it.

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

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, available `make` targets, and the CI pipeline.
