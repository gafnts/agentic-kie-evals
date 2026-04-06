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

## Running the Benchmark

Requires `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, or `OPENAI_API_KEY` depending on the models being evaluated, plus a `LANGCHAIN_API_KEY` to read from and write results to LangSmith.

```bash
# Dry run — see what would execute without making API calls
uv run python -m agentic_kie_evals.run_benckmark --dry-run

# Single quick test (one model / strategy / modality, 10 examples)
uv run python -m agentic_kie_evals.run_benckmark \
    --model claude-haiku --strategy single_pass --modality text \
    --max-concurrency 1 --limit 10

# Full matrix on the train split (default)
uv run python -m agentic_kie_evals.run_benckmark

# Final benchmark on the dev split
uv run python -m agentic_kie_evals.run_benckmark --split dev
```

### CLI reference

| Flag | Choices | Default | Description |
|---|---|---|---|
| `--model` | `claude-haiku`, `gemini-flash`, `gpt` | all | Restrict to a single model |
| `--strategy` | `single_pass`, `agentic` | both | Restrict to a single extraction strategy |
| `--modality` | `text`, `multimodal` | both | Restrict to a single modality (single-pass only) |
| `--split` | `train`, `dev`, `test` | `train` | Dataset split to evaluate against |
| `--max-concurrency` | int | `4` | Max concurrent evaluations |
| `--limit` | int | none | Cap the number of examples evaluated |
| `--dry-run` | — | false | Print the experiment matrix and exit |

The experiment matrix is `model × strategy × modality`. The `agentic` strategy does not accept a modality parameter and is always run without it. Each experiment is logged to LangSmith under the prefix `{model}--{strategy}--{modality}`.

## Evaluators

Evaluators live in `agentic_kie_evals.evaluators` and follow the LangSmith custom evaluator signature `(outputs, reference_outputs) -> {"key": str, "score": float}`. Normalization (lowercasing, whitespace trimming, trailing-period stripping) is applied to both sides before comparison.

| Evaluator | Field | Method | Score |
|---|---|---|---|
| `exact_effective_date` | `effective_date` | Exact match | 0 or 1 |
| `exact_jurisdiction` | `jurisdiction` | Exact match | 0 or 1 |
| `fuzzy_jurisdiction` | `jurisdiction` | SequenceMatcher ≥ 0.85 | 0 or 1 |
| `exact_term` | `term` | Exact match | 0 or 1 |
| `fuzzy_term` | `term` | SequenceMatcher ≥ 0.85 | 0 or 1 |
| `exact_party` | `party` | Set F1, exact string | 0–1 continuous |
| `fuzzy_party` | `party` | Set F1, SequenceMatcher ≥ 0.85 | 0–1 continuous |

`exact_party` and `fuzzy_party` compute precision and recall independently over the set of party names, then return their F1. Both `None` predictions and `None` references on scalar fields score 1.0 (true negative); a mismatch scores 0.0.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, available `make` targets, and the CI pipeline.
