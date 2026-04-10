# Agentic KIE Evals

[![CI](https://github.com/gafnts/agentic-kie-evals/actions/workflows/ci.yml/badge.svg)](https://github.com/gafnts/agentic-kie-evals/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

Evaluation suite for [Agentic KIE](https://github.com/gafnts/agentic-kie), benchmarking its single-pass and agentic strategies across LLM providers on the Kleister NDA dataset.

## Installation

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/). See [CONTRIBUTING.md](CONTRIBUTING.md) for full setup instructions, available `make` targets, and environment configuration.

## Dataset

This project uses the [Kleister NDA](https://github.com/applicaai/kleister-nda) dataset from Applica AI, which consists of NDA documents sourced from SEC Edgar, annotated with four entity types: `effective_date`, `jurisdiction`, `party`, and `term`.

Dataset preprocessing and delivery is handled by the Python package [kleister-nda-preparation](https://github.com/gafnts/kleister-nda-preparation). The preparation pipeline reads the original TSV partitions, transforms raw labels into structured records validated against a Pydantic schema, relocates the corresponding PDF documents, and writes the results as partitioned Parquet files.

> This step runs automatically as part of `make install`.

### Uploading the dataset to LangSmith

Before running the benchmark, the preprocessed Parquet files and their PDF attachments need to be uploaded to [LangSmith](https://smith.langchain.com/). The `upload_dataset.py` module supports several behaviors:

1. Dry run (validates parquet files and PDF paths, no API calls)
```bash
uv run python -m agentic_kie_evals.upload_dataset --dry-run
```

2. Upload all partitions
```bash
uv run python -m agentic_kie_evals.upload_dataset
```

3. Upload specific partitions
```bash
uv run python -m agentic_kie_evals.upload_dataset --partitions train dev-0
```

4. Delete and recreate the dataset from scratch
```bash
uv run python -m agentic_kie_evals.upload_dataset --recreate
```

> The upload script is idempotent: re-running it is safe. It reuses an existing dataset and deterministic example IDs prevent duplicates.

## Running the benchmark

The benchmark runner evaluates the full experiment matrix (`model × strategy × modality`) against the LangSmith dataset. Each run is scored by the evaluators and logged back to LangSmith under the prefix `{model}--{strategy}--{modality}`.

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

The `agentic` strategy does not accept a modality parameter and is always run without it.

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

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow, available `make` targets, and the CI pipeline.
