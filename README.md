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

1. Dry run (print the experiment matrix without making any API calls)
```bash
uv run python -m agentic_kie_evals.run_benchmark --dry-run
```

2. Single quick test (one model / strategy, 10 examples)
```bash
uv run python -m agentic_kie_evals.run_benchmark \
    --tier lite --model gemini --strategy single_pass --limit 10
```

3. Full matrix, lite tier (cost-optimised models) on the train split
```bash
uv run python -m agentic_kie_evals.run_benchmark
```

4. Full matrix, standard tier (full-capability models) on the dev split
```bash
uv run python -m agentic_kie_evals.run_benchmark \
    --tier standard --split dev
```

### CLI reference

| Flag | Choices | Default | Description |
|---|---|---|---|
| `--tier` | `lite`, `standard`, `flagship` | `lite` | Model tier: cost-optimised, full-capability, or top-capability |
| `--model` | `claude`, `gemini`, `gpt` | all | Restrict to a single model |
| `--strategy` | `single_pass`, `agentic` | both | Restrict to a single extraction strategy |
| `--split` | `train`, `dev`, `test` | `dev` | Dataset split to evaluate against |
| `--limit` | int | none | Cap the number of examples evaluated |
| `--max-concurrency` | int | `3` | Max concurrent evaluations |
| `--max-retries` | int | `6` | Max retries per extractor call |
| `--dry-run` | — | false | Print the experiment matrix and exit |

Modalities are configured via `SINGLE_PASS_MODALITIES` and `AGENTIC_MODALITIES` in `run_benchmark.py`.

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
