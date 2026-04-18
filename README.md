<h1 align="center">Agentic KIE Evals</h1>
<p align="center">
  Benchmarking single-pass and agentic extraction strategies across LLM providers on the Kleister NDA dataset.
</p>
<p align="center">
<a href="https://github.com/gafnts/agentic-kie-evals/actions/workflows/ci.yml"><img src="https://github.com/gafnts/agentic-kie-evals/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License"></a>
</p>

---

<p align="center">
Extracting structured fields from legal documents is deceptively hard. This project measures how well modern LLMs handle that on real NDA documents from the SEC Edgar database. The benchmark covers three model families (Claude, Gemini, and GPT) and scores each run in LangSmith using exact and fuzzy F1 evaluators.
</p>

## Contents

- [Dataset](#dataset)
- [Running the benchmark](#running-the-benchmark)
- [Evaluators](#evaluators)
- [Contributing](#contributing)

---

## Dataset

This project uses the [Kleister NDA](https://github.com/applicaai/kleister-nda) dataset from Applica AI, which consists of NDA documents sourced from SEC Edgar, annotated with four entity types: `effective_date`, `jurisdiction`, `party`, and `term`.

Dataset preprocessing and delivery is handled by the Python package [kleister-nda-preparation](https://github.com/gafnts/kleister-nda-preparation). The preparation pipeline reads the original TSV partitions, transforms raw labels into structured records validated against a Pydantic schema, relocates the corresponding PDF documents, and writes the results as partitioned Parquet files.

> [!NOTE]
> This step runs automatically as part of `make install`.

### Uploading the dataset to LangSmith

Before running the benchmark, the preprocessed Parquet files and their PDF attachments need to be uploaded to [LangSmith](https://smith.langchain.com/). The [upload_dataset.py](src/agentic_kie_evals/upload_dataset.py) module supports several behaviors:

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

> [!TIP]
> The upload script is idempotent: re-running it is safe. It reuses an existing dataset and deterministic example IDs prevent duplicates.

---

## Running the benchmark

The benchmark runner evaluates the full experiment matrix (`model × strategy × modality`) against the LangSmith dataset. Each run is scored by the evaluators and logged back to LangSmith.

1. Dry run (print the experiment matrix without making any API calls)
```bash
uv run python -m agentic_kie_evals.run_benchmark --dry-run
```

2. Single quick test (one model, one strategy, 10 examples)
```bash
uv run python -m agentic_kie_evals.run_benchmark \
    --tier lite --model gemini --strategy single_pass --limit 10
```

3. Full matrix, lite tier (cost-optimised models) on the dev split
```bash
uv run python -m agentic_kie_evals.run_benchmark
```

4. Full matrix, standard tier (full-capability models) on the dev split
```bash
uv run python -m agentic_kie_evals.run_benchmark --tier standard
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

> [!NOTE]
> Modalities are configured via `SINGLE_PASS_MODALITIES` and `AGENTIC_MODALITIES` in [run_benchmark.py](src/agentic_kie_evals/run_benchmark.py).

---

## Evaluators

Evaluators live in [evaluators.py](src/agentic_kie_evals/evaluators.py) and follow the LangSmith custom evaluator signature `(outputs, reference_outputs) -> {"key": str, "score": float}`.

| Evaluator | Field | Method | Score |
|---|---|---|---|
| `exact_effective_date_f1` | `effective_date` | Exact match | 0 or 1 |
| `exact_jurisdiction_f1` | `jurisdiction` | Exact match | 0 or 1 |
| `fuzzy_jurisdiction_f1` | `jurisdiction` | SequenceMatcher ≥ 0.85 | 0 or 1 |
| `exact_term_f1` | `term` | Exact match | 0 or 1 |
| `fuzzy_term_f1` | `term` | SequenceMatcher ≥ 0.85 | 0 or 1 |
| `exact_party_f1` | `party` | Set F1, exact string | 0–1 continuous |
| `fuzzy_party_f1` | `party` | Set F1, SequenceMatcher ≥ 0.85 | 0–1 continuous |
| `exact_f1` | all fields | Macro-average of exact F1 scores | 0–1 continuous |
| `fuzzy_f1` | all fields | Macro-average of fuzzy F1 scores | 0–1 continuous |

Normalization (lowercasing, whitespace trimming, trailing-period stripping) is applied to both sides before comparison.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow, available `make` targets, and the CI pipeline.
