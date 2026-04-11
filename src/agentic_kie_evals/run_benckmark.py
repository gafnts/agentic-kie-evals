"""
Benchmark runner for the Kleister NDA extraction evaluation.

Runs the experiment matrix (model x strategy x modality) against the
LangSmith dataset and scores each run with the evaluators defined in
evaluators.py.

Two model tiers are available via the --tier argument:
  - lite: Cost-optimised models (claude-haiku-4-5, gemini-2.5-flash, gpt-5.4-mini)
  - standard: Full-capability models (claude-sonnet-4-6, gemini-2.5-pro, gpt-5.4)
"""

from __future__ import annotations

import argparse
import logging
import tempfile
from collections.abc import Callable
from itertools import islice
from pathlib import Path
from typing import Any, Literal, cast

from agentic_kie.extractors import AgenticExtractor, SinglePassExtractor
from agentic_kie.loader import PDFLoader
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langsmith import Client, evaluate
from nda import NDA
from rich.logging import RichHandler
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn

from .evaluators import ALL_EVALUATORS

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
)
logger = logging.getLogger(__name__)


DATASET_NAME = "kleister-nda"

TIERS: dict[str, dict[str, Callable[[], BaseChatModel]]] = {
    "lite": {
        "claude": lambda: ChatAnthropic(model="claude-haiku-4-5"),  # type: ignore[call-arg]
        "gemini": lambda: ChatGoogleGenerativeAI(model="gemini-2.5-flash"),
        "gpt": lambda: ChatOpenAI(model="gpt-5.4-mini"),
    },
    "standard": {
        "claude": lambda: ChatAnthropic(model="claude-sonnet-4-6"),  # type: ignore[call-arg]
        "gemini": lambda: ChatGoogleGenerativeAI(model="gemini-2.5-pro"),
        "gpt": lambda: ChatOpenAI(model="gpt-5.4"),
    },
}

SINGLE_PASS_MODALITIES = ("text", "image")
AGENTIC_MODALITIES = ("multimodal",)


def make_target(
    extractor: SinglePassExtractor[NDA] | AgenticExtractor[NDA],
) -> Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]:
    """
    Create a LangSmith target function that captures the extractor.

    The target receives (inputs, attachments) as positional args. This
    is a LangSmith requirement for attachment-based evaluation. It reads
    the PDF bytes from the attachment, writes them to a temp file (since
    PDFLoader accepts a Path), runs extraction, and returns the result
    dict augmented with document metadata for downstream analysis.
    """

    def target(inputs: dict[str, Any], attachments: dict[str, Any]) -> dict[str, Any]:
        pdf_bytes = attachments["document"]["reader"].read()

        # PDFLoader requires a file path, so write bytes to a temp file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(pdf_bytes)
            tmp_path = Path(f.name)

        try:
            loader = PDFLoader()
            document = loader.load(tmp_path)
            result = extractor.extract(document)

            return {
                **result.model_dump(),
                "_page_count": document.page_count,
                "_char_count": len(document.full_text),
            }
        finally:
            tmp_path.unlink(missing_ok=True)

    return target


def run_experiment(
    extractor: SinglePassExtractor[NDA] | AgenticExtractor[NDA],
    *,
    model_name: str,
    strategy: str,
    modality: str = "n/a",
    splits: list[str],
    max_concurrency: int = 4,
    limit: int | None = None,
) -> None:
    """
    Run a single experiment against the LangSmith dataset.
    """
    experiment_prefix = f"{model_name}-{strategy}-{modality}"

    metadata = {
        "model_name": model_name,
        "strategy": strategy,
        "modality": modality,
    }

    client = Client()

    logger.info("Starting experiment: %s", experiment_prefix)

    evaluate(
        make_target(extractor),
        data=list(
            islice(
                client.list_examples(
                    dataset_name=DATASET_NAME,
                    splits=splits,
                    include_attachments=True,
                ),
                limit,
            )
        ),
        evaluators=ALL_EVALUATORS,
        experiment_prefix=experiment_prefix,
        metadata=metadata,
        max_concurrency=max_concurrency,
    )

    logger.info("Completed experiment: %s", experiment_prefix)


def build_experiment_matrix(
    tier: str,
    model_filter: str | None = None,
    strategy_filter: str | None = None,
    modality_filter: str | None = None,
) -> list[dict[str, str]]:
    """
    Build the list of experiments to run, optionally filtered.
    """
    experiments: list[dict[str, str]] = []

    for model_name in TIERS[tier]:
        if model_filter and model_name != model_filter:
            continue

        # Single-pass: model × modality
        if strategy_filter is None or strategy_filter == "single_pass":
            for modality in SINGLE_PASS_MODALITIES:
                if modality_filter and modality != modality_filter:
                    continue
                experiments.append(
                    {
                        "model_name": model_name,
                        "strategy": "single_pass",
                        "modality": modality,
                    }
                )

        # Agentic: model × modality
        if strategy_filter is None or strategy_filter == "agentic":
            for modality in AGENTIC_MODALITIES:
                if modality_filter and modality != modality_filter:
                    continue
                experiments.append(
                    {
                        "model_name": model_name,
                        "strategy": "agentic",
                        "modality": modality,
                    }
                )

    return experiments


def make_extractor(
    model_name: str, strategy: str, modality: str, tier: str
) -> SinglePassExtractor[NDA] | AgenticExtractor[NDA]:
    """
    Instantiate the appropriate extractor for an experiment.
    """
    model = TIERS[tier][model_name]()

    if strategy == "single_pass":
        return SinglePassExtractor(
            model=model,
            schema=NDA,
            modality=cast(Literal["text", "image", "multimodal"], modality),
        )
    elif strategy == "agentic":
        return AgenticExtractor(
            model=model,
            schema=NDA,
            modality=cast(Literal["text", "image", "multimodal"], modality),
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Kleister NDA extraction benchmark experiments.",
    )
    parser.add_argument(
        "--tier",
        choices=list(TIERS.keys()),
        default="lite",
        help="Model tier to use. Default: lite.",
    )
    parser.add_argument(
        "--model",
        choices=list(TIERS["lite"].keys()),
        default=None,
        help="Run only this model. Default: all models.",
    )
    parser.add_argument(
        "--strategy",
        choices=["single_pass", "agentic"],
        default=None,
        help="Run only this strategy. Default: both.",
    )
    parser.add_argument(
        "--modality",
        choices=["text", "image", "multimodal"],
        default=None,
        help="Run only this modality (single_pass only). Default: both.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "dev", "test"],
        default="train",
        help="Dataset split to evaluate against. Default: train.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=4,
        help="Max concurrent evaluations. Default: 4.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List experiments without executing.",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Max examples to evaluate."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    experiments = build_experiment_matrix(
        tier=args.tier,
        model_filter=args.model,
        strategy_filter=args.strategy,
        modality_filter=args.modality,
    )

    if not experiments:
        logger.warning("No experiments match the provided filters.")
        return

    logger.info(
        "Tier: %s | Experiment matrix: %d experiment(s)", args.tier, len(experiments)
    )
    for i, exp in enumerate(experiments, 1):
        logger.info(
            "  [%d] %s / %s / %s",
            i,
            exp["model_name"],
            exp["strategy"],
            exp["modality"],
        )

    if args.dry_run:
        logger.info("Dry run: Exiting without executing.")
        return

    splits = [args.split]

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
    ) as progress:
        task = progress.add_task("Running experiments", total=len(experiments))

        for exp in experiments:
            extractor = make_extractor(
                exp["model_name"], exp["strategy"], exp["modality"], args.tier
            )
            run_experiment(
                extractor,
                model_name=exp["model_name"],
                strategy=exp["strategy"],
                modality=exp["modality"],
                splits=splits,
                max_concurrency=args.max_concurrency,
                limit=args.limit,
            )
            progress.advance(task)

    logger.info("All experiments complete.")


if __name__ == "__main__":
    main()
