"""
Benchmark runner for the Kleister NDA extraction evaluation.

Runs the experiment matrix (model x strategy x modality) against the
LangSmith dataset and scores each run with the evaluators defined in
evaluators.py.
"""

import argparse
import logging
import tempfile
from collections.abc import Callable
from itertools import islice
from pathlib import Path
from typing import Any, Literal, cast

from agentic_kie.extractors.agent import AgenticExtractor
from agentic_kie.extractors.single_pass import SinglePassExtractor
from agentic_kie.loader import PDFLoader
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langsmith import Client, evaluate
from nda import NDA

from .evaluators import ALL_EVALUATORS

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DATASET_NAME = "kleister-nda"

MODELS: dict[str, Callable[[], BaseChatModel]] = {
    "claude-haiku": lambda: ChatAnthropic(model="claude-haiku-4-5"),  # type: ignore[call-arg]
    "gemini-flash": lambda: ChatGoogleGenerativeAI(model="gemini-2.5-flash"),
    "gpt": lambda: ChatOpenAI(model="gpt-4.1-mini"),
}

SINGLE_PASS_MODALITIES = ("text", "multimodal")


def make_target(
    extractor: SinglePassExtractor[NDA] | AgenticExtractor[NDA],
) -> Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]:
    """
    Create a LangSmith target function that captures the extractor.

    The target receives (inputs, attachments) as positional args — this
    is a LangSmith requirement for attachment-based evaluation. It reads
    the PDF bytes from the attachment, writes them to a temp file (since
    PDFLoader accepts a Path), runs extraction, and returns the result
    dict augmented with document metadata for downstream analysis.
    """

    def target(inputs: dict[str, Any], attachments: dict[str, Any]) -> dict[str, Any]:
        print(f"attachment keys: {list(attachments.keys())}")
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
    experiment_prefix = f"{model_name}--{strategy}--{modality}"

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
    model_filter: str | None = None,
    strategy_filter: str | None = None,
    modality_filter: str | None = None,
) -> list[dict[str, str]]:
    """
    Build the list of experiments to run, optionally filtered.
    """
    experiments: list[dict[str, str]] = []

    for model_name in MODELS:
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

        # Agentic: model only (no modality param)
        if (
            strategy_filter is None or strategy_filter == "agentic"
        ) and modality_filter is None:
            experiments.append(
                {
                    "model_name": model_name,
                    "strategy": "agentic",
                    "modality": "n/a",
                }
            )

    return experiments


def make_extractor(
    model_name: str, strategy: str, modality: str
) -> SinglePassExtractor[NDA] | AgenticExtractor[NDA]:
    """
    Instantiate the appropriate extractor for an experiment.
    """
    model = MODELS[model_name]()

    if strategy == "single_pass":
        return SinglePassExtractor(
            model=model,
            schema=NDA,
            modality=cast(Literal["text", "image", "multimodal"], modality),
        )
    elif strategy == "agentic":
        return AgenticExtractor(model=model, schema=NDA)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Kleister NDA extraction benchmark experiments.",
    )
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
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
        choices=["text", "multimodal"],
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
        model_filter=args.model,
        strategy_filter=args.strategy,
        modality_filter=args.modality,
    )

    if not experiments:
        logger.warning("No experiments match the provided filters.")
        return

    logger.info("Experiment matrix: %d experiment(s)", len(experiments))
    for i, exp in enumerate(experiments, 1):
        logger.info(
            "  [%d] %s / %s / %s",
            i,
            exp["model_name"],
            exp["strategy"],
            exp["modality"],
        )

    if args.dry_run:
        logger.info("Dry run — exiting without executing.")
        return

    splits = [args.split]

    for exp in experiments:
        extractor = make_extractor(exp["model_name"], exp["strategy"], exp["modality"])
        run_experiment(
            extractor,
            model_name=exp["model_name"],
            strategy=exp["strategy"],
            modality=exp["modality"],
            splits=splits,
            max_concurrency=args.max_concurrency,
            limit=args.limit,
        )

    logger.info("All experiments complete.")


if __name__ == "__main__":
    main()
