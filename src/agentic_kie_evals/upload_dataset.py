"""
Uploads the Kleister NDA dataset to LangSmith.

Reads the preprocessed Parquet files and PDF documents produced by the
kleister-nda-preparation package and creates a LangSmith dataset with
one example per document. Each example includes the structured labels
as outputs and the PDF as an attachment.

Partition-to-split mapping:
    train: train
    dev-0: dev
    test-A: test

The script is idempotent: it reuses an existing dataset and derives
deterministic example IDs from filenames, so re-running it will not
create duplicates.
"""

from __future__ import annotations

import argparse
import logging
import uuid
from pathlib import Path
from typing import Any
from uuid import UUID

import polars as pl
from dotenv import load_dotenv
from langsmith import Client
from rich.logging import RichHandler
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
)
logger = logging.getLogger(__name__)


STATIC_DIR = Path(__file__).parents[2] / "data" / "kleister-nda"

PARTITIONS: dict[str, str] = {
    "train": "train",
    "dev-0": "dev",
    "test-A": "test",
}

NAMESPACE_UUID = UUID("fcd0fe34-475f-4e1a-819a-85dde6f2fa71")
DEFAULT_DATASET_NAME = "kleister-nda"
DEFAULT_BATCH_SIZE = 20


def read_partition(partition_dir: str, split_name: str) -> list[dict[str, Any]]:
    """
    Read a partition's parquet and build LangSmith example dicts.
    """
    partition_path = STATIC_DIR / partition_dir
    parquet_path = partition_path / "data.parquet"
    documents_path = partition_path / "documents"

    df = pl.read_parquet(parquet_path)

    examples: list[dict[str, Any]] = []
    for row in df.iter_rows(named=True):
        filename: str = row["filename"]
        pdf_path = documents_path / filename

        if not pdf_path.exists():
            logger.warning("PDF not found for %s, skipping", filename)
            continue

        if "labels_schema" in row:
            labels: dict[str, Any] = row["labels_schema"]
            outputs = {
                "effective_date": labels.get("effective_date"),
                "jurisdiction": labels.get("jurisdiction"),
                "party": labels.get("party"),
                "term": labels.get("term"),
            }
        else:
            outputs = {
                "effective_date": None,
                "jurisdiction": None,
                "party": None,
                "term": None,
            }

        example: dict[str, Any] = {
            "id": uuid.uuid5(NAMESPACE_UUID, filename),
            "inputs": {"document_id": filename},
            "outputs": outputs,
            "attachments": {
                "document": {
                    "mime_type": "application/pdf",
                    "data": pdf_path,
                },
            },
            "metadata": {
                "partition": partition_dir,
                "source": "kleister-nda",
            },
            "split": split_name,
        }
        examples.append(example)

    return examples


def get_or_create_dataset(
    client: Client,
    dataset_name: str,
    *,
    recreate: bool = False,
) -> UUID:
    """
    Return dataset ID, creating the dataset if needed.
    """
    if recreate:
        try:
            client.delete_dataset(dataset_name=dataset_name)
            logger.info("Deleted existing dataset: %s", dataset_name)
        except Exception:
            pass

    try:
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description="Kleister NDA dataset for KIE evaluation",
        )
        logger.info("Created dataset: %s (%s)", dataset_name, dataset.id)
        return dataset.id
    except Exception:
        dataset = client.read_dataset(dataset_name=dataset_name)
        logger.info("Using existing dataset: %s (%s)", dataset_name, dataset.id)
        return UUID(str(dataset.id))


def upload_partition(
    client: Client,
    dataset_id: UUID | str,
    examples: list[dict[str, Any]],
    batch_size: int = DEFAULT_BATCH_SIZE,
    dry_run: bool = False,
) -> None:
    """
    Upload examples in batches to LangSmith.
    """
    total = len(examples)
    total_batches = (total + batch_size - 1) // batch_size

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("Uploading batches", total=total_batches)

        for i in range(0, total, batch_size):
            batch = examples[i : i + batch_size]

            if not dry_run:
                client.create_examples(
                    dataset_id=dataset_id,
                    examples=batch,
                    dangerously_allow_filesystem=True,
                )

            progress.advance(task)

    verb = "[DRY RUN] Would upload" if dry_run else "Uploaded"
    logger.info("%s %d examples in %d batches", verb, total, total_batches)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload Kleister NDA dataset to LangSmith"
    )
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_DATASET_NAME,
        help=f"LangSmith dataset name (default: {DEFAULT_DATASET_NAME})",
    )
    parser.add_argument(
        "--partitions",
        nargs="+",
        choices=list(PARTITIONS.keys()),
        default=list(PARTITIONS.keys()),
        help="Partitions to upload (default: all)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Upload batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate the dataset if it already exists",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate without uploading",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = Client()
    dataset_id: UUID | str

    if args.dry_run:
        logger.info("Dry run mode. Would create dataset: %s", args.dataset_name)
        dataset_id = "dry-run"
    else:
        dataset_id = get_or_create_dataset(
            client, args.dataset_name, recreate=args.recreate
        )

    for partition_dir in args.partitions:
        split_name = PARTITIONS[partition_dir]
        logger.info("Processing partition: %s → split: %s", partition_dir, split_name)

        examples = read_partition(partition_dir, split_name)
        logger.info("Found %d examples", len(examples))

        upload_partition(
            client=client,
            dataset_id=dataset_id,
            examples=examples,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
