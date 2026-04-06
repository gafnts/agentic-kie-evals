"""One-time script to upload the Kleister NDA dataset to LangSmith."""

from __future__ import annotations

import argparse
import uuid
from pathlib import Path
from typing import Any
from uuid import UUID

import polars as pl
from dotenv import load_dotenv
from langsmith import Client

STATIC_DIR = Path(__file__).parents[2] / "data" / "kleister-nda"

PARTITIONS: dict[str, str] = {
    "train": "train",
    "dev-0": "dev",
    "test-A": "test",
}

NAMESPACE_UUID = uuid.UUID("fcd0fe34-475f-4e1a-819a-85dde6f2fa71")

DEFAULT_DATASET_NAME = "kleister-nda"
DEFAULT_BATCH_SIZE = 20


def read_partition(partition_dir: str, split_name: str) -> list[dict[str, Any]]:
    """Read a partition's parquet and build LangSmith example dicts."""
    partition_path = STATIC_DIR / partition_dir
    parquet_path = partition_path / "data.parquet"
    documents_path = partition_path / "documents"

    df = pl.read_parquet(parquet_path)

    examples: list[dict[str, Any]] = []
    for row in df.iter_rows(named=True):
        filename: str = row["filename"]
        pdf_path = documents_path / filename

        if not pdf_path.exists():
            print(f"  WARNING: PDF not found for {filename}, skipping")
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
    """Return dataset ID, creating the dataset if needed."""
    if recreate:
        try:
            client.delete_dataset(dataset_name=dataset_name)
            print(f"Deleted existing dataset: {dataset_name}")
        except Exception:
            pass

    try:
        dataset = client.create_dataset(  # pyright: ignore[reportUnknownMemberType]
            dataset_name=dataset_name,
            description=(
                "Kleister NDA dataset (Applica AI). "
                "254 train / 83 dev / 203 test NDA documents "
                "from SEC Edgar. Four entity types: "
                "effective_date, jurisdiction, party, term."
            ),
        )
        print(f"Created dataset: {dataset_name} ({dataset.id})")
        return dataset.id
    except Exception:
        dataset = client.read_dataset(dataset_name=dataset_name)
        print(f"Using existing dataset: {dataset_name} ({dataset.id})")
        return UUID(str(dataset.id))


def upload_partition(
    client: Client,
    dataset_id: UUID | str,
    examples: list[dict[str, Any]],
    batch_size: int = DEFAULT_BATCH_SIZE,
    dry_run: bool = False,
) -> None:
    """Upload examples in batches to LangSmith."""
    total = len(examples)

    for i in range(0, total, batch_size):
        batch = examples[i : i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size

        if dry_run:
            print(
                f"  [DRY RUN] Batch {batch_num}/{total_batches}: {len(batch)} examples"
            )
            continue

        client.create_examples(  # pyright: ignore[reportUnknownMemberType]
            dataset_id=dataset_id,
            examples=batch,
            dangerously_allow_filesystem=True,
        )
        print(f"  Uploaded batch {batch_num}/{total_batches}: {len(batch)} examples")


def main() -> None:
    load_dotenv()

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
    args = parser.parse_args()

    client = Client()

    dataset_id: UUID | str
    if args.dry_run:
        print(f"[DRY RUN] Would create dataset: {args.dataset_name}")
        dataset_id = "dry-run"
    else:
        dataset_id = get_or_create_dataset(
            client, args.dataset_name, recreate=args.recreate
        )

    for partition_dir in args.partitions:
        split_name = PARTITIONS[partition_dir]
        print(f"\nProcessing partition: {partition_dir} → split: {split_name}")

        examples = read_partition(partition_dir, split_name)
        print(f"  Found {len(examples)} examples")

        upload_partition(
            client=client,
            dataset_id=dataset_id,
            examples=examples,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
