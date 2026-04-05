"""Tests for the upload_dataset module."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import polars as pl
import pytest
from pytest import CaptureFixture

from agentic_kie_evals.upload_dataset import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DATASET_NAME,
    NAMESPACE_UUID,
    PARTITIONS,
    get_or_create_dataset,
    read_partition,
    upload_partition,
)


@pytest.fixture
def fake_partition(tmp_path: Path) -> Path:
    """Create a fake partition directory with a parquet file and PDF stubs."""
    partition = tmp_path / "train"
    docs = partition / "documents"
    docs.mkdir(parents=True)

    # Create two PDFs
    (docs / "doc_a.pdf").write_bytes(b"%PDF-fake-a")
    (docs / "doc_b.pdf").write_bytes(b"%PDF-fake-b")
    # doc_c.pdf intentionally missing to test the skip-warning branch

    df = pl.DataFrame(
        {
            "filename": ["doc_a.pdf", "doc_b.pdf", "doc_c.pdf"],
            "labels_schema": [
                {
                    "effective_date": "2024-01-01",
                    "jurisdiction": "Delaware",
                    "party": "Acme Inc.",
                    "term": "2 years",
                },
                {
                    "effective_date": "2024-06-15",
                    "jurisdiction": "California",
                    "party": "Beta Corp.",
                    "term": "1 year",
                },
                {
                    "effective_date": None,
                    "jurisdiction": None,
                    "party": None,
                    "term": None,
                },
            ],
        }
    )
    df.write_parquet(partition / "data.parquet")
    return tmp_path


@pytest.fixture
def fake_partition_no_labels(tmp_path: Path) -> Path:
    """Partition whose parquet has no labels_schema column (test-A split)."""
    partition = tmp_path / "test-A"
    docs = partition / "documents"
    docs.mkdir(parents=True)

    (docs / "doc_x.pdf").write_bytes(b"%PDF-fake-x")

    df = pl.DataFrame({"filename": ["doc_x.pdf"]})
    df.write_parquet(partition / "data.parquet")
    return tmp_path


@pytest.fixture
def mock_client() -> MagicMock:
    return MagicMock(
        spec=["create_dataset", "read_dataset", "delete_dataset", "create_examples"]
    )


class TestReadPartition:
    def test_reads_examples_with_labels(self, fake_partition: Path) -> None:
        with patch("agentic_kie_evals.upload_dataset.STATIC_DIR", fake_partition):
            examples = read_partition("train", "train")

        # doc_c.pdf is missing on disk -> skipped
        assert len(examples) == 2

        ex = examples[0]
        assert ex["id"] == uuid.uuid5(NAMESPACE_UUID, "doc_a.pdf")
        assert ex["inputs"] == {"document_id": "doc_a.pdf"}
        assert ex["outputs"]["effective_date"] == "2024-01-01"
        assert ex["outputs"]["jurisdiction"] == "Delaware"
        assert ex["outputs"]["party"] == "Acme Inc."
        assert ex["outputs"]["term"] == "2 years"
        assert ex["split"] == "train"
        assert ex["metadata"] == {"partition": "train", "source": "kleister-nda"}
        assert ex["attachments"]["document"]["mime_type"] == "application/pdf"

    def test_skips_missing_pdf(
        self, fake_partition: Path, capsys: CaptureFixture[str]
    ) -> None:
        with patch("agentic_kie_evals.upload_dataset.STATIC_DIR", fake_partition):
            examples = read_partition("train", "train")

        assert len(examples) == 2
        captured = capsys.readouterr()
        assert "doc_c.pdf" in captured.out
        assert "WARNING" in captured.out

    def test_null_outputs_when_no_labels_column(
        self, fake_partition_no_labels: Path
    ) -> None:
        with patch(
            "agentic_kie_evals.upload_dataset.STATIC_DIR", fake_partition_no_labels
        ):
            examples = read_partition("test-A", "test")

        assert len(examples) == 1
        outputs = examples[0]["outputs"]
        assert outputs == {
            "effective_date": None,
            "jurisdiction": None,
            "party": None,
            "term": None,
        }
        assert examples[0]["split"] == "test"

    def test_deterministic_ids(self, fake_partition: Path) -> None:
        """Same filename always produces the same UUID5."""
        with patch("agentic_kie_evals.upload_dataset.STATIC_DIR", fake_partition):
            run1 = read_partition("train", "train")
            run2 = read_partition("train", "train")

        assert [e["id"] for e in run1] == [e["id"] for e in run2]


class TestGetOrCreateDataset:
    def test_creates_new_dataset(self, mock_client: MagicMock) -> None:
        mock_dataset = MagicMock()
        mock_dataset.id = uuid.uuid4()
        mock_client.create_dataset.return_value = mock_dataset

        result = get_or_create_dataset(mock_client, "my-dataset")

        mock_client.create_dataset.assert_called_once()
        assert result == mock_dataset.id

    def test_falls_back_to_existing(self, mock_client: MagicMock) -> None:
        mock_client.create_dataset.side_effect = Exception("already exists")
        mock_dataset = MagicMock()
        mock_dataset.id = uuid.uuid4()
        mock_client.read_dataset.return_value = mock_dataset

        result = get_or_create_dataset(mock_client, "my-dataset")

        mock_client.read_dataset.assert_called_once_with(dataset_name="my-dataset")
        assert result == uuid.UUID(str(mock_dataset.id))

    def test_recreate_deletes_first(self, mock_client: MagicMock) -> None:
        mock_dataset = MagicMock()
        mock_dataset.id = uuid.uuid4()
        mock_client.create_dataset.return_value = mock_dataset

        get_or_create_dataset(mock_client, "my-dataset", recreate=True)

        mock_client.delete_dataset.assert_called_once_with(dataset_name="my-dataset")
        mock_client.create_dataset.assert_called_once()

    def test_recreate_ignores_delete_failure(self, mock_client: MagicMock) -> None:
        mock_client.delete_dataset.side_effect = Exception("not found")
        mock_dataset = MagicMock()
        mock_dataset.id = uuid.uuid4()
        mock_client.create_dataset.return_value = mock_dataset

        result = get_or_create_dataset(mock_client, "my-dataset", recreate=True)

        assert result == mock_dataset.id


class TestUploadPartition:
    def test_uploads_in_batches(self, mock_client: MagicMock) -> None:
        examples: list[dict[str, Any]] = [{"id": i} for i in range(5)]
        dataset_id = uuid.uuid4()

        upload_partition(mock_client, dataset_id, examples, batch_size=2)

        assert mock_client.create_examples.call_count == 3  # 2+2+1
        # Verify batch sizes
        sizes = [
            len(c.kwargs["examples"])
            for c in mock_client.create_examples.call_args_list
        ]
        assert sizes == [2, 2, 1]

    def test_dry_run_skips_upload(
        self, mock_client: MagicMock, capsys: CaptureFixture[str]
    ) -> None:
        examples: list[dict[str, Any]] = [{"id": i} for i in range(3)]

        upload_partition(mock_client, "dry-run", examples, batch_size=2, dry_run=True)

        mock_client.create_examples.assert_not_called()
        captured = capsys.readouterr()
        assert "DRY RUN" in captured.out

    def test_empty_examples(self, mock_client: MagicMock) -> None:
        upload_partition(mock_client, uuid.uuid4(), [], batch_size=5)

        mock_client.create_examples.assert_not_called()

    def test_single_batch(self, mock_client: MagicMock) -> None:
        examples: list[dict[str, Any]] = [{"id": i} for i in range(3)]
        dataset_id = uuid.uuid4()

        upload_partition(mock_client, dataset_id, examples, batch_size=10)

        mock_client.create_examples.assert_called_once()
        assert len(mock_client.create_examples.call_args.kwargs["examples"]) == 3

    def test_passes_dataset_id_and_filesystem_flag(
        self, mock_client: MagicMock
    ) -> None:
        examples: list[dict[str, Any]] = [{"id": 1}]
        dataset_id = uuid.uuid4()

        upload_partition(mock_client, dataset_id, examples, batch_size=10)

        _, kwargs = mock_client.create_examples.call_args
        assert kwargs["dataset_id"] == dataset_id
        assert kwargs["dangerously_allow_filesystem"] is True


class TestMain:
    @patch("agentic_kie_evals.upload_dataset.upload_partition")
    @patch("agentic_kie_evals.upload_dataset.read_partition")
    @patch("agentic_kie_evals.upload_dataset.Client")
    @patch("agentic_kie_evals.upload_dataset.load_dotenv")
    def test_dry_run_skips_dataset_creation(
        self,
        mock_dotenv: MagicMock,
        mock_client_cls: MagicMock,
        mock_read: MagicMock,
        mock_upload: MagicMock,
    ) -> None:
        mock_read.return_value = [{"id": 1}]

        from agentic_kie_evals.upload_dataset import main

        with patch(
            "sys.argv",
            ["upload_dataset", "--dry-run", "--partitions", "train"],
        ):
            main()

        mock_dotenv.assert_called_once()
        # In dry-run mode, dataset_id is the string "dry-run"
        mock_upload.assert_called_once()
        _, kwargs = mock_upload.call_args
        assert kwargs["dataset_id"] == "dry-run"
        assert kwargs["dry_run"] is True

    @patch("agentic_kie_evals.upload_dataset.upload_partition")
    @patch("agentic_kie_evals.upload_dataset.read_partition")
    @patch("agentic_kie_evals.upload_dataset.get_or_create_dataset")
    @patch("agentic_kie_evals.upload_dataset.Client")
    @patch("agentic_kie_evals.upload_dataset.load_dotenv")
    def test_uploads_all_partitions_by_default(
        self,
        mock_dotenv: MagicMock,
        mock_client_cls: MagicMock,
        mock_get_or_create: MagicMock,
        mock_read: MagicMock,
        mock_upload: MagicMock,
    ) -> None:
        dataset_id = uuid.uuid4()
        mock_get_or_create.return_value = dataset_id
        mock_read.return_value = [{"id": 1}]

        from agentic_kie_evals.upload_dataset import main

        with patch("sys.argv", ["upload_dataset"]):
            main()

        # All three partitions should be processed
        assert mock_read.call_count == len(PARTITIONS)
        assert mock_upload.call_count == len(PARTITIONS)


class TestConstants:
    def test_partitions_mapping(self) -> None:
        assert PARTITIONS == {
            "train": "train",
            "dev-0": "dev",
            "test-A": "test",
        }

    def test_defaults(self) -> None:
        assert DEFAULT_DATASET_NAME == "kleister-nda"
        assert DEFAULT_BATCH_SIZE == 20

    def test_namespace_uuid_is_stable(self):
        assert uuid.UUID("fcd0fe34-475f-4e1a-819a-85dde6f2fa71") == NAMESPACE_UUID
