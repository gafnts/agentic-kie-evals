.PHONY: install lint format type check test

install:
	uv sync --all-groups --all-extras
	uv run pre-commit install
	uv run pre-commit install --hook-type pre-push

lint:
	uv run ruff check src tests

format:
	uv run ruff check src tests --fix

type:
	uv run mypy src tests

check:
	uv run pre-commit run --all-files

test:
	uv run pytest --cov --cov-branch --cov-report=xml -v
