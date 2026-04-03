# Contributing

## Development setup

The package requires Python 3.13 or later. Dependencies are managed with [uv](https://docs.astral.sh/uv/).

1. Clone the repository.

```bash
git clone https://github.com/gafnts/agentic-kie-evals.git
cd agentic-kie-evals
```

2. Install all dependencies, dev tools, and git hooks.

```bash
make install
```

## Available targets

| Target | Description |
|---|---|
| `make check` | Run the full pre-commit suite (lint, format, type check) |
| `make lint` | Run `ruff check` on `src` and `tests` |
| `make format` | Run `ruff check --fix` on `src` and `tests` |
| `make type` | Run `mypy` on `src` and `tests` |
| `make test` | Run `pytest` with branch coverage |


## CI pipeline

GitHub Actions runs two sequential jobs on every push and pull request to `main`:

1. **`lint-and-type-check`**: runs `pyproject-fmt --check`, `ruff check`, `ruff format --check`, and `mypy`.
2. **`test`**: runs `pytest` with branch coverage and uploads the `coverage.xml` report to Codecov.

Coverage is enforced at 95%.
