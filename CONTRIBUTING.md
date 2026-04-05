# Contributing

## Development setup

The project requires Python 3.13 or later. Dependencies are managed with [uv](https://docs.astral.sh/uv/).

1. Clone the repository.

```bash
git clone https://github.com/gafnts/agentic-kie-evals.git
cd agentic-kie-evals
```

2. Install all dependencies, dev tools, and git hooks.

```bash
make install
```

3. Create a `.env` file if you need to use API keys.

```bash
cp .env.example .env
```

## Available targets

| Target | Description |
|---|---|
| `make check` | Run the full pre-commit suite (lint, format, type check) |
| `make lint` | Run `ruff check` on `src` |
| `make format` | Run `ruff check --fix` on `src` |
| `make type` | Run `mypy` on `src` |


## CI pipeline

GitHub Actions runs one sequential job on every push and pull request to `main`:

1. **`lint-and-type-check`**: runs `pyproject-fmt --check`, `ruff check`, `ruff format --check`, and `mypy`.
