"""Smoke test validating that the package is importable and the toolchain is wired correctly."""

import agentic_kie_evals


def test_import():
    assert agentic_kie_evals is not None
