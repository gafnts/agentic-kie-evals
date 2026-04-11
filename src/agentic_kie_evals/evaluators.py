"""
Evaluators for the Kleister NDA extraction benchmark.

Each public evaluator follows the LangSmith custom evaluator signature:
    def evaluator(outputs: dict, reference_outputs: dict) -> dict

Returns {"key": str, "score": float} where score is a continuous value
in [0.0, 1.0].  Every entity field is scored using set-based F1: scalar
fields are treated as sets of size 0 or 1, while the party list field is
a variable-size set.

Normalization (lowercasing, whitespace and trailing-period stripping) is
applied to both sides before comparison.
"""

from __future__ import annotations

from collections.abc import Callable
from difflib import SequenceMatcher
from typing import Any

FUZZY_THRESHOLD = 0.85


def _normalize_for_eval(value: str | None) -> str | None:
    """
    Normalize a scalar value for evaluation comparison.

    Applied to both prediction and ground truth before scoring.
    Lowercases, strips whitespace, and strips trailing periods
    (but preserves periods in abbreviations like 'Inc.').
    """
    if value is None:
        return None
    return value.lower().strip().rstrip(".")


def _scalar_to_set(value: str | None) -> set[str]:
    """
    Convert a normalized scalar value to a singleton set (or empty).
    """
    if value is None:
        return set()
    return {value}


def _extract_party_names(party_field: list[dict[str, Any]] | None) -> set[str]:
    """
    Extract and normalize party name strings from the party field.

    The party field is a list of dicts like [{"name": "Nike_Inc."}, ...],
    which is the shape produced by NDA.model_dump().
    """
    if not party_field:
        return set()
    normalized: set[str] = set()
    for p in party_field:
        name = p.get("name")
        if name:
            n = _normalize_for_eval(name)
            if n:
                normalized.add(n)
    return normalized


def _best_fuzzy_match(candidate: str, reference_set: set[str]) -> bool:
    """
    Check if *candidate* fuzzy-matches any element in *reference_set*.
    """
    return any(
        SequenceMatcher(None, candidate, ref).ratio() >= FUZZY_THRESHOLD
        for ref in reference_set
    )


def _set_f1(predicted: set[str], expected: set[str], *, fuzzy: bool) -> float:
    """
    Compute F1 between two sets of strings.

    If *fuzzy* is True, matching uses ``SequenceMatcher`` with
    ``FUZZY_THRESHOLD``.  If False, matching is exact string equality.

    For fuzzy matching, precision and recall are computed independently
    because a fuzzy match is not necessarily symmetric — a predicted name
    might fuzzy-match a different expected name than vice versa.
    """
    if not predicted and not expected:
        return 1.0
    if not predicted or not expected:
        return 0.0

    if fuzzy:
        tp_precision = sum(1 for p in predicted if _best_fuzzy_match(p, expected))
        tp_recall = sum(1 for e in expected if _best_fuzzy_match(e, predicted))
    else:
        tp_precision = len(predicted & expected)
        tp_recall = tp_precision

    precision = tp_precision / len(predicted)
    recall = tp_recall / len(expected)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def _make_aggregate_evaluator(
    key: str,
    field_configs: list[tuple[str, bool, bool]],
) -> Callable[..., dict[str, Any]]:
    """
    Create a macro-average F1 evaluator across multiple fields.

    *field_configs* is a list of ``(field, fuzzy, is_set_field)`` tuples that
    define which variant to use for each field.  The score is the unweighted
    mean of the per-field F1 values.
    """

    def aggregate_eval(
        outputs: dict[str, Any], reference_outputs: dict[str, Any]
    ) -> dict[str, Any]:
        scores = []
        for field, fuzzy, is_set_field in field_configs:
            if is_set_field:
                predicted = _extract_party_names(outputs.get(field))
                expected = _extract_party_names(reference_outputs.get(field))
            else:
                predicted = _scalar_to_set(_normalize_for_eval(outputs.get(field)))
                expected = _scalar_to_set(
                    _normalize_for_eval(reference_outputs.get(field))
                )
            scores.append(_set_f1(predicted, expected, fuzzy=fuzzy))
        return {"key": key, "score": sum(scores) / len(scores)}

    aggregate_eval.__name__ = key
    return aggregate_eval


def _make_field_evaluators(
    field: str,
    *,
    fuzzy: bool,
    is_set_field: bool = False,
) -> list[Callable[..., dict[str, Any]]]:
    """
    Create precision, recall, and F1 evaluators for a single entity field.

    For scalar fields (*is_set_field* = False) the value is promoted to a
    singleton set so the same set-based logic applies uniformly.
    """
    prefix = "fuzzy" if fuzzy else "exact"

    def _get_sets(
        outputs: dict[str, Any],
        reference_outputs: dict[str, Any],
    ) -> tuple[set[str], set[str]]:
        if is_set_field:
            return (
                _extract_party_names(outputs.get(field)),
                _extract_party_names(reference_outputs.get(field)),
            )
        return (
            _scalar_to_set(_normalize_for_eval(outputs.get(field))),
            _scalar_to_set(_normalize_for_eval(reference_outputs.get(field))),
        )

    def f1_eval(
        outputs: dict[str, Any], reference_outputs: dict[str, Any]
    ) -> dict[str, Any]:
        predicted, expected = _get_sets(outputs, reference_outputs)
        return {
            "key": f"{prefix}_{field}_f1",
            "score": _set_f1(predicted, expected, fuzzy=fuzzy),
        }

    f1_eval.__name__ = f"{prefix}_{field}_f1"

    return [f1_eval]


ALL_EVALUATORS: list[Callable[..., dict[str, Any]]] = [
    *_make_field_evaluators("effective_date", fuzzy=False),
    *_make_field_evaluators("jurisdiction", fuzzy=False),
    *_make_field_evaluators("jurisdiction", fuzzy=True),
    *_make_field_evaluators("term", fuzzy=False),
    *_make_field_evaluators("term", fuzzy=True),
    *_make_field_evaluators("party", fuzzy=False, is_set_field=True),
    *_make_field_evaluators("party", fuzzy=True, is_set_field=True),
    _make_aggregate_evaluator(
        "exact_f1",
        [
            ("effective_date", False, False),
            ("jurisdiction", False, False),
            ("term", False, False),
            ("party", False, True),
        ],
    ),
    _make_aggregate_evaluator(
        "fuzzy_f1",
        [
            ("effective_date", False, False),
            ("jurisdiction", True, False),
            ("term", True, False),
            ("party", True, True),
        ],
    ),
]
