"""
Evaluators for the Kleister NDA extraction benchmark.

Each evaluator follows the LangSmith custom evaluator signature:
    def evaluator(outputs: dict, reference_outputs: dict) -> dict

Returns {"key": str, "score": float} where score is 0.0 or 1.0 for
scalar fields, and a continuous F1 score for the party list field.

Normalization is applied to both sides before comparison to handle
casing differences, trailing periods, and whitespace.
"""

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


def _score_none_aware(
    predicted: str | None,
    expected: str | None,
    compare_fn: Callable[[str, str], float],
) -> float:
    """
    Score two scalar values with None-awareness.

    Both None → 1.0 (true negative).
    One None  → 0.0 (miss or hallucination).
    Both present → delegate to compare_fn(predicted, expected).
    """
    if predicted is None and expected is None:
        return 1.0
    if predicted is None or expected is None:
        return 0.0
    return compare_fn(predicted, expected)


def _best_fuzzy_match(candidate: str, reference_set: set[str]) -> bool:
    """
    Check if candidate fuzzy-matches any element in reference_set.
    """
    return any(
        SequenceMatcher(None, candidate, ref).ratio() >= FUZZY_THRESHOLD
        for ref in reference_set
    )


def _set_f1(predicted: set[str], expected: set[str], *, fuzzy: bool) -> float:
    """
    Compute F1 between two sets of strings.

    If fuzzy=True, matching uses SequenceMatcher with FUZZY_THRESHOLD.
    If fuzzy=False, matching is exact string equality.

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
        tp_recall = tp_precision  # symmetric for exact match

    precision = tp_precision / len(predicted)
    recall = tp_recall / len(expected)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def exact_effective_date(
    outputs: dict[str, Any], reference_outputs: dict[str, Any]
) -> dict[str, Any]:
    """
    Exact match on effective_date after normalization.

    Date values are already in YYYY-MM-DD from the schema validator,
    so string comparison after lowercasing is sufficient.
    """
    predicted = _normalize_for_eval(outputs.get("effective_date"))
    expected = _normalize_for_eval(reference_outputs.get("effective_date"))
    score = _score_none_aware(predicted, expected, lambda p, e: float(p == e))
    return {"key": "exact_effective_date", "score": score}


def exact_jurisdiction(
    outputs: dict[str, Any], reference_outputs: dict[str, Any]
) -> dict[str, Any]:
    """
    Exact match on jurisdiction after normalization.
    """
    predicted = _normalize_for_eval(outputs.get("jurisdiction"))
    expected = _normalize_for_eval(reference_outputs.get("jurisdiction"))
    score = _score_none_aware(predicted, expected, lambda p, e: float(p == e))
    return {"key": "exact_jurisdiction", "score": score}


def fuzzy_jurisdiction(
    outputs: dict[str, Any], reference_outputs: dict[str, Any]
) -> dict[str, Any]:
    """
    Fuzzy match on jurisdiction using SequenceMatcher.
    """
    predicted = _normalize_for_eval(outputs.get("jurisdiction"))
    expected = _normalize_for_eval(reference_outputs.get("jurisdiction"))

    def _fuzzy(p: str, e: str) -> float:
        return float(SequenceMatcher(None, p, e).ratio() >= FUZZY_THRESHOLD)

    score = _score_none_aware(predicted, expected, _fuzzy)
    return {"key": "fuzzy_jurisdiction", "score": score}


def exact_term(
    outputs: dict[str, Any], reference_outputs: dict[str, Any]
) -> dict[str, Any]:
    """
    Exact match on term after normalization.
    """
    predicted = _normalize_for_eval(outputs.get("term"))
    expected = _normalize_for_eval(reference_outputs.get("term"))
    score = _score_none_aware(predicted, expected, lambda p, e: float(p == e))
    return {"key": "exact_term", "score": score}


def fuzzy_term(
    outputs: dict[str, Any], reference_outputs: dict[str, Any]
) -> dict[str, Any]:
    """
    Fuzzy match on term using SequenceMatcher.
    """
    predicted = _normalize_for_eval(outputs.get("term"))
    expected = _normalize_for_eval(reference_outputs.get("term"))

    def _fuzzy(p: str, e: str) -> float:
        return float(SequenceMatcher(None, p, e).ratio() >= FUZZY_THRESHOLD)

    score = _score_none_aware(predicted, expected, _fuzzy)
    return {"key": "fuzzy_term", "score": score}


def exact_party(
    outputs: dict[str, Any], reference_outputs: dict[str, Any]
) -> dict[str, Any]:
    """
    Set F1 over party names with exact matching.
    """
    predicted = _extract_party_names(outputs.get("party"))
    expected = _extract_party_names(reference_outputs.get("party"))
    score = _set_f1(predicted, expected, fuzzy=False)
    return {"key": "exact_party", "score": score}


def fuzzy_party(
    outputs: dict[str, Any], reference_outputs: dict[str, Any]
) -> dict[str, Any]:
    """
    Set F1 over party names with fuzzy matching.
    """
    predicted = _extract_party_names(outputs.get("party"))
    expected = _extract_party_names(reference_outputs.get("party"))
    score = _set_f1(predicted, expected, fuzzy=True)
    return {"key": "fuzzy_party", "score": score}


ALL_EVALUATORS: list[Callable[..., dict[str, Any]]] = [
    exact_effective_date,
    exact_jurisdiction,
    fuzzy_jurisdiction,
    exact_term,
    fuzzy_term,
    exact_party,
    fuzzy_party,
]
