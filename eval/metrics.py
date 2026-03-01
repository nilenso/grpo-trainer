"""Shared metric computation functions for evaluation."""

import math
from typing import Any

from .constants import (
    COMPILE_THRESHOLD,
    PASS_THRESHOLD,
    TEST_THRESHOLD,
    TYPE_CHECK_THRESHOLD,
)

# Fields that should be numeric (normalize from CSV string loading)
_NUMERIC_FIELDS = [
    "total_reward", "base_reward", "type_score", "compile_score", "test_score",
    "generation_time_sec", "completion_length",
    "prompt_tokens", "completion_tokens", "total_tokens", "tokens_per_sec",
]


def _safe_float(val: Any) -> float | None:
    """Convert a value to float, returning None for missing/empty/invalid."""
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _safe_mean(values: list[float | None]) -> float | None:
    """Compute mean of non-None values. Returns None if no valid values."""
    clean = [v for v in values if v is not None]
    return sum(clean) / len(clean) if clean else None


def compute_percentile(values: list[float], p: float) -> float | None:
    """
    Compute the p-th percentile using sorted linear interpolation.

    Args:
        values: List of numeric values (must be non-empty for a result)
        p: Percentile in [0, 100]

    Returns:
        Percentile value, or None for empty lists
    """
    if not values:
        return None
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]

    # Linear interpolation between closest ranks
    rank = (p / 100) * (n - 1)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    frac = rank - low
    return sorted_vals[low] + frac * (sorted_vals[high] - sorted_vals[low])


def _normalize_numeric_fields(results: list[dict[str, Any]]) -> None:
    """Normalize string values to float/None in place (for CSV loading)."""
    for r in results:
        for key in _NUMERIC_FIELDS:
            if key in r and isinstance(r[key], str):
                r[key] = _safe_float(r[key])


def compute_metrics(results: list[dict[str, Any]], name: str = "") -> dict[str, Any]:
    """
    Compute evaluation metrics from results.

    Args:
        results: List of result dictionaries
        name: Optional name for the result set

    Returns:
        Dictionary with computed metrics including percentile latencies,
        token aggregates, and GRPO-relevant behavioral metrics.
    """
    total = len(results)
    if total == 0:
        return {"total": 0, "name": name}

    _normalize_numeric_fields(results)

    passed = sum(1 for r in results if r["total_reward"] >= PASS_THRESHOLD)
    type_check_pass = sum(1 for r in results if r["type_score"] >= TYPE_CHECK_THRESHOLD)
    compiles = sum(1 for r in results if r["compile_score"] >= COMPILE_THRESHOLD)
    tests_pass = sum(1 for r in results if r["test_score"] >= TEST_THRESHOLD)

    gen_times = [
        r.get("generation_time_sec", 0) for r in results if r.get("generation_time_sec", 0) > 0
    ]

    # Token-related aggregates (from non-null values)
    tps_vals = [
        r["tokens_per_sec"] for r in results
        if r.get("tokens_per_sec") is not None
    ]
    prompt_tokens = [
        r["prompt_tokens"] for r in results
        if r.get("prompt_tokens") is not None
    ]
    comp_tokens = [
        r["completion_tokens"] for r in results
        if r.get("completion_tokens") is not None
    ]

    # Completion length stats
    comp_lengths = [
        r.get("completion_length", 0) for r in results
        if r.get("completion_length") and r.get("completion_length") > 0
    ]

    # Imperative ratio
    imperative_count = sum(1 for r in results if r.get("uses_imperative"))

    # Partial credit: average test_score across all results
    test_scores = [r.get("test_score", 0.0) for r in results if r.get("test_score") is not None]

    return {
        "name": name,
        "total": total,
        "passed": passed,
        "type_check_pass": type_check_pass,
        "compiles": compiles,
        "tests_pass": tests_pass,
        "pass_rate": passed / total * 100,
        "pass_at_1": round(passed / total * 100, 1),
        "type_check_rate": type_check_pass / total * 100,
        "compile_rate": compiles / total * 100,
        "test_pass_rate": tests_pass / total * 100,
        "avg_reward": sum(r["total_reward"] for r in results) / total,
        "avg_gen_time": sum(gen_times) / len(gen_times) if gen_times else 0.0,
        "total_gen_time": sum(gen_times),
        # Percentile latency
        "p50_gen_time": round(compute_percentile(gen_times, 50), 2) if gen_times else None,
        "p90_gen_time": round(compute_percentile(gen_times, 90), 2) if gen_times else None,
        "p99_gen_time": round(compute_percentile(gen_times, 99), 2) if gen_times else None,
        # Token throughput
        "avg_tokens_per_sec": (
            round(sum(tps_vals) / len(tps_vals), 1) if tps_vals else None
        ),
        "median_tokens_per_sec": (
            round(compute_percentile(tps_vals, 50), 1) if tps_vals else None
        ),
        # Token totals
        "total_prompt_tokens": sum(prompt_tokens) if prompt_tokens else None,
        "total_completion_tokens": sum(comp_tokens) if comp_tokens else None,
        "total_tokens": (
            (sum(prompt_tokens) + sum(comp_tokens))
            if prompt_tokens and comp_tokens
            else None
        ),
        "avg_prompt_tokens": (
            round(sum(prompt_tokens) / len(prompt_tokens), 1) if prompt_tokens else None
        ),
        "avg_completion_tokens": (
            round(sum(comp_tokens) / len(comp_tokens), 1) if comp_tokens else None
        ),
        # Partial credit
        "avg_tpr": round(sum(test_scores) / len(test_scores), 4) if test_scores else None,
        # Completion length
        "avg_completion_length": (
            round(sum(comp_lengths) / len(comp_lengths)) if comp_lengths else None
        ),
        "p50_completion_length": (
            round(compute_percentile(comp_lengths, 50)) if comp_lengths else None
        ),
        "p90_completion_length": (
            round(compute_percentile(comp_lengths, 90)) if comp_lengths else None
        ),
        # GRPO degradation signal
        "imperative_ratio": round(imperative_count / total, 3),
    }


def compute_failure_stages(results: list[dict[str, Any]]) -> dict[str, int]:
    """
    Compute failure stage breakdown from results.

    Args:
        results: List of result dictionaries

    Returns:
        Dictionary mapping failure stage to count
    """
    failure_stages: dict[str, int] = {}
    for r in results:
        stage = r.get("failure_stage", "")
        if stage:
            failure_stages[stage] = failure_stages.get(stage, 0) + 1
    return failure_stages


def compute_difficulty_stats(results: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """
    Compute pass/fail breakdown by difficulty level.

    Args:
        results: List of result dictionaries

    Returns:
        Dictionary mapping difficulty to {total, passed}
    """
    difficulty_stats: dict[str, dict[str, int]] = {}
    for r in results:
        diff = r.get("difficulty", "unknown") or "unknown"
        if diff not in difficulty_stats:
            difficulty_stats[diff] = {"total": 0, "passed": 0}
        difficulty_stats[diff]["total"] += 1
        if r["total_reward"] >= PASS_THRESHOLD:
            difficulty_stats[diff]["passed"] += 1
    return difficulty_stats
