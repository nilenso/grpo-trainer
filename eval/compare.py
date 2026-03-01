#!/usr/bin/env python3
"""
Compare evaluation results from two model runs.

Supports both CSV and enriched JSONL input (auto-detected by extension).

Usage:
    python -m eval.compare eval_runs/base/results.csv eval_runs/finetuned/results.csv
    python -m eval.compare eval_runs/base/completions.jsonl eval_runs/finetuned/completions.jsonl
"""

import csv
import json
import sys
from pathlib import Path
from typing import Any

from .constants import PASS_THRESHOLD
from .metrics import compute_failure_stages, compute_metrics


# Pipeline order for stage progression analysis
PIPELINE_ORDER = ["degenerate", "type_check", "compile", "execution", "pass"]


def _stage_rank(failure_stage: str) -> int:
    """
    Return pipeline rank for a failure stage.

    Higher rank = further in pipeline. Passed problems (empty stage) rank highest.
    """
    if not failure_stage:
        return PIPELINE_ORDER.index("pass")
    prefix = failure_stage.split(":")[0]
    try:
        return PIPELINE_ORDER.index(prefix)
    except ValueError:
        return -1  # unknown stages rank lowest


def load_results(path: str) -> list[dict[str, Any]]:
    """Load results from CSV or JSONL file (auto-detected by extension)."""
    p = Path(path)
    if p.suffix == ".jsonl":
        return load_results_jsonl(path)
    return load_results_csv(path)


def load_results_csv(path: str) -> list[dict[str, Any]]:
    """Load results from CSV file."""
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_results_jsonl(path: str) -> list[dict[str, Any]]:
    """Load results from enriched JSONL file, skipping _meta lines."""
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("_meta"):
                continue
            results.append(record)
    return results


def load_meta(path: str) -> dict[str, Any] | None:
    """Load _meta header from a JSONL file, or None if not present."""
    p = Path(path)
    if p.suffix != ".jsonl":
        return None
    with open(path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
        if first_line:
            record = json.loads(first_line)
            if record.get("_meta"):
                return record
    return None


def compute_metrics_with_failures(
    results: list[dict[str, Any]], name: str = ""
) -> dict[str, Any]:
    """Compute metrics including failure stage breakdown."""
    metrics = compute_metrics(results, name)
    metrics["failure_stages"] = compute_failure_stages(results)
    return metrics


def compute_per_problem_comparison(
    results_a: list[dict[str, Any]],
    results_b: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare results problem-by-problem, including reward deltas."""
    a_by_id = {r["id"]: r for r in results_a}
    b_by_id = {r["id"]: r for r in results_b}
    common_ids = set(a_by_id.keys()) & set(b_by_id.keys())

    improved = []   # B solved, A didn't
    regressed = []  # A solved, B didn't
    both_pass = []
    both_fail = []

    for pid in sorted(common_ids):
        ra, rb = a_by_id[pid], b_by_id[pid]
        reward_a = float(ra["total_reward"])
        reward_b = float(rb["total_reward"])
        a_pass = reward_a >= PASS_THRESHOLD
        b_pass = reward_b >= PASS_THRESHOLD
        reward_delta = round(reward_b - reward_a, 4)

        entry = {"id": pid, "reward_delta": reward_delta}

        if b_pass and not a_pass:
            improved.append(entry)
        elif a_pass and not b_pass:
            regressed.append(entry)
        elif a_pass and b_pass:
            both_pass.append(entry)
        else:
            both_fail.append(entry)

    return {
        "common_problems": len(common_ids),
        "improved": improved,
        "regressed": regressed,
        "both_pass": both_pass,
        "both_fail": both_fail,
        "improved_count": len(improved),
        "regressed_count": len(regressed),
        "net_change": len(improved) - len(regressed),
    }


def compute_stage_migration(
    results_a: list[dict[str, Any]],
    results_b: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    """
    Compute failure stage migration matrix.

    Returns {baseline_stage: {current_stage: count}} for common problems.
    Empty failure_stage (pass) is represented as "pass".
    """
    a_by_id = {r["id"]: r for r in results_a}
    b_by_id = {r["id"]: r for r in results_b}
    common_ids = set(a_by_id.keys()) & set(b_by_id.keys())

    migration: dict[str, dict[str, int]] = {}
    for pid in common_ids:
        stage_a = a_by_id[pid].get("failure_stage", "") or "pass"
        stage_b = b_by_id[pid].get("failure_stage", "") or "pass"
        if stage_a not in migration:
            migration[stage_a] = {}
        migration[stage_a][stage_b] = migration[stage_a].get(stage_b, 0) + 1

    return migration


def compute_stage_progression(
    results_a: list[dict[str, Any]],
    results_b: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Classify each problem as progressed / regressed / same in the pipeline.

    Pipeline: degenerate → type_check → compile → execution → pass.
    """
    a_by_id = {r["id"]: r for r in results_a}
    b_by_id = {r["id"]: r for r in results_b}
    common_ids = set(a_by_id.keys()) & set(b_by_id.keys())

    progressed = []
    regressed_stage = []
    same = []

    for pid in sorted(common_ids):
        stage_a = a_by_id[pid].get("failure_stage", "") or ""
        stage_b = b_by_id[pid].get("failure_stage", "") or ""
        rank_a = _stage_rank(stage_a)
        rank_b = _stage_rank(stage_b)

        if rank_b > rank_a:
            progressed.append(pid)
        elif rank_b < rank_a:
            regressed_stage.append(pid)
        else:
            same.append(pid)

    return {
        "progressed": progressed,
        "regressed": regressed_stage,
        "same": same,
        "progressed_count": len(progressed),
        "regressed_count": len(regressed_stage),
        "same_count": len(same),
    }


def compute_per_difficulty_deltas(
    results_a: list[dict[str, Any]],
    results_b: list[dict[str, Any]],
) -> dict[str, dict[str, float | None]]:
    """
    Compute pass rate delta broken down by difficulty level.

    Returns {difficulty: {pass_rate_a, pass_rate_b, delta}}.
    """
    def _by_difficulty(results: list[dict[str, Any]]) -> dict[str, list]:
        groups: dict[str, list] = {}
        for r in results:
            d = r.get("difficulty", "unknown") or "unknown"
            groups.setdefault(d, []).append(r)
        return groups

    a_groups = _by_difficulty(results_a)
    b_groups = _by_difficulty(results_b)
    all_diffs = sorted(set(a_groups.keys()) | set(b_groups.keys()))

    deltas: dict[str, dict[str, float | None]] = {}
    for diff in all_diffs:
        a_list = a_groups.get(diff, [])
        b_list = b_groups.get(diff, [])
        a_rate = (
            sum(1 for r in a_list if float(r["total_reward"]) >= PASS_THRESHOLD)
            / len(a_list) * 100 if a_list else None
        )
        b_rate = (
            sum(1 for r in b_list if float(r["total_reward"]) >= PASS_THRESHOLD)
            / len(b_list) * 100 if b_list else None
        )
        delta = (
            round(b_rate - a_rate, 1) if a_rate is not None and b_rate is not None else None
        )
        deltas[diff] = {
            "pass_rate_a": round(a_rate, 1) if a_rate is not None else None,
            "pass_rate_b": round(b_rate, 1) if b_rate is not None else None,
            "delta": delta,
            "count_a": len(a_list),
            "count_b": len(b_list),
        }
    return deltas


def compute_score_distributions(
    results: list[dict[str, Any]], bins: int = 20
) -> dict[str, list[int]]:
    """
    Return binned histograms for score fields and total_reward.

    Returns {field_name: [count_per_bin]} with `bins` buckets from 0.0 to 1.0.
    """
    fields = ["type_score", "compile_score", "test_score", "total_reward"]
    distributions: dict[str, list[int]] = {}

    for field in fields:
        counts = [0] * bins
        for r in results:
            val = r.get(field)
            if val is None:
                continue
            val = float(val)
            idx = min(int(val * bins), bins - 1)
            counts[idx] += 1
        distributions[field] = counts

    return distributions


def compute_length_distributions(
    results: list[dict[str, Any]], bin_size: int = 50, max_length: int = 2000
) -> list[int]:
    """
    Return binned histogram for completion_length.

    Bins are `bin_size` chars wide, from 0 to `max_length`.
    Last bin captures everything >= max_length.
    """
    num_bins = max_length // bin_size + 1
    counts = [0] * num_bins
    for r in results:
        length = r.get("completion_length")
        if length is None:
            continue
        length = int(length)
        idx = min(length // bin_size, num_bins - 1)
        counts[idx] += 1
    return counts


def compute_imperative_comparison(
    results_a: list[dict[str, Any]],
    results_b: list[dict[str, Any]],
) -> dict[str, int]:
    """
    Compare imperative style usage between two runs.

    Returns counts: both_imperative, both_functional,
    became_imperative, became_functional.
    """
    a_by_id = {r["id"]: r for r in results_a}
    b_by_id = {r["id"]: r for r in results_b}
    common_ids = set(a_by_id.keys()) & set(b_by_id.keys())

    both_imp = both_func = became_imp = became_func = 0
    for pid in common_ids:
        a_imp = bool(a_by_id[pid].get("uses_imperative"))
        b_imp = bool(b_by_id[pid].get("uses_imperative"))
        if a_imp and b_imp:
            both_imp += 1
        elif not a_imp and not b_imp:
            both_func += 1
        elif not a_imp and b_imp:
            became_imp += 1
        else:
            became_func += 1

    return {
        "both_imperative": both_imp,
        "both_functional": both_func,
        "became_imperative": became_imp,
        "became_functional": became_func,
    }


def compute_frozen_problems(
    results_a: list[dict[str, Any]],
    results_b: list[dict[str, Any]],
    reward_threshold: float = 0.05,
) -> list[str]:
    """
    Find problems where both runs fail with same failure stage and minimal reward delta.

    These are problems GRPO training couldn't move the needle on.
    """
    a_by_id = {r["id"]: r for r in results_a}
    b_by_id = {r["id"]: r for r in results_b}
    common_ids = set(a_by_id.keys()) & set(b_by_id.keys())

    frozen = []
    for pid in sorted(common_ids):
        ra, rb = a_by_id[pid], b_by_id[pid]
        reward_a = float(ra["total_reward"])
        reward_b = float(rb["total_reward"])

        # Both must fail
        if reward_a >= PASS_THRESHOLD or reward_b >= PASS_THRESHOLD:
            continue

        # Same failure stage
        stage_a = ra.get("failure_stage", "") or ""
        stage_b = rb.get("failure_stage", "") or ""
        if stage_a != stage_b:
            continue

        # Minimal reward delta
        if abs(reward_b - reward_a) < reward_threshold:
            frozen.append(pid)

    return frozen


def format_delta(val_a: float, val_b: float, is_pct: bool = True) -> str:
    """Format the delta between two values with direction indicator."""
    delta = val_b - val_a
    if abs(delta) < 0.1:
        indicator = "  "
    elif delta > 0:
        indicator = "↑ "
    else:
        indicator = "↓ "

    if is_pct:
        return f"{indicator}{delta:+.1f}%"
    return f"{indicator}{delta:+.4f}"


def _format_metric_row(
    label: str,
    val_a: float | None,
    val_b: float | None,
    is_pct: bool = True,
) -> str:
    """Format a single metric comparison row, handling None values."""
    if val_a is None or val_b is None:
        a_str = "N/A" if val_a is None else (f"{val_a:.1f}%" if is_pct else f"{val_a:.4f}")
        b_str = "N/A" if val_b is None else (f"{val_b:.1f}%" if is_pct else f"{val_b:.4f}")
        return f"│ {label:<19} │ {a_str:>10} │ {b_str:>10} │ {'N/A':>9} │"

    delta = format_delta(val_a, val_b, is_pct)
    if is_pct:
        return f"│ {label:<19} │ {val_a:>8.1f}%  │ {val_b:>8.1f}%  │ {delta:>9} │"
    return f"│ {label:<19} │ {val_a:>10.4f} │ {val_b:>10.4f} │ {delta:>9} │"


def print_comparison(
    metrics_a: dict[str, Any], metrics_b: dict[str, Any], comparison: dict[str, Any],
    results_a: list[dict[str, Any]] | None = None,
    results_b: list[dict[str, Any]] | None = None,
) -> None:
    """Print side-by-side comparison with extended metrics."""
    name_a = metrics_a["name"] or "Model A"
    name_b = metrics_b["name"] or "Model B"

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    print(f"Model A: {name_a}")
    print(f"Model B: {name_b}")
    print(f"Problems: {comparison['common_problems']}")
    print()

    # Main metrics table
    print("┌─────────────────────┬────────────┬────────────┬───────────┐")
    print(f"│ Metric              │ {'Model A':^10} │ {'Model B':^10} │ Delta     │")
    print("├─────────────────────┼────────────┼────────────┼───────────┤")

    rows = [
        ("Type Check Rate", "type_check_rate", True),
        ("Compile Rate", "compile_rate", True),
        ("Test Pass Rate", "test_pass_rate", True),
        ("pass@1", "pass_at_1", True),
        ("Avg Reward", "avg_reward", False),
        ("Avg TPR", "avg_tpr", False),
        ("Avg Tokens/sec", "avg_tokens_per_sec", False),
        ("Avg Comp. Tokens", "avg_completion_tokens", False),
        ("P50 Latency", "p50_gen_time", False),
        ("P90 Latency", "p90_gen_time", False),
    ]

    for label, key, is_pct in rows:
        val_a = metrics_a.get(key)
        val_b = metrics_b.get(key)
        print(_format_metric_row(label, val_a, val_b, is_pct))

    print("└─────────────────────┴────────────┴────────────┴───────────┘")

    # Problem-level changes
    print("\nProblem-Level Changes:")
    print(f"  Improved (B solved, A didn't): {comparison['improved_count']}")
    print(f"  Regressed (A solved, B didn't): {comparison['regressed_count']}")
    print(f"  Both pass: {len(comparison['both_pass'])}")
    print(f"  Both fail: {len(comparison['both_fail'])}")
    print(f"  Net change: {comparison['net_change']:+d}")

    # Show specific problems that changed
    if comparison["improved"]:
        ids = [e["id"] if isinstance(e, dict) else e for e in comparison["improved"][:10]]
        problems = ", ".join(ids)
        print(f"\n  Improved problems: {problems}", end="")
        if len(comparison["improved"]) > 10:
            print(f" ... (+{len(comparison['improved']) - 10} more)")
        else:
            print()

    if comparison["regressed"]:
        ids = [e["id"] if isinstance(e, dict) else e for e in comparison["regressed"][:10]]
        problems = ", ".join(ids)
        print(f"  Regressed problems: {problems}", end="")
        if len(comparison["regressed"]) > 10:
            print(f" ... (+{len(comparison['regressed']) - 10} more)")
        else:
            print()

    # Stage progression (if raw results available)
    if results_a is not None and results_b is not None:
        progression = compute_stage_progression(results_a, results_b)
        print("\nStage Progression:")
        print(f"  Progressed (further in pipeline): {progression['progressed_count']}")
        print(f"  Regressed (earlier in pipeline):  {progression['regressed_count']}")
        print(f"  Same stage:                       {progression['same_count']}")

    # Failure stage comparison
    print("\nFailure Breakdown:")
    all_stages = set(metrics_a["failure_stages"].keys()) | set(metrics_b["failure_stages"].keys())
    for stage in sorted(all_stages):
        count_a = metrics_a["failure_stages"].get(stage, 0)
        count_b = metrics_b["failure_stages"].get(stage, 0)
        delta = count_b - count_a
        delta_str = f"{delta:+d}" if delta != 0 else "0"
        print(f"  {stage}: {count_a} → {count_b} ({delta_str})")

    # Stage migration matrix (if raw results available)
    if results_a is not None and results_b is not None:
        migration = compute_stage_migration(results_a, results_b)
        if migration:
            print("\nStage Migration Matrix (rows=baseline, cols=current):")
            all_stages_m = sorted(
                set(s for row in migration.values() for s in row.keys()) | set(migration.keys())
            )
            # Header
            header = f"  {'':20}" + "".join(f"{s[:12]:>13}" for s in all_stages_m)
            print(header)
            for row_stage in all_stages_m:
                row_data = migration.get(row_stage, {})
                cells = "".join(
                    f"{row_data.get(col_stage, 0):>13}" for col_stage in all_stages_m
                )
                print(f"  {row_stage:20}{cells}")


def write_comparison_json(
    metrics_a: dict[str, Any],
    metrics_b: dict[str, Any],
    comparison: dict[str, Any],
    output_path: str,
    results_a: list[dict[str, Any]] | None = None,
    results_b: list[dict[str, Any]] | None = None,
) -> None:
    """Write comparison to JSON file with extended analysis."""
    data: dict[str, Any] = {
        "model_a": metrics_a,
        "model_b": metrics_b,
        "comparison": comparison,
    }

    if results_a is not None and results_b is not None:
        data["stage_migration"] = compute_stage_migration(results_a, results_b)
        data["stage_progression"] = compute_stage_progression(results_a, results_b)
        data["per_difficulty_deltas"] = compute_per_difficulty_deltas(results_a, results_b)
        data["imperative_comparison"] = compute_imperative_comparison(results_a, results_b)
        data["frozen_problems"] = compute_frozen_problems(results_a, results_b)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python -m eval.compare <results_a> <results_b> [output.json]")
        print("\nSupports both .csv and .jsonl files.")
        print("\nExample:")
        print("  python -m eval.compare eval_runs/base/completions.jsonl eval_runs/ft/completions.jsonl")
        sys.exit(1)

    path_a = sys.argv[1]
    path_b = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else None

    name_a = Path(path_a).parent.name
    name_b = Path(path_b).parent.name

    print(f"Loading {path_a}...")
    results_a = load_results(path_a)
    print(f"  Loaded {len(results_a)} results")

    print(f"Loading {path_b}...")
    results_b = load_results(path_b)
    print(f"  Loaded {len(results_b)} results")

    metrics_a = compute_metrics_with_failures(results_a, name_a)
    metrics_b = compute_metrics_with_failures(results_b, name_b)
    comparison = compute_per_problem_comparison(results_a, results_b)

    print_comparison(metrics_a, metrics_b, comparison, results_a, results_b)

    if output_path:
        write_comparison_json(metrics_a, metrics_b, comparison, output_path, results_a, results_b)
        print(f"\nComparison saved to: {output_path}")


if __name__ == "__main__":
    main()
