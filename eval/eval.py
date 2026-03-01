#!/usr/bin/env python3
"""
Evaluation script for OCaml code generation using a local LLM.

Reads problems from a HuggingFace dataset, generates solutions via OpenAI-compatible API
(vLLM, llama.cpp, etc.), evaluates them using the training reward system, and outputs
metrics to CSV.
"""

import csv
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from datasets import load_dataset

from .constants import FAILURE_STAGE_PATTERNS, PASS_THRESHOLD
from .metrics import compute_difficulty_stats, compute_failure_stages, compute_metrics
from .report import generate_html_report


from rlvr.environment import compute_reward_with_metadata, extract_code_block, prepend_signature

# Configuration via environment variables
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://localhost:8080/v1/chat/completions")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "local-model")
INPUT_DATASET = os.environ.get("INPUT_DATASET", "kiranpg/ocaml-eval-problems")

SYSTEM_PROMPT = "Respond only with runnable OCaml code (no prose)."

PROMPT_TEMPLATE = """You are an expert OCaml programmer. Complete the following OCaml function.
Respond with ONLY the function body wrapped in an ```ocaml``` code block.

{problem_text}"""


def call_openai_api(prompt: str) -> tuple[str, dict[str, Any]]:
    """
    Call an OpenAI-compatible API (vLLM, llama.cpp, etc.).

    Args:
        prompt: The user prompt to send

    Returns:
        Tuple of (response_text, usage_dict). usage_dict contains
        prompt_tokens, completion_tokens, total_tokens when available,
        plus tokens_per_sec from llama.cpp timings if present.

    Raises:
        requests.RequestException: On network errors
        ValueError: On unexpected response format
    """
    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }

    response = requests.post(OPENAI_BASE_URL, json=payload, timeout=300)
    response.raise_for_status()
    data = response.json()

    choices = data.get("choices")
    if not choices:
        raise ValueError("Unexpected response: missing 'choices'")
    message = choices[0].get("message")
    if not message or "content" not in message:
        raise ValueError("Unexpected response: missing 'message.content'")

    # Extract usage info (safe defaults)
    usage_raw = data.get("usage", {})
    usage: dict[str, Any] = {}
    if usage_raw:
        usage["prompt_tokens"] = usage_raw.get("prompt_tokens")
        usage["completion_tokens"] = usage_raw.get("completion_tokens")
        usage["total_tokens"] = usage_raw.get("total_tokens")

    # llama.cpp vendor extension: timings
    timings = data.get("timings", {})
    if timings and "predicted_per_second" in timings:
        usage["predicted_per_second"] = timings["predicted_per_second"]

    return message["content"].strip(), usage


def generate_solution(problem_text: str) -> tuple[str, float, dict[str, Any]]:
    """
    Generate a solution for a problem using the LLM.

    Args:
        problem_text: The problem description with function signature

    Returns:
        Tuple of (completion, generation_time_seconds, usage_dict)
    """
    prompt = PROMPT_TEMPLATE.format(problem_text=problem_text.strip())

    start_time = time.monotonic()
    completion, usage = call_openai_api(prompt)
    generation_time = time.monotonic() - start_time

    return completion, generation_time, usage


def map_reason_to_failure_stage(reason: str | None) -> str:
    """
    Map the reward system's reason to a failure stage.

    Args:
        reason: The reason string from _score_completion_vf

    Returns:
        Failure stage category string
    """
    if reason is None:
        return ""

    reason_lower = reason.lower()

    # Style prefix check
    if reason_lower.startswith("style:"):
        return "style"

    # Exception/fatal error (multi-condition)
    if "exception" in reason_lower or "fatal error" in reason_lower:
        return "execution:exception"

    # Pattern matching
    for pattern, stage in FAILURE_STAGE_PATTERNS:
        if pattern in reason_lower:
            return stage

    return f"other:{reason_lower[:30]}"


def evaluate_solution(pid: str, completion: str, tests: str) -> dict[str, Any]:
    """Evaluate a solution using the reward system."""
    info = {"tests": tests, "problem_id": pid}
    _, metadata = compute_reward_with_metadata(completion, info, {})
    return metadata


def read_problems(dataset_id: str, limit: int | None = None) -> list[dict[str, Any]]:
    """Read problems from HuggingFace dataset, optionally limiting the count."""
    split = "train"
    if limit is not None and limit > 0:
        split = f"train[:{limit}]"
    dataset = load_dataset(dataset_id, split=split)
    problems = [dict(row) for row in dataset]
    return problems


# Imperative OCaml patterns — GRPO-trained models tend to discover these
# as reliable ways to pass unit tests, degrading functional style.
_IMPERATIVE_PATTERNS = re.compile(
    r"""
    \bref\s          |   # ref keyword (ref x, not "reference")
    :=               |   # mutable assignment
    !\w              |   # dereference (e.g. !x, !count)
    \bfor\s          |   # for loop
    \bwhile\s        |   # while loop
    \bArray\.set\b   |   # array mutation
    \bBytes\.set\b       # bytes mutation
    """,
    re.VERBOSE,
)


def detect_imperative_style(code: str) -> bool:
    """
    Detect imperative OCaml patterns in generated code.

    Scans for ref/mutable assignment, loops, and mutation functions.
    This is a heuristic — `!` can be logical not in some contexts.

    Args:
        code: OCaml source code string

    Returns:
        True if any imperative pattern is found
    """
    return bool(_IMPERATIVE_PATTERNS.search(code))


def _base_scores(eval_result: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build base score fields from eval result or zeros for errors."""
    if eval_result is None:
        return {
            "total_reward": 0.0,
            "base_reward": 0.0,
            "type_score": 0.0,
            "compile_score": 0.0,
            "test_score": 0.0,
            "failure_stage": "generation_error",
        }
    return {
        "total_reward": eval_result["total_reward"],
        "base_reward": eval_result["base_reward"],
        "type_score": eval_result["type_score"],
        "compile_score": eval_result["compile_score"],
        "test_score": eval_result["test_score"],
        "failure_stage": map_reason_to_failure_stage(eval_result.get("reason")),
    }


def _compute_tokens_per_sec(
    usage: dict[str, Any], generation_time: float
) -> float | None:
    """Compute tokens/sec from usage and elapsed time."""
    # Prefer vendor-provided rate from llama.cpp timings
    if "predicted_per_second" in usage:
        return round(usage["predicted_per_second"], 1)
    # Fall back to completion_tokens / elapsed
    comp_tokens = usage.get("completion_tokens")
    if comp_tokens and generation_time > 0:
        return round(comp_tokens / generation_time, 1)
    return None


def build_result(
    pid: str,
    difficulty: str,
    eval_result: dict[str, Any] | None,
    generation_time: float = 0.0,
    completion: str = "",
    usage: dict[str, Any] | None = None,
    full_completion: str = "",
    run_id: str = "",
    timestamp: str = "",
) -> dict[str, Any]:
    """Build result dictionary from evaluation or error."""
    usage = usage or {}
    return {
        "id": pid,
        "difficulty": difficulty,
        **_base_scores(eval_result),
        "generation_time_sec": round(generation_time, 2),
        "completion_length": len(completion),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "tokens_per_sec": _compute_tokens_per_sec(usage, generation_time),
        "uses_imperative": detect_imperative_style(full_completion),
        "run_id": run_id,
        "timestamp": timestamp,
        "model": OPENAI_MODEL,
    }


def build_completion(
    pid: str,
    difficulty: str,
    problem_text: str,
    tests: str,
    eval_result: dict[str, Any] | None,
    generation_time: float = 0.0,
    completion: str = "",
    full_completion: str = "",
    error: str = "",
    usage: dict[str, Any] | None = None,
    run_id: str = "",
    timestamp: str = "",
) -> dict[str, Any]:
    """Build completion dictionary from evaluation or error."""
    usage = usage or {}
    result = {
        "id": pid,
        "difficulty": difficulty,
        "problem_text": problem_text,
        "tests": tests,
        "raw_completion": completion,
        "full_completion": full_completion,
        **_base_scores(eval_result),
        "generation_time_sec": round(generation_time, 2),
        "completion_length": len(completion),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "tokens_per_sec": _compute_tokens_per_sec(usage, generation_time),
        "uses_imperative": detect_imperative_style(full_completion),
        "run_id": run_id,
        "timestamp": timestamp,
        "model": OPENAI_MODEL,
    }
    if error:
        result["error"] = error
    return result


def process_single_problem(
    problem: dict[str, Any], run_id: str = "", timestamp: str = ""
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Process a single problem and return (result, completion) dictionaries."""
    pid = problem["id"]
    problem_text = problem["problem"]
    tests = problem["tests"]
    difficulty = problem.get("difficulty", "")

    common = {"run_id": run_id, "timestamp": timestamp}

    try:
        completion, generation_time, usage = generate_solution(problem_text)
    except Exception as exc:
        return (
            build_result(pid, difficulty, None, **common),
            build_completion(
                pid, difficulty, problem_text, tests, None, error=str(exc), **common
            ),
        )

    # Extract code from markdown blocks first, then prepend signature
    code = extract_code_block(completion)
    full_completion = prepend_signature(problem_text, code)
    eval_result = evaluate_solution(pid, full_completion, tests)

    return (
        build_result(
            pid, difficulty, eval_result, generation_time, completion,
            usage=usage, full_completion=full_completion, **common,
        ),
        build_completion(
            pid, difficulty, problem_text, tests, eval_result,
            generation_time, completion, full_completion,
            usage=usage, **common,
        ),
    )


def print_problem_status(i: int, total: int, pid: str, result: dict[str, Any]) -> None:
    """Print processing status for a single problem."""
    print(f"[{i + 1}/{total}] Processing {pid}...", end=" ", flush=True)
    if result["total_reward"] >= PASS_THRESHOLD:
        print(f"PASS (reward={result['total_reward']:.2f})")
    else:
        print(f"FAIL (reward={result['total_reward']:.2f}, stage={result['failure_stage']})")


def process_dataset(
    dataset_id: str,
    limit: int | None = None,
    run_id: str = "",
    timestamp: str = "",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Process problems in the dataset (optionally limiting the count)."""
    problems = read_problems(dataset_id, limit=limit)
    results = []
    completions = []

    for i, problem in enumerate(problems):
        result, completion = process_single_problem(problem, run_id, timestamp)
        results.append(result)
        completions.append(completion)
        print_problem_status(i, len(problems), problem["id"], result)

    return results, completions


RESULT_FIELDNAMES = [
    "id",
    "difficulty",
    "total_reward",
    "base_reward",
    "type_score",
    "compile_score",
    "test_score",
    "failure_stage",
    "generation_time_sec",
    "completion_length",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "tokens_per_sec",
    "uses_imperative",
    "run_id",
    "timestamp",
    "model",
]


def write_results(results: list[dict[str, Any]], output_path: str) -> None:
    """Write results to a CSV file."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(results)


def _build_meta(
    completions: list[dict[str, Any]],
    run_id: str,
    model: str,
    dataset: str,
    timestamp: str,
) -> dict[str, Any]:
    """Build _meta header line from completions using shared metric functions."""
    metrics = compute_metrics(completions)
    failure_stages = compute_failure_stages(completions)
    difficulty_stats = compute_difficulty_stats(completions)

    # Token aggregates (from non-null values)
    prompt_tokens = [c["prompt_tokens"] for c in completions if c.get("prompt_tokens") is not None]
    comp_tokens = [c["completion_tokens"] for c in completions if c.get("completion_tokens") is not None]
    tps_vals = [c["tokens_per_sec"] for c in completions if c.get("tokens_per_sec") is not None]
    comp_lengths = [c["completion_length"] for c in completions if c.get("completion_length")]
    imperative_count = sum(1 for c in completions if c.get("uses_imperative"))

    meta: dict[str, Any] = {
        "_meta": True,
        "run_id": run_id,
        "model": model,
        "dataset": dataset,
        "timestamp": timestamp,
        "total": metrics["total"],
        "passed": metrics["passed"],
        "pass_rate": round(metrics["pass_rate"], 1),
        "type_check_rate": round(metrics["type_check_rate"], 1),
        "compile_rate": round(metrics["compile_rate"], 1),
        "test_pass_rate": round(metrics["test_pass_rate"], 1),
        "avg_reward": round(metrics["avg_reward"], 4),
        "avg_gen_time": round(metrics["avg_gen_time"], 2),
        "total_prompt_tokens": sum(prompt_tokens) if prompt_tokens else None,
        "total_completion_tokens": sum(comp_tokens) if comp_tokens else None,
        "avg_tokens_per_sec": round(sum(tps_vals) / len(tps_vals), 1) if tps_vals else None,
        "avg_completion_length": round(sum(comp_lengths) / len(comp_lengths)) if comp_lengths else None,
        "imperative_ratio": round(imperative_count / len(completions), 3) if completions else 0.0,
        "failure_stages": failure_stages,
        "difficulty_stats": difficulty_stats,
    }
    return meta


def write_completions(
    completions: list[dict[str, Any]],
    output_path: str,
    run_id: str = "",
    model: str = "",
    dataset: str = "",
    timestamp: str = "",
) -> None:
    """Write completions to a JSONL file with _meta header line."""
    meta = _build_meta(completions, run_id, model, dataset, timestamp)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")
        for completion in completions:
            f.write(json.dumps(completion, ensure_ascii=False) + "\n")


def print_summary(results: list[dict[str, Any]]) -> None:
    """Print summary statistics."""
    total = len(results)
    if total == 0:
        print("No results to summarize.")
        return

    passed = sum(1 for r in results if r["total_reward"] >= PASS_THRESHOLD)
    pass_rate = passed / total * 100
    failure_stages = compute_failure_stages(results)

    gen_times = [r["generation_time_sec"] for r in results if r["generation_time_sec"] > 0]
    avg_gen_time = sum(gen_times) / len(gen_times) if gen_times else 0.0

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Total problems: {total}")
    print(f"Pass rate: {pass_rate:.1f}% ({passed}/{total})")
    print(f"Average generation time: {avg_gen_time:.2f}s")

    if failure_stages:
        print("\nFailure breakdown:")
        for stage, count in sorted(failure_stages.items(), key=lambda x: -x[1]):
            print(f"  {stage}: {count}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate OCaml code generation")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of problems to evaluate (omit for all)",
    )
    args = parser.parse_args()

    # Generate output directory with model name and timestamp
    model_name = OPENAI_MODEL.replace("/", "_").replace(":", "_")
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{model_name}_{ts_str}"
    iso_timestamp = datetime.now(timezone.utc).isoformat()
    run_dir = Path(f"eval_runs/{run_id}")

    # Create directory structure
    run_dir.mkdir(parents=True, exist_ok=True)

    results_path = run_dir / "results.csv"
    completions_path = run_dir / "completions.jsonl"

    print(f"Dataset: {INPUT_DATASET}")
    print(f"Model: {OPENAI_MODEL}")
    print(f"API URL: {OPENAI_BASE_URL}")
    print(f"Output directory: {run_dir}")
    if args.limit is not None and args.limit > 0:
        print(f"Limit: {args.limit} problems")
    print("-" * 50)

    results, completions = process_dataset(
        INPUT_DATASET, limit=args.limit, run_id=run_id, timestamp=iso_timestamp
    )
    write_results(results, str(results_path))
    write_completions(
        completions, str(completions_path),
        run_id=run_id, model=OPENAI_MODEL,
        dataset=INPUT_DATASET, timestamp=iso_timestamp,
    )
    generate_html_report(results, run_dir, OPENAI_MODEL, INPUT_DATASET)

    print("\nResults written to:")
    print(f"  CSV: {results_path}")
    print(f"  Completions: {completions_path}")
    print(f"  Report: {run_dir / 'report.html'}")
    print(f"\n  Open eval/viewer.html in a browser and upload {completions_path}")
    print_summary(results)


if __name__ == "__main__":
    main()
