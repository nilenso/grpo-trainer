# Implementation Plan: Eval Comparison Viewer + Enhanced Eval Metrics

## Overview

Build a **standalone static HTML viewer** (`eval/viewer.html`) that handles both single-run summary dashboards and two-run side-by-side comparison, purpose-built for comparing a **base/SFT model against an RLVR+GRPO trained model** to understand how training impacted capabilities. Enhance the eval script to capture token usage, latency, and GRPO-relevant behavioral metrics. All per-problem data lives in a single **completions JSONL** file (enriched with new fields + a metadata header line). The existing `report.py` / `report_template.html` pipeline is kept as-is for quick Jinja2-rendered reports.

### GRPO/RLVR Training Context

Research shows RLVR+GRPO training produces specific, predictable behavioral shifts:
- **Response length inflation** — GRPO's sequence-level loss normalization biases toward longer outputs
- **Reward distribution becomes bimodal** — problems migrate to "always solved" or "never solved"; the informative middle shrinks
- **Difficulty-tier asymmetry** — easy problems saturate fast, medium problems see the most improvement, hard problems can freeze without large rollout budgets
- **Imperative bias in functional languages** — RL-trained models discover that `ref`/loops reliably pass unit tests, degrading functional style (documented: GPT-5 produces imperative OCaml patterns in 80% of outputs vs 42% for GPT-4o)
- **Exploration narrows** — the model amplifies known high-reward paths, reducing solution diversity
- **All reasoning paths already existed in the base model** — RL sharpens probability over existing paths, doesn't create new ones

The viewer is designed to surface these phenomena.

---

## Files Affected

| File | Action | Purpose |
|---|---|---|
| `eval/eval.py` | Modify | Add token usage, tokens/sec; write run metadata header into JSONL; add run timestamp to every record |
| `eval/metrics.py` | Modify | Add percentile latency stats, token usage aggregates, partial test pass rate |
| `eval/constants.py` | Modify | Add new fieldnames for extended metrics |
| `eval/compare.py` | Modify | Extend comparison with delta metrics for new fields; support JSONL input |
| `eval/viewer.html` | **Create** | Standalone static HTML — single-run dashboard + two-run comparison viewer |
| `eval/report.py` | Keep | Unchanged unless new metrics need wiring to the Jinja2 report |
| `eval/report_template.html` | Keep | Unchanged unless new metrics need wiring to the Jinja2 report |

---

## Data Format: Enriched Completions JSONL

The completions JSONL becomes the **single source of truth** for the viewer. Each file has:

**Line 1 — Run metadata (`_meta: true` sentinel):**
```json
{
  "_meta": true,
  "run_id": "qwen2.5-ocamler_20260227_110000",
  "model": "qwen2.5-ocamler",
  "dataset": "kiranpg/ocaml-eval-problems",
  "timestamp": "2026-02-27T11:00:00+00:00",
  "total": 500, "passed": 340,
  "pass_rate": 68.0, "type_check_rate": 85.2, "compile_rate": 82.0, "test_pass_rate": 70.4,
  "avg_reward": 0.682, "avg_tpr": 0.71,
  "avg_gen_time": 1.43, "p50_gen_time": 1.21, "p90_gen_time": 2.87, "p99_gen_time": 5.12,
  "total_prompt_tokens": 125000, "total_completion_tokens": 89000, "avg_tokens_per_sec": 62.3,
  "avg_completion_length": 287, "p50_completion_length": 245, "p90_completion_length": 510,
  "imperative_ratio": 0.12,
  "failure_stages": {"compile": 45, "execution:test_fail": 80},
  "difficulty_stats": {"easy": {"total": 200, "passed": 170}}
}
```

**Lines 2+ — One per problem:**
```json
{
  "id": "medium_118", "difficulty": "medium",
  "problem_text": "...", "tests": "...",
  "raw_completion": "...", "full_completion": "...",
  "total_reward": 1.0, "base_reward": 0.65,
  "type_score": 0.25, "compile_score": 0.1, "test_score": 0.65,
  "failure_stage": "",
  "generation_time_sec": 1.43,
  "completion_length": 312,
  "prompt_tokens": 245, "completion_tokens": 89, "total_tokens": 334,
  "tokens_per_sec": 62.2,
  "uses_imperative": true,
  "run_id": "qwen2.5-ocamler_20260227_110000",
  "timestamp": "2026-02-27T11:00:00+00:00",
  "model": "qwen2.5-ocamler"
}
```

`run_id` (model + datetime) and `timestamp` appear in both the `_meta` header and every problem record, so individual lines are self-identifying even if the file is split or concatenated.

---

## Checkpoint 1: Enhance `eval.py` — Token Usage & Latency Capture

**Goal:** Extract `usage` and `timings` from API responses, write enriched JSONL with metadata header.

### Changes to `call_openai_api()`:
- Return `(content, usage_dict)` instead of just `content`
- Parse `response.json()["usage"]` → `{prompt_tokens, completion_tokens, total_tokens}` or `{}` if absent
- Parse `response.json().get("timings", {})` (llama.cpp vendor extension) — extract `predicted_per_second` when present
- Never fail if fields are missing — default to `{}`

### Changes to `generate_solution()`:
- Return `(completion, generation_time, usage)`
- Switch from `time.perf_counter()` to `time.monotonic()`

### New function `detect_imperative_style(code: str) -> bool`:
- Scan generated OCaml for imperative patterns: `ref `, `:=`, `!` (dereference), `for `, `while `, `Array.set`, `Bytes.set`
- Returns `True` if any are found — a key GRPO training degradation signal (RL-trained models discover imperative patterns pass unit tests reliably, at the cost of functional style)

### Changes to `build_result()` / `build_completion()`:
- Add fields: `prompt_tokens`, `completion_tokens`, `total_tokens`, `tokens_per_sec`, `uses_imperative`
- `tokens_per_sec`: computed as `completion_tokens / generation_time` when both are available, else `None`
- `uses_imperative`: from `detect_imperative_style(full_completion)`
- All timing/token fields nullable — `None` serialises as JSON `null`

### Changes to `write_completions()`:
- Compute aggregate metrics via `compute_metrics()` + `compute_failure_stages()` + `compute_difficulty_stats()`
- Write `_meta` JSON line first, then per-problem lines
- Include `run_id` and `timestamp` in meta and every record

### Changes to `main()`:
- Generate `run_id = f"{model_name}_{timestamp}"`
- Pass `run_id` and ISO timestamp through to builders
- Print path to `viewer.html` with usage hint: `"Open eval/viewer.html in a browser and upload this file"`

### Verification:
- `uv run python -m eval.eval --limit 3`, inspect JSONL for `_meta` line, new fields, and `null` for missing timing data

---

## Checkpoint 2: Enhance `metrics.py` — Richer Aggregates

**Goal:** Add percentile latency, token aggregates, and partial test pass rate.

### New helper:
```python
def compute_percentile(values: list[float], p: float) -> float | None
```
Simple sorted-interpolation percentile, no numpy. Returns `None` for empty lists.

### Changes to `compute_metrics()` — add:
- `p50_gen_time`, `p90_gen_time`, `p99_gen_time` — only computed when timing data exists; `None` otherwise
- `avg_tokens_per_sec`, `median_tokens_per_sec` — only from records where `tokens_per_sec is not None`
- `total_prompt_tokens`, `total_completion_tokens`, `total_tokens` — sum of non-null values
- `avg_prompt_tokens`, `avg_completion_tokens` — mean of non-null values
- `avg_tpr` — mean of `test_score` across all results (partial credit metric)
- `avg_completion_length` — mean of `completion_length`
- `p50_completion_length`, `p90_completion_length` — GRPO inflates response length; percentiles reveal the distribution shape
- `imperative_ratio` — fraction of completions where `uses_imperative` is true; key GRPO degradation signal

All new fields are `None` when insufficient data exists — the viewer shows "N/A".

### Verification:
- Call `compute_metrics()` with mock data (some records missing token fields), confirm graceful `None` values

---

## Checkpoint 3: Enhance `compare.py` — Extended Deltas

**Goal:** Add deltas for token usage and latency, plus deeper change-analysis metrics.

### Changes:
- Add comparison rows: `Avg Tokens/sec`, `Avg Completion Tokens`, `P50 Latency`, `P90 Latency`, `Avg TPR`
- Skip rows where either side is `None` (print "N/A" instead of delta)
- Track per-problem reward magnitude delta (not just binary flip) — include `reward_delta` in improved/regressed lists
- Support loading from enriched JSONL: skip lines where `_meta` is true, parse remaining as records
- **New: `compute_stage_migration(results_a, results_b)`** — returns a dict-of-dicts mapping `{baseline_stage: {current_stage: count}}` for the failure stage migration matrix
- **New: `compute_stage_progression(results_a, results_b)`** — given the pipeline order (`degenerate → type_check → compile → execution → pass`), classify each problem as progressed / regressed / same and return counts + lists
- **New: `compute_per_difficulty_deltas(results_a, results_b)`** — pass rate delta broken down by difficulty level
- **New: `compute_score_distributions(results)`** — return binned histograms for `type_score`, `compile_score`, `test_score`, `total_reward` (used by viewer for overlay charts)
- **New: `compute_length_distributions(results)`** — return binned histograms for `completion_length` (GRPO length inflation analysis)
- **New: `compute_imperative_comparison(results_a, results_b)`** — return counts: `{both_imperative, both_functional, became_imperative, became_functional}` — surfaces style degradation from GRPO training
- **New: `compute_frozen_problems(results_a, results_b)`** — return list of problem IDs where both runs fail with same failure stage and reward delta < 0.05 — these are problems GRPO couldn't move

### Verification:
- Compare two JSONL files via CLI, verify new rows appear; verify graceful N/A when one run has no token data
- Verify stage migration matrix output is correct against manual inspection

---

## Checkpoint 4: Build `eval/viewer.html` — The Standalone Viewer

**Goal:** A single static HTML file (no server, no build) that handles both single-run dashboards and two-run comparison.

### Theme:
- **Catppuccin Mocha** (dark) and **Catppuccin Latte** (light), matching `dashboard/completions.html`
- Toggle button in header, preference saved to `localStorage`

### Landing state:
- Centered drop zone: "Drop 1 JSONL for dashboard, or 2 for comparison"
- File input buttons as fallback
- Parses `_meta` line for instant header rendering before processing all records

---

### Mode 1 — Single Run Dashboard (1 JSONL):

**Top metric cards row:**
- Total Problems, Pass@1, Type Check Rate, Compile Rate, Test Pass Rate, Avg Reward
- **New cards:** Total Tokens, Avg Tokens/sec, P50 / P90 Latency, Avg Completion Length, Imperative Ratio
- Cards with `null` data show "N/A" in muted style

**Charts row (Chart.js):**
- Failure stage doughnut
- Pass rate by difficulty stacked bar
- **New:** Latency distribution histogram (bucketed generation times) — hidden if no timing data
- **New:** Token usage scatter (prompt vs completion tokens, colored by pass/fail) — hidden if no token data
- **New:** Completion length distribution histogram — reveals verbosity patterns
- **New:** Reward distribution histogram — shows whether solutions cluster at 0/1 (bimodal) or spread across partial scores

**Failure stage details table**

**Configuration/metadata table** (model, dataset, timing stats, token stats)

**Below dashboard: scrollable problem list** with expandable rows — click to reveal problem text, generated code (highlight.js), test cases (collapsible, default closed), score breakdown

---

### Mode 2 — Comparison View (2 JSONLs):

**Top bar:**
- File names with `run_id` / `timestamp` for each
- Summary deltas: `N compared | X improved | Y regressed | Z progressed (further in pipeline) | Δ pass rate | Δ latency | Δ tokens/sec`
- Delta values show "N/A" when timing/token data unavailable in either run

**Aggregate comparison section *(collapsible, default open)*:**

Paired metric cards with deltas, designed to surface improvements/regressions at a glance:

- **Pass rate Δ** — with arrow and color
- **Avg reward Δ** — captures partial improvements invisible to binary pass/fail
- **Avg TPR Δ** (average test pass rate) — partial credit; a model that fails fewer tests is improving even without flipping pass/fail
- **Stage progression summary** — stacked bar showing how many problems progressed / regressed / stayed in the pipeline (e.g. "45 progressed, 12 regressed, 443 same")
- **Failure stage migration matrix** — compact heatmap/table: rows = baseline failure stage, columns = current failure stage, cell = count. Instantly shows where problems moved (e.g. 20 problems went from `compile` → `execution:test_fail`, 5 went from `execution:test_fail` → `pass`). This is the single most informative view for understanding training impact.
- **Per-difficulty delta cards** — pass rate Δ broken down by easy/medium/hard; highlights if training helped one tier but hurt another
- **Score component histograms** — overlaid distributions of `type_score`, `compile_score`, `test_score` for baseline vs current; shows if the distribution shifted even when top-line pass rate didn't move much
- **Reward distribution overlay** — histogram of `total_reward` for both runs overlaid; reveals whether the model is producing more partial solutions or just flipping binary pass/fail
- **Latency / token comparison** — side-by-side if timing data available, N/A otherwise

**GRPO Training Impact panel *(collapsible, default open)*:**

Metrics specifically designed to surface RLVR+GRPO behavioral shifts:

- **Completion length distribution overlay** — histogram of `completion_length` for baseline vs current, overlaid. GRPO typically inflates response length; this reveals the magnitude and whether it's uniform or concentrated in certain difficulty tiers. Show median + p90 as vertical markers.
- **Imperative bias comparison** — bar chart: `imperative_ratio` for baseline vs current. If GRPO training increased imperative style usage, this surfaces it immediately. Clicking the bar filters the sidebar to show only imperative-style solutions.
- **Reward bimodality analysis** — reward distribution histograms overlaid. A healthy GRPO run shifts mass from the middle toward 1.0 (solved); an over-trained model becomes bimodal (mass at 0.0 and 1.0, hollow middle). Annotate the histogram with the fraction of problems in three bands: `[0, 0.3)` (clear fail), `[0.3, 0.7)` (partial), `[0.7, 1.0]` (pass/near-pass).
- **Difficulty tier impact table** — for each difficulty (easy/medium/hard): pass rate Δ, avg reward Δ, avg completion length Δ, imperative ratio Δ. Research shows easy problems saturate fast, medium problems improve most, hard problems can freeze — this table reveals if that pattern holds.
- **"Frozen" hard problems callout** — count of hard problems where both runs fail with identical failure stage and reward delta < 0.05. These are problems GRPO training couldn't move the needle on, suggesting they may need larger rollout budgets or curriculum adjustments.

**Left sidebar — Problem list:**
- Searchable, filterable list matched by `id`
- Each entry shows:
  - `#id`, difficulty badge
  - Reward delta (`+0.35`, `-1.00`) with color (green positive, red negative)
  - Stage transition label when failure stage changed (e.g. `compile → pass`, `pass → execution:test_fail`, `type_check:syntax → compile`)
  - Colored status badges: 🟢 improved, 🔴 regressed, ⚪ unchanged
- **Default sort: by absolute reward delta descending** — biggest changes surface first
- **Filters:**
  - Difficulty: easy / medium / hard / all
  - Change: improved / regressed / both pass / both fail / all
  - Failure stage (current run): dynamically populated from data (e.g. compile / execution:test_fail / type_check:syntax / all)
  - Failure stage (baseline run): same dynamic list — allows cross-filtering e.g. "baseline was compile, current is anything"
  - Stage transition: progressed (got further in pipeline even if still failing) / regressed (failed earlier) / same stage / all
  - Style: imperative (either run) / functional only (neither run) / became imperative (base functional → current imperative) / all
- **Sort:** reward delta (default) / absolute reward delta / latency delta / completion length delta / ID
- Count: "Showing 42 of 500"

**Stage transition logic:**
Pipeline order: `degenerate → type_check → compile → execution → pass`. A problem that moved from `compile` failure to `execution:test_fail` has *progressed* (got further) even though it still fails. This is a key insight for training — the model learned something even if it didn't fully solve the problem. The "stage transition" filter and label expose this.

**Right panel — Problem detail (side-by-side):**

All sections are **collapsible** (`<details>`/`<summary>`) so the user can focus on what matters. Default open/closed states noted below.

- Shared header: problem description + signature *(collapsible, default open)*
- Test cases *(collapsible, default closed)* — can be long, hidden until needed
- Two sub-panels, each showing:
  - Label ("Baseline" / "Current") + model name + run timestamp
  - Score pills with per-component deltas: type ✓/✗ (Δ), compile ✓/✗ (Δ), tests ✓/✗ (Δ), total reward (Δ) — deltas highlighted green/red when a component flipped between runs
  - Failure stage label (e.g. `compile:timeout`, `execution:test_fail`) — shown when the problem failed, hidden on pass
  - Stage transition arrow between the two panels when failure stage changed (e.g. `compile → execution:test_fail`)
  - `imperative` badge — shown when `uses_imperative` is true; orange/warning color to flag style degradation. When comparing, highlight if the style *changed* (e.g. base was functional, current uses imperative)
  - Stats row: generation time, completion length (with Δ), token count, tokens/sec (or "N/A") *(collapsible, default open)*
  - **Code panel:** Generated OCaml with highlight.js syntax highlighting *(collapsible, default open)*

Not a diff view — independent generations, so side-by-side with independent highlighting.

---

### Libraries (all CDN):
- `highlight.js` + `ocaml` language pack (~50KB)
- `Chart.js` (already used across project)
- Google Sans font (matching dashboard)

### Performance — large JSONL handling:

| Technique | Where | Why |
|---|---|---|
| **Streaming line parse** | File load | Parse JSONL line-by-line via `TextDecoder` + split on `\n`; don't `JSON.parse` the entire file as one string |
| **Extract `_meta` first** | File load | Render dashboard header instantly from line 1 before parsing remaining 1000+ lines |
| **Store lightweight index** | Memory | Sidebar list holds only `{id, difficulty, total_reward, generation_time_sec, failure_stage, tokens_per_sec}` per problem — full record (with `problem_text`, `raw_completion`, etc.) stays in a `Map` but code strings are only read on click |
| **Lazy highlight** | Code panels | Call `hljs.highlightElement()` only when a problem is selected, not for all 500+ upfront |
| **Debounced search** | Sidebar filter | 150ms debounce on keystroke filtering |
| **`requestAnimationFrame` batching** | Sidebar render | Render sidebar items in batches of 50 per frame to avoid blocking the main thread |
| **Dispose charts on mode switch** | Charts | Destroy Chart.js instances before creating new ones to prevent memory leaks |

### Verification:
- Upload 1 JSONL → dashboard renders with all cards/charts, N/A for missing data
- Upload 2 JSONLs → comparison view with sidebar, filters, side-by-side code
- Test with a large file (1000+ records) — confirm no jank on load or scroll
- Test theme toggle, all filter combinations, search

---

## Checkpoint 5: Integration & Wiring

- **Update** `eval/__init__.py` — no changes needed (report.py stays)
- **Update** `eval/eval.py` `main()` — keep `generate_html_report()` call for backward compat; add viewer.html usage hint to summary output
- Ensure `results.csv` is still written (lightweight summary for programmatic use / CI)
- Verify `scripts/run-eval.sh` still works unchanged

### Verification:
- Full end-to-end: `uv run python -m eval.eval --limit 5` → open `viewer.html` → upload the generated JSONL → inspect dashboard
- Comparison: upload two JSONLs from different eval runs → verify side-by-side, confirm `run_id` / timestamps distinguish them

---

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| vLLM doesn't return `timings` like llama.cpp | Compute `tokens_per_sec` from `usage.completion_tokens / elapsed`; all timing fields nullable, viewer shows N/A |
| `usage` field missing from some backends | Default to `None`; viewer shows "—" gracefully; no card/chart rendered when all null |
| Large JSONL (1000+ problems) slows browser | Streaming parse, lightweight sidebar index, lazy code highlighting, batched DOM rendering |
| Concatenated/split JSONL files lose context | `run_id` + `timestamp` on every record makes lines self-identifying |
| Old JSONL files lack new fields | Viewer treats missing fields as `null` → N/A; fully backward-compatible |
| Imperative detection false positives | Simple regex heuristic; `!` can be logical not in some contexts. Accept as approximation — flag in UI as "heuristic" with tooltip explaining the patterns matched |

## Out of Scope
- pass@k with k>1 (needs multiple generations per problem)
- CI regression testing integration
- Monaco/CodeMirror (overkill for read-only)
- Streaming API / TTFT measurement (follow-up — requires switching `call_openai_api` to streaming)
- Changes to `report.py` / `report_template.html` (only touched if the new metrics from checkpoints 1-2 need to flow into the existing Jinja2 report)
