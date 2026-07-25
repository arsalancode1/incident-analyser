# CLAUDE.md

Context for Claude Code. Read this before changing anything.

## What this is

The metrics tool family for an automated incident debugging agent, targeting a
planet-scale distributed database (Spanner-shaped: multi-region, multi-cell,
metrics in a distributed in-memory TSDB, plus logs, traces, and change tables).

**There is no LLM in this repo.** This is the deterministic layer an orchestrator
calls into. Keep it that way — see the architecture rule below.

## The architecture rule

Determinism belongs in the **tools**, not in the **steps**.

Hand-written playbooks ("check QPS, then check latency") fail on novel incidents
and rot silently. A fully free-form agent wanders, anchors on the first plausible
story, and burns context. So: the orchestrator plans freely and decides what to
call next; everything it calls is validated, budgeted, deterministic, and
returns compact summaries.

Practical consequence for anything you add here: if you find yourself encoding an
investigation *sequence*, stop. Encode the *capability* instead and let the
planner sequence it.

## Two invariants — do not weaken these

**1. A bad selector is an error, not a finding.**

`region="us-east1"` against a fleet using `us-east-1` must raise `ToolError`,
never return `[]`. An empty result is indistinguishable from "that region is
healthy," and an agent will confidently report the latter. This is the single
most dangerous failure mode in the whole system.

`ResultStatus.EMPTY_SELECTOR` exists only for this. `TSDBClient` implementations
must return it — not `OK` with zero series — when a filter matches no known
entity. Errors carry `did_you_mean` suggestions so a typo self-corrects in one turn.

**2. The agent never sees raw time series points.**

Points are token-expensive and models are bad at spotting steps in long float
lists. `summarize.py` reduces a series to ~25 tokens: stats, delta, change point,
shape. Anything you add that returns arrays of points to the tool boundary is a bug.

## Layout

```
metric_analysis/
  types.py        Series, Window, ResultStatus, ToolError
  catalog.py      metric semantics, schema validation, fuzzy suggestions
  tsdb.py         TSDBClient protocol + SyntheticTSDB (tests/replay)
  changepoint.py  onset detection (O(n) Welch scan), shape classification
  summarize.py    series -> compact summary; peer outlier ranking
  attribution.py  explain_delta: additive + exact ratio decomposition
  tools.py        agent-facing surface: validation, budgets, cache, evidence
```

## Commands

```bash
python3 test_metric_analysis.py    # 15 tests, no pytest needed
python3 demo_investigation.py      # end-to-end on a synthetic incident
```

Tests must stay dependency-light (numpy only) and runnable without a real TSDB.
`SyntheticTSDB` is the seed of the replay harness — same shape, with frozen
snapshots of real incidents swapped in later.

## The maths worth understanding before editing `attribution.py`

`attribute_ratio` decomposes a rate change **exactly**:

```
R_c - R_b = Σ (d_c/D_c)(r_c - r_b)      rate effect: the slice itself got worse
          + Σ (d_c/D_c - d_b/D_b) r_b   mix effect: traffic moved to a bad slice
```

These are different incidents with different fixes. Rate effect pages the service
owner; mix effect pages whoever changed routing. A test asserts the two sum to
the true delta within 1e-12 — if you touch this, that test must still pass.

`explain_delta` narrows greedily but has a `cohesion` guard: once past the
explanatory threshold it keeps absorbing values that are comparably large, and
refuses to narrow further when several share the blame. This exists because the
first version reported `job=frontend` alone while `txn-coordinator` was failing
just as hard — plausible, well-formatted, and wrong in a way that sends someone
to the wrong team. Do not remove it for tidier output.

## Conventions

- Type hints everywhere; `from __future__ import annotations`.
- Tool functions return JSON-serializable dicts with an `evidence_id`.
- Every tool result is recorded in the evidence ledger so downstream claims can
  cite it. New tools must do this too.
- Comments explain *why*, especially where a guardrail looks paranoid.
- No new runtime dependencies without discussion. numpy only, so far.

## Known gaps

- **Distribution metrics.** Percentiles aren't additive, so `explain_delta` on
  `spanner.rpc.latency` ranks by peer deviation × volume instead of decomposing.
  Needs a real approach if latency attribution matters.
- **Seasonal baselines.** `Window` supports same-time-last-week; nothing selects
  it automatically.
- **`search_metrics`** is keyword matching standing in for a vector index.
- **Single process.** No regional worker split yet; tool execution should
  eventually run next to the data with only summaries crossing regions.

## Next component

The change-log tool. `find_onset` already emits a precise timestamp whose entire
purpose is to be intersected with deploys, config pushes, and flag flips. In most
incidents that intersection *is* the root cause.
