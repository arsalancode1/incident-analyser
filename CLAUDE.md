# CLAUDE.md

Project-wide context. Read `DESIGN.md` when making architecture decisions —
it has the reasoning behind everything below.

## What we're building

An agent that automatically debugs production incidents on a planet-scale
distributed database (Spanner-shaped: jobs and services across many regions and
cells). Debugging signal lives in four places:

- **Metrics** — planet-scale distributed in-memory TSDB, high-cardinality tags
  (region, cell, job, version, task, alloc)
- **Logs** — code logs, queryable tables
- **Traces** — distributed traces per request
- **Changes** — rollouts, config pushes, flag flips, capacity ops, schema changes

**Success condition:** when a page fires, the investigation is already done by
the time on-call opens their laptop. The deliverable is an incident brief posted
to the incident channel — not a new dashboard, which would fail on adoption.

**Current state: only the metrics component exists.** Everything else below is
unbuilt. Don't assume a module is there because this file mentions it.

## The decision everything else follows from

**Determinism belongs in the tools, not in the steps.**

Hand-written playbooks ("check QPS, then latency, then...") serve one incident
class each, so maintenance scales with the number of failure modes — unbounded,
and they rot silently. A fully free-form agent avoids that but wanders, anchors
on the first plausible story, and gathers only confirming evidence.

So: the orchestrator plans freely and decides what to call next. Everything it
calls is schema-validated, budgeted, deterministic, and returns compact
summaries. A good tool serves every investigation touching that data source, so
maintenance scales with the number of data sources instead — small and stable.

**Practical consequence.** If you find yourself encoding an investigation
*sequence*, stop. Encode the *capability* and let the planner sequence it. The
one exception is a fixed opening sweep before any drill-down, which exists
specifically to prevent tunnel vision.

## Project-wide invariants

These bind every component, not just metrics.

1. **A bad selector is an error, not a finding.** A filter that matches no known
   entity must raise with `did_you_mean` suggestions, never return empty. An
   empty result is indistinguishable from "that slice is healthy," and the agent
   will confidently report the latter. This is the most dangerous failure mode
   in the system.

2. **The agent never sees raw data.** No point arrays, no log dumps, no full
   traces. Reduce in code first: series → stats and change points, logs →
   template frequency diffs, traces → span-level diffs against a fast baseline.
   Anything returning bulk data across the tool boundary is a bug.

3. **Every claim cites evidence.** Tool results carry an `evidence_id` recorded
   in an append-only ledger. A verifier pass — fresh model call that sees claims
   and raw results but *not* the agent's reasoning — strips unsupported claims
   rather than softening them.

4. **No dependency on the system being debugged.** If it stores state in the
   database it debugs, it fails exactly when needed. Audit the full dependency
   closure; any edge back into the debugged system is a bug.

5. **Read-only, budgeted, circuit-broken.** An agent fanning out aggressively
   during a P0 is a real way to turn one incident into two.

6. **"I don't know" is a first-class output.** If evals reward producing an
   answer, the system learns to fabricate one. Explicit "not checked" and
   "ruled out" lists are often more useful to on-call than the conclusion.

## Components

| Component | Status | Notes |
|---|---|---|
| Metrics tools | **built** | `metric_analysis/`, 29 tests; golden signals, hard fixtures |
| Stage-1 brief | **built** | `brief.py`; fixed sweep, materiality gate, no reasoning |
| Change log tool | **next** | rollouts, config, flags, capacity ops |
| Topology tool | not built | dependency graph; without it you can't get from a frontend symptom to a backend cause |
| Log tools | not built | template clustering, frequency diffs |
| Trace tools | not built | exemplars linked from latency buckets, span diffs |
| Harness | **built** | `incident_agent/`; loop control, dispatch, schemas, circuit breakers, replay record |
| Orchestrator | **built** | `incident_agent/`; system prompt, `conclude` schema, verifier pass |
| Skills | **built** | `skills/*.json` + `incident_agent/skills.py`; declarative, deterministic retrieval |
| Hard fixtures | **built** | `scenarios.py`; mix shift, overlapping faults |
| Eval harness | not built | replay over historical incidents, frozen snapshots |

Rough effort split: 80% harness and tools, 15% skills, 5% agent. Teams routinely
plan this backwards.

**Build order.** Change log is next because `find_onset` already emits a precise
timestamp whose entire purpose is to be intersected with deploys — in a large
fraction of incidents that intersection *is* the root cause. Topology after,
since symptom-to-cause traversal needs it.

## Skills are the contribution surface

When built: after a postmortem an SRE writes a declarative note — "Paxos leader
election storms show up as `raft.election_count` spiking with flat QPS; the tell
versus a network partition is `heartbeat_rtt` p50 staying normal while p99
doesn't" — and the agent retrieves it as a *prior* when the symptom matches.

Nobody maintains an execution graph. A missing contribution degrades the system
gracefully instead of breaking it. Keep skills declarative: the moment someone
writes "step 1, step 2" into a skill, we've reinvented the playbook.

## Rollout staging

1. **Tool layer + auto-generated brief.** No reasoning at all — onset, blast
   radius, correlated changes, top contributing dimensions. Saves on-call ten
   minutes per incident and proves the tool layer works.
2. **Hypothesis generation in shadow mode.** Runs live, output hidden, compared
   against what humans concluded.
3. **Advisory.** Trust earned per incident class.

Underclaim in naming. Anything called `AutoRCA` sets an expectation it can't
meet, and the first confidently wrong output turns the name into a joke.

## Conventions

- Python 3.11+, type hints, `from __future__ import annotations`.
- numpy is the only runtime dependency. Don't add more without discussion.
- Tool functions return JSON-serializable dicts including an `evidence_id`.
- Tests stay dependency-light and runnable without a real backend. Synthetic
  fixtures are the seed of the replay harness — same shape, frozen snapshots of
  real incidents swapped in later.
- Comments explain *why*, especially where a guardrail looks paranoid. When you
  add a guardrail, record the bug that motivated it — that's what stops a future
  session deleting it for tidier output.
- Run `python3 test_metric_analysis.py` and `python3 test_incident_agent.py` before
  reporting a change complete.
- The metrics package has no LLM in it and must not acquire one. Model access
  lives in `incident_agent/`, behind a provider-agnostic `ModelClient`; SDKs are
  imported lazily so numpy stays the only hard dependency.

## Further context

- `DESIGN.md` — full architecture reasoning, anti-hallucination scaffolding,
  operational constraints, evaluation strategy. Read when making design
  decisions; not needed every session.
- `metric_analysis/CLAUDE.md` — metrics component specifics. Loads automatically
  when working in that directory.
