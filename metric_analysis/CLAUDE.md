# Metrics component

Project-wide rules are in the root `CLAUDE.md`. This file covers what's specific
to the metrics tools. There is no LLM in this package — it's the deterministic
layer the orchestrator calls into. Keep it that way.

## Agent-facing surface (`tools.py`)

```
search_metrics(intent)                          ranked metric specs
describe_metric(metric)                          schema + semantics
list_field_values(metric, field, window)         live values; never guess these
series_summary(metric, filters, incident, base)  stats, delta, onset, shape
find_onset(metric, filters, window)              coarse scan then fine re-query
explain_delta(metric, fields, base, incident)    which slice moved it
peer_comparison(metric, peer_field, incident)    robust-z against siblings
correlation_scan(onset, window)                  what else changed then
```

New tools must validate against the catalog, charge the budget, and record an
`evidence_id`.

## Read `attribution.py` before editing it

`attribute_ratio` decomposes a rate change **exactly**:

```
R_c - R_b = Σ (d_c/D_c)(r_c - r_b)      rate effect: the slice itself got worse
          + Σ (d_c/D_c - d_b/D_b) r_b   mix effect: traffic moved to a bad slice
```

These are different incidents with different fixes. Rate effect pages the service
owner; mix effect pages whoever changed routing. Collapsing them into one number
sends people to the wrong team. A test asserts the two sum to the true delta
within 1e-12 — if you touch this, that test must still pass.

`explain_delta` narrows greedily with a `min_share` guard (default 0.10): once
past the explanatory threshold it keeps absorbing any value still responsible
for at least that share of the change, and refuses to narrow further when
several share the blame.

**Why it's there.** The first version reported `job=frontend` alone at 0.706
explanatory power while `txn-coordinator` at 0.294 was failing just as hard.
Plausible, well-formatted, and wrong in a way that sends someone to the wrong
team. No prompt could have caught this — it was arithmetic in the tool layer,
which is the whole argument for keeping arithmetic out of the model.

**Why the floor is absolute, not relative.** The guard originally compared each
value against 25% of the last one taken, which puts the bar at whatever height
the biggest contributor happens to land on: an 0.81/0.19 split dropped a broken
slice while 0.70/0.29 kept it. Materiality doesn't depend on how big the largest
contributor was. Don't reintroduce a relative test, and don't remove the guard
for tidier output.

## Fixture reproducibility

`SyntheticTSDB` must be byte-identical across processes — it's the seed of the
replay harness, and a fixture that reshapes itself makes replay meaningless.

Use `stable_hash()` from `tsdb.py`, never Python's builtin `hash()`, which is
salted per process (PEP 456). That bug produced a test failing roughly one run
in six, which is worse than a hard failure because it reads as noise. Two tests
guard it now; don't weaken them, and don't paper over failures with
`PYTHONHASHSEED`.

## Time series specifics

- **Two-pass onset detection.** Coarse buckets to locate, fine re-query in a
  narrow band to pin. The precise timestamp is what gets intersected with the
  deploy log, so the second query pays for itself.
- **Peer baselines beat temporal baselines.** "8x the median of its 47 siblings"
  survives a traffic spike that fools a week-over-week delta.
- **Aggregation is pushed server-side.** Never pull per-task series and roll up
  in the client.
- **Progressive narrowing**, never full-cardinality queries: global → region →
  cell → job → task.

## Wiring a real TSDB

Implement `TSDBClient.fetch` and `.field_values` (see `tsdb.py`). Two hard
requirements: server-side aggregation, and `fetch` must return `EMPTY_SELECTOR`
— not `OK` with zero series — when the filter matches no known entity. That
second one is the most likely thing to be quietly wrong in a real backend, and
it silently corrupts every conclusion downstream.

Keep `SyntheticTSDB` working; it's what the tests and replay harness run against.

Generate the catalog nightly from the metric registry. Auto-draft descriptions,
have owners review them. The agent is close to useless without good descriptions
— this is the part that needs real investment and the part teams underinvest in.

## Known gaps

- **Distribution metrics.** Percentiles aren't additive, so `explain_delta` on
  `spanner.rpc.latency` ranks by peer deviation × volume instead of decomposing.
  Needs a real approach if latency attribution matters.
- **Seasonal baselines.** `Window` supports same-time-last-week; nothing selects
  it automatically.
- **`search_metrics`** is keyword matching standing in for a vector index.
- **Single process.** Tool execution should eventually run next to the data in
  each region, with only summaries crossing regions.
