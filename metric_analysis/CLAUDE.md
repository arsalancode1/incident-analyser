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

## The opening sweep and the materiality gate

`brief.py` runs a fixed sweep: describe, fleet summary, every golden signal,
peers, then — only if the change is established as real — attribution, onset,
correlation. The gate is the important part. `explain_delta` decomposes whatever
delta it is handed and explanatory power is a *share* of that delta, so on a
flat metric some slice still owns 70% of the noise. The first version of this
brief reported a confident location for a completely healthy fleet. Three
independent signals each suffice: a significant fleet change point, a delta past
a floor, or a peer outlier — fleet delta alone would miss one small cell on fire.

Golden signals are swept per-signal and judged on their own terms, not for
correlation with the symptom, because the paged metric is one view of the
service and errors can be flat while traffic collapses. They are configured, not
hardcoded; the four classic names are a default.

`mechanism` (rate versus mix) is computed **once over the joint partition** of
all requested dimensions, not per narrowing level. Cell names repeat across
regions, so grouping by cell alone pools a shifting cell with its healthy
namesakes and a pure traffic shift reappears as a rate effect. A test pins the
verdict across three field orderings.

## Fixtures worth failing

`scenarios.py` carries three: `bad_rollout` (the original, kept byte-identical
as the replay seed), `mix_shift` (traffic moves toward an always-bad cell;
nothing degrades — reporting it as degradation pages the wrong team), and
`overlapping_faults` (two unrelated faults; naming one culprit is the failure).
The last two found real bugs within minutes of existing. A fixture that cannot
be failed measures the fixture, not the system.

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

## Configuring the catalog

`demo_catalog()` is a Python fixture because tests need it byte-identical and
file-free. Real catalogs are JSON under `catalog/`, loaded by
`config.load_catalog()` — a file or a directory of files, merged in sorted
order. Adding a metric is a data change, no code and no deploy.

Validate before it reaches an incident: `python3 -m metric_analysis.check catalog/`.
Validation is strict on purpose. A typo'd `denominator` silently degrades ratio
attribution to additive on a rate metric, which is correct-looking output with
wrong arithmetic, and a P0 is the worst time to discover it.

`fields` maps a tag to a list of values or to `null`. A list means "these are
all of them" and anything else is rejected with suggestions. `null` means
resolved live via `list_field_values` — correct for task, alloc, instance, where
a nightly snapshot is stale within minutes and a stale enumeration rejects
filters that are actually valid. **For `null` fields invariant 1 moves one layer
down**: the catalog cannot check the value, so `TSDBClient.fetch` must return
`EMPTY_SELECTOR`. Give dynamic fields a `field_cardinality` hint or the guard
assumes 1000.

`golden_signals` is configurable and the names are yours — a queueing system
might add `queue_depth`. Only tag metrics describing health from the outside;
tagging diagnostic detail means it competes for attention on every incident.

## Wiring a real TSDB

Implement `TSDBClient.fetch` and `.field_values` (see `tsdb.py`). Two hard
requirements: server-side aggregation, and `fetch` must return `EMPTY_SELECTOR`
— not `OK` with zero series — when the filter matches no known entity. That
second one is the most likely thing to be quietly wrong in a real backend, and
it silently corrupts every conclusion downstream.

**Run `contract.check_contract` before trusting any output from a new backend.**
It checks the behaviour the analysis layer assumes and cannot verify per-query:
the EMPTY_SELECTOR contract, NO_DATA versus EMPTY_SELECTOR, sample ordering and
alignment, group_by labelling, and determinism. Verified to catch each of those
by injecting them into the reference client. It also distinguishes a
misconfigured probe from a backend defect — a suite that cries wolf gets skipped
exactly when it matters.

`monarch.py` is a scaffold for Monarch: query shaping, filter routing between
target and metric fields, alignment by metric kind, and an existence-probe
strategy for EMPTY_SELECTOR. It has never run against a real instance — supply
an executor, then run the contract suite.

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
