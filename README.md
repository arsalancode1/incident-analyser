# metric-analysis

The metrics tool family for the incident debugging agent. No LLM anywhere in
here — this is the deterministic layer the orchestrator calls into.

```
python3 test_metric_analysis.py     # 17 tests
python3 demo_investigation.py       # end-to-end on a synthetic incident
```

## Layout

| Module | Role |
|---|---|
| `types.py` | `Series`, `Window`, `ResultStatus`, `ToolError` |
| `catalog.py` | Metric semantics, schema, validation, fuzzy suggestions |
| `tsdb.py` | `TSDBClient` protocol + `SyntheticTSDB` for tests and replay |
| `changepoint.py` | Onset detection (O(n) Welch scan) and shape classification |
| `summarize.py` | Series → ~25 tokens; peer outlier ranking |
| `attribution.py` | `explain_delta`: additive and exact ratio decomposition |
| `tools.py` | Agent-facing surface: validation, budgets, cache, evidence ledger |

## Agent-facing tools

```
search_metrics(intent)                          -> ranked metric specs
describe_metric(metric)                          -> schema + semantics
list_field_values(metric, field, window)         -> live values; never guess these
series_summary(metric, filters, incident, base)  -> stats, delta, onset, shape
find_onset(metric, filters, window)              -> coarse scan then fine re-query
explain_delta(metric, fields, base, incident)    -> which slice moved it
peer_comparison(metric, peer_field, incident)    -> robust-z against siblings
correlation_scan(onset, window)                  -> what else changed then
```

## Two invariants worth defending in code review

**A bad selector is an error, not a finding.** `region="us-east1"` against a
fleet using `us-east-1` must raise, never return `[]`. An empty result is
indistinguishable from "that region is healthy", and an agent will happily
report the latter. `ResultStatus.EMPTY_SELECTOR` exists solely for this.

**Rate effect and mix effect are different incidents.** `attribute_ratio`
decomposes a ratio change exactly:

```
R_c - R_b = Σ (d_c/D_c)(r_c - r_b)        rate effect: the slice got worse
          + Σ (d_c/D_c - d_b/D_b) r_b     mix effect: traffic moved to a bad slice
```

The first pages the service owner. The second pages whoever changed routing.
Collapsing them into one number sends people to the wrong team.

## Wiring to a real TSDB

Implement `TSDBClient.fetch` and `.field_values` (see `tsdb.py`). Two hard
requirements: aggregation must be pushed server-side, and `fetch` must return
`EMPTY_SELECTOR` — not `OK` with zero series — when the filter matches no known
entity.

Generate the catalog nightly from your metric registry. Auto-draft descriptions,
have owners review them. The agent is close to useless without good
descriptions; this is the part that needs real investment.

## Not built yet

- Distribution metrics: percentiles are not additive, so `explain_delta` on
  `spanner.rpc.latency` currently ranks by peer deviation × volume rather than
  decomposing. Needs a proper approach if latency attribution matters.
- Seasonal baselines: `Window` supports same-time-last-week, but nothing picks
  it automatically.
- Regional worker split. Everything runs in one process today.
- The `search_metrics` stub is keyword matching. Swap in a vector index.
