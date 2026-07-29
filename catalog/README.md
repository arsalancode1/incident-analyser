# Metric catalog configuration

Add metrics here. No code change, no deploy of `metric_analysis`.

```python
from metric_analysis.config import load_catalog
catalog = load_catalog("catalog/")        # merges every *.json
catalog = load_catalog("catalog/rpc.json")  # or a single file
```

A directory is the shape that survives contact with an organisation: one file
per team, owned by the people who own those metrics. Files load in sorted order,
so the result is deterministic.

## Checking a file after editing it

```bash
python3 -m metric_analysis.check catalog/
```

Validation is strict and errors name the file and metric. That is deliberate — a
typo'd `denominator` silently degrades ratio attribution to additive on a rate
metric, which produces correct-looking output with wrong arithmetic. Failing at
load is far cheaper than discovering it during a P0.

## Fields

`fields` maps a tag name to either a list of values or `null`.

```json
"fields": {
  "region": ["us-east-1", "eu-west-4"],
  "task": null
}
```

- **A list** means "these are all of them". A filter naming anything else is
  rejected at the tool boundary with `did_you_mean` suggestions.
- **`null`** means the values are resolved live via `list_field_values`. Use it
  for anything high-cardinality — task, alloc, instance. A nightly snapshot of
  task names is stale within minutes, and a stale enumeration is worse than
  none because it rejects filters that are actually valid.

For `null` fields the cardinality guard needs a size estimate. Give it one:

```json
"field_cardinality": {"task": 5000}
```

Without a hint it assumes 1000, which is high on purpose — the guard should
refuse a grouping it cannot size rather than discover the blast radius live.

**Invariant 1 still holds for dynamic fields**, but one layer down: the catalog
cannot check the value, so your `TSDBClient.fetch` must return
`EMPTY_SELECTOR` — not `OK` with zero series — when a filter matches no known
entity. This is the single most likely thing to be quietly wrong in a real
backend, and it silently corrupts every conclusion downstream.

## Golden signals

`golden_signal` on a metric marks it as one of the signals swept on every
incident, independently of which metric paged. Declare the order once:

```json
"golden_signals": ["errors", "latency", "traffic", "saturation"]
```

The names are yours — a queueing system might add `queue_depth`, a data pipeline
`freshness`. Order is the order they appear in the brief. If a metric declares a
signal missing from this list, loading fails rather than silently reordering it.

Only mark metrics that describe service health from the outside. Diagnostic
detail like `tablet.split_rate` is not something you would page on, and marking
it as a signal means it competes for attention on every incident.

## The two fields that decide whether any of this works

`description` and `interpretation` are the highest-leverage investment in the
system. A metric name alone tells a model almost nothing and it will invent the
rest. Write what the number counts, and what it means when it moves:

```json
"description": "Failed Spanner RPCs, counted at the serving task. Includes ABORTED from lock contention, which clients usually retry and which may not be user-visible.",
"interpretation": "Divide by spanner.rpc.count for a rate; the raw count follows the diurnal cycle on its own. A rise in rate with flat count is a serving problem; a rise in both is usually load."
```

Generate these nightly from your metric registry, auto-draft them, and have
owners review. This is the part teams underinvest in.
