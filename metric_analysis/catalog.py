"""Metric semantics, schema, and validation.

Two jobs. First, tell the agent what a metric *means* -- the descriptions here
are the single highest-leverage investment in the whole system, because a metric
name alone tells a model almost nothing and it will invent the rest. Second,
reject malformed selectors before they reach the TSDB, with suggestions
attached, so a typo costs one turn instead of producing a confident wrong answer
(invariant 1).

In production this is generated nightly from the metric registry, with
auto-drafted descriptions that owners review. `demo_catalog()` is the fixture
that stands in for it in tests and replay.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Any

from .types import ToolError

# Fixed order so the sweep is deterministic and reads the way an engineer scans
# a dashboard: what is broken, how slow, how much load, how full.
GOLDEN_SIGNALS = ("errors", "latency", "traffic", "saturation")

# The synthetic fleet's vocabulary. `SyntheticTSDB` imports this rather than
# keeping its own copy: if the catalog and the TSDB disagree about what exists,
# invariant 1 becomes untestable -- a filter the catalog accepts would come back
# empty from storage, which is exactly the failure mode the invariant forbids.
#
# Cell names repeat across regions on purpose. If cell names were globally
# unique, `cell=fb` would identify a region implicitly and attribution could
# short-circuit the region dimension; keeping them shared means region and cell
# are genuinely independent dimensions, as they are in a real multi-region fleet
# where you must say which region's cell `fb` you mean.
DEMO_FLEET: dict[str, list[str]] = {
    "region": ["us-east-1", "us-central-1", "eu-west-4", "asia-east-1"],
    "cell": ["aa", "ab", "ba", "bb", "fa", "fb"],
    "job": ["frontend", "txn-coordinator", "tablet-server"],
    "task": [f"{i:03d}" for i in range(8)],
    "version": ["v2.40", "v2.41"],
}


@dataclass
class MetricSpec:
    """What a metric is, what its tags are, and how to reason about it."""

    name: str
    kind: str  # counter | gauge | distribution
    unit: str
    description: str
    # Tag name -> allowed values, or None for a field whose values are resolved
    # live. A planet-scale fleet has far too many task names to enumerate in a
    # nightly snapshot, and a stale enumeration is worse than none: it rejects
    # filters that are actually valid. For dynamic fields, invariant 1 is
    # enforced one layer down, by the TSDB returning EMPTY_SELECTOR.
    fields: dict[str, list[str] | None] = field(default_factory=dict)
    # Estimates for dynamic fields, used by the cardinality guard.
    field_cardinality: dict[str, int] = field(default_factory=dict)
    interpretation: str = ""
    denominator: str | None = None
    percentile: str | None = None
    # Which of the four golden signals this metric represents, if any. Most
    # metrics are None: `spanner.tablet.split_rate` is diagnostic detail, not a
    # signal you would page on. Only the handful that describe service health
    # from the outside get a role here.
    golden_signal: str | None = None

    # Assumed size of a dynamic field when nothing better is configured. High
    # on purpose: the cardinality guard should refuse an unbounded grouping it
    # cannot size, not wave it through and discover the blast radius live.
    DEFAULT_DYNAMIC_CARDINALITY = 1000

    def cardinality_of(self, field_name: str) -> int:
        values = self.fields.get(field_name)
        if values is not None:
            return len(values)
        return self.field_cardinality.get(field_name, self.DEFAULT_DYNAMIC_CARDINALITY)

    @property
    def aggregation(self) -> str:
        """How to combine series when grouping. Counters sum; everything else
        averages. Distributions *should* merge sketches -- averaging quantiles
        is wrong, and it is the known gap called out in CLAUDE.md."""
        return "sum" if self.kind == "counter" else "avg"

    def summary(self) -> dict[str, Any]:
        return {
            "metric": self.name,
            "kind": self.kind,
            "unit": self.unit,
            "aggregation": self.aggregation,
            "description": self.description,
            "interpretation": self.interpretation,
            "denominator": self.denominator,
            "percentile": self.percentile,
            # Cardinality only. Values come from list_field_values, which reads
            # live -- a catalog snapshot of task names is stale within minutes.
            "fields": {
                k: {"cardinality": self.cardinality_of(k), "dynamic": v is None}
                for k, v in self.fields.items()
            },
        }


class MetricCatalog:
    def __init__(
        self, specs: list[MetricSpec], signal_order: tuple[str, ...] = GOLDEN_SIGNALS
    ) -> None:
        self._specs: dict[str, MetricSpec] = {s.name: s for s in specs}
        # Configurable so a fleet with its own vocabulary -- "queue_depth",
        # "freshness" -- can order its own signals without editing code.
        self.signal_order = tuple(signal_order)

    def names(self) -> list[str]:
        return list(self._specs)

    def get(self, metric: str) -> MetricSpec:
        spec = self._specs.get(metric)
        if spec is None:
            raise ToolError(
                "unknown_metric",
                f"No metric named {metric!r} in the catalog.",
                difflib.get_close_matches(metric, self.names(), n=5, cutoff=0.5),
            )
        return spec

    def validate_filter(self, metric: str, filters: dict[str, str]) -> None:
        """Invariant 1 lives here.

        A filter naming something that does not exist is a malformed query. It
        must raise, never quietly select nothing, because zero series and zero
        problems look identical downstream.
        """
        spec = self.get(metric)
        for k, v in filters.items():
            if k not in spec.fields:
                raise ToolError(
                    "unknown_field",
                    f"{k!r} is not a tag on {metric!r}.",
                    list(spec.fields),
                )
            allowed = spec.fields[k]
            if allowed is None:
                # Values resolved live, so there is no list to check against.
                # The TSDB must return EMPTY_SELECTOR for a filter matching no
                # entity -- that contract is what keeps invariant 1 intact here.
                continue
            if v not in allowed:
                near = difflib.get_close_matches(v, allowed, n=5, cutoff=0.5)
                raise ToolError(
                    "unknown_field_value",
                    f"{v!r} is not a value of {k!r} on {metric!r}. Returning an "
                    f"empty result here would be indistinguishable from a healthy "
                    f"one, so this is an error.",
                    near or allowed[:8],
                )

    def group_by_check(self, metric: str, group_by: list[str]) -> None:
        spec = self.get(metric)
        bad = [f for f in group_by if f not in spec.fields]
        if bad:
            raise ToolError(
                "unknown_field",
                f"Cannot group {metric!r} by {bad}: not tags on this metric.",
                list(spec.fields),
            )

    def estimate_cardinality(
        self,
        metric: str,
        filters: dict[str, str],
        group_by: list[str],
    ) -> int:
        """Upper bound on returned series. Deliberately an over-estimate: the
        guard should refuse a query that *might* blow up, not discover it did."""
        spec = self.get(metric)
        n = 1
        for g in group_by:
            if g in filters:
                continue
            values = spec.fields.get(g)
            n *= max(1, len(values) if values is not None else spec.cardinality_of(g))
        return n

    def golden_signals(self) -> dict[str, list[str]]:
        """Signal role -> metrics carrying it, in GOLDEN_SIGNALS order.

        A signal can legitimately map to several metrics in a real fleet (two
        serving paths, two error counters), so this is a list per signal rather
        than one name.
        """
        out: dict[str, list[str]] = {}
        for spec in self._specs.values():
            if spec.golden_signal:
                out.setdefault(spec.golden_signal, []).append(spec.name)
        ordered = [s for s in self.signal_order if s in out]
        # Signals configured on a metric but absent from signal_order still
        # surface, just last -- silently dropping one would be the kind of
        # quiet omission this project treats as a bug.
        ordered += sorted(s for s in out if s not in self.signal_order)
        return {sig: sorted(out[sig]) for sig in ordered}

    def search(self, intent: str, limit: int = 10) -> list[dict[str, Any]]:
        """Keyword matching standing in for a vector index (see 'Known gaps').

        Scores name matches above prose matches, since an agent that already
        half-knows the name should land on it.
        """
        terms = [w for w in re.split(r"[^a-z0-9.]+", intent.lower()) if len(w) > 2]
        scored: list[tuple[int, MetricSpec]] = []
        for spec in self._specs.values():
            prose = f"{spec.description} {spec.interpretation}".lower()
            name = spec.name.lower()
            score = sum(prose.count(w) for w in terms) + 3 * sum(1 for w in terms if w in name)
            if score:
                scored.append((score, spec))
        scored.sort(key=lambda x: (-x[0], x[1].name))
        return [
            {
                "metric": s.name,
                "kind": s.kind,
                "unit": s.unit,
                "description": s.description,
                "denominator": s.denominator,
                "match_score": sc,
            }
            for sc, s in scored[:limit]
        ]


def _f(*names: str) -> dict[str, list[str]]:
    return {n: list(DEMO_FLEET[n]) for n in names}


def demo_catalog() -> MetricCatalog:
    """A small Spanner-shaped catalog. Descriptions are written the way a real
    one has to be: what the number counts, and what it means when it moves."""
    return MetricCatalog(
        [
            MetricSpec(
                name="spanner.rpc.errors",
                kind="counter",
                unit="errors",
                description=(
                    "Spanner RPCs that terminated with a non-OK status, counted at the "
                    "serving task. Includes ABORTED from lock contention, which is "
                    "usually retried by the client and is not always user-visible."
                ),
                interpretation=(
                    "Divide by spanner.rpc.count for an error rate; the absolute count "
                    "tracks traffic and will follow the diurnal cycle on its own. A rise "
                    "in rate with flat spanner.rpc.count is a serving problem; a rise in "
                    "both is usually load."
                ),
                fields=_f("region", "cell", "job", "task", "version"),
                denominator="spanner.rpc.count",
                golden_signal="errors",
            ),
            MetricSpec(
                name="spanner.rpc.count",
                kind="counter",
                unit="requests",
                description="All Spanner RPCs accepted by a serving task, any status.",
                interpretation=(
                    "The denominator for error and retry rates, and the traffic signal "
                    "in its own right. Strongly diurnal, so week-over-week beats "
                    "hour-over-hour for judging whether a move is real."
                ),
                fields=_f("region", "cell", "job", "task", "version"),
                golden_signal="traffic",
            ),
            MetricSpec(
                name="spanner.rpc.latency",
                kind="distribution",
                unit="ms",
                description="Server-side RPC latency distribution, p99 exported.",
                interpretation=(
                    "p99 moving while p50 stays flat points at a subset of requests or a "
                    "subset of tasks, not a fleet-wide slowdown. Percentiles are not "
                    "additive, so explain_delta cannot decompose this metric and falls "
                    "back to ranking by peer deviation."
                ),
                fields=_f("region", "cell", "job", "task", "version"),
                percentile="p99",
                golden_signal="latency",
            ),
            MetricSpec(
                name="spanner.paxos.leader_elections",
                kind="counter",
                unit="elections",
                description="Paxos leader elections started by a tablet replica.",
                interpretation=(
                    "A steady low rate is normal rebalancing. A spike means replicas are "
                    "losing leases: usually a task restarting, a network partition, or a "
                    "machine going away. Frequently precedes user-visible errors by a "
                    "minute or two, which makes it useful in a correlation scan."
                ),
                fields=_f("region", "cell", "task"),
            ),
            MetricSpec(
                name="spanner.lock.wait_time",
                kind="gauge",
                unit="ms",
                description="Mean time a transaction spends waiting to acquire locks.",
                interpretation=(
                    "Rises with write contention on a hot key range. Distinguishes a "
                    "contention incident from a capacity one: capacity shows up in CPU "
                    "and latency together, contention shows up here first."
                ),
                fields=_f("region", "cell", "job"),
                golden_signal="saturation",
            ),
            MetricSpec(
                name="spanner.tablet.split_rate",
                kind="gauge",
                unit="splits/s",
                description="Rate at which the load balancer is splitting tablets.",
                interpretation=(
                    "Elevated splitting is the system reacting to a hot range. It is a "
                    "response to load imbalance, not usually a cause of one."
                ),
                fields=_f("region", "cell"),
            ),
            MetricSpec(
                name="spanner.storage.read_bytes",
                kind="counter",
                unit="bytes",
                description="Bytes read from the storage layer to serve queries.",
                interpretation=(
                    "A jump with flat spanner.rpc.count means queries got more expensive "
                    "-- a plan change or an index that stopped being used."
                ),
                fields=_f("region", "cell", "job"),
            ),
            MetricSpec(
                name="spanner.task.restarts",
                kind="counter",
                unit="restarts",
                description="Task restarts observed by the cluster manager.",
                interpretation=(
                    "Near zero in steady state, so any sustained non-zero value is worth "
                    "explaining. Restarts clustered in one cell right after a rollout is "
                    "the signature of a bad build."
                ),
                fields=_f("region", "cell", "job", "task", "version"),
            ),
        ]
    )
