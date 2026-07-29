"""Monarch adapter.

**This is a scaffold, not a working client.** Monarch is internal to Google;
there is no public SDK, and nothing here has been run against a real instance.
What it does provide is the part that is knowable without one: the query shape,
the aggregation and alignment rules the analysis layer depends on, and -- most
importantly -- a concrete strategy for the EMPTY_SELECTOR contract, which is the
piece a naive integration silently gets wrong.

Supply `execute` and this becomes a `TSDBClient`. Then run
`metric_analysis.contract.check_contract` against it before trusting any output.

    from metric_analysis.monarch import MonarchTSDB, MonarchQuery

    def execute(q: MonarchQuery) -> list[tuple[dict[str, str], list[float], list[float]]]:
        ...  # your query client; return (labels, timestamps, values) per series

    tsdb = MonarchTSDB(execute=execute, catalog=catalog)

Why the seam is a query *description* rather than a query string: the tool layer
forbids raw query strings anywhere the agent can reach, and building one here
would put string construction one import away from the model. A structured
description is also what lets `check_contract` and the replay harness record
what was asked without parsing anything back out.

Notes on mapping this layer onto Monarch, from its published design:

  * **Alignment before grouping.** Monarch requires an alignment step before
    `group_by`. `step_s` is the alignment period. Counters align with `delta`,
    gauges with `mean`; getting this backwards makes a counter's window sum
    depend on resolution, and then the coarse and fine passes of `find_onset`
    disagree about the same instant.

  * **Target fields versus metric fields.** Monarch separates fields that
    identify the entity (cluster, job, task) from labels on the metric itself.
    The catalog does not distinguish them because the analysis does not care --
    but the query builder does, so `target_fields` declares which is which. Get
    it wrong and filters land in the wrong clause.

  * **Distributions are native, and that is an opportunity.** Monarch stores
    latency as bucketed distributions rather than pre-computed percentiles. The
    "percentiles aren't additive" gap in CLAUDE.md exists because a pre-computed
    p99 cannot be re-aggregated. With real distributions it can: merging buckets
    across a slice gives a true percentile, which would let `explain_delta`
    decompose latency properly instead of ranking by peer deviation. That work
    is not done here, but this is where it would go -- see `extract_percentile`.

  * **Regional evaluation.** Monarch pushes evaluation down to zones. That lines
    up with the operational constraint of running analysis next to the data with
    only summaries crossing regions, and it is why `points_scanned` should
    report what the zones scanned rather than what came back.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import numpy as np

from .catalog import MetricCatalog
from .types import QueryResult, ResultStatus, Series, ToolError, Window

# One series per (labels, timestamps, values).
RawSeries = tuple[dict[str, str], list[float], list[float]]


@dataclass(frozen=True)
class MonarchQuery:
    """A query as a description, never as a string."""

    metric: str
    window: Window
    align_period_s: float
    align_fn: str                      # "delta" | "mean" | "max"
    target_filters: dict[str, str] = field(default_factory=dict)
    metric_filters: dict[str, str] = field(default_factory=dict)
    group_by: tuple[str, ...] = ()
    reduce_fn: str = "sum"             # "sum" | "mean" | "merge"
    percentile: float | None = None    # set when the metric is a distribution
    purpose: str = "analysis"          # "analysis" | "existence_probe"

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "window": self.window.to_dict(),
            "align": f"{self.align_fn}({self.align_period_s:g}s)",
            "target_filters": dict(self.target_filters),
            "metric_filters": dict(self.metric_filters),
            "group_by": list(self.group_by),
            "reduce": self.reduce_fn,
            "percentile": self.percentile,
            "purpose": self.purpose,
        }


class QueryExecutor(Protocol):
    def __call__(self, query: MonarchQuery) -> list[RawSeries]: ...


def align_fn_for(kind: str) -> str:
    """Counters accumulate, so they align with delta; everything else averages.

    Aligning a counter with mean reports a rate where the analysis expects a
    count, which makes every window sum resolution-dependent and quietly breaks
    the two-pass onset detection.
    """
    return {"counter": "delta", "gauge": "mean", "distribution": "mean"}.get(kind, "mean")


def reduce_fn_for(kind: str) -> str:
    return {"counter": "sum", "gauge": "mean", "distribution": "merge"}.get(kind, "mean")


def extract_percentile(buckets: Any, percentile: float) -> float:
    """Placeholder for real distribution handling.

    Monarch distributions are bucketed histograms, which *are* mergeable -- so a
    percentile computed after merging across a slice is a true percentile, not
    an average of percentiles. Implementing this properly is what would close
    the latency-attribution gap in CLAUDE.md.

    Left unimplemented rather than approximated: an averaged quantile looks
    plausible and is wrong in a way nothing downstream can detect.
    """
    raise NotImplementedError(
        "Distribution handling needs Monarch's bucketer schema. Until then, configure "
        "distribution metrics to export a pre-computed percentile and accept that "
        "explain_delta cannot decompose them."
    )


class MonarchTSDB:
    """`TSDBClient` over a Monarch query executor.

    `target_fields` names the tags Monarch treats as target identity rather than
    metric labels. Everything not listed is sent as a metric field.
    """

    def __init__(
        self,
        execute: QueryExecutor,
        catalog: MetricCatalog,
        target_fields: tuple[str, ...] = ("region", "cell", "job", "task"),
        existence_probe_window_s: float = 7 * 86400.0,
        strict_existence: bool = True,
    ) -> None:
        self.execute = execute
        self.catalog = catalog
        self.target_fields = tuple(target_fields)
        self.existence_probe_window_s = existence_probe_window_s
        self.strict_existence = strict_existence
        self._existence_cache: dict[tuple[str, tuple[tuple[str, str], ...]], bool] = {}
        self._queries = 0
        self._points = 0

    # -- filter routing ----------------------------------------------------
    def _split_filters(self, filters: dict[str, str]) -> tuple[dict[str, str], dict[str, str]]:
        target = {k: v for k, v in filters.items() if k in self.target_fields}
        metric = {k: v for k, v in filters.items() if k not in self.target_fields}
        return target, metric

    def _build(
        self,
        metric: str,
        filters: dict[str, str],
        window: Window,
        step_s: float,
        group_by: list[str],
        purpose: str = "analysis",
    ) -> MonarchQuery:
        spec = self.catalog.get(metric)
        target, metric_filters = self._split_filters(filters)
        return MonarchQuery(
            metric=metric,
            window=window,
            align_period_s=float(step_s),
            align_fn=align_fn_for(spec.kind),
            target_filters=target,
            metric_filters=metric_filters,
            group_by=tuple(group_by),
            reduce_fn=reduce_fn_for(spec.kind),
            percentile=float(spec.percentile.lstrip("p")) / 100.0
            if spec.percentile and spec.percentile.startswith("p")
            else None,
            purpose=purpose,
        )

    # -- the EMPTY_SELECTOR contract ---------------------------------------
    #
    # Storage cannot distinguish these on its own: a filter matching no entity
    # and an entity reporting no points both come back empty. So on the empty
    # path only, re-ask over a much wider window. Data anywhere in it means the
    # entities are real and this window was genuinely quiet (NO_DATA); still
    # nothing means the selector never matched (EMPTY_SELECTOR).
    #
    # One extra query, paid only when a result is empty. Cheap against the
    # alternative, which is reporting a typo'd filter as evidence that a slice
    # is healthy.
    #
    # The probe has a known false positive: an entity decommissioned last month
    # still exists inside the wide window, so its absence today reads as
    # NO_DATA. That errs toward "it stopped reporting", which is the safe
    # direction -- it prompts a look rather than a shrug.

    def _existence_key(self, metric: str, filters: dict[str, str]) -> tuple[str, tuple[tuple[str, str], ...]]:
        return (metric, tuple(sorted(filters.items())))

    def fetch(
        self,
        metric: str,
        filters: dict[str, str],
        window: Window,
        step_s: float,
        group_by: list[str],
    ) -> QueryResult:
        query = self._build(metric, filters, window, step_s, group_by)
        self._queries += 1
        raw = self.execute(query)

        if not raw:
            if not filters:
                # No filter to be wrong about, so nothing was excluded -- the
                # metric simply has no data in this window.
                return QueryResult(ResultStatus.NO_DATA, [], 0, "No data for an unfiltered query.")
            if not self.strict_existence:
                return QueryResult(
                    ResultStatus.NO_DATA, [], 0,
                    "Empty result; existence probing disabled, so this may be a bad selector.",
                )
            key = self._existence_key(metric, filters)
            if key not in self._existence_cache:
                probe_window = Window(
                    window.end - self.existence_probe_window_s, window.end
                )
                probe = self._build(
                    metric, filters, probe_window, max(step_s, 3600.0), [],
                    purpose="existence_probe",
                )
                self._queries += 1
                self._existence_cache[key] = bool(self.execute(probe))
            if self._existence_cache[key]:
                return QueryResult(
                    ResultStatus.NO_DATA, [], 0,
                    f"Entities matching {filters} exist but reported nothing in this window.",
                )
            return QueryResult(
                ResultStatus.EMPTY_SELECTOR, [], 0,
                f"Filter {filters} matched no entity for {metric!r} in the last "
                f"{self.existence_probe_window_s / 86400:.0f} days.",
            )

        series = []
        for labels, timestamps, values in raw:
            t = np.asarray(timestamps, dtype=float)
            v = np.asarray(values, dtype=float)
            order = np.argsort(t, kind="stable")
            series.append(Series(dict(labels), t[order], v[order]))
        points = sum(len(s) for s in series)
        self._points += points
        return QueryResult(ResultStatus.OK, series, points)

    def field_values(self, metric: str, field_name: str, window: Window) -> list[str]:
        spec = self.catalog.get(metric)
        if field_name not in spec.fields:
            raise ToolError(
                "unknown_field", f"{field_name!r} is not a tag on {metric!r}.", list(spec.fields)
            )
        query = self._build(metric, {}, window, 3600.0, [field_name], purpose="existence_probe")
        self._queries += 1
        seen = {
            labels.get(field_name)
            for labels, _, _ in self.execute(query)
            if labels.get(field_name)
        }
        return sorted(seen)

    def stats(self) -> dict[str, Any]:
        return {"queries": self._queries, "points_returned": self._points}


def _make_executor_stub() -> Callable[[MonarchQuery], list[RawSeries]]:
    """A stub that fails loudly. Better than one returning [], which would look
    exactly like a healthy fleet with nothing wrong."""

    def execute(query: MonarchQuery) -> list[RawSeries]:
        raise NotImplementedError(
            f"No Monarch executor wired. Query was: {query.to_dict()}. Supply an execute "
            f"callable to MonarchTSDB, then run metric_analysis.contract.check_contract "
            f"against it before trusting any output."
        )

    return execute
