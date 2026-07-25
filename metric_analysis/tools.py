"""The agent-facing surface.

Every function here is a tool the orchestrator may call. Rules enforced at this
boundary, not above it:

  1. No raw query strings. Structured params only, validated against the catalog.
  2. Malformed queries raise ToolError with suggestions. They never return [].
  3. Cost is estimated before execution and charged against a budget.
  4. Results are compact summaries; raw points are capped and opt-in.
  5. Every result gets an ID and is recorded, so downstream claims can cite it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from .attribution import explain_delta as _explain_delta
from .catalog import MetricCatalog
from .changepoint import detect_change_point
from .summarize import peer_outliers, summarize_series
from .tsdb import TSDBClient
from .types import QueryResult, ResultStatus, Series, ToolError, Window


@dataclass
class Budget:
    max_queries: int = 60
    max_points: int = 4_000_000
    queries: int = 0
    points: int = 0

    def charge(self, points: int) -> None:
        self.queries += 1
        self.points += points
        if self.queries > self.max_queries:
            raise ToolError("budget_exceeded", f"Query budget of {self.max_queries} exhausted.")
        if self.points > self.max_points:
            raise ToolError("budget_exceeded", f"Point budget of {self.max_points} exhausted.")


@dataclass
class Evidence:
    """Append-only record. Any claim in the final report must cite an id here."""

    records: dict[str, dict[str, Any]] = field(default_factory=dict)

    def add(self, tool: str, params: dict[str, Any], result: Any) -> str:
        h = hashlib.sha1(
            json.dumps([tool, params], sort_keys=True, default=str).encode()
        ).hexdigest()[:10]
        eid = f"ev_{h}"
        self.records[eid] = {"id": eid, "tool": tool, "params": params, "result": result}
        return eid

    def get(self, eid: str) -> dict[str, Any] | None:
        return self.records.get(eid)


class MetricTools:
    def __init__(
        self,
        catalog: MetricCatalog,
        tsdb: TSDBClient,
        budget: Budget | None = None,
        evidence: Evidence | None = None,
        max_series: int = 400,
    ):
        self.catalog = catalog
        self.tsdb = tsdb
        self.budget = budget or Budget()
        self.evidence = evidence or Evidence()
        self.max_series = max_series
        self._cache: dict[str, QueryResult] = {}

    # -- discovery ---------------------------------------------------------
    def search_metrics(self, intent: str, limit: int = 10) -> dict[str, Any]:
        hits = self.catalog.search(intent, limit)
        out = {"query": intent, "matches": hits}
        out["evidence_id"] = self.evidence.add("search_metrics", {"intent": intent}, out)
        return out

    def describe_metric(self, metric: str) -> dict[str, Any]:
        # Recorded like every other tool result: a report that leans on what a
        # metric *means* ("this counter excludes retries") is making a claim,
        # and it needs an id to cite for it.
        out = self.catalog.get(metric).summary()
        out["evidence_id"] = self.evidence.add("describe_metric", {"metric": metric}, out)
        return out

    def list_field_values(self, metric: str, field_name: str, window: Window) -> dict[str, Any]:
        """The agent must resolve field values through this, never from memory."""
        spec = self.catalog.get(metric)
        if field_name not in spec.fields:
            raise ToolError(
                "unknown_field",
                f"{field_name!r} is not a tag on {metric!r}.",
                list(spec.fields),
            )
        vals = self.tsdb.field_values(metric, field_name, window)
        out = {"metric": metric, "field": field_name, "values": vals, "count": len(vals)}
        out["evidence_id"] = self.evidence.add(
            "list_field_values", {"metric": metric, "field": field_name}, out
        )
        return out

    # -- fetching ----------------------------------------------------------
    def _fetch(
        self,
        metric: str,
        filters: dict[str, str],
        window: Window,
        step_s: float,
        group_by: list[str],
    ) -> QueryResult:
        self.catalog.validate_filter(metric, filters)
        self.catalog.group_by_check(metric, group_by)

        est = self.catalog.estimate_cardinality(metric, filters, group_by)
        if est > self.max_series:
            raise ToolError(
                "cardinality_too_high",
                f"That grouping would return ~{est} series (cap {self.max_series}). "
                f"Add a filter or drop a group_by field.",
                [f"filter on {g} first" for g in group_by],
            )

        ck = json.dumps([metric, filters, window.start, window.end, step_s, group_by], sort_keys=True)
        if ck in self._cache:
            return self._cache[ck]

        res = self.tsdb.fetch(metric, filters, window, step_s, group_by)
        self.budget.charge(res.points_scanned)

        if res.status is ResultStatus.EMPTY_SELECTOR:
            raise ToolError(
                "empty_selector",
                f"Filter {filters} matched zero known entities for {metric!r}. "
                f"This is a malformed query, not a healthy signal.",
                [],
            )
        self._cache[ck] = res
        return res

    # -- analysis ----------------------------------------------------------
    def series_summary(
        self,
        metric: str,
        filters: dict[str, str],
        incident: Window,
        baseline: Window | None = None,
        group_by: list[str] | None = None,
        step_s: float = 60.0,
    ) -> dict[str, Any]:
        """Compact summary per series: stats, delta, change point, shape."""
        group_by = group_by or []
        span = Window(min(incident.start, baseline.start if baseline else incident.start), incident.end)
        res = self._fetch(metric, filters, span, step_s, group_by)

        if res.status is ResultStatus.NO_DATA or not res.series:
            out = {
                "metric": metric,
                "filters": filters,
                "status": "no_data",
                "note": "Query was valid and matched real entities, but no points were reported.",
            }
            out["evidence_id"] = self.evidence.add("series_summary", {"metric": metric, "filters": filters}, out)
            return out

        summaries = [summarize_series(s, incident, baseline) for s in res.series]
        summaries.sort(key=lambda d: -abs(d.get("delta_pct") or 0))
        out = {
            "metric": metric,
            "filters": filters,
            "group_by": group_by,
            "series_count": len(summaries),
            "series": summaries[:20],
            "truncated": len(summaries) > 20,
        }
        out["evidence_id"] = self.evidence.add(
            "series_summary",
            {"metric": metric, "filters": filters, "group_by": group_by},
            out,
        )
        return out

    def find_onset(
        self,
        metric: str,
        filters: dict[str, str],
        window: Window,
        coarse_step_s: float = 300.0,
        fine_step_s: float = 15.0,
    ) -> dict[str, Any]:
        """Two-pass onset detection: coarse scan to locate, fine re-query to pin.

        The precise timestamp is what you intersect with the deploy log, so it's
        worth the second query.
        """
        coarse = self._fetch(metric, filters, window, coarse_step_s, [])
        if not coarse.series:
            return {"status": "no_data", "metric": metric}
        cp = detect_change_point(coarse.series[0].timestamps, coarse.series[0].values)
        if cp is None or not cp.significant:
            out = {"metric": metric, "filters": filters, "onset": None,
                   "note": "No significant change point in this window."}
            out["evidence_id"] = self.evidence.add("find_onset", {"metric": metric, "filters": filters}, out)
            return out

        band = Window(cp.timestamp - 6 * coarse_step_s, cp.timestamp + 6 * coarse_step_s)
        fine = self._fetch(metric, filters, band, fine_step_s, [])
        refined = cp
        if fine.series and len(fine.series[0]) > 12:
            r = detect_change_point(fine.series[0].timestamps, fine.series[0].values)
            if r and r.significant:
                refined = r

        out = {
            "metric": metric,
            "filters": filters,
            "onset": refined.to_dict(),
            "resolution_s": fine_step_s,
            "next_step": "Intersect this timestamp with the change log for the affected entities.",
        }
        out["evidence_id"] = self.evidence.add("find_onset", {"metric": metric, "filters": filters}, out)
        return out

    def explain_delta(
        self,
        metric: str,
        fields: list[str],
        baseline: Window,
        incident: Window,
        filters: dict[str, str] | None = None,
        step_s: float = 60.0,
        use_denominator: bool = True,
    ) -> dict[str, Any]:
        """Which slice moved the metric. The workhorse."""
        filters = filters or {}
        self.catalog.group_by_check(metric, fields)
        spec = self.catalog.get(metric)
        span = Window(min(baseline.start, incident.start), max(baseline.end, incident.end))

        pools: dict[str, list[Series]] = {}
        pools[metric] = self._fetch(metric, filters, span, step_s, fields).series
        den = spec.denominator if (use_denominator and spec.denominator) else None
        if den:
            pools[den] = self._fetch(den, filters, span, step_s, fields).series

        out = _explain_delta(pools, metric, fields, baseline, incident, denominator=den)
        out["filters"] = filters
        out["evidence_id"] = self.evidence.add(
            "explain_delta", {"metric": metric, "fields": fields, "filters": filters}, out
        )
        return out

    def peer_comparison(
        self,
        metric: str,
        peer_field: str,
        incident: Window,
        filters: dict[str, str] | None = None,
        step_s: float = 60.0,
    ) -> dict[str, Any]:
        """Rank entities against their structural siblings. Seasonality-immune."""
        filters = filters or {}
        res = self._fetch(metric, filters, incident, step_s, [peer_field])
        outliers = peer_outliers(res.series, incident)
        out = {
            "metric": metric,
            "peer_field": peer_field,
            "peers_compared": len(res.series),
            "outliers": outliers,
            "note": "Empty outlier list means all peers are behaving alike, not that all are healthy.",
        }
        out["evidence_id"] = self.evidence.add(
            "peer_comparison", {"metric": metric, "peer_field": peer_field, "filters": filters}, out
        )
        return out

    def correlation_scan(
        self,
        onset: float,
        window: Window,
        candidate_metrics: list[str] | None = None,
        filters: dict[str, str] | None = None,
        tolerance_s: float = 600.0,
        step_s: float = 60.0,
    ) -> dict[str, Any]:
        """What else changed at that moment.

        Scans candidates for change points clustering near `onset` and ranks by
        temporal proximity, with metrics that moved *first* ranked higher --
        precedence is weak evidence of causality, and worth surfacing.
        """
        filters = filters or {}
        candidates = candidate_metrics or self.catalog.names()
        hits = []
        for m in candidates:
            try:
                res = self._fetch(m, filters, window, step_s, [])
            except ToolError:
                continue
            if not res.series:
                continue
            cp = detect_change_point(res.series[0].timestamps, res.series[0].values)
            if cp is None or not cp.significant:
                continue
            lag = cp.timestamp - onset
            if abs(lag) > tolerance_s:
                continue
            hits.append(
                {
                    "metric": m,
                    "change_point": cp.to_dict(),
                    "lag_s": round(lag, 1),
                    "relation": "preceded" if lag < -step_s else ("concurrent" if abs(lag) <= step_s else "followed"),
                }
            )
        hits.sort(key=lambda h: (h["lag_s"], abs(h["lag_s"])))
        out = {"onset": onset, "tolerance_s": tolerance_s, "correlated": hits, "scanned": len(candidates)}
        out["evidence_id"] = self.evidence.add("correlation_scan", {"onset": onset}, out)
        return out

    def usage(self) -> dict[str, Any]:
        return {
            "queries": self.budget.queries,
            "points_scanned": self.budget.points,
            "cache_entries": len(self._cache),
            "evidence_records": len(self.evidence.records),
        }
