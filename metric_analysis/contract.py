"""Conformance checks every `TSDBClient` must pass.

Run this against a new backend before trusting a single brief from it. The
analysis layer above makes assumptions it cannot verify per-query, and a backend
that quietly violates one produces confident, well-formatted, wrong output --
which is worse than an outage, because nobody notices.

The check that matters most is EMPTY_SELECTOR. Every other violation here
degrades quality; that one inverts a conclusion. A filter naming an entity that
does not exist must raise, not return zero series, because zero series and "that
slice is healthy" are indistinguishable by the time they reach a model, and it
will report the reassuring one. A real backend has no reason to distinguish
these on its own -- both are "no data" to storage -- so this is the single most
likely thing to be silently wrong in an integration.

Usage:

    from metric_analysis.contract import check_contract, format_report
    print(format_report(check_contract(my_client, my_catalog, probe)))

`probe` describes one metric the backend really has, plus a filter that is
genuinely valid and one that is genuinely nonsense, because a conformance suite
cannot invent those for an unknown fleet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .catalog import MetricCatalog
from .types import QueryResult, ResultStatus, ToolError, Window


@dataclass
class Probe:
    """What the suite needs to know about a fleet it has never seen."""

    metric: str
    window: Window
    step_s: float = 60.0
    # A filter that matches real entities right now.
    valid_filter: dict[str, str] = field(default_factory=dict)
    # A filter whose *values* are nonsense but whose *tags* are real. This is
    # the one that must raise; supply something no entity could ever match.
    bogus_filter: dict[str, str] = field(default_factory=dict)
    # A tag with few enough values to group by without a cardinality explosion.
    group_by_field: str = ""
    # A window far enough in the past that retention has dropped it, if the
    # backend has retention. Used to separate NO_DATA from EMPTY_SELECTOR.
    empty_window: Window | None = None


@dataclass
class Violation:
    check: str
    severity: str  # "critical" | "major" | "minor"
    detail: str
    consequence: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "check": self.check,
            "severity": self.severity,
            "detail": self.detail,
            "consequence": self.consequence,
        }


def _call(client: Any, probe: Probe, filters: dict[str, str], group_by: list[str],
          window: Window | None = None) -> QueryResult | ToolError | Exception:
    try:
        return client.fetch(probe.metric, filters, window or probe.window, probe.step_s, group_by)
    except ToolError as e:
        return e
    except Exception as e:  # noqa: BLE001 - the suite reports, it does not crash
        return e


def check_contract(client: Any, catalog: MetricCatalog, probe: Probe) -> list[Violation]:
    """Return every violation found. An empty list means the client conforms."""
    v: list[Violation] = []

    # -- 1. the one that inverts conclusions ------------------------------
    if probe.bogus_filter:
        got = _call(client, probe, probe.bogus_filter, [])
        if isinstance(got, ToolError):
            pass  # raising is acceptable: the caller learns it was malformed
        elif isinstance(got, Exception):
            v.append(Violation(
                "empty_selector", "major",
                f"bogus filter raised {type(got).__name__}: {got}",
                "An unexpected exception ends the investigation instead of letting the "
                "agent correct a typo. Raise ToolError or return EMPTY_SELECTOR.",
            ))
        elif got.series:
            # The client is behaving correctly and the probe is wrong. Blaming
            # the backend here would be worse than useless: it would send
            # someone hunting a bug in code that is fine, and a suite that cries
            # wolf gets skipped exactly when it matters.
            v.append(Violation(
                "probe_configuration", "major",
                f"bogus_filter {probe.bogus_filter} returned {len(got.series)} series, so it "
                f"matches real entities and cannot test the EMPTY_SELECTOR contract",
                "Not a client fault. Supply a filter whose values genuinely exist nowhere "
                "-- a misspelled region, a version never built -- or this check is a no-op "
                "and the most dangerous failure mode goes untested.",
            ))
        elif got.status is not ResultStatus.EMPTY_SELECTOR:
            v.append(Violation(
                "empty_selector", "critical",
                f"filter {probe.bogus_filter} matching no entity returned "
                f"status={got.status.value} with zero series, expected EMPTY_SELECTOR",
                "THE most dangerous failure in the system. Zero series is "
                "indistinguishable from a healthy slice, so the agent will report a "
                "typo'd filter as evidence that everything is fine. Every conclusion "
                "downstream of this backend is unsafe until it is fixed.",
            ))

    # -- 2. valid filter must actually work -------------------------------
    got = _call(client, probe, probe.valid_filter, [])
    if isinstance(got, Exception):
        v.append(Violation(
            "valid_query", "critical",
            f"valid filter {probe.valid_filter} raised {type(got).__name__}: {got}",
            "Nothing else can be checked, and no investigation can run.",
        ))
        return v
    if got.status is ResultStatus.EMPTY_SELECTOR:
        v.append(Violation(
            "valid_query", "critical",
            f"valid filter {probe.valid_filter} was reported EMPTY_SELECTOR",
            "Real slices being rejected as nonexistent. Either the probe filter is "
            "wrong or existence checking is inverted; both block every query.",
        ))
        return v
    if got.status is ResultStatus.OK and not got.series:
        v.append(Violation(
            "status_consistency", "critical",
            "status=OK with zero series",
            "OK means 'here is data'. Zero series must be NO_DATA (entity exists, "
            "reported nothing) or EMPTY_SELECTOR (no such entity). Collapsing the "
            "three loses the distinction the whole design rests on.",
        ))

    # -- 3. absence of data is not absence of entity ----------------------
    if probe.empty_window is not None:
        got_empty = _call(client, probe, probe.valid_filter, [], window=probe.empty_window)
        if isinstance(got_empty, QueryResult) and got_empty.status is ResultStatus.EMPTY_SELECTOR:
            v.append(Violation(
                "no_data_vs_empty_selector", "major",
                "a real entity with no points in the window returned EMPTY_SELECTOR",
                "'Stopped reporting' is a symptom worth investigating -- a task that went "
                "silent. Reporting it as a malformed query hides a real signal.",
            ))

    # -- 4. shape of what comes back --------------------------------------
    if isinstance(got, QueryResult) and got.series:
        s = got.series[0]
        t = s.timestamps
        if t.size and not np.all(np.diff(t) > 0):
            v.append(Violation(
                "ordering", "major", "timestamps are not strictly ascending",
                "Change point detection scans in order; unsorted input makes the onset "
                "timestamp meaningless, and that timestamp is what gets intersected "
                "with the deploy log.",
            ))
        if t.size > 2:
            steps = np.diff(t)
            if float(np.max(np.abs(steps - probe.step_s))) > probe.step_s * 0.5:
                v.append(Violation(
                    "alignment", "major",
                    f"samples are not aligned to step_s={probe.step_s} "
                    f"(observed spacing {float(np.median(steps)):.1f}s)",
                    "Unaligned samples make window sums resolution-dependent, so the "
                    "coarse and fine passes of find_onset disagree about the same instant.",
                ))
        if s.values.size != t.size:
            v.append(Violation(
                "shape", "critical", "values and timestamps have different lengths",
                "Every downstream calculation indexes them together.",
            ))
        if got.points_scanned <= 0:
            v.append(Violation(
                "cost_accounting", "minor",
                "points_scanned is zero for a query that returned data",
                "The budget charges on this. Reporting zero means an agent can fan out "
                "without limit during a P0, which is a real way to turn one incident "
                "into two.",
            ))

    # -- 5. grouping ------------------------------------------------------
    if probe.group_by_field:
        grouped = _call(client, probe, probe.valid_filter, [probe.group_by_field])
        if isinstance(grouped, QueryResult) and grouped.series:
            missing = [
                s.labels for s in grouped.series if probe.group_by_field not in s.labels
            ]
            if missing:
                v.append(Violation(
                    "group_by_labels", "critical",
                    f"{len(missing)} series lack the {probe.group_by_field!r} label they "
                    f"were grouped by",
                    "Attribution reads the grouped field off the labels. Without it every "
                    "slice collapses to '<unset>' and the decomposition is meaningless.",
                ))
            keys = [s.labels.get(probe.group_by_field) for s in grouped.series]
            if len(keys) != len(set(keys)):
                v.append(Violation(
                    "group_by_uniqueness", "major",
                    f"duplicate values for {probe.group_by_field!r} across series",
                    "One series per distinct value is assumed. Duplicates double-count "
                    "in every sum, inflating whichever slice happens to be split.",
                ))

    # -- 6. determinism ---------------------------------------------------
    again = _call(client, probe, probe.valid_filter, [])
    if isinstance(got, QueryResult) and isinstance(again, QueryResult):
        if len(again.series) != len(got.series):
            v.append(Violation(
                "determinism", "major",
                f"identical queries returned {len(got.series)} then {len(again.series)} series",
                "Replay and evaluation both assume a query is reproducible. A backend "
                "that varies makes a failed eval indistinguishable from a flaky one.",
            ))
        elif got.series and again.series:
            a, b = got.series[0].finite(), again.series[0].finite()
            if a.size == b.size and a.size and not np.allclose(a, b, rtol=1e-9, equal_nan=True):
                v.append(Violation(
                    "determinism", "minor",
                    "identical queries returned different values",
                    "Expected for a live window still filling. If this fires on a closed "
                    "historical window, replay cannot be trusted.",
                ))

    # -- 7. does the catalog agree with the backend? ----------------------
    try:
        spec = catalog.get(probe.metric)
    except ToolError:
        v.append(Violation(
            "catalog_agreement", "major",
            f"{probe.metric!r} is not in the catalog",
            "The catalog is what validates filters before they reach storage. A metric "
            "the backend serves but the catalog does not know is unreachable by the agent.",
        ))
        return v

    for tag in probe.valid_filter:
        if tag not in spec.fields:
            v.append(Violation(
                "catalog_agreement", "major",
                f"tag {tag!r} works against the backend but is absent from the catalog",
                "Filters on it are rejected before they are tried, so a real dimension is "
                "invisible to the agent.",
            ))
    return v


def format_report(violations: list[Violation]) -> str:
    if not violations:
        return "PASS — client conforms. Note this checks behaviour, not correctness of data."
    order = {"critical": 0, "major": 1, "minor": 2}
    ranked = sorted(violations, key=lambda x: order.get(x.severity, 9))
    lines = [f"FAIL — {len(ranked)} violation(s)\n"]
    for x in ranked:
        lines.append(f"[{x.severity.upper()}] {x.check}")
        lines.append(f"    what: {x.detail}")
        lines.append(f"    why it matters: {x.consequence}\n")
    if any(x.severity == "critical" for x in ranked):
        lines.append("Do not run investigations against this backend until the critical "
                     "violations are fixed. Its output would look correct and be wrong.")
    return "\n".join(lines)
