"""Rollout stage 1: the auto-generated incident brief. No reasoning at all.

This is the fixed opening sweep the root CLAUDE.md carves out as the single
exception to "don't encode investigation sequences". It exists precisely
*because* it is not adaptive: running the same six questions every time is what
stops an investigation from tunnelling on the first plausible story. Everything
below the sweep -- which slice, which peers, what else moved -- is still decided
by the tools, not by this file.

What it deliberately does not do:

  * No causal claims. It reports that a rollout label changed at the same
    moment errors rose. It does not say the rollout caused them. Precedence is
    weak evidence and is labelled as such.
  * No invention. If onset detection finds nothing significant, the brief says
    so. "I don't know" is a first-class output (invariant 6), and a brief that
    fabricates a conclusion is worse than no brief.
  * No hidden failures. If a step raises, the brief degrades and records the
    step under `not_checked` rather than dying. Losing the whole brief during a
    P0 because one query blew a budget is the wrong trade.

The `not_checked` list is not an apology. On-call needs to know that the change
log was never consulted, because otherwise "no deploys mentioned" reads as "no
deploys happened".
"""

from __future__ import annotations

import time
from typing import Any, Callable

from .tools import MetricTools
from .types import ToolError, Window, _r

SCHEMA_VERSION = 1

# Dimensions carrying enough cardinality to be worth a per-task breakdown are
# excluded from the opening sweep: the cardinality guard would reject the query
# anyway, and blast radius is a question about cells and jobs, not tasks.
_SWEEP_EXCLUDED_DIMENSIONS = ("task", "alloc")
_MAX_SWEEP_DIMENSIONS = 4

# Signal sources the agent has no tools for yet. Naming them is the point --
# see the module docstring.
_UNBUILT_SOURCES = [
    ("change log", "cannot confirm whether a rollout, config push or flag flip landed at onset"),
    ("logs", "no template frequency diff, so new error signatures are unknown"),
    ("traces", "no span-level diff, so the slow path inside the request is unknown"),
    ("topology", "no dependency graph, so a downstream cause cannot be traversed to"),
]


def _utc(ts: float | None) -> str | None:
    if ts is None:
        return None
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


class _Sweep:
    """Runs each step, absorbing tool failures into a recorded skip."""

    def __init__(self) -> None:
        self.evidence: list[str] = []
        self.skipped: list[dict[str, str]] = []

    def run(self, name: str, fn: Callable[[], dict[str, Any]]) -> dict[str, Any] | None:
        try:
            out = fn()
        except ToolError as e:
            # Budget exhaustion and cardinality refusals are expected operating
            # conditions, not bugs. Record and carry on.
            self.skipped.append({"step": name, "reason": f"{e.kind}: {e.message}"})
            return None
        eid = out.get("evidence_id")
        if eid and eid not in self.evidence:
            self.evidence.append(eid)
        return out


def build_brief(
    tools: MetricTools,
    symptom_metric: str,
    incident: Window,
    baseline: Window | None = None,
    dimensions: list[str] | None = None,
    peer_field: str = "cell",
    scan_window: Window | None = None,
    min_delta_pct: float = 25.0,
    now: float | None = None,
) -> dict[str, Any]:
    """Run the fixed sweep and assemble a brief.

    `now` is injectable and omitted when None so the output stays byte-identical
    under replay; a wall-clock stamp would defeat the whole point of the
    reproducibility work in the fixture layer.
    """
    baseline = baseline or Window(incident.start - 4 * incident.duration, incident.start - 3 * incident.duration)
    scan_window = scan_window or Window(incident.start - 2 * incident.duration, incident.end)
    sweep = _Sweep()

    # -- 1. what is this metric ------------------------------------------
    spec = sweep.run("describe_metric", lambda: tools.describe_metric(symptom_metric))
    if dimensions is None:
        fields = list((spec or {}).get("fields", {}))
        dimensions = [f for f in fields if f not in _SWEEP_EXCLUDED_DIMENSIONS][:_MAX_SWEEP_DIMENSIONS]

    # -- 2. how bad, fleet-wide ------------------------------------------
    summary = sweep.run(
        "series_summary",
        lambda: tools.series_summary(symptom_metric, {}, incident, baseline),
    )

    # -- 3. the other golden signals --------------------------------------
    # The page names one metric, but that metric is one view of the service.
    # Errors can be flat while traffic collapses, and a brief anchored only on
    # the paged metric would report "errors normal" and miss the outage. Each
    # signal is checked on its own terms rather than for correlation with the
    # symptom, so an independent problem is still visible.
    signals = _golden_signal_sweep(
        tools, sweep, symptom_metric, incident, baseline, min_delta_pct
    )

    # -- 4. how wide is it ------------------------------------------------
    # Runs before attribution and unfiltered, for two reasons. A small cell on
    # fire can be invisible in a fleet-wide aggregate, so peer deviation is part
    # of deciding whether anything happened at all; and filtering peers by the
    # implicated region would make this depend on attribution, which is exactly
    # the circularity that lets a brief confirm its own premise.
    peers = sweep.run(
        "peer_comparison",
        lambda: tools.peer_comparison(symptom_metric, peer_field, incident),
    )

    # -- 5. did anything actually happen? ---------------------------------
    # The gate that stops the most dangerous output this component can produce.
    # explain_delta decomposes whatever delta it is given, and explanatory power
    # is a *share* of that delta -- so on a flat metric some slice still owns
    # 70% of the noise, and the brief names a healthy cell with a straight face.
    # Attribution is only meaningful once the change is established as real.
    materiality = _materiality(summary, peers, min_delta_pct)
    if not materiality["material"]:
        for step in ("explain_delta", "find_onset", "correlation_scan"):
            sweep.skipped.append({"step": step, "reason": materiality["reason"]})
        return _finish(
            _assemble(
                symptom_metric=symptom_metric,
                incident=incident,
                baseline=baseline,
                summary=summary,
                attribution=None,
                narrowed={},
                spans={},
                onset=None,
                onset_ts=None,
                peers=peers,
                peer_field=peer_field,
                correlated=None,
                signals=signals,
                materiality=materiality,
                sweep=sweep,
                tools=tools,
            ),
            now,
        )

    # -- 6. which slice moved it ------------------------------------------
    attribution = (
        sweep.run(
            "explain_delta",
            lambda: tools.explain_delta(symptom_metric, dimensions, baseline, incident),
        )
        if dimensions
        else None
    )
    narrowed: dict[str, str] = dict((attribution or {}).get("narrowed_to") or {})
    spans: dict[str, list[str]] = dict((attribution or {}).get("spans") or {})

    # Onset and correlation are asked about the implicated slice when there is
    # one -- a fleet-wide aggregate smears the step and costs precision on the
    # single most valuable fact in the brief.
    slice_filter = {k: v for k, v in narrowed.items() if k in (peer_field, "region", "cell")}

    # -- 7. when did it start ---------------------------------------------
    onset = sweep.run(
        "find_onset",
        lambda: tools.find_onset(symptom_metric, slice_filter, scan_window),
    )
    onset_ts = ((onset or {}).get("onset") or {}).get("timestamp")

    # -- 8. what else moved at that moment --------------------------------
    correlated = None
    if onset_ts is not None:
        correlated = sweep.run(
            "correlation_scan",
            lambda: tools.correlation_scan(onset_ts, scan_window, filters=slice_filter),
        )
    else:
        sweep.skipped.append(
            {"step": "correlation_scan", "reason": "no onset to scan around"}
        )

    return _finish(
        _assemble(
            symptom_metric=symptom_metric,
            incident=incident,
            baseline=baseline,
            summary=summary,
            attribution=attribution,
            narrowed=narrowed,
            spans=spans,
            onset=onset,
            onset_ts=onset_ts,
            peers=peers,
            peer_field=peer_field,
            correlated=correlated,
            signals=signals,
            materiality=materiality,
            sweep=sweep,
            tools=tools,
        ),
        now,
    )


def _finish(brief: dict[str, Any], now: float | None) -> dict[str, Any]:
    if now is not None:
        brief["generated_at"] = _utc(now)
    return brief


def _golden_signal_sweep(
    tools: MetricTools,
    sweep: _Sweep,
    symptom_metric: str,
    incident: Window,
    baseline: Window,
    min_delta_pct: float,
) -> list[dict[str, Any]]:
    """Check every golden signal fleet-wide, not just the one that paged.

    One cheap `series_summary` per signal. Deliberately shallower than the
    symptom's gate -- no peer comparison, so a signal qualifies on a change point
    or a delta past the floor. The symptom earns the deeper check because it is
    the metric that will be attributed; the others only need to answer "is this
    also abnormal", and paying four peer queries to sharpen an answer nobody
    drills into is not worth the fan-out during a P0.

    A signal whose query fails is reported as unknown rather than healthy. This
    is invariant 1 in a different costume: silence must never read as normal.
    """
    out: list[dict[str, Any]] = []
    for signal, metrics in tools.catalog.golden_signals().items():
        for metric in metrics:
            summary = sweep.run(
                f"golden_signal:{signal}",
                lambda m=metric: tools.series_summary(m, {}, incident, baseline),
            )
            if summary is None:
                out.append(
                    {"signal": signal, "metric": metric, "material": None,
                     "is_symptom": metric == symptom_metric,
                     "note": "query failed; treated as unknown, not healthy"}
                )
                continue
            top = (summary.get("series") or [{}])[0]
            verdict = _materiality(summary, None, min_delta_pct)
            out.append(
                {
                    "signal": signal,
                    "metric": metric,
                    "material": verdict["material"],
                    "delta_pct": top.get("delta_pct"),
                    "shape": top.get("shape"),
                    "is_symptom": metric == symptom_metric,
                    "evidence_id": summary.get("evidence_id"),
                }
            )
    return out


def _materiality(
    summary: dict[str, Any] | None,
    peers: dict[str, Any] | None,
    min_delta_pct: float,
) -> dict[str, Any]:
    """Decide whether the symptom moved enough to be worth attributing.

    Three independent signals, any one of which is enough. Fleet delta alone
    would miss a single small cell on fire, whose contribution disappears into a
    planet-scale aggregate -- which is why peer deviation counts on its own.
    """
    top = ((summary or {}).get("series") or [{}])[0]
    delta = top.get("delta_pct")
    has_cp = bool(top.get("change_point"))
    outliers = len((peers or {}).get("outliers") or [])

    reasons = []
    if has_cp:
        reasons.append("fleet series has a significant change point")
    if delta is not None and abs(delta) >= min_delta_pct:
        reasons.append(f"fleet delta {delta:+.0f}% exceeds the {min_delta_pct:.0f}% floor")
    if outliers:
        reasons.append(f"{outliers} peer outlier(s) against siblings")

    if reasons:
        return {"material": True, "reason": "; ".join(reasons), "delta_pct": delta,
                "peer_outliers": outliers, "threshold_pct": min_delta_pct}
    observed = f"{delta:+.0f}%" if delta is not None else "no baseline comparison"
    return {
        "material": False,
        "reason": (
            f"symptom did not move materially ({observed}, no significant change point, "
            f"no peer outliers); attributing noise would name a healthy slice"
        ),
        "delta_pct": delta,
        "peer_outliers": outliers,
        "threshold_pct": min_delta_pct,
    }


def _assemble(
    *,
    symptom_metric: str,
    incident: Window,
    baseline: Window,
    summary: dict[str, Any] | None,
    attribution: dict[str, Any] | None,
    narrowed: dict[str, str],
    spans: dict[str, list[str]],
    onset: dict[str, Any] | None,
    onset_ts: float | None,
    peers: dict[str, Any] | None,
    peer_field: str,
    correlated: dict[str, Any] | None,
    signals: list[dict[str, Any]],
    materiality: dict[str, Any],
    sweep: _Sweep,
    tools: MetricTools,
) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    ruled_out: list[str] = []
    material = materiality["material"]

    # -- magnitude --------------------------------------------------------
    top = ((summary or {}).get("series") or [{}])[0]
    delta_pct = top.get("delta_pct")
    shape = top.get("shape")
    if summary and delta_pct is not None:
        findings.append(
            {
                "kind": "magnitude",
                "statement": (
                    f"{symptom_metric} is {_direction(delta_pct)} {abs(delta_pct):.0f}% fleet-wide "
                    f"against the baseline window ({shape})."
                ),
                "evidence": [summary.get("evidence_id")],
            }
        )
    elif summary:
        findings.append(
            {
                "kind": "magnitude",
                "statement": f"{symptom_metric} reported data but no baseline to compare against.",
                "evidence": [summary.get("evidence_id")],
            }
        )

    # -- onset ------------------------------------------------------------
    if onset_ts is not None:
        cp = (onset or {}).get("onset") or {}
        findings.append(
            {
                "kind": "onset",
                "statement": (
                    f"Onset {_utc(onset_ts)} (±{_r(onset.get('resolution_s'), 0)}s), "
                    f"{cp.get('shape')}, {cp.get('before')} → {cp.get('after')}, "
                    f"confidence {cp.get('confidence')}."
                ),
                "evidence": [onset.get("evidence_id")],
            }
        )
    elif onset is not None:
        findings.append(
            {
                "kind": "onset",
                "statement": "No significant change point in the scan window; onset unknown.",
                "evidence": [onset.get("evidence_id")],
            }
        )

    # -- where ------------------------------------------------------------
    if attribution:
        eid = attribution.get("evidence_id")
        if narrowed or spans:
            where = ", ".join(f"{k}={v}" for k, v in narrowed.items())
            for k, vs in spans.items():
                where += f"{', ' if where else ''}{k} in {{{', '.join(vs)}}}"
            findings.append(
                {"kind": "location", "statement": f"Change concentrates in {where}.", "evidence": [eid]}
            )
            ruled_out.append("Fleet-wide degradation: the change concentrates in a proper subset.")
            if spans:
                findings.append(
                    {
                        "kind": "location",
                        "statement": (
                            "Multiple values share the blame at the last level, so this is not a "
                            "single-slice fault."
                        ),
                        "evidence": [eid],
                    }
                )
            mode_note = _mode_note(attribution)
            if mode_note:
                findings.append({"kind": "mechanism", "statement": mode_note, "evidence": [eid]})
        else:
            findings.append(
                {
                    "kind": "location",
                    "statement": (
                        "No dimension concentrated the change; it looks fleet-wide or the signal "
                        "is diffuse."
                    ),
                    "evidence": [eid],
                }
            )

    # -- blast radius -----------------------------------------------------
    blast: dict[str, Any] = {}
    if peers:
        n_out, n_peers = len(peers.get("outliers") or []), peers.get("peers_compared") or 0
        blast = {
            "peer_field": peer_field,
            "outliers": n_out,
            "compared": n_peers,
            "worst": (peers.get("outliers") or [{}])[0].get("ratio_to_peers"),
        }
        if n_peers:
            verb = "is an outlier" if n_out == 1 else "are outliers"
            stmt = f"{n_out} of {n_peers} {peer_field}s {verb} against their siblings"
            worst = blast["worst"]
            stmt += f"; worst is {worst}x its peer median." if worst else "."
            findings.append({"kind": "blast_radius", "statement": stmt, "evidence": [peers.get("evidence_id")]})
            if n_out == 0:
                ruled_out.append(
                    f"Single-{peer_field} fault: no {peer_field} stands out from its siblings."
                )

    # -- what else moved --------------------------------------------------
    changes: list[dict[str, Any]] = []
    if correlated:
        eid = correlated.get("evidence_id")
        for hit in (correlated.get("correlated") or [])[:6]:
            if hit["metric"] == symptom_metric:
                continue
            changes.append(
                {
                    "metric": hit["metric"],
                    "relation": hit["relation"],
                    "lag_s": hit["lag_s"],
                    "delta_pct": (hit.get("change_point") or {}).get("delta_pct"),
                }
            )
        leading = [c for c in changes if c["relation"] == "preceded"]
        if leading:
            findings.append(
                {
                    "kind": "correlation",
                    "statement": (
                        f"{len(leading)} metric{'' if len(leading) == 1 else 's'} moved "
                        f"*before* onset: "
                        + ", ".join(f"{c['metric']} ({c['lag_s']:+.0f}s)" for c in leading)
                        + ". Precedence is weak evidence of causality, not proof."
                    ),
                    "evidence": [eid],
                }
            )
        scanned = correlated.get("scanned") or 0
        quiet = scanned - len(correlated.get("correlated") or [])
        if quiet > 0:
            ruled_out.append(
                f"{quiet} of {scanned} scanned metrics showed no significant change near onset."
            )

    # -- other golden signals ---------------------------------------------
    others = [s for s in signals if not s["is_symptom"]]
    for sig in [s for s in others if s["material"]]:
        findings.append(
            {
                "kind": "golden_signal",
                "statement": (
                    f"{sig['signal']} ({sig['metric']}) is also independently material: "
                    f"{sig['delta_pct']:+.0f}% fleet-wide, {sig['shape']}."
                ),
                "evidence": [sig["evidence_id"]],
            }
        )
    quiet = [s["signal"] for s in others if not s["material"]]
    if quiet:
        # Negative signal results are worth as much as positive ones here: "the
        # other three signals are normal" is what separates a contained fault
        # from the early edge of a general outage.
        ruled_out.append(f"No material fleet-wide movement in: {', '.join(quiet)}.")

    # -- confidence -------------------------------------------------------
    # An immaterial change is not low confidence in a conclusion; there is no
    # conclusion. Saying "none" rather than "low" keeps the brief from reading
    # as a weak finding when the honest answer is that nothing happened.
    located = bool(narrowed or spans)
    if not material:
        confidence = "none — no material change to explain"
        findings.insert(
            0,
            {
                "kind": "materiality",
                "statement": f"Not investigated further: {materiality['reason']}.",
                "evidence": [(summary or {}).get("evidence_id")],
            },
        )
    elif onset_ts is not None and located:
        confidence = "high"
    elif onset_ts is not None or located:
        confidence = "medium"
    else:
        confidence = "low"

    not_checked = [{"source": s, "consequence": why} for s, why in _UNBUILT_SOURCES]
    not_checked += [{"source": s["step"], "consequence": s["reason"]} for s in sweep.skipped]

    return {
        "schema_version": SCHEMA_VERSION,
        "stage": "1-automated-brief",
        "reasoning": "none — deterministic tool output only",
        "symptom_metric": symptom_metric,
        "windows": {"incident": incident.to_dict(), "baseline": baseline.to_dict()},
        "headline": _headline(symptom_metric, delta_pct, narrowed, spans, onset_ts, material),
        "confidence": confidence,
        "materiality": materiality,
        "onset": {"timestamp": _r(onset_ts, 1), "utc": _utc(onset_ts)} if onset_ts else None,
        "location": {"narrowed_to": narrowed or None, "spans": spans or None},
        "golden_signals": signals,
        "blast_radius": blast or None,
        "correlated_changes": changes,
        "findings": findings,
        "ruled_out": ruled_out,
        "not_checked": not_checked,
        "next_steps": _next_steps(narrowed, spans, onset_ts),
        "budget": tools.usage(),
        "evidence_ids": sweep.evidence,
    }


def _direction(pct: float) -> str:
    return "up" if pct >= 0 else "down"


def _mode_note(attribution: dict[str, Any]) -> str | None:
    """Rate effect and mix effect are different incidents with different fixes;
    surfacing which one dominates is the single most actionable line here."""
    if attribution.get("mode") != "ratio":
        return None
    path = attribution.get("attribution_path") or []
    if not path:
        return None
    top = (path[-1].get("top_values") or [{}])[0]
    rate, mix = abs(top.get("rate_effect") or 0.0), abs(top.get("mix_effect") or 0.0)
    if rate > mix * 3:
        return "Driven by the slice itself degrading, not by a traffic shift (pages the service owner)."
    if mix > rate * 3:
        return (
            "Driven by traffic shifting toward an already-worse slice, not new degradation "
            "(pages whoever changed routing)."
        )
    return None


def _headline(
    metric: str,
    delta_pct: float | None,
    narrowed: dict[str, str],
    spans: dict[str, list[str]],
    onset_ts: float | None,
    material: bool,
) -> str:
    if not material:
        return f"No significant movement in {metric} over the incident window."
    parts = [metric]
    if delta_pct is not None:
        parts.append(f"{_direction(delta_pct)} {abs(delta_pct):.0f}%")
    # Fixed order: narrowed_to is keyed by whichever dimension attribution
    # picked first, so reading it in insertion order yields "fb/eu-west-4".
    where = "/".join(narrowed[k] for k in ("region", "cell") if k in narrowed)
    if where:
        parts.append(f"in {where}")
    elif spans:
        parts.append("across " + ", ".join(f"{k}={'|'.join(v)}" for k, v in spans.items()))
    if onset_ts is not None:
        parts.append(f"since {_utc(onset_ts)}")
    return " ".join(parts) + "."


def _next_steps(
    narrowed: dict[str, str], spans: dict[str, list[str]], onset_ts: float | None
) -> list[str]:
    steps = []
    if onset_ts is not None:
        target = ", ".join(f"{k}={v}" for k, v in narrowed.items()) or "the affected entities"
        steps.append(f"Intersect {_utc(onset_ts)} with the change log for {target}.")
    if spans:
        steps.append(
            "Several values are implicated at the last level; check whether they share a "
            "dependency rather than treating them as separate faults."
        )
    if not narrowed and not spans:
        steps.append("No slice isolated. Widen the baseline or check a different symptom metric.")
    return steps


def render_brief(brief: dict[str, Any]) -> str:
    """Plain text for the incident channel. Not a dashboard -- a separate UI
    fails on adoption, so the brief has to be readable where the page lands."""
    L: list[str] = []
    L.append(f"INCIDENT BRIEF — {brief['headline']}")
    L.append(f"confidence: {brief['confidence']}   ·   reasoning: {brief['reasoning']}")
    L.append("")

    L.append("WHAT WE KNOW")
    for f in brief["findings"]:
        cites = " ".join(e for e in f["evidence"] if e)
        L.append(f"  • {f['statement']}  [{cites}]")
    if not brief["findings"]:
        L.append("  • Nothing conclusive. The sweep completed but found no significant signal.")
    L.append("")

    if brief.get("golden_signals"):
        L.append("GOLDEN SIGNALS (fleet-wide, each judged on its own terms)")
        for s in brief["golden_signals"]:
            mark = {True: "MOVED ", False: "  ok  ", None: "  ??  "}[s["material"]]
            tag = "  <- paged on this" if s["is_symptom"] else ""
            if s["material"] is None:
                L.append(f"  [{mark}] {s['signal']:<11} {s['metric']}: {s['note']}{tag}")
            else:
                L.append(
                    f"  [{mark}] {s['signal']:<11} {s['metric']}: "
                    f"{s['delta_pct']:+.1f}% ({s['shape']}){tag}"
                )
        L.append("")

    if brief["correlated_changes"]:
        L.append("MOVED AT THE SAME TIME (correlation, not causation)")
        for c in brief["correlated_changes"]:
            d = f", {c['delta_pct']:+.0f}%" if c.get("delta_pct") is not None else ""
            L.append(f"  • {c['metric']}: {c['relation']} by {c['lag_s']:+.0f}s{d}")
        L.append("")

    if brief["ruled_out"]:
        L.append("RULED OUT")
        for r in brief["ruled_out"]:
            L.append(f"  • {r}")
        L.append("")

    L.append("NOT CHECKED — absence of evidence here is not evidence of absence")
    for n in brief["not_checked"]:
        L.append(f"  • {n['source']}: {n['consequence']}")
    L.append("")

    if brief["next_steps"]:
        L.append("SUGGESTED NEXT STEPS")
        for s in brief["next_steps"]:
            L.append(f"  • {s}")
        L.append("")

    b = brief["budget"]
    L.append(f"cost: {b['queries']} queries, {b['points_scanned']:,} points scanned")
    return "\n".join(L)
