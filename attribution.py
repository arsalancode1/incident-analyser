"""Multi-dimensional contribution analysis.

The question is never 'did the metric move'. It's 'which slice moved it'.
One call here replaces ten to twenty agent-driven group-by queries, and it does
the arithmetic deterministically instead of asking a language model to.

Two modes:
  * additive  - counters. Adtributor-style explanatory power + surprise.
  * ratio     - error rates, hit rates. Exact decomposition of the rate change
                into a *rate effect* and a *mix effect*, which are different
                incidents with different fixes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .changepoint import _r
from .types import Series, Window


@dataclass
class SliceContribution:
    field: str
    value: str
    baseline: float
    current: float
    explanatory_power: float   # share of the total delta this slice accounts for
    surprise: float            # how much its *share* of the total shifted
    rate_effect: float | None = None
    mix_effect: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "slice": f"{self.field}={self.value}",
            "baseline": _r(self.baseline),
            "current": _r(self.current),
            "explanatory_power": _r(self.explanatory_power, 3),
            "surprise": _r(self.surprise, 3),
        }
        if self.rate_effect is not None:
            d["rate_effect"] = _r(self.rate_effect, 3)
            d["mix_effect"] = _r(self.mix_effect, 3)
        return d


def _agg(series: list[Series], w: Window, field: str) -> dict[str, float]:
    """Sum a metric per value of one field over a window."""
    out: dict[str, float] = {}
    for s in series:
        v = s.slice(w).finite()
        if len(v) == 0:
            continue
        key = s.labels.get(field, "<unset>")
        out[key] = out.get(key, 0.0) + float(np.sum(v))
    return out


def _js_surprise(p: float, q: float) -> float:
    """Per-element Jensen-Shannon contribution between two share values."""
    if p <= 0 and q <= 0:
        return 0.0
    m = 0.5 * (p + q)
    t = 0.0
    if p > 0:
        t += 0.5 * p * np.log2(p / m)
    if q > 0:
        t += 0.5 * q * np.log2(q / m)
    return float(t)


def attribute_additive(
    series: list[Series],
    field: str,
    baseline: Window,
    incident: Window,
) -> list[SliceContribution]:
    b = _agg(series, baseline, field)
    c = _agg(series, incident, field)

    # Normalize for unequal window lengths before comparing totals.
    scale = incident.duration / baseline.duration if baseline.duration else 1.0
    b = {k: v * scale for k, v in b.items()}

    B, C = sum(b.values()), sum(c.values())
    delta = C - B
    out = []
    for k in set(b) | set(c):
        bv, cv = b.get(k, 0.0), c.get(k, 0.0)
        ep = (cv - bv) / delta if abs(delta) > 1e-12 else 0.0
        sp = _js_surprise(bv / B if B else 0.0, cv / C if C else 0.0)
        out.append(SliceContribution(field, k, bv, cv, ep, sp))
    out.sort(key=lambda s: (-abs(s.explanatory_power), -s.surprise))
    return out


def attribute_ratio(
    num_series: list[Series],
    den_series: list[Series],
    field: str,
    baseline: Window,
    incident: Window,
) -> list[SliceContribution]:
    """Exact decomposition of a ratio change.

        R_c - R_b = SUM_v [ (d_c/D_c)(r_c - r_b) ]  <- rate effect
                  + SUM_v [ (d_c/D_c - d_b/D_b) r_b ]  <- mix effect

    Rate effect means that slice genuinely got worse. Mix effect means traffic
    moved toward a slice that was always worse. Conflating them sends you after
    the wrong on-call.
    """
    nb, nc = _agg(num_series, baseline, field), _agg(num_series, incident, field)
    db, dc = _agg(den_series, baseline, field), _agg(den_series, incident, field)

    Db, Dc = sum(db.values()), sum(dc.values())
    Nb, Nc = sum(nb.values()), sum(nc.values())
    Rb = Nb / Db if Db else 0.0
    Rc = Nc / Dc if Dc else 0.0
    total_delta = Rc - Rb

    out = []
    for k in set(db) | set(dc):
        d_b, d_c = db.get(k, 0.0), dc.get(k, 0.0)
        n_b, n_c = nb.get(k, 0.0), nc.get(k, 0.0)
        r_b = n_b / d_b if d_b else 0.0
        r_c = n_c / d_c if d_c else 0.0
        share_b = d_b / Db if Db else 0.0
        share_c = d_c / Dc if Dc else 0.0

        rate_eff = share_c * (r_c - r_b)
        mix_eff = (share_c - share_b) * r_b
        contrib = rate_eff + mix_eff
        ep = contrib / total_delta if abs(total_delta) > 1e-15 else 0.0
        sp = _js_surprise(
            n_b / Nb if Nb else 0.0,
            n_c / Nc if Nc else 0.0,
        )
        out.append(
            SliceContribution(field, k, r_b, r_c, ep, sp, rate_effect=rate_eff, mix_effect=mix_eff)
        )
    out.sort(key=lambda s: (-abs(s.explanatory_power), -s.surprise))
    return out


def explain_delta(
    series_by_metric: dict[str, list[Series]],
    metric: str,
    fields: list[str],
    baseline: Window,
    incident: Window,
    denominator: str | None = None,
    teep: float = 0.67,
    cohesion: float = 0.25,
    max_depth: int = 4,
) -> dict[str, Any]:
    """Find the smallest slice specification that explains the change.

    Greedy recursive narrowing: pick the dimension whose top values explain the
    most delta with the fewest slices, commit to those values, recurse into the
    remaining dimensions within that subset. This mirrors what a human does
    with a dashboard, and it is exponentially cheaper than a full cuboid scan.
    """
    num = series_by_metric[metric]
    den = series_by_metric.get(denominator) if denominator else None
    mode = "ratio" if den else "additive"

    def subset(pool: list[Series], sel: dict[str, str]) -> list[Series]:
        return [s for s in pool if all(s.labels.get(k) == v for k, v in sel.items())]

    selection: dict[str, str] = {}
    spans: dict[str, list[str]] = {}
    path: list[dict[str, Any]] = []
    remaining = list(fields)

    for _ in range(max_depth):
        if not remaining:
            break
        n_sub = subset(num, selection)
        d_sub = subset(den, selection) if den else None
        if not n_sub:
            break

        best = None
        for f in remaining:
            if f in selection:
                continue
            if mode == "ratio":
                contribs = attribute_ratio(n_sub, d_sub, f, baseline, incident)
            else:
                contribs = attribute_additive(n_sub, f, baseline, incident)
            if len(contribs) < 2:
                continue
            # Smallest prefix of values reaching the explanatory threshold --
            # but once past it, keep absorbing values that are comparably large.
            # Without this, a 0.70/0.29 split reports only the first value and
            # silently drops a second slice that is failing just as hard.
            cum, take = 0.0, []
            for c in contribs:
                if c.explanatory_power <= 0:
                    break
                if cum >= teep and c.explanatory_power < cohesion * take[-1].explanatory_power:
                    break
                take.append(c)
                cum += c.explanatory_power
            if not take or cum < teep:
                continue
            # prefer fewer slices, then higher concentration
            score = (len(take), -cum)
            if best is None or score < best[0]:
                best = (score, f, take, contribs)

        if best is None:
            break
        _, fname, take, contribs = best
        path.append(
            {
                "field": fname,
                "implicated_values": [c.value for c in take],
                "explains": _r(sum(c.explanatory_power for c in take), 3),
                "top_values": [c.to_dict() for c in contribs[:4]],
            }
        )
        remaining = [f for f in remaining if f != fname]
        if len(take) == 1:
            selection[fname] = take[0].value
        else:
            # Several values share the blame. Report them rather than picking
            # one, and stop -- drilling into an arbitrary branch would be a guess.
            spans[fname] = [c.value for c in take]
            break

    return {
        "metric": metric,
        "mode": mode,
        "baseline_window": baseline.to_dict(),
        "incident_window": incident.to_dict(),
        "narrowed_to": selection or None,
        "spans": spans or None,
        "attribution_path": path,
        "interpretation": _interpret(path, selection, spans, mode),
    }


def _interpret(
    path: list[dict[str, Any]],
    sel: dict[str, str],
    spans: dict[str, list[str]],
    mode: str,
) -> str:
    if not path:
        return "No dimension concentrated the change; it looks fleet-wide or the signal is diffuse."

    parts = []
    if sel:
        parts.append(", ".join(f"{k}={v}" for k, v in sel.items()))
    for k, vs in spans.items():
        parts.append(f"{k} in {{{', '.join(vs)}}}")
    if not parts:
        return "Change is spread across multiple values; no single slice dominates."

    note = f"Change concentrates in {'; '.join(parts)}."
    if spans:
        note += " Multiple values at the last level are affected comparably, so this is not a single-slice fault."
    if mode == "ratio":
        top = path[-1]["top_values"][0]
        re_, me_ = abs(top.get("rate_effect") or 0), abs(top.get("mix_effect") or 0)
        if re_ > me_ * 3:
            note += " Driven by the slice itself degrading, not by a traffic shift."
        elif me_ > re_ * 3:
            note += " Driven by traffic shifting toward an already-worse slice, not new degradation."
    return note
