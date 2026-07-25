"""Series -> compact summary, and peer outlier ranking.

This module is where invariant 2 is enforced. A 4-hour series at 60s resolution
is 240 floats; serialised for a model that is on the order of 1500 tokens, and
having spent them the model still has to eyeball a step change in a list of
numbers, which it is bad at. Everything here reduces a series to a couple of
dozen tokens that answer the questions actually asked of it: how big, which
direction, when did it start, what shape.

`peer_outliers` implements the other half of the philosophy: compare an entity
to its structural siblings rather than to its own past. "8x the median of its 47
siblings" survives a traffic spike that a week-over-week delta reports as an
incident.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .changepoint import detect_change_point
from .types import Series, Window, _r

# 0.6745 is the 75th percentile of the standard normal; dividing the MAD by it
# (equivalently multiplying by 1.4826) puts the scale on the same footing as a
# standard deviation for normally distributed data, so the threshold below reads
# like a z-score.
_MAD_TO_SIGMA = 1.4826

# Higher than a conventional 3.0 because peer groups are small and the cost of a
# false outlier is an on-call engineer investigating a healthy cell.
PEER_Z_THRESHOLD = 3.5


def summarize_series(
    series: Series,
    incident: Window,
    baseline: Window | None = None,
) -> dict[str, Any]:
    """Everything worth knowing about one series, and nothing else."""
    current = series.slice(incident).finite()
    out: dict[str, Any] = {"labels": series.labels}

    if current.size == 0:
        # The series exists but reported nothing in this window. That is a
        # finding -- a task that stopped exporting is a symptom, not an absence.
        out.update({"status": "no_points_in_window", "delta_pct": None})
        return out

    q50, q95 = (float(x) for x in np.percentile(current, [50, 95]))
    mean_c = float(current.mean())
    out.update(
        {
            "n": int(current.size),
            "mean": _r(mean_c),
            "p50": _r(q50),
            "p95": _r(q95),
            "max": _r(float(current.max())),
            "last": _r(float(current[-1])),
        }
    )

    delta_pct: float | None = None
    if baseline is not None:
        base = series.slice(baseline).finite()
        if base.size:
            mean_b = float(base.mean())
            out["baseline_mean"] = _r(mean_b)
            if abs(mean_b) > 1e-12:
                delta_pct = 100.0 * (mean_c - mean_b) / abs(mean_b)
        else:
            # Common and important: a label value that did not exist in the
            # baseline, e.g. a version that only shipped during the incident.
            # Saying so beats reporting an infinite increase from zero.
            out["baseline_mean"] = None
            out["baseline_note"] = "series reported no points in the baseline window"
    out["delta_pct"] = _r(delta_pct, 1)

    cp = detect_change_point(series.timestamps, series.values)
    if cp is not None and cp.significant:
        out["change_point"] = cp.to_dict()
        out["shape"] = cp.shape
    else:
        out["change_point"] = None
        out["shape"] = "no significant change"
    return out


def peer_outliers(
    series: list[Series],
    window: Window,
    z_threshold: float = PEER_Z_THRESHOLD,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Rank entities against their siblings using a median/MAD robust z-score.

    Median and MAD rather than mean and standard deviation because the outlier
    we are hunting is itself in the sample: a single cell at 8x drags the mean
    up and inflates the variance enough to hide itself.
    """
    stats: list[tuple[Series, float]] = []
    for s in series:
        v = s.slice(window).finite()
        if v.size:
            stats.append((s, float(v.mean())))

    # With one or two peers "outlier" has no meaning; anything can be called
    # anomalous relative to a single sibling.
    if len(stats) < 3:
        return []

    values = np.array([m for _, m in stats], dtype=float)
    median = float(np.median(values))
    scale = _MAD_TO_SIGMA * float(np.median(np.abs(values - median)))
    if scale <= 1e-12:
        # MAD collapses when more than half the peers are identical, which is
        # normal for near-zero counters. Fall back to the standard deviation.
        sd = float(values.std(ddof=1))
        scale = sd if sd > 1e-12 else 0.0
    if scale == 0.0:
        return []  # every peer identical: nothing to rank

    out: list[dict[str, Any]] = []
    for i, (s, m) in enumerate(stats):
        z = (m - median) / scale
        if abs(z) < z_threshold:
            continue
        peer_median = float(np.median(np.delete(values, i)))
        out.append(
            {
                "labels": s.labels,
                "value": _r(m),
                "peer_median": _r(peer_median),
                "ratio_to_peers": _r(m / peer_median, 2) if abs(peer_median) > 1e-12 else None,
                "robust_z": _r(z, 1),
                "peers": len(stats) - 1,
            }
        )
    out.sort(key=lambda d: -abs(d["robust_z"] or 0.0))
    return out[:limit]
