"""Onset detection and shape classification.

This runs in code, not in the model. Two reasons. Models are unreliable at
spotting a step in a list of floats -- they anchor on the largest value rather
than the largest *change*. And the onset timestamp is the fact you intersect
with the deploy log, so it needs to be a number with an error bar, not a
paraphrase.

The scan is a Welch t-test over every admissible split, computed from prefix
sums so the whole thing is O(n) rather than O(n^2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .types import _r

# Re-exported: attribution.py imports _r from here, and rounding for output is
# close enough to this module's job to leave that import alone.
__all__ = ["ChangePoint", "detect_change_point", "classify_shape", "_r"]

# A split has to beat this to count. Deliberately far above a nominal 95%
# critical value: we test every split point, so the null distribution is the
# maximum of ~n correlated t-statistics, not a single one. Under flat noise that
# maximum routinely reaches 3-4. Reporting a change point that isn't there is
# worse than missing a weak one -- it sends the agent to intersect a meaningless
# timestamp with the deploy log and find a spurious "cause".
T_THRESHOLD = 6.0

# And a change has to be large enough to matter, not merely certain. With
# thousands of points, a 1% drift is statistically overwhelming and
# operationally noise. This is what keeps a slow diurnal ramp from being
# reported as an onset.
MIN_RELATIVE_EFFECT = 0.12


@dataclass
class ChangePoint:
    timestamp: float
    index: int
    t_stat: float
    before_mean: float
    after_mean: float
    delta_pct: float | None
    shape: str
    significant: bool

    @property
    def confidence(self) -> str:
        a = abs(self.t_stat)
        return "high" if a >= 12 else ("medium" if a >= T_THRESHOLD else "low")

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": _r(self.timestamp, 1),
            "before": _r(self.before_mean),
            "after": _r(self.after_mean),
            "delta_pct": _r(self.delta_pct, 1),
            "shape": self.shape,
            "t_stat": _r(self.t_stat, 1),
            "confidence": self.confidence,
            "significant": self.significant,
        }


def classify_shape(values: np.ndarray, k: int) -> str:
    """What the series did after the break. The label changes what you ask next.

    step  -> something switched: a push, a flag, a failover.
    ramp  -> something is filling up: a queue, a cache, a disk.
    spike -> it already recovered; look for what retried or restarted.
    """
    before, after = values[:k], values[k:]
    if before.size == 0 or after.size == 0:
        return "unknown"
    m_before, m_after = float(before.mean()), float(after.mean())
    amp = abs(m_after - m_before)
    if amp == 0.0:
        return "flat"

    if after.size >= 6:
        tail = after[-max(3, after.size // 4) :]
        if abs(float(tail.mean()) - m_before) < 0.35 * amp:
            return "spike"
        x = np.arange(after.size, dtype=float)
        slope = float(np.polyfit(x, after, 1)[0])
        # A trend that covers most of the step height means the level never
        # settled -- it is still moving, which is a ramp, not a step.
        if abs(slope) * after.size > 0.6 * amp:
            return "ramp"
    return "step_up" if m_after > m_before else "step_down"


def detect_change_point(
    timestamps: np.ndarray,
    values: np.ndarray,
    min_segment: int | None = None,
    t_threshold: float = T_THRESHOLD,
    min_relative_effect: float = MIN_RELATIVE_EFFECT,
) -> ChangePoint | None:
    """Strongest single level shift in the series, or None if too short.

    Returns a ChangePoint even when it fails the significance bars, with
    `significant=False`, so callers can see what the best candidate was. Callers
    that only want real onsets check `.significant`.
    """
    t = np.asarray(timestamps, dtype=float)
    v = np.asarray(values, dtype=float)
    keep = np.isfinite(t) & np.isfinite(v)
    t, v = t[keep], v[keep]
    n = int(v.size)
    if n < 12:
        return None

    # Both segments need enough samples for a variance estimate. Without a
    # floor, the split one point from the end wins on every noisy series.
    if min_segment is None:
        min_segment = max(4, n // 20)
    if 2 * min_segment + 1 > n:
        min_segment = max(2, n // 4)

    c1 = np.cumsum(v)
    c2 = np.cumsum(v * v)
    ks = np.arange(min_segment, n - min_segment + 1)
    if ks.size == 0:
        return None

    n_left = ks.astype(float)
    n_right = float(n) - n_left
    s_left = c1[ks - 1]
    s_right = c1[-1] - s_left
    q_left = c2[ks - 1]
    q_right = c2[-1] - q_left

    m_left = s_left / n_left
    m_right = s_right / n_right
    # Population variance from prefix sums, corrected to the sample estimate.
    var_left = np.maximum(q_left / n_left - m_left**2, 0.0) * n_left / np.maximum(n_left - 1, 1)
    var_right = np.maximum(q_right / n_right - m_right**2, 0.0) * n_right / np.maximum(n_right - 1, 1)

    se = np.sqrt(var_left / n_left + var_right / n_right)
    # A perfectly clean step has zero variance on both sides and infinite t.
    # Floor the denominator relative to the signal so it stays a large finite
    # number instead of a NaN.
    floor = 1e-12 * max(1.0, float(np.max(np.abs(v))))
    t_stat = (m_right - m_left) / np.maximum(se, floor)

    i = int(np.argmax(np.abs(t_stat)))
    k = int(ks[i])
    before, after = float(m_left[i]), float(m_right[i])
    amp = abs(after - before)
    rel = amp / max(abs(before), abs(after), 1e-12)
    delta_pct = 100.0 * (after - before) / abs(before) if abs(before) > 1e-12 else None

    return ChangePoint(
        timestamp=float(t[k]),
        index=k,
        t_stat=float(t_stat[i]),
        before_mean=before,
        after_mean=after,
        delta_pct=delta_pct,
        shape=classify_shape(v, k),
        significant=bool(abs(t_stat[i]) >= t_threshold and rel >= min_relative_effect),
    )
