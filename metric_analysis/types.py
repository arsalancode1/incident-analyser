"""Core value types shared by every module.

`Series` is deliberately thin: labels plus two parallel float arrays. Points live
here so the analysis modules can work on them -- they must never reach the tool
boundary (invariant 2). Anything crossing that boundary goes through
`summarize.py` first.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


def _r(x: Any, digits: int = 4) -> float | None:
    """Round for JSON output, mapping non-finite values to None.

    json.dumps happily emits bare NaN/Infinity, which is not valid JSON and
    which downstream parsers reject. Every number leaving a tool goes through
    here.
    """
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return round(f, digits)


class ResultStatus(Enum):
    """Why a query returned what it returned.

    The distinction between NO_DATA and EMPTY_SELECTOR is invariant 1 and the
    whole reason this enum exists:

      NO_DATA        the selector named real entities; they reported nothing.
                     A finding. Possibly interesting -- a task that stopped
                     exporting is a symptom.
      EMPTY_SELECTOR the selector named nothing that exists. A malformed query.
                     Never a finding. `MetricTools` turns this into a ToolError
                     with suggestions, because "no such region" and "that region
                     is healthy" are indistinguishable once they reach a model,
                     and it will report the reassuring one.
    """

    OK = "ok"
    NO_DATA = "no_data"
    EMPTY_SELECTOR = "empty_selector"


class ToolError(Exception):
    """Every failure the agent can act on. `suggestions` is what makes a typo
    self-correcting in one turn instead of becoming a wrong conclusion."""

    def __init__(self, kind: str, message: str, suggestions: list[str] | None = None) -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.suggestions: list[str] = list(suggestions or [])

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": self.kind,
            "message": self.message,
            "did_you_mean": self.suggestions,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"ToolError({self.kind!r}, {self.message!r}, {self.suggestions!r})"


@dataclass(frozen=True)
class Window:
    """A half-open time interval [start, end) in epoch seconds.

    Half-open so that adjacent windows tile without double-counting the shared
    endpoint -- baseline and incident windows are routinely adjacent, and a
    duplicated sample lands in both sums otherwise.
    """

    start: float
    end: float

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ToolError(
                "bad_window",
                f"Window end {self.end} precedes start {self.start}.",
            )

    @property
    def duration(self) -> float:
        return float(self.end - self.start)

    def shifted(self, by_s: float) -> Window:
        return Window(self.start + by_s, self.end + by_s)

    def previous_week(self) -> Window:
        """Same clock time, seven days earlier. The seasonal baseline exists;
        nothing selects it automatically yet (see 'Known gaps' in CLAUDE.md)."""
        return self.shifted(-7 * 86400.0)

    def mask(self, timestamps: np.ndarray) -> np.ndarray:
        return (timestamps >= self.start) & (timestamps < self.end)

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": _r(self.start, 1),
            "end": _r(self.end, 1),
            "duration_s": _r(self.duration, 1),
        }


class Series:
    """One labelled time series. Timestamps are epoch seconds, ascending.

    NaN is meaningful: it marks 'this series did not exist at this timestamp',
    which happens routinely when a label changes value -- a task restarting on a
    new build ends one `version=` series and begins another. `finite()` is how
    every consumer drops those gaps, so a version label that only appears in the
    incident window aggregates correctly instead of summing as zero.
    """

    __slots__ = ("labels", "timestamps", "values")

    def __init__(
        self,
        labels: dict[str, str],
        timestamps: np.ndarray,
        values: np.ndarray,
    ) -> None:
        self.labels: dict[str, str] = dict(labels)
        self.timestamps = np.asarray(timestamps, dtype=float)
        self.values = np.asarray(values, dtype=float)
        if self.timestamps.shape != self.values.shape:
            raise ToolError(
                "bad_series",
                f"timestamps {self.timestamps.shape} and values {self.values.shape} "
                f"have different shapes.",
            )

    def __len__(self) -> int:
        return int(self.timestamps.size)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Series({self.labels}, n={len(self)})"

    def key(self, fields: list[str]) -> tuple[str, ...]:
        return tuple(self.labels.get(f, "<unset>") for f in fields)

    def slice(self, w: Window) -> Series:
        m = w.mask(self.timestamps)
        return Series(self.labels, self.timestamps[m], self.values[m])

    def finite(self) -> np.ndarray:
        """Values with gaps dropped. Not a Series -- callers want the numbers."""
        return self.values[np.isfinite(self.values)]


@dataclass
class QueryResult:
    """What a `TSDBClient.fetch` returns.

    `points_scanned` is the *raw* cost on the storage side, not the size of what
    came back. It is what the budget charges against, because a query that
    aggregates a million points server-side into one series still cost a million
    points of someone's fan-out.
    """

    status: ResultStatus
    series: list[Series] = field(default_factory=list)
    points_scanned: int = 0
    note: str = ""

    @property
    def ok(self) -> bool:
        return self.status is ResultStatus.OK
