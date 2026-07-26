"""TSDB access: the client protocol, and a synthetic implementation.

`SyntheticTSDB` is not a mock. It is the seed of the replay harness: same
interface, same aggregation semantics, same EMPTY_SELECTOR contract, with a
generated fleet standing in for frozen snapshots of real incidents. Tests that
pass against it are the tests that will run against those snapshots later, so it
has to behave like the real thing in the ways that matter -- particularly in
distinguishing "nothing there" from "no such thing".

The generated fleet carries one injected incident: a build rolls out to two
cells in every region, and in one specific cell it degrades. That shape is
deliberate. A fault that lines up perfectly with a single dimension is easy and
teaches the analysis layer nothing; here version, region and cell each explain
part of the story and only their intersection is the actual answer.

Requirements for a real implementation:

  * Aggregation is pushed server-side. Never ship points to the orchestrator to
    sum them there -- that is the whole cost model of the system.
  * `fetch` returns EMPTY_SELECTOR, not OK with zero series, when the filter
    matches no known entity (invariant 1).
  * `points_scanned` reports storage-side cost, not response size, so the budget
    charges for fan-out rather than for what survived aggregation.
"""

from __future__ import annotations

import hashlib
import itertools
from typing import Any, Iterator, Protocol

import numpy as np

from .catalog import MetricCatalog, MetricSpec, demo_catalog
from .types import QueryResult, ResultStatus, Series, ToolError, Window


def stable_hash(*parts: str) -> int:
    """Deterministic across processes.

    Python's builtin hash() is salted per process (PEP 456), so using it for
    fixture offsets makes the synthetic fleet reshape itself on every run. That
    is fatal for a replay harness, whose entire purpose is reproducibility --
    and it shows up as a test that fails roughly one run in six, which is worse
    than a hard failure because it reads as noise.
    """
    return int.from_bytes(
        hashlib.blake2b("\x00".join(parts).encode(), digest_size=8).digest(), "big"
    )


class TSDBClient(Protocol):
    def fetch(
        self,
        metric: str,
        filters: dict[str, str],
        window: Window,
        step_s: float,
        group_by: list[str],
    ) -> QueryResult: ...

    def field_values(self, metric: str, field: str, window: Window) -> list[str]: ...


# Relative traffic weights. Nothing depends on the exact numbers; they exist so
# slices differ in size, which is what makes the mix-effect term in
# attribute_ratio non-trivial.
_REGION_WEIGHT = {"us-east-1": 1.3, "us-central-1": 1.0, "eu-west-4": 0.9, "asia-east-1": 0.8}
_REGION_PHASE = {"us-east-1": 0.0, "us-central-1": 0.5, "eu-west-4": 2.6, "asia-east-1": 4.4}
_CELL_WEIGHT = {"aa": 1.0, "ab": 0.95, "ba": 1.05, "bb": 0.9, "fa": 1.0, "fb": 1.1}
_JOB_WEIGHT = {"frontend": 1.0, "txn-coordinator": 0.8, "tablet-server": 1.2}

# Diurnal swing. Kept modest on purpose: a deep cycle inside a three-hour window
# looks like a ramp to the change point detector, and the point of the fixture
# is to test onset detection against a real step, not against the time of day.
_DIURNAL_AMPLITUDE = 0.08


class SyntheticTSDB:
    """Deterministic synthetic fleet with one injected incident.

    Determinism is by construction, not by seeding a stream: every value is a
    pure function of (seed, metric, entity, timestamp). Two fetches covering the
    same instant at different resolutions therefore agree, which matters because
    `find_onset` deliberately re-queries a narrow band at finer resolution and
    would otherwise be comparing two different universes.
    """

    ROLLED_CELLS = ("fa", "fb")
    FAULT_REGION = "eu-west-4"
    FAULT_CELL = "fb"
    # Error rates the faulted jobs jump to. Chosen so their contributions are
    # comparable -- frontend carries more traffic, txn-coordinator degrades
    # harder -- which is what exercises the min_share guard in explain_delta.
    # tablet-server runs in the same cell and stays healthy, so the guard has
    # something it must exclude.
    FAULT_JOBS = {"frontend": 0.12, "txn-coordinator": 0.14}
    NEW_VERSION = "v2.41"
    OLD_VERSION = "v2.40"
    # Leader elections destabilise slightly before RPCs start failing. Real
    # incidents have this structure and correlation_scan ranks on it, so the
    # fixture needs at least one genuinely leading indicator.
    PAXOS_LEAD_S = 180.0

    def __init__(
        self,
        seed: int = 0,
        fault_start: float | None = None,
        rollout_start: float | None = None,
        catalog: MetricCatalog | None = None,
    ) -> None:
        self.seed = int(seed)
        self.fault_start = fault_start
        self.rollout_start = rollout_start
        self.catalog = catalog or demo_catalog()
        self._queries = 0
        self._series_returned = 0
        self._points_generated = 0

    # -- determinism helpers ----------------------------------------------
    def _u(self, *key: Any) -> float:
        """Uniform [0,1) keyed by content rather than by call order.

        Built on stable_hash for the reason given there: an RNG stream would
        also depend on how many points were requested and in what order, so the
        same entity would jitter differently between a coarse and a fine query.
        """
        return stable_hash(str(self.seed), *(str(k) for k in key)) / 2.0**64

    def _wiggle(self, t: np.ndarray, amplitude: float, *key: Any) -> np.ndarray:
        """Smooth pseudo-noise: a few incommensurate sinusoids.

        Not drawn from an RNG, because an RNG stream depends on how many points
        you asked for -- the same entity would jitter differently at 15s and
        300s resolution and the refined onset query would disagree with the
        coarse one.
        """
        out = np.zeros_like(t)
        for i in range(3):
            period = 120.0 + 780.0 * self._u("period", i, *key)
            phase = 2.0 * np.pi * self._u("phase", i, *key)
            out += np.sin(2.0 * np.pi * t / period + phase)
        return amplitude * out / 3.0

    def _entity_key(self, ent: dict[str, str]) -> tuple[str, ...]:
        return tuple(f"{k}={ent[k]}" for k in sorted(ent))

    # -- fleet enumeration -------------------------------------------------
    def _dims(self, spec: MetricSpec) -> list[str]:
        # `version` is not an independent dimension: it is a consequence of
        # where the rollout has reached, so it is applied per entity below.
        return [f for f in spec.fields if f != "version"]

    def _entities(self, spec: MetricSpec, filters: dict[str, str]) -> list[dict[str, str]]:
        dims = self._dims(spec)
        pools = [[filters[d]] if d in filters else spec.fields[d] for d in dims]
        return [dict(zip(dims, combo)) for combo in itertools.product(*pools)]

    def _version_segments(
        self,
        ent: dict[str, str],
        t: np.ndarray,
        want: str | None,
    ) -> Iterator[tuple[str, np.ndarray]]:
        """Yield (version, mask) pairs for one entity.

        A task that restarts onto a new build ends one labelled series and
        begins another; it does not retroactively relabel its history. Modelling
        that honestly is what makes the version dimension informative, and it is
        why NaN gaps have to be a first-class thing everywhere downstream.
        """
        rolled = self.rollout_start is not None and ent.get("cell") in self.ROLLED_CELLS
        if not rolled:
            if want in (None, self.OLD_VERSION):
                yield self.OLD_VERSION, np.ones_like(t, dtype=bool)
            return
        for version, mask in (
            (self.OLD_VERSION, t < self.rollout_start),
            (self.NEW_VERSION, t >= self.rollout_start),
        ):
            if want not in (None, version):
                continue
            if mask.any():
                yield version, mask

    # -- signal models -----------------------------------------------------
    def _faulted(self, ent: dict[str, str]) -> bool:
        return ent.get("region") == self.FAULT_REGION and ent.get("cell") == self.FAULT_CELL

    def _traffic(self, ent: dict[str, str], t: np.ndarray, step_s: float) -> np.ndarray:
        """Requests in each step. Scaled by step so window sums are resolution
        independent -- otherwise a coarse and a fine query disagree about how
        much traffic there was."""
        key = self._entity_key(ent)
        w = (
            _REGION_WEIGHT.get(ent.get("region", ""), 1.0)
            * _CELL_WEIGHT.get(ent.get("cell", ""), 1.0)
            * _JOB_WEIGHT.get(ent.get("job", ""), 1.0)
        )
        scale = 0.9 + 0.2 * self._u("scale", *key)
        phase = _REGION_PHASE.get(ent.get("region", ""), 0.0)
        diurnal = 1.0 + _DIURNAL_AMPLITUDE * np.sin(2.0 * np.pi * t / 86400.0 + phase)
        rps = 100.0 * w * scale * diurnal * (1.0 + self._wiggle(t, 0.02, "traffic", *key))
        return rps * step_s

    def _error_rate(self, ent: dict[str, str], t: np.ndarray) -> np.ndarray:
        key = self._entity_key(ent)
        base = 0.0015 + 0.001 * self._u("err", *key)
        rate = base * (1.0 + self._wiggle(t, 0.05, "err", *key))
        job = ent.get("job")
        if self.fault_start is not None and self._faulted(ent) and job in self.FAULT_JOBS:
            faulted = self.FAULT_JOBS[job] * (1.0 + self._wiggle(t, 0.04, "fault", *key))
            rate = np.where(t >= self.fault_start, faulted, rate)
        return rate

    def _values(self, spec: MetricSpec, ent: dict[str, str], t: np.ndarray, step_s: float) -> np.ndarray:
        key = self._entity_key(ent)
        name = spec.name

        if name == "spanner.rpc.count":
            return self._traffic(ent, t, step_s)

        if name == "spanner.rpc.errors":
            return self._traffic(ent, t, step_s) * self._error_rate(ent, t)

        if name == "spanner.rpc.latency":
            base = 18.0 + 8.0 * self._u("lat", *key)
            v = base * (1.0 + self._wiggle(t, 0.04, "lat", *key))
            if self.fault_start is not None and self._faulted(ent) and ent.get("job") in self.FAULT_JOBS:
                v = np.where(t >= self.fault_start, 55.0 + 10.0 * self._u("lat2", *key), v)
            return v

        if name == "spanner.paxos.leader_elections":
            base = 0.02 * (0.8 + 0.4 * self._u("pax", *key))
            v = base * (1.0 + self._wiggle(t, 0.06, "pax", *key))
            if self.fault_start is not None and self._faulted(ent):
                v = np.where(t >= self.fault_start - self.PAXOS_LEAD_S, 0.5 * base / 0.02, v)
            return v * step_s

        if name == "spanner.lock.wait_time":
            base = 3.0 + 1.5 * self._u("lock", *key)
            v = base * (1.0 + self._wiggle(t, 0.05, "lock", *key))
            if self.fault_start is not None and self._faulted(ent) and ent.get("job") in self.FAULT_JOBS:
                v = np.where(t >= self.fault_start, 40.0 + 6.0 * self._u("lock2", *key), v)
            return v

        if name == "spanner.tablet.split_rate":
            base = 0.3 * (0.7 + 0.6 * self._u("split", *key))
            return base * (1.0 + self._wiggle(t, 0.08, "split", *key))

        if name == "spanner.storage.read_bytes":
            return self._traffic(ent, t, step_s) * (4.0e3 + 2.0e3 * self._u("bytes", *key))

        if name == "spanner.task.restarts":
            base = 0.0005 * (0.5 + self._u("restart", *key))
            v = np.full_like(t, base)
            if self.rollout_start is not None and self._faulted(ent):
                v = np.where(t >= self.rollout_start, 0.01, v)
            return v * step_s * (1.0 + self._wiggle(t, 0.05, "restart", *key))

        # Unknown metric in the catalog but not modelled here. Flat is honest.
        return np.full_like(t, 1.0)

    # -- TSDBClient --------------------------------------------------------
    def fetch(
        self,
        metric: str,
        filters: dict[str, str],
        window: Window,
        step_s: float,
        group_by: list[str],
    ) -> QueryResult:
        spec = self.catalog.get(metric)
        self._queries += 1

        entities = self._entities(spec, {k: v for k, v in filters.items() if k != "version"})
        want_version = filters.get("version")
        if want_version is not None and "version" not in spec.fields:
            raise ToolError("unknown_field", f"'version' is not a tag on {metric!r}.", list(spec.fields))

        t = np.arange(window.start, window.end, float(step_s), dtype=float)
        if t.size == 0:
            return QueryResult(ResultStatus.NO_DATA, [], 0, "Window is shorter than one step.")

        rows: list[tuple[dict[str, str], np.ndarray]] = []
        for ent in entities:
            values = self._values(spec, ent, t, float(step_s))
            if "version" not in spec.fields:
                rows.append((dict(ent), values))
                continue
            for version, mask in self._version_segments(ent, t, want_version):
                rows.append(({**ent, "version": version}, np.where(mask, values, np.nan)))

        if not rows:
            # Invariant 1. The filter named something the fleet has never heard
            # of -- most often a version that has not shipped anywhere. Zero
            # series here would read as "healthy" to anything downstream.
            return QueryResult(
                ResultStatus.EMPTY_SELECTOR,
                [],
                0,
                f"Filter {filters} matched no known entity for {metric!r}.",
            )

        points = len(rows) * int(t.size)
        self._points_generated += points
        series = self._group(rows, t, group_by, spec.aggregation)
        self._series_returned += len(series)

        if all(not np.isfinite(s.values).any() for s in series):
            # Entities exist, but none reported in this window. A finding, not a
            # malformed query -- note the different status.
            return QueryResult(ResultStatus.NO_DATA, series, points, "Known entities reported no points.")
        return QueryResult(ResultStatus.OK, series, points)

    @staticmethod
    def _group(
        rows: list[tuple[dict[str, str], np.ndarray]],
        t: np.ndarray,
        group_by: list[str],
        aggregation: str,
    ) -> list[Series]:
        """Aggregate server-side, the way a real TSDB must.

        NaN-aware throughout: a timestamp where every member of a group is
        absent stays NaN rather than becoming zero, because zero errors and no
        report are different facts.
        """
        buckets: dict[tuple[str, ...], list[np.ndarray]] = {}
        labels_for: dict[tuple[str, ...], dict[str, str]] = {}
        for labels, values in rows:
            key = tuple(labels.get(g, "<unset>") for g in group_by)
            buckets.setdefault(key, []).append(values)
            labels_for.setdefault(key, {g: labels.get(g, "<unset>") for g in group_by})

        out: list[Series] = []
        for key in sorted(buckets):
            stack = np.vstack(buckets[key])
            finite = np.isfinite(stack)
            count = finite.sum(axis=0)
            total = np.where(finite, stack, 0.0).sum(axis=0)
            combined = total / np.maximum(count, 1) if aggregation == "avg" else total
            out.append(Series(labels_for[key], t, np.where(count > 0, combined, np.nan)))
        return out

    def field_values(self, metric: str, field: str, window: Window) -> list[str]:
        spec = self.catalog.get(metric)
        if field not in spec.fields:
            raise ToolError("unknown_field", f"{field!r} is not a tag on {metric!r}.", list(spec.fields))
        if field != "version":
            return list(spec.fields[field])

        # Versions are resolved against the window, not the schema: which builds
        # were running is a question about time, and the answer is what makes a
        # rollout visible in the first place.
        if self.rollout_start is None:
            return [self.OLD_VERSION]
        present = []
        if window.start < self.rollout_start:
            present.append(self.OLD_VERSION)
        if window.end > self.rollout_start:
            present.extend([self.OLD_VERSION, self.NEW_VERSION])
        return sorted(set(present))

    def stats(self) -> dict[str, Any]:
        return {
            "queries": self._queries,
            "series_returned": self._series_returned,
            "points_generated": self._points_generated,
        }
