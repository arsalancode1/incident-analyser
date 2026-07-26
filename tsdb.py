"""TSDB access.

`TSDBClient` is the seam you implement against your real planet-scale store.
`SyntheticTSDB` generates a reproducible fleet with an injectable fault, which
is what the replay harness and unit tests run against.
"""

from __future__ import annotations

import hashlib
import itertools
from typing import Any, Protocol

import numpy as np

from .types import Labels, QueryResult, ResultStatus, Series, Window


def stable_hash(*parts: str) -> int:
    """Deterministic across processes.

    Python's builtin hash() is salted per process (PEP 456), so using it for
    fixture offsets makes the synthetic fleet reshape itself on every run. That
    is fatal for a replay harness, whose entire purpose is reproducibility --
    and it shows up as a test that fails one run in six.
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
    ) -> QueryResult:
        """Return one series per distinct combination of `group_by` values.

        Aggregation MUST happen server-side. Never pull per-task series to the
        client and roll up here.
        """
        ...

    def field_values(self, metric: str, fieldname: str, window: Window) -> list[str]:
        ...


class SyntheticTSDB:
    """Deterministic synthetic fleet.

    Baseline: diurnal request volume, low steady error rate, per-cell offsets.
    Fault: a version rollout degrades one (region, cell) from `fault_start`.
    """

    REGIONS = ["us-east-1", "us-central-1", "eu-west-4", "asia-south-1"]
    CELLS = ["aa", "ab", "ba", "fb"]
    JOBS = ["frontend", "txn-coordinator", "tablet-server"]

    def __init__(
        self,
        seed: int = 7,
        fault_start: float | None = None,
        fault_region: str = "eu-west-4",
        fault_cell: str = "fb",
        fault_version: str = "v2.41",
        fault_error_rate: float = 0.11,
        rollout_start: float | None = None,
    ):
        self.rng = np.random.default_rng(seed)
        self.fault_start = fault_start
        self.fault_region = fault_region
        self.fault_cell = fault_cell
        self.fault_version = fault_version
        self.fault_error_rate = fault_error_rate
        self.rollout_start = rollout_start if rollout_start is not None else fault_start
        self.queries = 0
        self.points_served = 0

    # -- fleet layout ------------------------------------------------------
    def _entities(self) -> list[Labels]:
        out = []
        for r, c, j in itertools.product(self.REGIONS, self.CELLS, self.JOBS):
            out.append({"region": r, "cell": c, "job": j})
        return out

    def _version_of(self, e: Labels, t: np.ndarray) -> np.ndarray:
        """Only the faulty cell moves to the new version, at rollout_start."""
        base = np.full(t.shape, "v2.40", dtype=object)
        if self.rollout_start and e["region"] == self.fault_region and e["cell"] == self.fault_cell:
            base[t >= self.rollout_start] = self.fault_version
        return base

    def _rates(self, e: Labels, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (requests_per_step, errors_per_step)."""
        h = stable_hash(e["region"], e["cell"], e["job"]) % 1000 / 1000.0
        # diurnal volume, region-phased
        phase = self.REGIONS.index(e["region"]) * (np.pi / 2)
        diurnal = 1.0 + 0.35 * np.sin(2 * np.pi * t / 86400.0 + phase)
        scale = {"frontend": 4000, "txn-coordinator": 1500, "tablet-server": 900}[e["job"]]
        req = scale * (0.7 + 0.6 * h) * diurnal
        req = req * (1 + 0.03 * self.rng.standard_normal(t.shape))
        req = np.clip(req, 1.0, None)

        err_rate = np.full(t.shape, 0.002 + 0.0015 * h)
        if (
            self.fault_start
            and e["region"] == self.fault_region
            and e["cell"] == self.fault_cell
            and e["job"] in ("frontend", "txn-coordinator")
        ):
            hit = t >= self.fault_start
            err_rate[hit] = self.fault_error_rate * (0.85 + 0.3 * h)
        err_rate = np.clip(err_rate * (1 + 0.08 * self.rng.standard_normal(t.shape)), 0, 1)
        return req, req * err_rate

    def _latency(self, e: Labels, t: np.ndarray) -> np.ndarray:
        h = stable_hash(e["cell"], e["job"]) % 1000 / 1000.0
        base = 18 + 9 * h
        lat = base * (1 + 0.06 * self.rng.standard_normal(t.shape))
        if (
            self.fault_start
            and e["region"] == self.fault_region
            and e["cell"] == self.fault_cell
            and e["job"] in ("frontend", "txn-coordinator")
        ):
            lat[t >= self.fault_start] *= 6.5
        return lat

    def _paxos(self, e: Labels, t: np.ndarray) -> np.ndarray:
        v = 0.4 + 0.3 * (stable_hash(e["cell"]) % 100) / 100.0
        out = np.clip(v + 0.15 * self.rng.standard_normal(t.shape), 0, None)
        if (
            self.fault_start
            and e["region"] == self.fault_region
            and e["cell"] == self.fault_cell
        ):
            out[t >= self.fault_start] += 9.0
        return out

    # -- client interface --------------------------------------------------
    def fetch(
        self,
        metric: str,
        filters: dict[str, str],
        window: Window,
        step_s: float,
        group_by: list[str],
    ) -> QueryResult:
        self.queries += 1
        t = np.arange(window.start, window.end, step_s, dtype=float)
        if len(t) == 0:
            return QueryResult(ResultStatus.NO_DATA, note="window shorter than step")

        buckets: dict[tuple[str, ...], np.ndarray] = {}
        matched_any = False

        for e in self._entities():
            versions = self._version_of(e, t)
            # version is time-varying; approximate by majority within window
            vmaj = "v2.40"
            if len(versions):
                vals, counts = np.unique(versions.astype(str), return_counts=True)
                vmaj = str(vals[np.argmax(counts)])
            full = dict(e)
            full["version"] = vmaj

            if any(full.get(k) != v for k, v in filters.items()):
                continue
            matched_any = True

            if metric == "spanner.rpc.requests":
                vals_arr = self._rates(e, t)[0]
            elif metric == "spanner.rpc.errors":
                vals_arr = self._rates(e, t)[1]
            elif metric == "spanner.rpc.latency":
                vals_arr = self._latency(e, t)
            elif metric == "spanner.paxos.leader_elections":
                vals_arr = self._paxos(e, t)
            else:
                return QueryResult(ResultStatus.NO_DATA, note=f"no data for {metric}")

            key = tuple(full.get(g, "") for g in group_by)
            if metric == "spanner.rpc.latency":
                # distributions: take the max across members as a p99 stand-in
                buckets[key] = np.maximum(buckets[key], vals_arr) if key in buckets else vals_arr
            else:
                buckets[key] = buckets.get(key, 0.0) + vals_arr

        if not matched_any:
            return QueryResult(
                ResultStatus.EMPTY_SELECTOR,
                note="filter matched zero known entities; check field values",
            )

        series = []
        for key, vals_arr in buckets.items():
            labels = {g: k for g, k in zip(group_by, key)}
            labels.update(filters)
            series.append(Series(labels, t.copy(), np.asarray(vals_arr, dtype=float), metric))

        pts = sum(len(s) for s in series)
        self.points_served += pts
        return QueryResult(ResultStatus.OK, series, points_scanned=pts)

    def field_values(self, metric: str, fieldname: str, window: Window) -> list[str]:
        vals = set()
        for e in self._entities():
            if fieldname in e:
                vals.add(e[fieldname])
        if fieldname == "version":
            vals = {"v2.40"}
            if self.rollout_start and self.rollout_start < window.end:
                vals.add(self.fault_version)
        if fieldname == "task":
            vals = {f"task-{i}" for i in range(8)}
        return sorted(vals)

    def stats(self) -> dict[str, Any]:
        return {"queries": self.queries, "points_served": self.points_served}
