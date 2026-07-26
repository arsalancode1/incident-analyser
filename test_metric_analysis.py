"""Self-contained tests. Run: python3 test_metric_analysis.py

These double as the seed of the replay harness: same shape, synthetic fixtures
swapped for frozen snapshots of real incidents.
"""

import numpy as np

from metric_analysis import Budget, MetricTools, SyntheticTSDB, ToolError, Window, demo_catalog
from metric_analysis.attribution import attribute_ratio
from metric_analysis.changepoint import detect_change_point
from metric_analysis.summarize import peer_outliers
from metric_analysis.types import Series

NOW = 1_750_000_000.0
PASSED, FAILED = [], []


def check(name):
    def deco(fn):
        try:
            fn()
            PASSED.append(name)
        except Exception as e:  # noqa: BLE001
            FAILED.append((name, f"{type(e).__name__}: {e}"))
        return fn
    return deco


def mk(labels, val, n=60, start=NOW - 3600, step=60.0):
    t = np.arange(start, start + n * step, step, dtype=float)
    v = np.full(len(t), float(val)) if np.isscalar(val) else np.asarray(val, dtype=float)
    return Series(labels, t, v)


def tools(**kw):
    return MetricTools(demo_catalog(), SyntheticTSDB(seed=3, **kw), budget=Budget(max_queries=200))


# --------------------------------------------------------------- guardrails
@check("unknown metric raises with suggestions")
def _():
    try:
        demo_catalog().get("spanner.rpc.error")
        raise AssertionError("should have raised")
    except ToolError as e:
        assert e.kind == "unknown_metric"
        assert "spanner.rpc.errors" in e.suggestions


@check("misspelled field value raises, does not return empty")
def _():
    try:
        demo_catalog().validate_filter("spanner.rpc.errors", {"region": "us-east1"})
        raise AssertionError("should have raised")
    except ToolError as e:
        assert e.kind == "unknown_field_value"
        assert "us-east-1" in e.suggestions


@check("field not present on metric is rejected")
def _():
    try:
        demo_catalog().validate_filter("spanner.rpc.errors", {"datacentre": "x"})
        raise AssertionError("should have raised")
    except ToolError as e:
        assert e.kind == "unknown_field"


@check("cardinality guard blocks an unbounded grouping")
def _():
    t = tools()
    try:
        t.series_summary(
            "spanner.rpc.errors", {}, Window(NOW - 600, NOW),
            group_by=["region", "cell", "job", "task"],
        )
        raise AssertionError("should have raised")
    except ToolError as e:
        assert e.kind == "cardinality_too_high"


@check("budget exhaustion raises")
def _():
    t = MetricTools(demo_catalog(), SyntheticTSDB(seed=1), budget=Budget(max_queries=2))
    try:
        for i in range(6):
            t.series_summary("spanner.rpc.errors", {}, Window(NOW - 600 - i, NOW - i))
        raise AssertionError("should have raised")
    except ToolError as e:
        assert e.kind == "budget_exceeded"


# ------------------------------------------------------- attribution maths
@check("ratio decomposition is exact (rate + mix effects sum to the delta)")
def _():
    base, inc = Window(0, 600), Window(600, 1200)
    num, den = [], []
    rng = np.random.default_rng(0)
    for cell in ["aa", "ab", "ba", "fb"]:
        d = rng.uniform(500, 2000)
        r_b, r_c = rng.uniform(0.001, 0.02), rng.uniform(0.001, 0.2)
        dv = np.concatenate([np.full(10, d), np.full(10, d * rng.uniform(0.5, 1.5))])
        nv = np.concatenate([np.full(10, d * r_b), dv[10:] * r_c])
        t = np.arange(0, 1200, 60.0)
        num.append(Series({"cell": cell}, t, nv))
        den.append(Series({"cell": cell}, t, dv))

    contribs = attribute_ratio(num, den, "cell", base, inc)
    total = sum(c.rate_effect + c.mix_effect for c in contribs)

    Nb = sum(float(np.sum(s.slice(base).values)) for s in num)
    Db = sum(float(np.sum(s.slice(base).values)) for s in den)
    Nc = sum(float(np.sum(s.slice(inc).values)) for s in num)
    Dc = sum(float(np.sum(s.slice(inc).values)) for s in den)
    expected = Nc / Dc - Nb / Db
    assert abs(total - expected) < 1e-12, f"{total} != {expected}"


@check("pure traffic shift is reported as mix effect, not degradation")
def _():
    base, inc = Window(0, 600), Window(600, 1200)
    t = np.arange(0, 1200, 60.0)
    # 'good' rate 0.001, 'bad' rate 0.10 -- neither changes. Traffic moves.
    good_d = np.concatenate([np.full(10, 9000.0), np.full(10, 5000.0)])
    bad_d = np.concatenate([np.full(10, 1000.0), np.full(10, 5000.0)])
    num = [Series({"cell": "good"}, t, good_d * 0.001), Series({"cell": "bad"}, t, bad_d * 0.10)]
    den = [Series({"cell": "good"}, t, good_d), Series({"cell": "bad"}, t, bad_d)]

    contribs = {c.value: c for c in attribute_ratio(num, den, "cell", base, inc)}
    bad = contribs["bad"]
    assert abs(bad.rate_effect) < 1e-9, f"rate effect should vanish, got {bad.rate_effect}"
    assert bad.mix_effect > 0.01, f"mix effect should dominate, got {bad.mix_effect}"


@check("genuine degradation is reported as rate effect, not mix")
def _():
    base, inc = Window(0, 600), Window(600, 1200)
    t = np.arange(0, 1200, 60.0)
    d = np.full(20, 5000.0)  # traffic constant
    bad_rate = np.concatenate([np.full(10, 0.002), np.full(10, 0.15)])
    num = [Series({"cell": "good"}, t, d * 0.002), Series({"cell": "bad"}, t, d * bad_rate)]
    den = [Series({"cell": "good"}, t, d), Series({"cell": "bad"}, t, d)]

    contribs = {c.value: c for c in attribute_ratio(num, den, "cell", base, inc)}
    bad = contribs["bad"]
    assert bad.rate_effect > 0.05, bad.rate_effect
    assert abs(bad.mix_effect) < 1e-9, bad.mix_effect


@check("explain_delta finds the injected culprit end to end")
def _():
    onset = NOW - 40 * 60
    t = tools(fault_start=onset, rollout_start=onset)
    out = t.explain_delta(
        "spanner.rpc.errors", ["region", "cell", "job", "version"],
        baseline=Window(NOW - 4 * 3600, NOW - 3 * 3600),
        incident=Window(NOW - 30 * 60, NOW),
    )
    assert out["narrowed_to"]["region"] == "eu-west-4", out["narrowed_to"]
    assert out["narrowed_to"]["cell"] == "fb", out["narrowed_to"]
    jobs = set((out["spans"] or {}).get("job", []))
    assert jobs == {"frontend", "txn-coordinator"}, jobs
    assert "tablet-server" not in jobs


# --------------------------------------------------------------- detection
@check("change point lands within one sample of the true step")
def _():
    t = np.arange(0, 600, 1.0)
    v = np.concatenate([np.full(400, 10.0), np.full(200, 40.0)])
    v = v + np.random.default_rng(5).normal(0, 0.5, len(v))
    cp = detect_change_point(t, v)
    assert cp is not None and cp.significant
    assert abs(cp.timestamp - 400) <= 1.0, cp.timestamp


@check("flat noise produces no significant change point")
def _():
    t = np.arange(0, 400, 1.0)
    v = np.random.default_rng(9).normal(50, 2.0, len(t))
    cp = detect_change_point(t, v)
    assert cp is None or not cp.significant, cp


@check("onset detection is accurate through the tool layer")
def _():
    onset = NOW - 50 * 60
    t = tools(fault_start=onset, rollout_start=onset)
    out = t.find_onset(
        "spanner.rpc.errors", {"region": "eu-west-4", "cell": "fb"},
        Window(NOW - 3 * 3600, NOW),
    )
    assert out["onset"] is not None
    assert abs(out["onset"]["timestamp"] - onset) <= 60, out["onset"]


@check("peer comparison isolates the outlier cell")
def _():
    series = [mk({"cell": c}, 10.0 + i * 0.3) for i, c in enumerate("abcdefgh")]
    series.append(mk({"cell": "fb"}, 95.0))
    out = peer_outliers(series, Window(NOW - 3600, NOW))
    assert out and out[0]["labels"]["cell"] == "fb", out
    assert out[0]["ratio_to_peers"] > 5


@check("no-data is distinguishable from a bad selector")
def _():
    t = tools()
    res = t.series_summary("spanner.paxos.leader_elections", {}, Window(NOW - 600, NOW))
    assert res.get("status") != "no_data"
    try:
        t.series_summary("spanner.rpc.errors", {"cell": "zz"}, Window(NOW - 600, NOW))
        raise AssertionError("should have raised")
    except ToolError as e:
        assert e.kind == "unknown_field_value"


# ----------------------------------------------------- replay reproducibility
@check("stable_hash is constant across processes")
def _():
    from metric_analysis.tsdb import stable_hash
    # Builtin hash() would return a different value every run (PEP 456 salting),
    # silently reshaping the synthetic fleet and making replay meaningless.
    assert stable_hash("eu-west-4", "fb", "frontend") == 1561409042999120442


@check("synthetic fixture is byte-identical across separate processes")
def _():
    import subprocess
    import sys

    prog = (
        "import json;"
        "from metric_analysis import MetricTools, SyntheticTSDB, Window, demo_catalog;"
        "N=1750000000.0;O=N-2400;"
        "t=MetricTools(demo_catalog(), SyntheticTSDB(seed=11, fault_start=O, rollout_start=O));"
        "r=t.explain_delta('spanner.rpc.errors',['region','cell','job'],"
        "Window(N-14400,N-10800),Window(N-1800,N));"
        "print(json.dumps([r['narrowed_to'], r['spans'], r['attribution_path']], sort_keys=True))"
    )
    runs = {
        subprocess.run(
            [sys.executable, "-c", prog], capture_output=True, text=True, check=True
        ).stdout
        for _ in range(3)
    }
    assert len(runs) == 1, f"fixture is not reproducible across processes: {len(runs)} variants"


@check("every tool result carries an evidence id")
def _():
    onset = NOW - 40 * 60
    t = tools(fault_start=onset, rollout_start=onset)
    for res in [
        t.series_summary("spanner.rpc.errors", {}, Window(NOW - 1800, NOW)),
        t.explain_delta("spanner.rpc.errors", ["region", "cell"],
                        Window(NOW - 4 * 3600, NOW - 3 * 3600), Window(NOW - 1800, NOW)),
        t.peer_comparison("spanner.rpc.errors", "cell", Window(NOW - 1800, NOW)),
        t.list_field_values("spanner.rpc.errors", "region", Window(NOW - 1800, NOW)),
    ]:
        assert "evidence_id" in res
        assert t.evidence.get(res["evidence_id"]) is not None


# ------------------------------------------------------- stage-1 brief
@check("brief locates the injected incident and pins the onset")
def _():
    from metric_analysis.brief import build_brief

    onset = NOW - 40 * 60
    t = tools(fault_start=onset, rollout_start=onset)
    b = build_brief(t, "spanner.rpc.errors", Window(NOW - 1800, NOW), Window(NOW - 4 * 3600, NOW - 3 * 3600))
    assert b["confidence"] == "high", b["confidence"]
    loc = b["location"]["narrowed_to"]
    assert loc["region"] == "eu-west-4" and loc["cell"] == "fb", loc
    assert abs(b["onset"]["timestamp"] - onset) <= 60, b["onset"]
    # The headline must read region/cell, not whichever dimension attribution
    # happened to commit to first.
    assert "eu-west-4/fb" in b["headline"], b["headline"]


@check("brief refuses to attribute a flat metric")
def _():
    from metric_analysis.brief import build_brief

    # No fault injected. explain_delta would still hand back a slice owning most
    # of the noise; naming it would be the single most damaging thing this
    # component can do, so the gate must stop before attribution runs.
    t = tools()
    b = build_brief(t, "spanner.rpc.errors", Window(NOW - 1800, NOW), Window(NOW - 4 * 3600, NOW - 3 * 3600))
    assert b["materiality"]["material"] is False, b["materiality"]
    assert b["location"]["narrowed_to"] is None, b["location"]
    assert b["onset"] is None
    assert b["confidence"].startswith("none"), b["confidence"]
    skipped = {n["source"] for n in b["not_checked"]}
    assert {"explain_delta", "find_onset", "correlation_scan"} <= skipped, skipped


@check("every brief finding cites a resolvable evidence id")
def _():
    from metric_analysis.brief import build_brief

    onset = NOW - 40 * 60
    t = tools(fault_start=onset, rollout_start=onset)
    b = build_brief(t, "spanner.rpc.errors", Window(NOW - 1800, NOW), Window(NOW - 4 * 3600, NOW - 3 * 3600))
    assert b["findings"]
    for f in b["findings"]:
        ids = [e for e in f["evidence"] if e]
        assert ids, f"finding cites nothing: {f['statement']}"
        for eid in ids:
            assert t.evidence.get(eid) is not None, f"dangling evidence id {eid}"


@check("brief degrades instead of dying when the budget runs out")
def _():
    from metric_analysis.brief import build_brief

    onset = NOW - 40 * 60
    # Losing the whole brief during a P0 because one query blew a budget is the
    # wrong trade; the sweep absorbs the failure and records it.
    t = MetricTools(
        demo_catalog(),
        SyntheticTSDB(seed=3, fault_start=onset, rollout_start=onset),
        budget=Budget(max_queries=1),
    )
    b = build_brief(t, "spanner.rpc.errors", Window(NOW - 1800, NOW), Window(NOW - 4 * 3600, NOW - 3 * 3600))
    reasons = " ".join(n["consequence"] for n in b["not_checked"])
    assert "budget_exceeded" in reasons, reasons
    assert b["headline"]


@check("brief output is JSON-clean and free of raw point arrays")
def _():
    import json

    from metric_analysis.brief import build_brief, render_brief

    onset = NOW - 40 * 60
    t = tools(fault_start=onset, rollout_start=onset)
    b = build_brief(t, "spanner.rpc.errors", Window(NOW - 1800, NOW), Window(NOW - 4 * 3600, NOW - 3 * 3600))
    s = json.dumps(b, default=str)
    assert "NaN" not in s and "Infinity" not in s
    assert isinstance(render_brief(b), str)

    def scan(node):
        if isinstance(node, dict):
            return sum(scan(v) for v in node.values())
        if isinstance(node, list):
            nums = sum(1 for x in node if isinstance(x, (int, float)))
            return (1 if nums > 8 else 0) + sum(scan(v) for v in node)
        return 0

    assert scan(b) == 0, "brief leaked an array of raw points"


if __name__ == "__main__":
    for name in PASSED:
        print(f"  PASS  {name}")
    for name, err in FAILED:
        print(f"  FAIL  {name}\n          {err}")
    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    raise SystemExit(1 if FAILED else 0)
