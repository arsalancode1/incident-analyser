"""End-to-end walkthrough on a synthetic incident.

Scenario: v2.41 rolls out to eu-west-4/fb at T-40m and degrades RPC error rate.
This is the sequence of tool calls an orchestrator would make. Nothing here
needs an LLM -- that's the point. The tools do the work; the model only decides
which one to call next.
"""

import json
import time

from metric_analysis import Budget, MetricTools, SyntheticTSDB, ToolError, Window, demo_catalog

NOW = 1_750_000_000.0
ONSET_TRUTH = NOW - 40 * 60


def show(title, obj):
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")
    print(json.dumps(obj, indent=2, default=str))


def main():
    catalog = demo_catalog()
    tsdb = SyntheticTSDB(seed=11, fault_start=ONSET_TRUTH, rollout_start=ONSET_TRUTH)
    tools = MetricTools(catalog, tsdb, budget=Budget(max_queries=60))

    incident = Window(NOW - 30 * 60, NOW)
    baseline = Window(NOW - 4 * 3600, NOW - 3 * 3600)

    # 0. A typo must fail loudly, not silently look healthy.
    try:
        tools.series_summary("spanner.rpc.errors", {"region": "us-east1"}, incident, baseline)
    except ToolError as e:
        show("0. Guardrail: misspelled field value is rejected, not silently empty", e.to_dict())

    # 1. Orient: how bad, fleet-wide, and what shape?
    show(
        "1. Fleet-wide error signal",
        tools.series_summary("spanner.rpc.errors", {}, incident, baseline),
    )

    # 2. Attribute the change. One call, no manual drill-down.
    attribution = tools.explain_delta(
        metric="spanner.rpc.errors",
        fields=["region", "cell", "job", "version"],
        baseline=baseline,
        incident=incident,
    )
    show("2. explain_delta: which slice moved the error rate", attribution)

    culprit = attribution["narrowed_to"] or {}

    # 3. Confirm against peers -- immune to the diurnal cycle.
    show(
        "3. Peer comparison across cells in the implicated region",
        tools.peer_comparison(
            "spanner.rpc.errors",
            peer_field="cell",
            incident=incident,
            filters={"region": culprit["region"]} if "region" in culprit else {},
        ),
    )

    # 4. Pin the onset precisely (coarse scan, then fine re-query).
    onset = tools.find_onset(
        "spanner.rpc.errors",
        filters={k: v for k, v in culprit.items() if k in ("region", "cell")},
        window=Window(NOW - 3 * 3600, NOW),
    )
    show("4. Onset detection", onset)

    if onset.get("onset"):
        detected = onset["onset"]["timestamp"]
        print(f"\n  ground truth onset : {ONSET_TRUTH:.0f}")
        print(f"  detected onset     : {detected:.0f}")
        print(f"  error              : {abs(detected - ONSET_TRUTH):.0f}s")

        # 5. What else moved at the same moment?
        show(
            "5. Correlation scan around the onset",
            tools.correlation_scan(
                onset=detected,
                window=Window(NOW - 3 * 3600, NOW),
                filters={k: v for k, v in culprit.items() if k in ("region", "cell")},
            ),
        )

    show("Resource usage for the whole investigation", tools.usage())
    show("Underlying TSDB load", tsdb.stats())


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nwall clock: {time.time() - t0:.2f}s")
