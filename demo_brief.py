"""Rollout stage 1: the auto-generated brief, on two fleets.

Run: python3 demo_brief.py

There is no model here. Every line of both briefs is deterministic tool output,
which is the point of shipping this before any reasoning: if the brief is not
useful on its own, an agent layered on top will not rescue it.

The second fleet is the one worth staring at. It is healthy, and the brief has
to say so. Attribution will happily decompose pure noise and hand back a slice
owning most of it -- so the interesting question for stage 1 is never "does it
find the fault", it is "does it stay quiet when there isn't one".
"""

import json
import sys

from metric_analysis import Budget, MetricTools, SyntheticTSDB, Window, demo_catalog
from metric_analysis.brief import build_brief, render_brief

NOW = 1_750_000_000.0
ONSET = NOW - 40 * 60
INCIDENT = Window(NOW - 30 * 60, NOW)
BASELINE = Window(NOW - 4 * 3600, NOW - 3 * 3600)


def run(title: str, tsdb: SyntheticTSDB, as_json: bool) -> None:
    tools = MetricTools(demo_catalog(), tsdb, budget=Budget(max_queries=60))
    brief = build_brief(tools, "spanner.rpc.errors", INCIDENT, BASELINE)
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
    print(json.dumps(brief, indent=2, default=str) if as_json else render_brief(brief))


def main() -> None:
    as_json = "--json" in sys.argv
    run(
        "A. Bad rollout: v2.41 degrades eu-west-4/fb at T-40m",
        SyntheticTSDB(seed=11, fault_start=ONSET, rollout_start=ONSET),
        as_json,
    )
    run(
        "B. Healthy fleet: nothing wrong. The brief must not invent a culprit.",
        SyntheticTSDB(seed=4),
        as_json,
    )
    print(
        "\nNote how B costs a fraction of A: the materiality gate stops before\n"
        "attribution, so a quiet fleet is cheap as well as honest.\n"
    )


if __name__ == "__main__":
    main()
