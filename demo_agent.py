"""The agent loop over the metrics tools.

    python3 demo_agent.py                            # scripted model, no key needed
    python3 demo_agent.py --model anthropic:claude-opus-5
    python3 demo_agent.py --model openai:gpt-4o
    python3 demo_agent.py --model openai:llama-3.3-70b --base-url http://localhost:8000/v1
    python3 demo_agent.py --json                     # the full replay record

With no --model the run uses a scripted client: a fixed sequence of tool calls
standing in for a model, so the harness is demonstrable with no key, no network
and no cost. It exercises the same code path a real model takes -- including a
deliberately malformed filter, to show that a bad selector comes back as a
correctable error rather than ending the investigation.
"""

import argparse
import json
import os
import sys

from incident_agent import ModelResponse, ScriptedModel, ToolCall, build_model, investigate
from incident_agent.loop import render_investigation
from metric_analysis import Budget, MetricTools, SyntheticTSDB, Window, demo_catalog

NOW = 1_750_000_000.0
ONSET = NOW - 40 * 60
INCIDENT = Window(NOW - 30 * 60, NOW)
BASELINE = Window(NOW - 4 * 3600, NOW - 3 * 3600)


def scripted() -> ScriptedModel:
    """A plausible investigation, hand-written. This is what a model would do."""
    return ScriptedModel(
        [
            ModelResponse(
                text="The sweep implicates eu-west-4/fb on v2.41. First, which versions exist?",
                tool_calls=[
                    ToolCall("c1", "list_field_values", {"metric": "spanner.rpc.errors", "field": "version"})
                ],
            ),
            ModelResponse(
                text="Now a deliberate typo, to show error recovery.",
                tool_calls=[
                    ToolCall("c2", "peer_comparison", {
                        "metric": "spanner.rpc.errors", "peer_field": "cell",
                        "filters": {"region": "eu-west4"}})
                ],
            ),
            ModelResponse(
                text="Corrected from the suggestions.",
                tool_calls=[
                    ToolCall("c3", "peer_comparison", {
                        "metric": "spanner.rpc.errors", "peer_field": "cell",
                        "filters": {"region": "eu-west-4"}})
                ],
            ),
            ModelResponse(
                text="Falsification: if the rollout is the cause, an un-rolled cell "
                     "in the same region should show no onset.",
                tool_calls=[
                    ToolCall("c4", "find_onset", {
                        "metric": "spanner.rpc.errors",
                        "filters": {"region": "eu-west-4", "cell": "aa"}})
                ],
            ),
            ModelResponse(
                tool_calls=[
                    ToolCall("c5", "conclude", {
                        "summary": "Errors on spanner.rpc.errors concentrate in eu-west-4/fb from "
                                   "14:26:40Z, coincident with v2.41 reaching that cell.",
                        "hypotheses": [
                            {"statement": "The v2.41 rollout to eu-west-4/fb introduced the "
                                          "regression; frontend and txn-coordinator both degraded "
                                          "while tablet-server in the same cell did not.",
                             "confidence": "medium",
                             "evidence_ids": ["__ATTRIBUTION__"]},
                        ],
                        "falsification": "If the rollout were the cause, an un-rolled cell in the "
                                         "same region should show no onset. Cell aa shows none, "
                                         "which is consistent. This does not exclude a cause "
                                         "specific to fb that coincided with the push.",
                        "ruled_out": ["Fleet-wide degradation", "A pure traffic shift (mix effect ~0)"],
                        "not_checked": ["change log — no tool exists, so the rollout is unconfirmed",
                                        "logs", "traces", "topology"],
                    })
                ],
            ),
        ]
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", help="provider:model, e.g. anthropic:claude-opus-5")
    p.add_argument("--base-url", help="for OpenAI-compatible endpoints")
    p.add_argument("--verify", action="store_true", help="run the verification pass")
    p.add_argument("--json", action="store_true", help="dump the full replay record")
    args = p.parse_args()

    tools = MetricTools(
        demo_catalog(),
        SyntheticTSDB(seed=11, fault_start=ONSET, rollout_start=ONSET),
        budget=Budget(max_queries=80),
    )

    if args.model:
        kwargs = {"base_url": args.base_url} if args.base_url else {}
        try:
            model = build_model(args.model, **kwargs)
        except RuntimeError as e:
            print(f"{e}\n", file=sys.stderr)
            return
        verifier = build_model(args.model, **kwargs) if args.verify else None
    else:
        model = scripted()
        verifier = None
        # The scripted conclusion cites the attribution result, whose id is only
        # known once the sweep has run. Resolve it the way a real model would:
        # by reading it off a tool result.
        probe = MetricTools(demo_catalog(), SyntheticTSDB(seed=11, fault_start=ONSET, rollout_start=ONSET))
        eid = probe.explain_delta(
            "spanner.rpc.errors", ["region", "cell", "job", "version"], BASELINE, INCIDENT
        )["evidence_id"]
        for resp in model._script:
            for call in resp.tool_calls:
                for h in call.arguments.get("hypotheses", []):
                    h["evidence_ids"] = [eid]
        # stderr, so `--json` stays pipeable into jq and the replay harness.
        print("(no --model given: using a scripted client, no API key required)\n", file=sys.stderr)

    inv = investigate(
        tools, model, "spanner.rpc.errors", INCIDENT, BASELINE,
        max_turns=10, verifier=verifier,
    )

    if args.json:
        print(json.dumps(inv.to_dict(), indent=2, default=str))
        return

    print(render_investigation(inv))
    print("\n--- tool calls the agent made after the sweep ---")
    for i in inv.invocations:
        flag = "ERROR" if i.is_error else "  ok "
        detail = i.result.get("error") or i.evidence_id or ""
        print(f"  [{flag}] {i.name:18} {json.dumps(i.arguments, default=str)[:70]:70} {detail}")


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY") and "--model" in sys.argv:
        print("note: no ANTHROPIC_API_KEY in the environment\n", file=sys.stderr)
    main()
