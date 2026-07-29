"""Harness and orchestrator tests. Run: python3 test_incident_agent.py

Every test runs offline against ScriptedModel. That is not a compromise for CI's
sake -- it is the point. The loop's control flow (error recovery, circuit
breaking, budget exhaustion, falsification enforcement) is exactly the part that
must not depend on a vendor being reachable or on a model happening to behave a
particular way today.
"""

import json

from incident_agent import (
    Dispatcher,
    ModelResponse,
    ScriptedModel,
    ToolCall,
    investigate,
    render_investigation,
    validate,
    verify,
)
from metric_analysis import Budget, MetricTools, SyntheticTSDB, ToolError, Window, demo_catalog

NOW = 1_750_000_000.0
ONSET = NOW - 40 * 60
INCIDENT = Window(NOW - 1800, NOW)
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


def tools(**kw):
    return MetricTools(
        demo_catalog(),
        SyntheticTSDB(seed=11, fault_start=ONSET, rollout_start=ONSET, **kw),
        budget=Budget(max_queries=200),
    )


def conclude_call(cid="z", **over):
    payload = {
        "summary": "s",
        "hypotheses": [{"statement": "h", "confidence": "low", "evidence_ids": []}],
        "falsification": "checked the un-rolled cell",
    }
    payload.update(over)
    return ModelResponse(tool_calls=[ToolCall(cid, "conclude", payload)])


# ----------------------------------------------------------------- schemas
@check("schema validation rejects a bad enum with suggestions")
def _():
    try:
        validate({"type": "string", "enum": ["incident", "baseline"]}, "yesterday")
        raise AssertionError("should have raised")
    except ToolError as e:
        assert e.kind == "bad_arguments"
        assert "incident" in e.suggestions


@check("schema validation rejects unknown and missing fields")
def _():
    schema = {
        "type": "object",
        "properties": {"metric": {"type": "string"}},
        "required": ["metric"],
        "additionalProperties": False,
    }
    for bad in ({}, {"metric": "m", "sneaky": 1}, {"metric": 5}):
        try:
            validate(schema, bad)
            raise AssertionError(f"should have raised for {bad}")
        except ToolError as e:
            assert e.kind == "bad_arguments", e.kind


# ---------------------------------------------------------------- dispatch
@check("a bad filter value returns a correctable error, it does not raise")
def _():
    # The whole reason the metrics layer refuses to return [] for a bad selector
    # is so the loop can hand the error back and let the model retry.
    d = Dispatcher(tools=tools(), windows={"incident": INCIDENT, "baseline": INCIDENT, "scan": INCIDENT})
    inv = d.dispatch(ToolCall("c1", "series_summary", {"metric": "spanner.rpc.errors",
                                                       "filters": {"region": "us-east1"}}))
    assert inv.is_error
    assert inv.result["error"] == "unknown_field_value"
    assert "us-east-1" in inv.result["did_you_mean"]


@check("an unknown tool name is an error, not a crash")
def _():
    d = Dispatcher(tools=tools(), windows={"incident": INCIDENT, "baseline": INCIDENT, "scan": INCIDENT})
    inv = d.dispatch(ToolCall("c1", "drop_table", {}))
    assert inv.is_error and inv.result["error"] == "unknown_tool"


@check("concluding without a falsification test is rejected")
def _():
    d = Dispatcher(tools=tools(), windows={"incident": INCIDENT, "baseline": INCIDENT, "scan": INCIDENT})
    inv = d.dispatch(ToolCall("c", "conclude", {"summary": "s", "hypotheses": [], "falsification": "  "}))
    assert inv.is_error and inv.result["error"] == "missing_falsification"
    assert d.concluded is None


@check("a fabricated evidence id is rejected")
def _():
    d = Dispatcher(tools=tools(), windows={"incident": INCIDENT, "baseline": INCIDENT, "scan": INCIDENT})
    inv = d.dispatch(ToolCall("c", "conclude", {
        "summary": "s", "falsification": "f",
        "hypotheses": [{"statement": "h", "confidence": "high", "evidence_ids": ["ev_deadbeef"]}]}))
    assert inv.is_error and inv.result["error"] == "unknown_evidence"


@check("an uncited hypothesis is rejected but an empty hypothesis list is allowed")
def _():
    d = Dispatcher(tools=tools(), windows={"incident": INCIDENT, "baseline": INCIDENT, "scan": INCIDENT})
    bad = d.dispatch(ToolCall("c", "conclude", {
        "summary": "s", "falsification": "f",
        "hypotheses": [{"statement": "it was the rollout", "confidence": "high", "evidence_ids": []}]}))
    assert bad.is_error and bad.result["error"] == "uncited_hypothesis"
    # "I don't know" must stay reachable, or the rule becomes a trap that
    # pressures the model into inventing a citation.
    ok = d.dispatch(ToolCall("c2", "conclude", {"summary": "s", "falsification": "f", "hypotheses": []}))
    assert not ok.is_error, ok.result


# -------------------------------------------------------------------- loop
@check("the opening sweep runs before the model gets a turn")
def _():
    model = ScriptedModel([conclude_call(hypotheses=[])])
    inv = investigate(tools(), model, "spanner.rpc.errors", INCIDENT)
    # The model's first message must already contain the brief, so it cannot
    # anchor on a hypothesis formed before seeing the fixed sweep.
    first = model.calls[0]["messages"][0]["content"][0]["text"]
    assert "opening sweep" in first and "eu-west-4" in first
    assert inv.brief["location"]["narrowed_to"]["cell"] == "fb"


@check("investigation completes and records a replayable trace")
def _():
    model = ScriptedModel([
        ModelResponse(tool_calls=[ToolCall("c1", "find_onset", {
            "metric": "spanner.rpc.errors", "filters": {"region": "eu-west-4", "cell": "fb"}})]),
        conclude_call(hypotheses=[]),
    ])
    inv = investigate(tools(), model, "spanner.rpc.errors", INCIDENT)
    assert inv.stop_reason == "concluded", inv.stop_reason
    assert inv.conclusion is not None
    d = inv.to_dict()
    json.dumps(d, default=str)  # the record must survive serialisation
    assert d["invocations"][0]["tool"] == "find_onset"
    assert d["brief"] and d["transcript"] and d["model"] == "scripted"
    assert isinstance(render_investigation(inv), str)


@check("repeating an identical call trips the circuit breaker")
def _():
    call = ToolCall("c1", "find_onset", {"metric": "spanner.rpc.errors", "filters": {}})
    model = ScriptedModel([
        ModelResponse(tool_calls=[call]),
        ModelResponse(tool_calls=[ToolCall("c2", "find_onset", {"metric": "spanner.rpc.errors", "filters": {}})]),
        conclude_call(hypotheses=[]),
    ])
    inv = investigate(tools(), model, "spanner.rpc.errors", INCIDENT)
    dupes = [i for i in inv.invocations if i.result.get("error") == "duplicate_call"]
    assert len(dupes) == 1, [i.result for i in inv.invocations]


@check("tool call budget is enforced")
def _():
    # Real cell names, so the calls are refused for budget rather than for being
    # malformed -- otherwise this test passes for the wrong reason.
    script = [ModelResponse(tool_calls=[ToolCall(f"c{i}", "find_onset", {
        "metric": "spanner.rpc.errors", "filters": {"cell": c}})])
        for i, c in enumerate(["aa", "ab", "ba", "bb", "fa", "fb"])]
    script.append(conclude_call(hypotheses=[]))
    inv = investigate(tools(), ScriptedModel(script), "spanner.rpc.errors", INCIDENT,
                      max_turns=12, max_tool_calls=2)
    refused = [i for i in inv.invocations if i.result.get("error") == "budget_exceeded"]
    assert refused, [i.result.get("error") for i in inv.invocations]


@check("a model outage keeps the deterministic brief")
def _():
    class Broken:
        name = "broken"

        def complete(self, *a, **kw):
            raise RuntimeError("503 from provider")

    inv = investigate(tools(), Broken(), "spanner.rpc.errors", INCIDENT)
    assert inv.conclusion is None
    assert inv.stop_reason.startswith("model_error"), inv.stop_reason
    # Losing the model must not lose the brief; on-call still gets something.
    assert inv.brief["location"]["narrowed_to"]["cell"] == "fb"
    assert "eu-west-4" in render_investigation(inv)


@check("prose without a tool call is nudged once, then the loop stops")
def _():
    model = ScriptedModel([ModelResponse(text="I think it was the rollout."),
                           ModelResponse(text="Still thinking.")])
    inv = investigate(tools(), model, "spanner.rpc.errors", INCIDENT, max_turns=2)
    assert inv.stop_reason == "no_tool_calls", inv.stop_reason
    nudge = model.calls[1]["messages"][-1]["content"][0]["text"]
    assert "conclude" in nudge


# ---------------------------------------------------------------- verifier
@check("verifier strips an unsupported claim instead of softening it")
def _():
    model = ScriptedModel([
        ModelResponse(tool_calls=[ToolCall("c1", "find_onset", {
            "metric": "spanner.rpc.errors", "filters": {"region": "eu-west-4", "cell": "fb"}})]),
        ModelResponse(tool_calls=[ToolCall("c2", "conclude", {
            "summary": "s", "falsification": "f",
            "hypotheses": [
                {"statement": "supported one", "confidence": "high", "evidence_ids": ["PLACEHOLDER"]},
                {"statement": "invented one", "confidence": "high", "evidence_ids": ["PLACEHOLDER"]},
            ]})]),
    ])
    t = tools()
    # Resolve the placeholder to the real id the first call will produce.
    real = t.find_onset("spanner.rpc.errors", {"region": "eu-west-4", "cell": "fb"},
                        Window(NOW - 3600, NOW))["evidence_id"]
    for r in model._script:
        for c in r.tool_calls:
            for h in c.arguments.get("hypotheses", []):
                h["evidence_ids"] = [real]

    verifier = ScriptedModel([ModelResponse(text=json.dumps({"verdicts": [
        {"index": 0, "verdict": "SUPPORTED", "reason": "onset matches"},
        {"index": 1, "verdict": "UNSUPPORTED", "reason": "nothing states this"}]}))])
    inv = investigate(t, model, "spanner.rpc.errors", INCIDENT, verifier=verifier)
    assert inv.conclusion is not None, inv.stop_reason
    kept = [h["statement"] for h in inv.conclusion["hypotheses"]]
    stripped = [h["statement"] for h in inv.conclusion.get("stripped_by_verifier", [])]
    assert kept == ["supported one"], kept
    assert stripped == ["invented one"], stripped


@check("verifier never sees the agent's reasoning")
def _():
    verifier = ScriptedModel([ModelResponse(text='{"verdicts": []}')])
    inv_model = ScriptedModel([
        ModelResponse(text="SECRET_CHAIN_OF_THOUGHT: I am guessing wildly here.",
                      tool_calls=[ToolCall("c1", "find_onset", {"metric": "spanner.rpc.errors", "filters": {}})]),
        ModelResponse(tool_calls=[ToolCall("c2", "conclude", {
            "summary": "s", "falsification": "f",
            "hypotheses": [{"statement": "h", "confidence": "low", "evidence_ids": []}]})]),
    ])
    t = tools()
    inv = investigate(t, inv_model, "spanner.rpc.errors", INCIDENT, verifier=verifier)
    # The uncited hypothesis is refused, so nothing reaches the verifier; when it
    # does run, the payload must carry no trace of the reasoning.
    blob = json.dumps(verifier.calls, default=str)
    assert "SECRET_CHAIN_OF_THOUGHT" not in blob
    assert inv.invocations[-1].result.get("error") == "uncited_hypothesis"


@check("a verifier outage does not silently mark claims verified")
def _():
    class Broken:
        name = "broken-verifier"

        def complete(self, *a, **kw):
            raise RuntimeError("timeout")

    from incident_agent.dispatch import ToolInvocation
    out = verify(Broken(), {"hypotheses": [{"statement": "h", "evidence_ids": ["ev_1"]}]},
                 [ToolInvocation("c", "find_onset", {}, {"evidence_id": "ev_1"}, False, "ev_1")])
    assert out["verified"] is False and "error" in out


# ------------------------------------------------------------------ models
@check("model spec strings select a provider without importing an SDK")
def _():
    from incident_agent.models import build_model

    for spec in ("anthropic:claude-opus-5", "openai:gpt-4o"):
        try:
            build_model(spec)
        except RuntimeError as e:
            assert "not installed" in str(e), e   # expected: SDK absent here
    try:
        build_model("nope:x")
        raise AssertionError("should have raised")
    except ValueError as e:
        assert "Unknown provider" in str(e)


@check("canonical transcript translates to both provider formats")
def _():
    from incident_agent.models import AnthropicModel, OpenAIModel, ModelResponse as MR, assistant_turn, tool_results, user_text

    msgs = [user_text("hi"),
            assistant_turn(MR(text="calling", tool_calls=[ToolCall("t1", "find_onset", {"metric": "m"})])),
            tool_results([("t1", {"onset": 1.0}, False)])]

    a = AnthropicModel.__dict__["_to_provider"].__func__(msgs)
    assert a[1]["content"][1]["type"] == "tool_use"
    assert a[2]["content"][0]["type"] == "tool_result" and a[2]["content"][0]["tool_use_id"] == "t1"

    o = OpenAIModel.__dict__["_to_provider"].__func__("sys", msgs)
    assert o[0]["role"] == "system"
    assert o[2]["tool_calls"][0]["function"]["name"] == "find_onset"
    assert o[3]["role"] == "tool" and o[3]["tool_call_id"] == "t1"


# ------------------------------------------------------------------ skills
@check("a skill that encodes steps is rejected at load")
def _():
    from incident_agent.skills import SkillError, load_skill

    base = {"id": "x", "title": "t", "incident_class": "c"}
    # The whole architecture exists to avoid playbooks; a skill smuggling one in
    # would reintroduce maintenance that scales with failure modes.
    for bad in ("Step 1: check the error rate.",
                "1. Look at elections.",
                "Then run peer_comparison on the cell."):
        try:
            load_skill({**base, "tell": [bad]})
            raise AssertionError(f"should have rejected: {bad}")
        except SkillError as e:
            assert "procedure" in str(e), e
    # Declarative prose using the same words must still load.
    ok = load_skill({**base, "tell": [
        "Elections spike while QPS stays flat, and latency degrades before errors do."]})
    assert ok.tell


@check("a skill with an unknown field is rejected")
def _():
    from incident_agent.skills import SkillError, load_skill

    try:
        load_skill({"id": "x", "title": "t", "incident_class": "c", "steps": ["do a thing"]})
        raise AssertionError("should have raised")
    except SkillError as e:
        assert "unknown fields" in str(e)


@check("retrieval ranks the matching incident class first, with reasons")
def _():
    from incident_agent.skills import retrieve
    from metric_analysis.brief import build_brief

    t = tools()
    b = build_brief(t, "spanner.rpc.errors", INCIDENT, Window(NOW - 4 * 3600, NOW - 3 * 3600))
    priors = retrieve(b)
    assert priors, "expected priors for a version-correlated step change"
    assert priors[0]["skill"].startswith("bad-rollout"), [p["skill"] for p in priors]
    assert priors[0]["matched_because"], priors[0]
    # Competing classes must still surface -- a single prior is tunnel vision
    # with citations attached.
    assert len(priors) > 1, priors


@check("retrieval degrades to nothing rather than forcing a match")
def _():
    from incident_agent.skills import retrieve

    # A brief with no material signals and no location should match nothing.
    empty = {"symptom_metric": "spanner.storage.read_bytes", "golden_signals": [],
             "correlated_changes": [], "location": {"narrowed_to": None, "spans": None}}
    assert retrieve(empty) == []


@check("priors reach the model as priors, and land in the replay record")
def _():
    model = ScriptedModel([conclude_call(hypotheses=[])])
    inv = investigate(tools(), model, "spanner.rpc.errors", INCIDENT)
    assert inv.priors and inv.to_dict()["priors"] == [p["skill"] for p in inv.priors]
    prompt = model.calls[0]["messages"][0]["content"][0]["text"]
    assert "bad-rollout" in prompt
    # Framing is the guardrail: a note that reads as guidance becomes a
    # procedure the model follows.
    assert "not instructions" in prompt
    assert "may be wrong here" in prompt


@check("skills can be supplied explicitly, including none at all")
def _():
    model = ScriptedModel([conclude_call(hypotheses=[])])
    inv = investigate(tools(), model, "spanner.rpc.errors", INCIDENT, skills=[])
    assert inv.priors == []
    prompt = model.calls[0]["messages"][0]["content"][0]["text"]
    assert "bad-rollout" not in prompt


@check("the shipped skill library loads and is well formed")
def _():
    from incident_agent.skills import load_skills

    skills = load_skills()
    assert len(skills) >= 3, len(skills)
    for s in skills:
        assert s.tell, f"{s.id} has no observations"
        assert s.confusable_with, f"{s.id} names nothing it is confused with"
        assert s.provenance.get("postmortem"), f"{s.id} has no provenance"


@check("the wrong prior is demoted when its distinguishing evidence is absent")
def _():
    from incident_agent.skills import retrieve
    from metric_analysis.brief import build_brief
    from metric_analysis.scenarios import bad_rollout, mix_shift

    base = Window(NOW - 4 * 3600, NOW - 3 * 3600)

    t1 = MetricTools(demo_catalog(), bad_rollout(ONSET), budget=Budget(max_queries=90))
    rollout_priors = retrieve(build_brief(t1, "spanner.rpc.errors", INCIDENT, base))
    assert rollout_priors[0]["skill"].startswith("bad-rollout"), [p["skill"] for p in rollout_priors]

    # Same symptom metric, same material signal, same step_up shape -- all the
    # generic criteria bad-rollout matches on. But no rollout happened and the
    # change is mix-driven. Ranking it first here is exactly the failure this
    # scoring was rebalanced to prevent.
    t2 = MetricTools(demo_catalog(), mix_shift(ONSET), budget=Budget(max_queries=90))
    mix_priors = retrieve(build_brief(t2, "spanner.rpc.errors", INCIDENT, base))
    assert mix_priors, "expected the traffic-shift prior to match"
    assert mix_priors[0]["skill"].startswith("routing-shift"), [p["skill"] for p in mix_priors]
    assert not any(p["skill"].startswith("bad-rollout") for p in mix_priors), [p["skill"] for p in mix_priors]


@check("a contradicted criterion is reported as counting against the prior")
def _():
    from incident_agent.skills import retrieve
    from metric_analysis.brief import build_brief
    from metric_analysis.scenarios import bad_rollout

    t = MetricTools(demo_catalog(), bad_rollout(ONSET), budget=Budget(max_queries=90))
    priors = retrieve(build_brief(t, "spanner.rpc.errors", INCIDENT, Window(NOW - 4 * 3600, NOW - 3 * 3600)))
    against = [r for p in priors for r in p["matched_because"] if r.startswith("counts against")]
    # An unexplained prior is one the agent cannot sensibly discount, so the
    # reasons must carry the negative evidence too, not only the positive.
    assert against, [p["matched_because"] for p in priors]


if __name__ == "__main__":
    for name in PASSED:
        print(f"  PASS  {name}")
    for name, err in FAILED:
        print(f"  FAIL  {name}\n          {err}")
    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    raise SystemExit(1 if FAILED else 0)
