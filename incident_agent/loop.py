"""Loop control: the harness that drives a model over the metrics tools.

Structure of an investigation:

    1. Fixed opening sweep      deterministic, no model involved
    2. Model-driven drill-down  bounded turns, bounded tool calls
    3. Conclusion               schema-validated, falsification required
    4. Verification             fresh model call, reasoning withheld

Step 1 is not an optimisation. Running the same sweep every time is what stops
an investigation anchoring on whichever hypothesis the model happened to form
first, and it means a failed or budget-starved model still leaves on-call with a
usable brief instead of nothing.

Everything the loop does lands in a JSON-serialisable record. DESIGN.md asks for
that on day one rather than retrofitted after the first bad call, and this is
day one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from metric_analysis.brief import build_brief
from metric_analysis.tools import MetricTools
from metric_analysis.types import Window

from .dispatch import Dispatcher, ToolInvocation
from .models import ModelClient, assistant_turn, tool_results, user_text
from .prompts import SYSTEM_PROMPT, opening_message
from .schemas import TOOL_SCHEMAS

SCHEMA_VERSION = 1


@dataclass
class Investigation:
    """The permanent replayable record of one investigation."""

    symptom_metric: str
    windows: dict[str, Any]
    brief: dict[str, Any]
    model: str
    transcript: list[dict[str, Any]] = field(default_factory=list)
    invocations: list[ToolInvocation] = field(default_factory=list)
    conclusion: dict[str, Any] | None = None
    verification: dict[str, Any] | None = None
    stop_reason: str = "unknown"
    turns: int = 0
    usage: dict[str, int] = field(default_factory=dict)
    tool_budget: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "symptom_metric": self.symptom_metric,
            "windows": self.windows,
            "model": self.model,
            "stop_reason": self.stop_reason,
            "turns": self.turns,
            "usage": self.usage,
            "tool_budget": self.tool_budget,
            "brief": self.brief,
            "conclusion": self.conclusion,
            "verification": self.verification,
            "invocations": [i.to_dict() for i in self.invocations],
            "transcript": self.transcript,
        }


def investigate(
    tools: MetricTools,
    model: ModelClient,
    symptom_metric: str,
    incident: Window,
    baseline: Window | None = None,
    scan_window: Window | None = None,
    max_turns: int = 10,
    max_tool_calls: int = 25,
    verifier: ModelClient | None = None,
) -> Investigation:
    """Run an investigation. Never raises on model or tool misbehaviour.

    A partial investigation with an honest stop_reason is worth more during a P0
    than an exception, and the opening brief is already in the record by the time
    the model gets its first turn.
    """
    baseline = baseline or Window(
        incident.start - 4 * incident.duration, incident.start - 3 * incident.duration
    )
    scan_window = scan_window or Window(incident.start - 2 * incident.duration, incident.end)
    windows = {"incident": incident, "baseline": baseline, "scan": scan_window}

    brief = build_brief(tools, symptom_metric, incident, baseline, scan_window=scan_window)
    inv = Investigation(
        symptom_metric=symptom_metric,
        windows={k: v.to_dict() for k, v in windows.items()},
        brief=brief,
        model=getattr(model, "name", "unknown"),
    )

    dispatcher = Dispatcher(tools=tools, windows=windows)
    budget_note = (
        f"You have at most {max_turns} turns and {max_tool_calls} tool calls."
    )
    messages: list[dict[str, Any]] = [
        user_text(opening_message(json.dumps(brief, indent=2, default=str), symptom_metric, budget_note))
    ]

    seen_calls: set[str] = set()
    tool_calls_made = 0

    for turn in range(max_turns):
        inv.turns = turn + 1
        try:
            response = model.complete(SYSTEM_PROMPT, messages, TOOL_SCHEMAS)
        except Exception as e:  # noqa: BLE001 - provider outage must not lose the brief
            inv.stop_reason = f"model_error: {type(e).__name__}: {e}"
            break

        _accumulate(inv.usage, response.usage)
        inv.transcript.append({"role": "assistant", **response.to_dict()})
        messages.append(assistant_turn(response))

        if not response.tool_calls:
            # No tools and no conclusion: the model has stopped without using the
            # one structured exit available to it. Prompt once, then give up
            # rather than looping on prose.
            if turn < max_turns - 1:
                messages.append(
                    user_text(
                        "You did not call a tool. If you are finished, call `conclude` -- "
                        "including the falsification test. If you are not, call a tool."
                    )
                )
                continue
            inv.stop_reason = "no_tool_calls"
            break

        results = []
        for call in response.tool_calls:
            if tool_calls_made >= max_tool_calls:
                results.append(
                    _refuse(
                        inv, call, "budget_exceeded", "Tool call budget spent. Call `conclude`."
                    )
                )
                continue

            # Circuit breaker. A model that repeats an identical call is stuck,
            # and letting it burn the budget re-reading the same result helps
            # nobody. Telling it what it already has is more useful than a bare
            # refusal.
            fingerprint = f"{call.name}:{json.dumps(call.arguments, sort_keys=True, default=str)}"
            if fingerprint in seen_calls and call.name != "conclude":
                results.append(
                    _refuse(
                        inv,
                        call,
                        "duplicate_call",
                        "You already made this exact call; its result is above. "
                        "Ask a different question or call `conclude`.",
                    )
                )
                continue
            seen_calls.add(fingerprint)

            invocation = dispatcher.dispatch(call)
            tool_calls_made += 1
            inv.invocations.append(invocation)
            results.append((call.id, invocation.result, invocation.is_error))

        messages.append(tool_results(results))
        inv.transcript.append({"role": "tool_results", "results": [r[1] for r in results]})

        if dispatcher.concluded is not None:
            inv.conclusion = dispatcher.concluded
            inv.stop_reason = "concluded"
            break
    else:
        inv.stop_reason = "max_turns"

    if inv.stop_reason == "unknown":
        inv.stop_reason = "max_turns"
    inv.tool_budget = tools.usage()

    if inv.conclusion is not None and verifier is not None:
        from .verifier import verify  # local import keeps the module graph acyclic

        inv.verification = verify(verifier, inv.conclusion, dispatcher.invocations)
        _apply_verdicts(inv)

    return inv


def _refuse(
    inv: Investigation, call: Any, error: str, message: str
) -> tuple[str, dict[str, Any], bool]:
    """Record a refused call and shape the reply the model sees.

    Refusals belong in the replay record as much as executions do: what the
    agent *tried* and was stopped from doing is exactly what an eval needs to
    distinguish a model that got lucky from one that was well-behaved.
    """
    result = {"error": error, "message": message}
    inv.invocations.append(
        ToolInvocation(
            call_id=call.id,
            name=call.name,
            arguments=call.arguments,
            result=result,
            is_error=True,
        )
    )
    return call.id, result, True


def _accumulate(total: dict[str, int], delta: dict[str, int]) -> None:
    for k, v in (delta or {}).items():
        total[k] = total.get(k, 0) + int(v)


def _apply_verdicts(inv: Investigation) -> None:
    """Strip unsupported hypotheses rather than softening them.

    A hedged wrong claim is still wrong and is harder for a reader to catch, so
    removal is the correct outcome. The stripped claims stay in the record --
    what the agent wanted to say but could not support is exactly the signal an
    eval needs.
    """
    verification = inv.verification or {}
    unsupported = {v["index"] for v in verification.get("verdicts", []) if v["verdict"] == "UNSUPPORTED"}
    if not unsupported or not inv.conclusion:
        return
    kept, stripped = [], []
    for i, h in enumerate(inv.conclusion.get("hypotheses", [])):
        (stripped if i in unsupported else kept).append(h)
    inv.conclusion["hypotheses"] = kept
    inv.conclusion["stripped_by_verifier"] = stripped


def render_investigation(inv: Investigation) -> str:
    """Text for the incident channel: the brief, then what the agent added."""
    from metric_analysis.brief import render_brief

    out = [render_brief(inv.brief), "", "=" * 72, "AGENT INVESTIGATION", "=" * 72]
    c = inv.conclusion
    if not c:
        out.append(f"No conclusion reached (stop reason: {inv.stop_reason}).")
        out.append("The deterministic brief above still stands on its own.")
        return "\n".join(out)

    out.append(c.get("summary", ""))
    out.append("")
    hypotheses = c.get("hypotheses") or []
    if hypotheses:
        out.append("HYPOTHESES")
        for h in hypotheses:
            out.append(f"  [{h['confidence']}] {h['statement']}  [{' '.join(h['evidence_ids'])}]")
    else:
        out.append("HYPOTHESES: none the evidence supports.")
    out.append("")
    out.append(f"FALSIFICATION TEST\n  {c.get('falsification', '(none stated)')}")
    for label, key in (("RULED OUT", "ruled_out"), ("NOT CHECKED", "not_checked")):
        if c.get(key):
            out.append("")
            out.append(label)
            out += [f"  • {x}" for x in c[key]]
    if c.get("stripped_by_verifier"):
        out.append("")
        out.append("STRIPPED BY VERIFIER (claims the evidence did not support)")
        out += [f"  • {h['statement']}" for h in c["stripped_by_verifier"]]
    out.append("")
    out.append(
        f"model: {inv.model} · turns: {inv.turns} · tool calls: {len(inv.invocations)} "
        f"· queries: {inv.tool_budget.get('queries')} · stop: {inv.stop_reason}"
    )
    return "\n".join(out)
