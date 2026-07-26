"""Tool dispatch: model-emitted call -> validated MetricTools invocation.

The load-bearing behaviour here is that **tool errors are returned to the model,
not raised**. A filter value that does not exist comes back as a structured
error carrying `did_you_mean`, the model corrects itself on the next turn, and
the investigation continues. Raising instead would convert every typo into a
failed investigation -- and the whole reason the metrics layer refuses to return
empty results for bad selectors is so this loop can recover from them.

Everything is read-only. There is no tool here that can mutate the system being
debugged, which is not an accident: an agent fanning out during a P0 is already
a risk, and one that can also act is a different category of risk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from metric_analysis.tools import MetricTools
from metric_analysis.types import ToolError, Window

from .models import ToolCall
from .schemas import SCHEMA_BY_NAME, validate


@dataclass
class ToolInvocation:
    """One dispatched call, as it appears in the replay record."""

    call_id: str
    name: str
    arguments: dict[str, Any]
    result: dict[str, Any]
    is_error: bool
    evidence_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "tool": self.name,
            "arguments": self.arguments,
            "is_error": self.is_error,
            "evidence_id": self.evidence_id,
            "result": self.result,
        }


@dataclass
class Dispatcher:
    tools: MetricTools
    windows: dict[str, Window]
    invocations: list[ToolInvocation] = field(default_factory=list)
    concluded: dict[str, Any] | None = None

    def _window(self, name: str | None, default: str) -> Window:
        return self.windows[name or default]

    def dispatch(self, call: ToolCall) -> ToolInvocation:
        try:
            result, is_error = self._run(call), False
        except ToolError as e:
            # The shape a model can act on: what went wrong, and what to try
            # instead. This is why the metrics layer raises on bad selectors
            # rather than returning [].
            result, is_error = e.to_dict(), True
        except Exception as e:  # noqa: BLE001 - an agent must not die on one bad call
            result, is_error = {"error": "internal", "message": f"{type(e).__name__}: {e}"}, True

        inv = ToolInvocation(
            call_id=call.id,
            name=call.name,
            arguments=call.arguments,
            result=result,
            is_error=is_error,
            evidence_id=result.get("evidence_id") if isinstance(result, dict) else None,
        )
        self.invocations.append(inv)
        return inv

    def _run(self, call: ToolCall) -> dict[str, Any]:
        schema = SCHEMA_BY_NAME.get(call.name)
        if schema is None:
            raise ToolError(
                "unknown_tool", f"No tool named {call.name!r}.", list(SCHEMA_BY_NAME)
            )
        validate(schema, call.arguments)
        a = call.arguments
        t = self.tools

        if call.name == "search_metrics":
            return t.search_metrics(a["intent"])
        if call.name == "describe_metric":
            return t.describe_metric(a["metric"])
        if call.name == "list_field_values":
            return t.list_field_values(a["metric"], a["field"], self._window(a.get("window"), "scan"))
        if call.name == "series_summary":
            return t.series_summary(
                a["metric"],
                a.get("filters", {}),
                self.windows["incident"],
                self.windows["baseline"],
                group_by=a.get("group_by") or [],
            )
        if call.name == "find_onset":
            return t.find_onset(a["metric"], a.get("filters", {}), self.windows["scan"])
        if call.name == "explain_delta":
            return t.explain_delta(
                a["metric"],
                a["fields"],
                self.windows["baseline"],
                self.windows["incident"],
                filters=a.get("filters", {}),
            )
        if call.name == "peer_comparison":
            return t.peer_comparison(
                a["metric"], a["peer_field"], self.windows["incident"], filters=a.get("filters", {})
            )
        if call.name == "correlation_scan":
            return t.correlation_scan(
                float(a["onset"]), self.windows["scan"], filters=a.get("filters", {})
            )
        if call.name == "conclude":
            return self._conclude(a)
        raise ToolError("unknown_tool", f"Tool {call.name!r} is declared but not wired.")

    def _conclude(self, a: dict[str, Any]) -> dict[str, Any]:
        """Accept a conclusion only if it cites evidence that actually exists.

        A hypothesis citing an id the ledger has never seen is a fabricated
        citation, and it is worse than an uncited one because it looks rigorous.
        The model is told which ids were bad so it can fix the claim rather than
        drop it.
        """
        known = set(self.tools.evidence.records)
        hypotheses = a.get("hypotheses", [])

        # Invariant 3 is absolute: every claim cites evidence. This is not a
        # trap -- an empty `hypotheses` list is explicitly valid, so a model
        # with nothing to cite has an honest exit. What it may not do is assert
        # a cause and leave the citation blank.
        uncited = [h["statement"] for h in hypotheses if not (h.get("evidence_ids") or [])]
        if uncited:
            raise ToolError(
                "uncited_hypothesis",
                f"These hypotheses cite no evidence: {uncited}. Cite the evidence_id of a "
                f"tool result that supports each one, or drop it -- concluding with an empty "
                f"hypotheses list and an honest 'not checked' list is a valid outcome.",
            )

        dangling: list[str] = []
        for h in hypotheses:
            dangling += [e for e in h.get("evidence_ids", []) if e not in known]
        if dangling:
            raise ToolError(
                "unknown_evidence",
                f"These evidence ids were never returned by a tool: {sorted(set(dangling))}. "
                f"Cite only ids you received, or state the claim as unsupported.",
                sorted(known)[:12],
            )
        if not a.get("falsification", "").strip():
            raise ToolError(
                "missing_falsification",
                "State what you would expect if your leading hypothesis were wrong, and "
                "what you found when you checked.",
            )
        self.concluded = dict(a)
        return {"accepted": True, "note": "Investigation closed."}
