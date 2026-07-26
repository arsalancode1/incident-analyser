"""The verification pass.

A fresh model call that sees the claims and the raw tool results but *not* the
reasoning that produced them. Withholding the reasoning is the entire point: a
coherent chain of thought is precisely what makes an unsupported conclusion
persuasive, and a verifier that reads it will be talked round by it.

It strips claims rather than softening them. Hedging a wrong claim leaves it in
the report wearing a disguise, where it still misdirects on-call but is harder
to notice.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .dispatch import ToolInvocation
from .models import ModelClient, user_text
from .prompts import VERIFIER_PROMPT


def _extract_json(text: str) -> dict[str, Any] | None:
    """Models fence JSON, prefix it with prose, or emit it bare. Try in order."""
    for candidate in (text, *re.findall(r"```(?:json)?\s*(.*?)```", text, re.S)):
        try:
            parsed = json.loads(candidate.strip())
        except (json.JSONDecodeError, AttributeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def verify(
    model: ModelClient,
    conclusion: dict[str, Any],
    invocations: list[ToolInvocation],
) -> dict[str, Any]:
    """Judge each hypothesis against the evidence it cites.

    Only the cited results are shown, plus the ids of everything else that was
    gathered -- a claim resting on a result the agent never actually looked at
    should not be rescued by handing the verifier the whole investigation.
    """
    hypotheses = conclusion.get("hypotheses") or []
    if not hypotheses:
        return {"verdicts": [], "note": "No hypotheses to verify."}

    by_id = {i.evidence_id: i for i in invocations if i.evidence_id}
    cited: set[str] = set()
    for h in hypotheses:
        cited.update(h.get("evidence_ids") or [])

    evidence = {
        eid: {"tool": by_id[eid].name, "arguments": by_id[eid].arguments, "result": by_id[eid].result}
        for eid in sorted(cited)
        if eid in by_id
    }
    claims = [
        {"index": i, "statement": h["statement"], "cites": h.get("evidence_ids", [])}
        for i, h in enumerate(hypotheses)
    ]

    payload = (
        "CLAIMS TO VERIFY:\n"
        + json.dumps(claims, indent=2, default=str)
        + "\n\nEVIDENCE (raw tool results for the cited ids):\n"
        + json.dumps(evidence, indent=2, default=str)
        + "\n\nIds gathered but not cited by any claim: "
        + json.dumps(sorted(set(by_id) - cited))
    )

    try:
        response = model.complete(VERIFIER_PROMPT, [user_text(payload)], [])
    except Exception as e:  # noqa: BLE001
        # A verifier outage must not silently promote unverified claims to
        # verified. Report the failure and leave the conclusion untouched.
        return {"verdicts": [], "error": f"{type(e).__name__}: {e}", "verified": False}

    parsed = _extract_json(response.text)
    if not parsed or "verdicts" not in parsed:
        return {"verdicts": [], "error": "verifier returned unparseable output", "verified": False,
                "raw": response.text[:500]}

    verdicts = [
        {
            "index": int(v.get("index", -1)),
            "verdict": "UNSUPPORTED" if str(v.get("verdict", "")).upper() != "SUPPORTED" else "SUPPORTED",
            "reason": v.get("reason", ""),
        }
        for v in parsed["verdicts"]
        if isinstance(v, dict)
    ]
    return {
        "verdicts": verdicts,
        "verified": True,
        "model": getattr(model, "name", "unknown"),
        "unsupported": sum(1 for v in verdicts if v["verdict"] == "UNSUPPORTED"),
    }
