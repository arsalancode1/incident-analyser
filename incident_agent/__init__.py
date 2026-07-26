"""Harness and orchestrator for the incident debugging agent.

This is the layer with the model in it. The metrics package deliberately has
none, and that separation is load-bearing: everything deterministic lives below
this line, so the model's contribution can be evaluated -- and swapped -- without
touching the analysis.

    from incident_agent import build_model, investigate, render_investigation

    inv = investigate(tools, build_model("anthropic:claude-opus-5"),
                      "spanner.rpc.errors", incident_window)
    print(render_investigation(inv))
"""

from __future__ import annotations

from .dispatch import Dispatcher, ToolInvocation
from .loop import Investigation, investigate, render_investigation
from .models import (
    AnthropicModel,
    ModelClient,
    ModelResponse,
    OpenAIModel,
    ScriptedModel,
    ToolCall,
    build_model,
)
from .prompts import SYSTEM_PROMPT, VERIFIER_PROMPT
from .schemas import TOOL_NAMES, TOOL_SCHEMAS, validate
from .verifier import verify

__all__ = [
    "SYSTEM_PROMPT",
    "TOOL_NAMES",
    "TOOL_SCHEMAS",
    "VERIFIER_PROMPT",
    "AnthropicModel",
    "Dispatcher",
    "Investigation",
    "ModelClient",
    "ModelResponse",
    "OpenAIModel",
    "ScriptedModel",
    "ToolCall",
    "ToolInvocation",
    "build_model",
    "investigate",
    "render_investigation",
    "validate",
    "verify",
]
