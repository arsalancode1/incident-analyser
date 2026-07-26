"""Provider-agnostic model access.

The orchestrator must not care which vendor is behind it. Three reasons this is
worth the indirection rather than calling one SDK directly:

  * Evaluation. Comparing models on the replay harness is the only way to learn
    whether the expensive one is actually better at *sequencing* -- which is the
    only job the model has here. That question is unanswerable if the provider
    is welded into the loop.
  * Testing. `ScriptedModel` runs the entire agent loop with no network and no
    API key, so the loop's control flow is covered by ordinary unit tests.
    Without it, every test of the harness needs a live vendor.
  * The dependency rule. numpy is this project's only runtime dependency. SDKs
    are imported lazily inside the adapter that needs them, so installing one is
    opt-in and the package still imports cleanly with neither present.

Messages are held in a canonical provider-neutral form and translated at the
edge. That canonical form is also what lands in the replay record, so a
transcript captured against one provider can be replayed against another.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field
from typing import Any, Protocol

# Sensible default. Sequencing tool calls is not a task that needs the largest
# model, and stage-1 targets 60-90s to first brief, so a faster model is a
# defensible swap -- measure it on the replay harness before assuming.
DEFAULT_ANTHROPIC_MODEL = "claude-opus-5"
DEFAULT_OPENAI_MODEL = "gpt-4o"


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}


@dataclass
class ModelResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"
    usage: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "tool_calls": [t.to_dict() for t in self.tool_calls],
            "stop_reason": self.stop_reason,
            "usage": self.usage,
        }


class ModelClient(Protocol):
    """What the loop needs from a model. Deliberately one method."""

    name: str

    def complete(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse: ...


# -- canonical message helpers -------------------------------------------
# Blocks are: {"type": "text"|"tool_call"|"tool_result", ...}


def user_text(text: str) -> dict[str, Any]:
    return {"role": "user", "content": [{"type": "text", "text": text}]}


def assistant_turn(response: ModelResponse) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    if response.text:
        content.append({"type": "text", "text": response.text})
    for call in response.tool_calls:
        content.append(
            {"type": "tool_call", "id": call.id, "name": call.name, "arguments": call.arguments}
        )
    return {"role": "assistant", "content": content}


def tool_results(results: list[tuple[str, dict[str, Any], bool]]) -> dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {"type": "tool_result", "id": tid, "content": payload, "is_error": is_error}
            for tid, payload, is_error in results
        ],
    }


class ScriptedModel:
    """Replays a fixed list of responses. No network, no key, no vendor.

    This is what makes the loop testable. It is also the seam the replay harness
    plugs into: feed it the assistant turns captured from a real investigation
    and the harness re-runs that investigation deterministically against frozen
    tool results.
    """

    def __init__(self, script: list[ModelResponse], name: str = "scripted") -> None:
        self.name = name
        self._script = list(script)
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        # Snapshot, not alias. The loop appends to `messages` in place, so
        # storing the reference would make every recorded call show the final
        # conversation rather than the one that produced this response --
        # silently useless for replay and for debugging the loop.
        self.calls.append(
            {
                "system": system,
                "messages": copy.deepcopy(messages),
                "tools": [t["name"] for t in tools],
            }
        )
        if not self._script:
            # Running dry means the loop took a path the script did not
            # anticipate. Failing loudly beats silently ending the turn and
            # looking like the model chose to stop.
            raise RuntimeError(
                f"ScriptedModel exhausted after {len(self.calls)} calls; the loop asked for more."
            )
        return self._script.pop(0)


class AnthropicModel:
    """Adapter for the Anthropic Messages API. SDK imported lazily."""

    def __init__(
        self,
        model: str = DEFAULT_ANTHROPIC_MODEL,
        api_key: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        client: Any = None,
    ) -> None:
        self.name = f"anthropic:{model}"
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        if client is not None:
            self._client = client
        else:
            try:
                import anthropic  # noqa: PLC0415 - lazy on purpose, see module docstring
            except ImportError as e:  # pragma: no cover - depends on environment
                raise RuntimeError(
                    "The anthropic SDK is not installed. `pip install anthropic`, or use a "
                    "different ModelClient. It is deliberately not a dependency of this project."
                ) from e
            self._client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    @staticmethod
    def _to_provider(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for m in messages:
            content = []
            for b in m["content"]:
                if b["type"] == "text":
                    content.append({"type": "text", "text": b["text"]})
                elif b["type"] == "tool_call":
                    content.append(
                        {"type": "tool_use", "id": b["id"], "name": b["name"], "input": b["arguments"]}
                    )
                elif b["type"] == "tool_result":
                    content.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": b["id"],
                            "content": json.dumps(b["content"], default=str),
                            "is_error": b["is_error"],
                        }
                    )
            out.append({"role": m["role"], "content": content})
        return out

    def complete(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system,
            messages=self._to_provider(messages),
            tools=[
                {"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
                for t in tools
            ],
        )
        text, calls = "", []
        for block in resp.content:
            if block.type == "text":
                text += block.text
            elif block.type == "tool_use":
                calls.append(ToolCall(id=block.id, name=block.name, arguments=dict(block.input)))
        return ModelResponse(
            text=text,
            tool_calls=calls,
            stop_reason=resp.stop_reason or "end_turn",
            usage={
                "input_tokens": getattr(resp.usage, "input_tokens", 0),
                "output_tokens": getattr(resp.usage, "output_tokens", 0),
            },
        )


class OpenAIModel:
    """Adapter for OpenAI-compatible chat completions. SDK imported lazily.

    `base_url` makes this reach any OpenAI-compatible endpoint -- vLLM, Together,
    Groq, a local server -- so "configurable models" does not mean "two vendors".
    """

    def __init__(
        self,
        model: str = DEFAULT_OPENAI_MODEL,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        client: Any = None,
    ) -> None:
        self.name = f"openai:{model}"
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        if client is not None:
            self._client = client
        else:
            try:
                import openai  # noqa: PLC0415 - lazy on purpose, see module docstring
            except ImportError as e:  # pragma: no cover - depends on environment
                raise RuntimeError(
                    "The openai SDK is not installed. `pip install openai`, or use a different "
                    "ModelClient. It is deliberately not a dependency of this project."
                ) from e
            self._client = openai.OpenAI(
                api_key=api_key or os.environ.get("OPENAI_API_KEY"), base_url=base_url
            )

    @staticmethod
    def _to_provider(system: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for m in messages:
            texts = [b["text"] for b in m["content"] if b["type"] == "text"]
            calls = [b for b in m["content"] if b["type"] == "tool_call"]
            results = [b for b in m["content"] if b["type"] == "tool_result"]
            if m["role"] == "assistant":
                msg: dict[str, Any] = {"role": "assistant", "content": "\n".join(texts) or None}
                if calls:
                    msg["tool_calls"] = [
                        {
                            "id": c["id"],
                            "type": "function",
                            "function": {
                                "name": c["name"],
                                "arguments": json.dumps(c["arguments"], default=str),
                            },
                        }
                        for c in calls
                    ]
                out.append(msg)
                continue
            # Tool results are their own role in this API, so a canonical user
            # turn carrying results becomes several provider messages.
            for r in results:
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": r["id"],
                        "content": json.dumps(r["content"], default=str),
                    }
                )
            if texts:
                out.append({"role": "user", "content": "\n".join(texts)})
        return out

    def complete(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=self._to_provider(system, messages),
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t["description"],
                        "parameters": t["parameters"],
                    },
                }
                for t in tools
            ],
        )
        choice = resp.choices[0]
        calls = []
        for tc in choice.message.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                # Malformed JSON from the model is a tool error, not a crash --
                # dispatch will hand it back so the model can correct itself.
                args = {"__malformed__": tc.function.arguments}
            calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))
        usage = getattr(resp, "usage", None)
        return ModelResponse(
            text=choice.message.content or "",
            tool_calls=calls,
            stop_reason=choice.finish_reason or "stop",
            usage={
                "input_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
                "output_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
            },
        )


def build_model(spec: str, **kwargs: Any) -> ModelClient:
    """Build a client from a `provider:model` string.

        build_model("anthropic:claude-opus-5")
        build_model("openai:gpt-4o")
        build_model("openai:llama-3.3-70b", base_url="http://localhost:8000/v1")
    """
    provider, _, model = spec.partition(":")
    provider = provider.strip().lower()
    if provider == "anthropic":
        return AnthropicModel(model=model or DEFAULT_ANTHROPIC_MODEL, **kwargs)
    if provider in ("openai", "openai-compatible"):
        return OpenAIModel(model=model or DEFAULT_OPENAI_MODEL, **kwargs)
    raise ValueError(f"Unknown provider {provider!r}. Known: anthropic, openai.")
