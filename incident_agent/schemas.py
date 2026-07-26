"""The closed vocabulary the model is allowed to speak.

Two design decisions worth defending, both aimed at shrinking the surface where
a model can be confidently wrong:

**Windows are named, not numeric.** The model selects "incident", "baseline" or
"scan"; the harness resolves those to real epoch ranges. A model asked to supply
timestamps will invent plausible ones, and an investigation anchored on a
hallucinated window produces real-looking analysis of the wrong hour. Naming
them removes the failure mode entirely rather than validating against it.

**Concluding is a tool call.** Parsing a conclusion out of prose means accepting
whatever shape the model felt like emitting. As a schema-validated call, the
falsification test and the "not checked" list are *required fields* -- the model
cannot conclude without stating what would have changed its mind. That is
mandatory falsification enforced by the type system rather than by a prompt
asking nicely.

The validator here is a deliberately small JSON-Schema subset. `jsonschema` is a
fine library and this is not the place to add a dependency for it.
"""

from __future__ import annotations

from typing import Any

from metric_analysis.types import ToolError

WINDOW_NAMES = ("incident", "baseline", "scan")

_FILTERS = {
    "type": "object",
    "description": (
        "Tag filters, e.g. {\"region\": \"eu-west-4\"}. Values must come from "
        "list_field_values, never from memory. A value that does not exist is an "
        "error, not an empty result."
    ),
    "additionalProperties": {"type": "string"},
}

_WINDOW = {
    "type": "string",
    "enum": list(WINDOW_NAMES),
    "description": "Which named window to use. You cannot supply raw timestamps.",
}


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    }


TOOL_SCHEMAS: list[dict[str, Any]] = [
    _tool(
        "search_metrics",
        "Find metrics by intent. Use when you do not know the metric name.",
        {"intent": {"type": "string", "description": "What you are looking for, in words."}},
        ["intent"],
    ),
    _tool(
        "describe_metric",
        "Semantics of a metric: what it counts, how to interpret a move, its denominator.",
        {"metric": {"type": "string"}},
        ["metric"],
    ),
    _tool(
        "list_field_values",
        "Live values of a tag. Resolve every filter value through this before using it.",
        {"metric": {"type": "string"}, "field": {"type": "string"}, "window": _WINDOW},
        ["metric", "field"],
    ),
    _tool(
        "series_summary",
        "Stats, delta against baseline, change point and shape for a metric slice.",
        {
            "metric": {"type": "string"},
            "filters": _FILTERS,
            "group_by": {"type": "array", "items": {"type": "string"}},
        },
        ["metric"],
    ),
    _tool(
        "find_onset",
        "Precise onset timestamp for a slice: coarse scan, then a fine re-query. "
        "The single most valuable fact in an incident.",
        {"metric": {"type": "string"}, "filters": _FILTERS},
        ["metric"],
    ),
    _tool(
        "explain_delta",
        "Which slice moved the metric. Decomposes the change across the given "
        "dimensions and narrows to the smallest specification that explains it. "
        "One call replaces a dozen group-by queries; prefer it over manual drilling.",
        {
            "metric": {"type": "string"},
            "fields": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Dimensions to decompose across, coarsest first.",
            },
            "filters": _FILTERS,
        },
        ["metric", "fields"],
    ),
    _tool(
        "peer_comparison",
        "Rank entities against their structural siblings (robust z-score). "
        "Immune to traffic spikes and the diurnal cycle, unlike a temporal delta.",
        {"metric": {"type": "string"}, "peer_field": {"type": "string"}, "filters": _FILTERS},
        ["metric", "peer_field"],
    ),
    _tool(
        "correlation_scan",
        "What else changed near a timestamp. Metrics that moved first rank higher; "
        "precedence is weak evidence of causality, not proof.",
        {
            "onset": {"type": "number", "description": "Epoch seconds, from find_onset."},
            "filters": _FILTERS,
        },
        ["onset"],
    ),
    _tool(
        "conclude",
        "End the investigation. You must state what would have falsified your "
        "leading hypothesis and whether you tested it. 'I don't know' is a valid "
        "and respected answer -- an unsupported hypothesis is worse than none.",
        {
            "summary": {
                "type": "string",
                "description": "One or two sentences an on-call engineer reads first.",
            },
            "hypotheses": {
                "type": "array",
                "description": "Ranked. Empty is valid if the evidence does not support one.",
                "items": {
                    "type": "object",
                    "properties": {
                        "statement": {"type": "string"},
                        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                        "evidence_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Ledger ids returned by the tools you relied on.",
                        },
                    },
                    "required": ["statement", "confidence", "evidence_ids"],
                    "additionalProperties": False,
                },
            },
            "falsification": {
                "type": "string",
                "description": (
                    "What you would expect to see if your leading hypothesis were WRONG, "
                    "and what you found when you checked. Required."
                ),
            },
            "ruled_out": {"type": "array", "items": {"type": "string"}},
            "not_checked": {
                "type": "array",
                "items": {"type": "string"},
                "description": "What you could not examine. Often more useful than the conclusion.",
            },
        },
        ["summary", "hypotheses", "falsification"],
    ),
]

TOOL_NAMES = tuple(t["name"] for t in TOOL_SCHEMAS)
SCHEMA_BY_NAME = {t["name"]: t["parameters"] for t in TOOL_SCHEMAS}

_TYPES: dict[str, tuple[type, ...]] = {
    "object": (dict,),
    "array": (list,),
    "string": (str,),
    "number": (int, float),
    "integer": (int,),
    "boolean": (bool,),
}


def validate(schema: dict[str, Any], value: Any, path: str = "arguments") -> None:
    """Minimal JSON-Schema subset check. Raises ToolError so the failure travels
    back to the model in the same shape as any other tool error, letting it
    self-correct instead of ending the investigation."""
    expected = schema.get("type")
    if expected:
        types = _TYPES[expected]
        # bool is an int subclass in Python; a boolean where a number belongs is
        # a real mistake and should not silently pass.
        if isinstance(value, bool) and expected in ("number", "integer"):
            raise ToolError("bad_arguments", f"{path}: expected {expected}, got boolean.")
        if not isinstance(value, types):
            raise ToolError(
                "bad_arguments", f"{path}: expected {expected}, got {type(value).__name__}."
            )

    if "enum" in schema and value not in schema["enum"]:
        raise ToolError(
            "bad_arguments",
            f"{path}: {value!r} is not one of {schema['enum']}.",
            list(schema["enum"]),
        )

    if expected == "object":
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                raise ToolError("bad_arguments", f"{path}: missing required field {key!r}.", list(props))
        extra = schema.get("additionalProperties", True)
        for key, sub in value.items():
            if key in props:
                validate(props[key], sub, f"{path}.{key}")
            elif isinstance(extra, dict):
                validate(extra, sub, f"{path}.{key}")
            elif extra is False:
                raise ToolError("bad_arguments", f"{path}: unexpected field {key!r}.", list(props))

    if expected == "array":
        item_schema = schema.get("items")
        if item_schema:
            for i, item in enumerate(value):
                validate(item_schema, item, f"{path}[{i}]")
