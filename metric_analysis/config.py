"""Load the metric catalog from configuration files.

`demo_catalog()` is a fixture written in Python because tests need it to be
byte-identical and file-free. A real fleet's catalog is neither: it changes when
metrics do, it is owned by the people who own the metrics, and it should be
generated nightly from the metric registry rather than edited by whoever is
touching the analysis code that week.

So: JSON on disk, one file per team or one file for everything, loaded and
validated here. Adding a metric is a data change with no code change and no
deploy of this package.

The validation is deliberately strict and its errors name the file and metric.
A catalog with a typo'd `denominator` produces an investigation that silently
falls back to additive attribution on a rate metric -- correct-looking output,
wrong arithmetic. Failing at load is much cheaper than discovering that during
a P0.

Two fields deserve more care than they usually get:

  `description` / `interpretation`  the agent is close to useless without these.
      A metric name alone tells a model almost nothing and it will invent the
      rest. Auto-draft them from the registry, then have owners review.
  `fields`      a list of values means "these are all of them, reject anything
      else". `null` means "resolved live" -- correct for anything
      high-cardinality like task or alloc, where a nightly snapshot is stale
      within minutes and a stale enumeration rejects filters that are valid.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .catalog import GOLDEN_SIGNALS, MetricCatalog, MetricSpec

_KINDS = ("counter", "gauge", "distribution")

_METRIC_KEYS = {
    "name", "kind", "unit", "description", "interpretation", "fields",
    "field_cardinality", "denominator", "percentile", "golden_signal",
}
_TOP_KEYS = {"metrics", "golden_signals", "version"}


class CatalogError(Exception):
    """Raised at load time, never during an investigation."""


def _fail(where: str, message: str) -> None:
    raise CatalogError(f"{where}: {message}")


def _spec_from_dict(data: dict[str, Any], where: str) -> MetricSpec:
    if not isinstance(data, dict):
        _fail(where, f"expected an object, got {type(data).__name__}")
    unknown = set(data) - _METRIC_KEYS
    if unknown:
        _fail(where, f"unknown keys {sorted(unknown)}; known keys are {sorted(_METRIC_KEYS)}")

    name = data.get("name")
    if not name or not isinstance(name, str):
        _fail(where, "missing required string field 'name'")
    where = f"{where} [{name}]"

    kind = data.get("kind")
    if kind not in _KINDS:
        _fail(where, f"kind must be one of {list(_KINDS)}, got {kind!r}")

    for required in ("unit", "description"):
        if not data.get(required):
            _fail(where, f"missing required field {required!r}")
    if not data.get("interpretation"):
        # Not fatal, but worth saying out loud: this is the field that decides
        # whether the agent can reason about a move at all.
        pass

    raw_fields = data.get("fields") or {}
    if not isinstance(raw_fields, dict):
        _fail(where, "'fields' must be an object mapping tag name -> values or null")
    fields: dict[str, list[str] | None] = {}
    for tag, values in raw_fields.items():
        if values is None:
            fields[tag] = None
        elif isinstance(values, list) and all(isinstance(v, str) for v in values):
            if not values:
                _fail(where, f"field {tag!r} has an empty value list; use null for live resolution")
            fields[tag] = list(values)
        else:
            _fail(where, f"field {tag!r} must be a list of strings or null, got {type(values).__name__}")

    cardinality = data.get("field_cardinality") or {}
    if not isinstance(cardinality, dict) or not all(
        isinstance(v, int) and v > 0 for v in cardinality.values()
    ):
        _fail(where, "'field_cardinality' must map tag name -> positive integer")
    unknown_hints = set(cardinality) - set(fields)
    if unknown_hints:
        _fail(where, f"field_cardinality names tags that are not fields: {sorted(unknown_hints)}")

    signal = data.get("golden_signal")
    if signal is not None and not isinstance(signal, str):
        _fail(where, f"golden_signal must be a string or null, got {type(signal).__name__}")

    return MetricSpec(
        name=name,
        kind=kind,
        unit=data["unit"],
        description=data["description"],
        interpretation=data.get("interpretation", ""),
        fields=fields,
        field_cardinality={k: int(v) for k, v in cardinality.items()},
        denominator=data.get("denominator"),
        percentile=data.get("percentile"),
        golden_signal=signal,
    )


def catalog_from_dict(data: dict[str, Any], where: str = "<catalog>") -> MetricCatalog:
    unknown = set(data) - _TOP_KEYS
    if unknown:
        _fail(where, f"unknown top-level keys {sorted(unknown)}")

    metrics = data.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        _fail(where, "'metrics' must be a non-empty list")

    specs = [_spec_from_dict(m, where) for m in metrics]
    names = [s.name for s in specs]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        _fail(where, f"duplicate metric names: {dupes}")

    known = set(names)
    for spec in specs:
        # A typo'd denominator silently degrades ratio attribution to additive
        # on a rate metric: plausible output, wrong arithmetic. Catch it here.
        if spec.denominator and spec.denominator not in known:
            _fail(
                f"{where} [{spec.name}]",
                f"denominator {spec.denominator!r} is not a metric in this catalog",
            )
        if spec.denominator == spec.name:
            _fail(f"{where} [{spec.name}]", "denominator cannot be the metric itself")

    order = data.get("golden_signals") or list(GOLDEN_SIGNALS)
    if not isinstance(order, list) or not all(isinstance(s, str) for s in order):
        _fail(where, "'golden_signals' must be a list of strings")

    declared = {s.golden_signal for s in specs if s.golden_signal}
    unlisted = sorted(declared - set(order))
    if unlisted:
        # Not fatal -- MetricCatalog.golden_signals() appends them last rather
        # than dropping them -- but almost always a mistake worth surfacing.
        _fail(
            where,
            f"metrics declare golden signals absent from 'golden_signals': {unlisted}. "
            f"Add them to the list to fix their position in the sweep.",
        )

    return MetricCatalog(specs, signal_order=tuple(order))


def load_catalog(path: str | Path) -> MetricCatalog:
    """Load from a JSON file, or merge every ``*.json`` in a directory.

    A directory is the shape that survives contact with an organisation: one
    file per team, each owned by the people who own those metrics, merged at
    load. Files are read in sorted order so the result is deterministic.
    """
    p = Path(path)
    if p.is_dir():
        files = sorted(p.glob("*.json"))
        if not files:
            raise CatalogError(f"{p}: directory contains no .json catalog files")
        merged: dict[str, Any] = {"metrics": []}
        for file in files:
            data = _read(file)
            unknown = set(data) - _TOP_KEYS
            if unknown:
                _fail(str(file), f"unknown top-level keys {sorted(unknown)}")
            merged["metrics"].extend(data.get("metrics") or [])
            if "golden_signals" in data:
                if "golden_signals" in merged and merged["golden_signals"] != data["golden_signals"]:
                    _fail(
                        str(file),
                        "conflicting 'golden_signals' across files. Declare the order in exactly "
                        "one file so the sweep order is unambiguous.",
                    )
                merged["golden_signals"] = data["golden_signals"]
        return catalog_from_dict(merged, where=str(p))
    return catalog_from_dict(_read(p), where=str(p))


def _read(file: Path) -> dict[str, Any]:
    try:
        data = json.loads(file.read_text())
    except FileNotFoundError as e:
        raise CatalogError(f"{file}: no such catalog file") from e
    except json.JSONDecodeError as e:
        raise CatalogError(f"{file}: invalid JSON at line {e.lineno}: {e.msg}") from e
    if not isinstance(data, dict):
        raise CatalogError(f"{file}: top level must be an object")
    return data


def catalog_to_dict(catalog: MetricCatalog) -> dict[str, Any]:
    """Serialise a catalog back to config form.

    Mainly so `demo_catalog()` can be dumped as a starting point to edit, rather
    than making someone write the first file from scratch against a schema they
    have not seen.
    """
    metrics = []
    for name in sorted(catalog.names()):
        spec = catalog.get(name)
        entry: dict[str, Any] = {
            "name": spec.name,
            "kind": spec.kind,
            "unit": spec.unit,
            "description": spec.description,
            "interpretation": spec.interpretation,
            "fields": {k: (list(v) if v is not None else None) for k, v in spec.fields.items()},
        }
        if spec.field_cardinality:
            entry["field_cardinality"] = dict(spec.field_cardinality)
        for optional in ("denominator", "percentile", "golden_signal"):
            if getattr(spec, optional):
                entry[optional] = getattr(spec, optional)
        metrics.append(entry)
    return {"version": 1, "golden_signals": list(catalog.signal_order), "metrics": metrics}


def describe_catalog(catalog: MetricCatalog) -> str:
    """One-line-per-metric summary, for checking a config after editing it."""
    lines = [f"{len(catalog.names())} metrics, signals: {', '.join(catalog.signal_order)}"]
    signals = catalog.golden_signals()
    by_metric = {m: s for s, ms in signals.items() for m in ms}
    for name in sorted(catalog.names()):
        spec = catalog.get(name)
        role = f"  [{by_metric[name]}]" if name in by_metric else ""
        dyn = [k for k, v in spec.fields.items() if v is None]
        tags = ", ".join(
            f"{k}({spec.cardinality_of(k)}{'*' if v is None else ''})"
            for k, v in spec.fields.items()
        )
        den = f" / {spec.denominator}" if spec.denominator else ""
        lines.append(f"  {name}{den}{role}")
        lines.append(f"      {spec.kind}, {spec.unit} · tags: {tags}")
        if dyn:
            lines.append(f"      * resolved live: {', '.join(dyn)}")
    return "\n".join(lines)
