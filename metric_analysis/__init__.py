"""Deterministic metrics tools for an automated incident debugging agent.

There is no LLM in this package. An orchestrator plans freely and decides what
to call; everything here is validated, budgeted, deterministic, and returns
compact summaries. See CLAUDE.md for the architecture rule and the two
invariants that constrain anything added here.
"""

from __future__ import annotations

from .attribution import (
    SliceContribution,
    attribute_additive,
    attribute_ratio,
    explain_delta,
)
from .brief import build_brief, render_brief
from .catalog import DEMO_FLEET, GOLDEN_SIGNALS, MetricCatalog, MetricSpec, demo_catalog
from .config import CatalogError, catalog_from_dict, catalog_to_dict, load_catalog
from .changepoint import ChangePoint, classify_shape, detect_change_point
from .summarize import peer_outliers, summarize_series
from .tools import Budget, Evidence, MetricTools
from .scenarios import SCENARIOS, bad_rollout, mix_shift, overlapping_faults
from .tsdb import Fault, SyntheticTSDB, TrafficShift, TSDBClient, stable_hash
from .types import QueryResult, ResultStatus, Series, ToolError, Window

__all__ = [
    "DEMO_FLEET",
    "GOLDEN_SIGNALS",
    "SCENARIOS",
    "Fault",
    "TrafficShift",
    "Budget",
    "CatalogError",
    "ChangePoint",
    "Evidence",
    "MetricCatalog",
    "MetricSpec",
    "MetricTools",
    "QueryResult",
    "ResultStatus",
    "Series",
    "SliceContribution",
    "SyntheticTSDB",
    "TSDBClient",
    "ToolError",
    "Window",
    "attribute_additive",
    "bad_rollout",
    "mix_shift",
    "overlapping_faults",
    "attribute_ratio",
    "build_brief",
    "catalog_from_dict",
    "catalog_to_dict",
    "load_catalog",
    "classify_shape",
    "demo_catalog",
    "detect_change_point",
    "explain_delta",
    "peer_outliers",
    "render_brief",
    "stable_hash",
    "summarize_series",
]
