"""Skills: incident-class priors contributed after postmortems.

The contribution surface. After an incident an SRE writes down what the failure
*looked like* -- not what to do about it -- and the agent retrieves that note as
a prior when a new incident presents the same shape.

Two properties make this cheap to maintain, and both are load-bearing:

**Skills are declarative, never procedural.** A skill says "elections spike while
QPS stays flat, and the tell versus a partition is heartbeat p50 staying normal";
it does not say "first check elections, then check heartbeat". The moment a skill
encodes a sequence it becomes a playbook, and playbooks are what this whole
architecture exists to avoid: they serve one incident class each, so maintenance
scales with the number of failure modes. The schema has no field for steps, and
`_reject_procedure` refuses text that smuggles them in anyway.

**A missing skill degrades, it does not break.** Retrieval returning nothing is
an ordinary outcome -- the agent investigates exactly as it would have. Nobody
maintains an execution graph, so nobody can leave a hole in one.

Retrieval is deterministic and runs in code, with no model involved. Which
priors surfaced is then part of the replay record, so a change in the skill
library is visible in eval results rather than being an invisible confound.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"

# Weights are ordered by how much each signal narrows the space, not tuned. A
# skill written for this exact symptom metric is far stronger evidence than one
# that merely mentions a metric which also moved.
_W_SYMPTOM = 3
_W_SIGNAL = 2
_W_SHAPE = 2
_W_CORRELATED = 1
_W_DIMENSION = 1
MIN_SCORE = 3
MAX_RESULTS = 3

# Phrasings that mean a sequence is being prescribed. Kept narrow on purpose:
# ordinary declarative prose uses "then" and "next" freely, and a lint that
# fires on those would push contributors into fighting the validator instead of
# writing clearly.
_PROCEDURE_PATTERNS = [
    re.compile(r"\bstep\s+\d", re.I),
    re.compile(r"^\s*\d+[.)]\s+", re.M),
    re.compile(r"\b(?:then|next|after that|finally)[,:]?\s+(?:run|call|query|check|look|fetch)\b", re.I),
    re.compile(r"\bfirst[,:]?\s+(?:run|call|query|fetch)\b", re.I),
]


class SkillError(Exception):
    pass


@dataclass
class Skill:
    id: str
    title: str
    incident_class: str
    version: int = 1
    match: dict[str, Any] = field(default_factory=dict)
    tell: list[str] = field(default_factory=list)
    confusable_with: list[dict[str, str]] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)

    def stamp(self) -> str:
        """Identity for the replay record: which revision of which note ran."""
        return f"{self.id}@v{self.version}"

    def to_prior(self, score: int, reasons: list[str]) -> dict[str, Any]:
        return {
            "skill": self.stamp(),
            "title": self.title,
            "incident_class": self.incident_class,
            "matched_because": reasons,
            "score": score,
            "tell": self.tell,
            "confusable_with": self.confusable_with,
            "provenance": self.provenance,
        }


def _reject_procedure(skill_id: str, texts: list[str]) -> None:
    for text in texts:
        for pattern in _PROCEDURE_PATTERNS:
            if pattern.search(text):
                raise SkillError(
                    f"skill {skill_id!r} reads as a procedure: {text[:80]!r}. Skills describe "
                    f"what an incident class looks like, not what to do about it -- a skill "
                    f"encoding steps is a playbook, and its maintenance cost scales with the "
                    f"number of failure modes rather than data sources."
                )


def load_skill(data: dict[str, Any]) -> Skill:
    for required in ("id", "title", "incident_class"):
        if not data.get(required):
            raise SkillError(f"skill is missing required field {required!r}")
    unknown = set(data) - {
        "id", "title", "incident_class", "version", "match", "tell",
        "confusable_with", "provenance",
    }
    if unknown:
        # Notably there is no "steps" or "procedure" key to add; rejecting
        # unknown keys is what keeps one from being introduced quietly.
        raise SkillError(f"skill {data['id']!r} has unknown fields: {sorted(unknown)}")

    skill = Skill(
        id=data["id"],
        title=data["title"],
        incident_class=data["incident_class"],
        version=int(data.get("version", 1)),
        match=data.get("match") or {},
        tell=list(data.get("tell") or []),
        confusable_with=list(data.get("confusable_with") or []),
        provenance=data.get("provenance") or {},
    )
    _reject_procedure(
        skill.id, skill.tell + [c.get("discriminator", "") for c in skill.confusable_with]
    )
    return skill


def load_skills(directory: str | Path | None = None) -> list[Skill]:
    """Load every skill in a directory, sorted by id for determinism.

    A missing directory is not an error: running with no skills at all is a
    supported configuration, and it is also the control arm when measuring
    whether skills help.
    """
    path = Path(directory or DEFAULT_SKILLS_DIR)
    if not path.is_dir():
        return []
    skills = []
    for file in sorted(path.glob("*.json")):
        try:
            skills.append(load_skill(json.loads(file.read_text())))
        except json.JSONDecodeError as e:
            raise SkillError(f"{file.name} is not valid JSON: {e}") from e
    return skills


def _brief_features(brief: dict[str, Any]) -> dict[str, Any]:
    """Reduce a brief to the handful of facts skills match against."""
    material = {
        s["signal"] for s in brief.get("golden_signals") or [] if s.get("material")
    }
    correlated = {c["metric"] for c in brief.get("correlated_changes") or []}
    location = brief.get("location") or {}
    dimensions = set(location.get("narrowed_to") or {}) | set(location.get("spans") or {})
    shapes = {
        s.get("shape")
        for s in brief.get("golden_signals") or []
        if s.get("is_symptom") and s.get("shape")
    }
    return {
        "symptom_metric": brief.get("symptom_metric"),
        "material_signals": material,
        "correlated_metrics": correlated,
        "dimensions": dimensions,
        "shapes": shapes,
    }


def retrieve(
    brief: dict[str, Any],
    skills: list[Skill] | None = None,
    min_score: int = MIN_SCORE,
    limit: int = MAX_RESULTS,
) -> list[dict[str, Any]]:
    """Rank skills against a brief. Deterministic; no model involved.

    Scoring is additive over independent observations rather than a similarity
    metric, so a retrieved prior can always say *why* it matched -- an
    unexplained prior is one the agent cannot sensibly discount.
    """
    skills = load_skills() if skills is None else skills
    features = _brief_features(brief)
    scored: list[tuple[int, str, dict[str, Any]]] = []

    for skill in skills:
        m, score, reasons = skill.match, 0, []
        if features["symptom_metric"] in (m.get("symptom_metrics") or []):
            score += _W_SYMPTOM
            reasons.append(f"symptom metric is {features['symptom_metric']}")
        for signal in sorted(features["material_signals"] & set(m.get("material_signals") or [])):
            score += _W_SIGNAL
            reasons.append(f"{signal} is materially abnormal")
        for shape in sorted(features["shapes"] & set(m.get("shapes") or [])):
            score += _W_SHAPE
            reasons.append(f"symptom onset shape is {shape}")
        for metric in sorted(features["correlated_metrics"] & set(m.get("correlated_metrics") or [])):
            score += _W_CORRELATED
            reasons.append(f"{metric} moved near onset")
        for dim in sorted(features["dimensions"] & set(m.get("narrowed_dimensions") or [])):
            score += _W_DIMENSION
            reasons.append(f"change concentrates on {dim}")
        if score >= min_score:
            scored.append((score, skill.id, skill.to_prior(score, reasons)))

    # Sort by score, then id -- ties must not depend on filesystem ordering.
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [prior for _, _, prior in scored[:limit]]
