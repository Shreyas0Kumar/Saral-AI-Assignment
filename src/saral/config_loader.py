"""Load and validate the YAML configuration.

Config lives in YAML rather than Python constants for two reasons: a lexicon
change shows up as a reviewable data diff, and every file's sha256 goes into the
run manifest so a metrics number can be traced to the config that produced it.

The cost of YAML is that it is untyped, so everything here is validated into
frozen dataclasses at load time. A malformed lexicon fails immediately instead
of silently misclassifying half the corpus.

This module is impure (it reads files) and therefore lives outside ``core/``.
``core/`` receives the loaded objects as parameters.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from saral.contracts.taxonomy import Adjacency, RoleFamily
from saral.core.hashing import file_sha256
from saral.core.normalize import norm_skill, norm_text

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


@dataclass(frozen=True)
class LexiconConfig:
    version: str
    #: normalised pattern -> family
    patterns: dict[str, RoleFamily]
    ambiguous: frozenset[str]
    context: dict[RoleFamily, dict[str, float]]
    context_min_score: float
    context_min_margin: float
    manager_titles: tuple[str, ...]
    staff_titles: tuple[str, ...]


@dataclass(frozen=True)
class SkillConfig:
    version: str
    #: normalised variant -> canonical
    aliases: dict[str, str]
    off_domain: frozenset[str]
    #: canonical skill -> the family a *declared* skill points at
    skill_families: dict[str, RoleFamily] = field(default_factory=dict)
    claimed_family_min_skills: int = 2


@dataclass(frozen=True)
class WeightConfig:
    version: str
    components: dict[str, float]
    must_have_points: float
    good_to_have_points: float
    claimed_only_credit: float
    seniority: dict[str, float]
    location: dict[str, float]
    metros: dict[str, frozenset[str]]
    tenure: dict[str, float]
    shipping_points: float
    shipping_keywords: tuple[str, ...]
    missing_must_have_penalty: float
    role_mismatch_cap: float


@dataclass(frozen=True)
class ExtractConfig:
    """Everything ``core.extract`` needs, with no filesystem access."""

    lexicon: LexiconConfig
    skills: SkillConfig
    adjacency: Adjacency
    #: half-life in months for recency weighting of experience entries
    recency_half_life_months: float = 36.0
    #: an alt family must score at least this fraction of the primary
    alt_family_ratio: float = 0.35
    max_alt_families: int = 2
    config_hashes: dict[str, str] = field(default_factory=dict)


def _read(path: Path) -> tuple[dict[str, Any], str]:
    data = path.read_bytes()
    return yaml.safe_load(data.decode("utf-8")), file_sha256(data)


def _family(name: str) -> RoleFamily:
    try:
        return RoleFamily(name)
    except ValueError as exc:  # pragma: no cover - config error path
        raise ValueError(f"unknown role_family in config: {name!r}") from exc


def load_lexicon(path: Path) -> tuple[LexiconConfig, str]:
    raw, digest = _read(path)
    patterns: dict[str, RoleFamily] = {}
    for family_name, entries in (raw.get("patterns") or {}).items():
        family = _family(family_name)
        for entry in entries or []:
            key = norm_text(entry)
            if not key:
                continue
            if key in patterns and patterns[key] != family:
                raise ValueError(
                    f"lexicon pattern {key!r} maps to both "
                    f"{patterns[key].value} and {family.value}"
                )
            patterns[key] = family

    ambiguous = frozenset(norm_text(a) for a in (raw.get("ambiguous") or []))
    overlap = ambiguous & patterns.keys()
    if overlap:
        raise ValueError(f"patterns and ambiguous overlap: {sorted(overlap)}")

    context: dict[RoleFamily, dict[str, float]] = {}
    for family_name, terms in (raw.get("context") or {}).items():
        context[_family(family_name)] = {
            norm_text(term): float(weight) for term, weight in (terms or {}).items()
        }

    overrides = raw.get("seniority_overrides") or {}
    cfg = LexiconConfig(
        version=str(raw.get("version", "0")),
        patterns=patterns,
        ambiguous=ambiguous,
        context=context,
        context_min_score=float(raw.get("context_min_score", 1.0)),
        context_min_margin=float(raw.get("context_min_margin", 0.5)),
        manager_titles=tuple(norm_text(t) for t in overrides.get("manager", [])),
        staff_titles=tuple(norm_text(t) for t in overrides.get("staff_plus", [])),
    )
    return cfg, digest


def load_skills(path: Path) -> tuple[SkillConfig, str]:
    raw, digest = _read(path)
    aliases: dict[str, str] = {}
    for canonical, variants in (raw.get("aliases") or {}).items():
        canon = norm_text(canonical)
        for variant in variants or []:
            key = norm_text(variant)
            if key and key != canon:
                aliases[key] = canon
    off_domain = frozenset(
        norm_skill(s, aliases) for s in (raw.get("off_domain") or []) if norm_skill(s, aliases)
    )
    skill_families: dict[str, RoleFamily] = {}
    for family_name, skills in (raw.get("skill_families") or {}).items():
        family = _family(family_name)
        for skill in skills or []:
            key = norm_skill(skill, aliases)
            # First declaration wins, so a skill listed under two families is a
            # config bug that shows up as a diff rather than silently flipping.
            if key:
                skill_families.setdefault(key, family)
    return SkillConfig(
        version=str(raw.get("version", "0")),
        aliases=aliases,
        off_domain=off_domain,
        skill_families=skill_families,
        claimed_family_min_skills=int(raw.get("claimed_family_min_skills", 2)),
    ), digest


def load_adjacency(path: Path) -> tuple[Adjacency, str]:
    raw, digest = _read(path)
    for key in (raw.get("pairs") or {}):
        a, _, b = key.partition("|")
        _family(a.strip())
        _family(b.strip())
    return (
        Adjacency(
            identity=float(raw.get("identity", 1.0)),
            pairs=raw.get("pairs") or {},
            em_prior_ic=float(raw.get("em_prior_ic", 0.5)),
            default=float(raw.get("default", 0.0)),
        ),
        digest,
    )


def load_weights(path: Path) -> tuple[WeightConfig, str]:
    raw, digest = _read(path)
    components = {k: float(v) for k, v in (raw.get("components") or {}).items()}
    total = sum(components.values())
    if abs(total - 100.0) > 1e-6:
        raise ValueError(f"weights must sum to 100, got {total}")

    skill = raw.get("skill_overlap") or {}
    if abs(
        float(skill.get("must_have_points", 0)) + float(skill.get("good_to_have_points", 0))
        - components.get("skill_overlap", 0)
    ) > 1e-6:
        raise ValueError("skill_overlap sub-weights must sum to the skill_overlap component")

    loc = raw.get("location_fit") or {}
    metros = {
        name: frozenset(norm_text(c) for c in cities)
        for name, cities in (loc.get("metros") or {}).items()
    }
    ship = raw.get("evidence_of_shipping") or {}
    pen = raw.get("penalties") or {}
    return (
        WeightConfig(
            version=str(raw.get("version", "0")),
            components=components,
            must_have_points=float(skill.get("must_have_points", 0)),
            good_to_have_points=float(skill.get("good_to_have_points", 0)),
            claimed_only_credit=float(skill.get("claimed_only_credit", 0.3)),
            seniority={k: float(v) for k, v in (raw.get("seniority_fit") or {}).items()},
            location={
                k: float(v) for k, v in loc.items() if k != "metros"
            },
            metros=metros,
            tenure={k: float(v) for k, v in (raw.get("tenure_stability") or {}).items()},
            shipping_points=float(ship.get("points", 0)),
            shipping_keywords=tuple(norm_text(k) for k in (ship.get("keywords") or [])),
            missing_must_have_penalty=float(pen.get("missing_must_have", 0)),
            role_mismatch_cap=float(pen.get("role_mismatch_cap", 100)),
        ),
        digest,
    )


@functools.lru_cache(maxsize=4)
def load_all(config_dir: str | None = None) -> tuple[ExtractConfig, WeightConfig]:
    """Load every config file. Cached: the hot path must not re-parse YAML."""
    base = Path(config_dir) if config_dir else CONFIG_DIR
    lexicon, h_lex = load_lexicon(base / "lexicon.yaml")
    skills, h_skill = load_skills(base / "skill_aliases.yaml")
    adjacency, h_adj = load_adjacency(base / "adjacency.yaml")
    weights, h_w = load_weights(base / "weights.yaml")
    extract = ExtractConfig(
        lexicon=lexicon,
        skills=skills,
        adjacency=adjacency,
        config_hashes={
            "lexicon.yaml": h_lex,
            "skill_aliases.yaml": h_skill,
            "adjacency.yaml": h_adj,
            "weights.yaml": h_w,
        },
    )
    return extract, weights
