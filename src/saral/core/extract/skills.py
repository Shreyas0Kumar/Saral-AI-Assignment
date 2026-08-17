"""Evidence tiering: the rule that a cosine baseline structurally cannot express.

A declared skill is **evidenced** if it appears in an experience entry's
``skills_used``, as a token in an entry's ``description``, or in an education
record's ``skills``. Otherwise it is *claimed and unverified*.

    skill_noise_ratio = |claimed_unverified| / |declared|

An embedding of ``text_context_full`` sees "Machine Learning" in a skills list
and moves the vector; it has no notion of whether the work history corroborates
it. That single distinction is what fixes SDB_10019 (mechanical engineer
listing ML), SDB_10023 (HR listing Excel and Python) and SDB_10010 (three months
of experience, thirty-one skills).

The known false-negative mode is honest and stated: real engineers with terse
one-line descriptions get penalised. Mitigations are that unverified skills
still earn partial must-have credit at scoring time rather than zero, and that
weak evidence lowers ``confidence`` rather than the score directly.

The headline and the ``about`` text are deliberately **not** evidence. They are
the self-description this whole component exists to distrust.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from saral.core.normalize import norm_skill, tokens


@dataclass
class SkillProfile:
    declared: list[str] = field(default_factory=list)
    core_skills: list[str] = field(default_factory=list)
    claimed_unverified: list[str] = field(default_factory=list)
    off_domain: list[str] = field(default_factory=list)
    #: canonical skill -> where it was corroborated (for `evidence=` reason codes)
    evidence: dict[str, str] = field(default_factory=dict)

    @property
    def skill_noise_ratio(self) -> float:
        if not self.declared:
            return 0.0
        return round(len(self.claimed_unverified) / len(self.declared), 3)


def _entry_evidence_terms(entry: dict) -> tuple[set[str], list[str]]:
    """Return ``(normalised skills_used, description tokens)`` for one entry."""
    used = {norm_skill(s) for s in (entry.get("skills_used") or [])}
    used.discard("")
    return used, tokens(entry.get("description"))


def build_skill_profile(
    declared_raw: list[str],
    experience: list[dict],
    education: list[dict],
    aliases: dict[str, str],
    off_domain_set: frozenset[str],
    entry_months: dict[int, int] | None = None,
    entry_age_months: dict[int, int] | None = None,
    half_life_months: float = 36.0,
) -> SkillProfile:
    entry_months = entry_months or {}
    entry_age_months = entry_age_months or {}

    declared: list[str] = []
    seen: set[str] = set()
    for raw in declared_raw:
        canonical = norm_skill(raw, aliases)
        if canonical and canonical not in seen:
            seen.add(canonical)
            declared.append(canonical)

    # Build the evidence index once.
    evidence_source: dict[str, str] = {}
    recency_weight: dict[str, float] = {}

    for index, entry in enumerate(experience):
        used_raw, desc_tokens = _entry_evidence_terms(entry)
        used = {norm_skill(u, aliases) for u in used_raw}
        used.discard("")
        desc_norm = " ".join(desc_tokens)
        months = entry_months.get(index, entry.get("duration_months") or 0)
        age = entry_age_months.get(index, 0)
        weight = float(months) * (0.5 ** (age / half_life_months)) if half_life_months else float(months)

        for skill in declared:
            if skill in used:
                evidence_source.setdefault(skill, f"skills_used@{entry.get('company_name') or '?'}")
                recency_weight[skill] = recency_weight.get(skill, 0.0) + weight
                continue
            # Multi-word skills must appear as a phrase; single tokens as a token.
            if " " in skill or "." in skill or "/" in skill:
                if skill in desc_norm:
                    evidence_source.setdefault(skill, f"description@{entry.get('company_name') or '?'}")
                    recency_weight[skill] = recency_weight.get(skill, 0.0) + weight * 0.8
            elif skill in desc_tokens:
                evidence_source.setdefault(skill, f"description@{entry.get('company_name') or '?'}")
                recency_weight[skill] = recency_weight.get(skill, 0.0) + weight * 0.8

    for record in education:
        edu_skills = {norm_skill(s, aliases) for s in (record.get("skills") or [])}
        for skill in declared:
            if skill and skill in edu_skills:
                evidence_source.setdefault(skill, f"education@{record.get('school_name') or '?'}")
                recency_weight.setdefault(skill, 0.0)

    core = [s for s in declared if s in evidence_source]
    core.sort(key=lambda s: (-recency_weight.get(s, 0.0), s))
    claimed = [s for s in declared if s not in evidence_source]
    off_domain = [s for s in declared if s in off_domain_set]

    return SkillProfile(
        declared=declared,
        core_skills=core,
        claimed_unverified=claimed,
        off_domain=off_domain,
        evidence=evidence_source,
    )
