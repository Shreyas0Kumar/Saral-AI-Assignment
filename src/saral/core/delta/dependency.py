"""Which signals a field group can affect.

This map is the justification the brief asks for when it says "be able to
justify what 'changed' means". A recompute is triggered only when a field with a
non-empty dependency set actually moved -- and "moved" is decided by the
*normalised* hash, which is why the rocket-emoji case resolves to noise without
a special case anywhere.

Honest scope note: the map decides **whether** to recompute and reports **which**
signals were affected. The implementation then recomputes the whole
`SignalRecord` for a dirty candidate, because extraction is sub-millisecond and
partial recomputation would add dependency-ordered evaluation and stale-subfield
tracking to save microseconds. The saving that matters is candidate-level --
recomputing 6 of 25 rather than 25 of 25. Claiming partial recompute I did not
build would be worse than reporting this plainly. See DECISIONS.md D12.
"""

from __future__ import annotations

DEPENDS: dict[str, frozenset[str]] = {
    "headline": frozenset(
        {"role_family", "seniority", "switch_intent", "confidence", "reason_codes"}
    ),
    "about": frozenset({"switch_intent", "confidence", "reason_codes"}),
    "skills": frozenset(
        {
            "core_skills",
            "claimed_skills_unverified",
            "skill_noise_ratio",
            "confidence",
            "reason_codes",
        }
    ),
    "experience": frozenset(
        {
            "role_family",
            "role_family_alt",
            "seniority",
            "years_total",
            "years_relevant",
            "tenure_stability",
            "core_skills",
            "claimed_skills_unverified",
            "skill_noise_ratio",
            "switch_intent",
            "confidence",
            "reason_codes",
        }
    ),
    "education": frozenset({"core_skills", "claimed_skills_unverified", "confidence"}),
    # Location is used at scoring time only. It never enters a SignalRecord, so
    # a move from Bengaluru to Berlin changes every ranking and no signal.
    "location": frozenset(),
    "is_open_to_work": frozenset({"switch_intent", "reason_codes"}),
}


def affected_signals(field_groups: set[str]) -> set[str]:
    out: set[str] = set()
    for group in field_groups:
        out |= DEPENDS.get(group, frozenset())
    return out


def requires_recompute(field_groups: set[str]) -> bool:
    return bool(affected_signals(field_groups))


def requires_rescore(field_groups: set[str]) -> bool:
    """Scoring depends on the signals plus location."""
    return requires_recompute(field_groups) or "location" in field_groups
