"""The closed set of reason codes.

Reason codes are the product surface: a recruiter reads them, and so does a
support engineer debugging a complaint. That only works if they are a *closed*
vocabulary -- otherwise every new branch invents a new string and the UI ends up
rendering forty spellings of the same idea.

Every emitted code must match exactly one template here. ``tests/test_reason_codes.py``
asserts that over the whole of ``out/``.
"""

from __future__ import annotations

import re

from saral.contracts.versions import REASON_VOCAB_VERSION

#: (template name, compiled matcher, human-readable gloss)
_TEMPLATES: list[tuple[str, str, str]] = [
    # --- signal-level (emitted by extraction) ----------------------------
    ("headline_declares_transition", r"^headline_declares_transition$",
     "the headline says the person is switching field"),
    ("headline_seniority_inflated", r"^headline_seniority_inflated$",
     "headline claims a level above the one the work history supports"),
    ("headline_family_mismatch", r"^headline_family_mismatch:[a-z_]+!=[a-z_]+$",
     "headline implies one role family, the work history says another"),
    ("skills_not_evidenced_in_experience", r"^skills_not_evidenced_in_experience$",
     "declared skills do not appear anywhere in the work history"),
    ("zero_role_experience_despite_skill_claim",
     r"^zero_role_experience_despite_skill_claim$",
     "claims skills for a family in which they have zero months"),
    ("off_domain_skills", r"^off_domain_skills:\d+$",
     "count of declared skills that map to no engineering concept at all"),
    ("open_to_work", r"^open_to_work:(true|false)$", "the platform flag, verbatim"),
    ("switch_intent", r"^switch_intent:(low|medium|high)$", "banded switch intent"),
    ("tenure_flag", r"^tenure_flag:(hopper|moderate|stable)$", "tenure stability band"),
    ("career_break", r"^career_break$", "an explicit break / returning-to-work signal"),
    ("low_confidence_extraction", r"^low_confidence_extraction$",
     "extraction confidence below 0.5; treat downstream numbers with care"),
    ("fallback_classifier_used", r"^fallback_classifier_used:[a-z_0-9]+$",
     "the lexicon abstained and a fallback classifier decided the family"),
    ("no_experience_descriptions", r"^no_experience_descriptions$",
     "not a single experience entry carried a description"),
    ("date_parse_failure", r"^date_parse_failure$", "at least one date could not be parsed"),
    ("duration_mismatch", r"^duration_mismatch$",
     "stated duration_months disagrees with the computed span by more than 2 months"),

    # --- ranking-level (emitted by scoring) ------------------------------
    ("role_match", r"^role_match:[a-z_]+$", "candidate family matches the job family"),
    ("role_adjacent", r"^role_adjacent:[a-z_]+~[a-z_]+=[0-9.]+$",
     "candidate family is adjacent, not identical, to the job family"),
    ("role_mismatch", r"^role_mismatch:[a-z_]+!=[a-z_]+$", "no adjacency at all"),
    ("years_in_band", r"^years_in_band:[0-9.]+ in \[[0-9.]+,[0-9.]+\]$",
     "relevant years sit inside the job's band"),
    ("years_below_min", r"^years_below_min:[0-9.]+<[0-9.]+$", "under the job's minimum"),
    ("years_above_max", r"^years_above_max:[0-9.]+>[0-9.]+$", "over the job's maximum"),
    ("must_have_met", r"^must_have_met:[^|]+(\|evidence=.+)?$",
     "a must-have is satisfied, optionally with the evidence snippet"),
    ("must_have_claimed_only", r"^must_have_claimed_only:[^|]+$",
     "must-have satisfied only by a declared skill with no corroboration in the work history"),
    ("must_have_missing", r"^must_have_missing:[^|]+$", "a must-have is not satisfied"),
    ("good_to_have_met", r"^good_to_have_met:[^|]+$", "a good-to-have is satisfied"),
    ("evidence_of_scale", r"^evidence_of_scale:.+$",
     "a production-scale phrase found in an experience description"),
    ("no_production_ownership", r"^no_production_ownership$",
     "the job asks for production ownership and nothing in the history evidences it"),
    ("location_exact", r"^location_exact:[a-z .]+$", "same city as the job"),
    ("location_remote_ok", r"^location_remote_ok$", "the job is remote"),
    ("location_mismatch", r"^location_mismatch:[a-z .]+vs[a-z .]+$", "different city"),
    ("score_capped_role_mismatch", r"^score_capped_role_mismatch:\d+$",
     "role_match scored zero, so the total was capped"),
]

_COMPILED: list[tuple[str, re.Pattern[str]]] = [
    (name, re.compile(pattern)) for name, pattern, _ in _TEMPLATES
]

VOCAB_GLOSSARY: dict[str, str] = {name: gloss for name, _, gloss in _TEMPLATES}

__all__ = ["validate", "template_of", "VOCAB_GLOSSARY", "REASON_VOCAB_VERSION"]


def template_of(code: str) -> str | None:
    """Return the name of the template ``code`` matches, or ``None``."""
    for name, matcher in _COMPILED:
        if matcher.match(code):
            return name
    return None


def validate(codes: list[str]) -> list[str]:
    """Return the subset of ``codes`` that match no template."""
    return [c for c in codes if template_of(c) is None]
