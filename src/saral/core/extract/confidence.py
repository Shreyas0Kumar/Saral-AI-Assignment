"""How much the extractor trusts its own output for this row.

"Rows the model is unsure about should say so." Confidence is a multiplicative
penalty stack over two families of evidence:

* **agreement** -- does the profile contradict itself? A headline implying a
  different family than the work history, a skills list mostly unevidenced, a
  claimed level the history does not support.
* **completeness** -- was there enough to read? No descriptions anywhere, dates
  that would not parse, stated durations disagreeing with the spans, a single
  short entry.

Multiplicative rather than additive so penalties compound instead of racing to
zero, and clamped to [0.05, 0.99] because neither certainty nor total ignorance
is ever the right claim.
"""

from __future__ import annotations

from dataclasses import dataclass

FLOOR = 0.05
CEILING = 0.99
LOW_CONFIDENCE_THRESHOLD = 0.50

PENALTIES: dict[str, float] = {
    # agreement
    "headline_family_mismatch": 0.15,
    "high_skill_noise": 0.10,
    "headline_seniority_inflated": 0.10,
    # completeness
    "no_experience_descriptions": 0.20,
    "duration_mismatch": 0.15,
    "date_parse_failure": 0.10,
    "fallback_classifier_used": 0.15,
    "thin_history": 0.20,
    "unclassified_entries": 0.15,
}


@dataclass(frozen=True)
class ConfidenceResult:
    value: float
    applied: dict[str, float]

    @property
    def is_low(self) -> bool:
        return self.value < LOW_CONFIDENCE_THRESHOLD


def noise_penalty_scale(skill_noise_ratio: float) -> float:
    """Graded, not a cliff.

    Under a strict evidence rule most real profiles score 0.5-0.8: one-line
    descriptions simply do not mention every declared skill. A hard threshold at
    0.5 therefore fires on nearly every row, and a penalty that fires on
    everything discriminates nothing. Scaling from 0.5 (no penalty) to 1.0 (full
    penalty) keeps the *ordering* -- which is where the information actually is
    -- without punishing the whole corpus for having terse profiles.
    """
    if skill_noise_ratio <= 0.5:
        return 0.0
    return min(1.0, (skill_noise_ratio - 0.5) / 0.5)


def compute_confidence(
    flags: set[str], scales: dict[str, float] | None = None
) -> ConfidenceResult:
    scales = scales or {}
    value = 1.0
    applied: dict[str, float] = {}
    for flag in sorted(flags):
        penalty = PENALTIES.get(flag)
        if penalty is None:
            continue
        penalty *= scales.get(flag, 1.0)
        if penalty <= 0:
            continue
        applied[flag] = round(penalty, 4)
        value *= 1.0 - penalty
    value = max(FLOOR, min(CEILING, value))
    return ConfidenceResult(value=round(value, 3), applied=applied)
