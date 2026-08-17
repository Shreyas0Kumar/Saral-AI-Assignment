"""Aggregate per-entry title classifications into one role family.

Classification happens **per experience entry**, never over the profile blob.
SDB_10019 is the proof: headline says "Transitioning to Data Science", the
skills list says "Machine Learning", and the work history says six years of
AutoCAD. A blob classifier lands him in a data role. Per-entry classification
plus duration weighting lands him where the work actually is.

    score[f] = sum over entries of  months_i * 0.5**(age_i / 36) * conf_i

The 36-month half-life is roughly one job cycle in Indian tech: a role left
three years ago counts half, six years ago a quarter. It is a stated prior, not
a tuned parameter -- tuning it on 55 labels would produce a number fitted to
noise. It is what makes SDB_10020 (Yahoo SWE 46m, Uber Senior ML 45m, Atlassian
EM 55m current) resolve to `engineering_manager` rather than `ml_engineer`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from saral.contracts.taxonomy import RoleFamily
from saral.core.dates import Span, age_months


@dataclass(frozen=True)
class EntryClassification:
    index: int
    family: RoleFamily | None
    confidence: float
    source: str
    evidence: str
    months: int
    age_months: int

    @property
    def weight(self) -> float:
        return float(self.months)


def decay(age: int, half_life_months: float) -> float:
    if half_life_months <= 0:
        return 1.0
    return 0.5 ** (age / half_life_months)


def family_scores(
    entries: list[EntryClassification],
    half_life_months: float = 36.0,
) -> dict[RoleFamily, float]:
    scores: dict[RoleFamily, float] = {}
    for entry in entries:
        if entry.family is None:
            continue
        contribution = entry.weight * decay(entry.age_months, half_life_months) * entry.confidence
        scores[entry.family] = scores.get(entry.family, 0.0) + contribution
    return scores


def resolve_family(
    scores: dict[RoleFamily, float],
    alt_ratio: float = 0.35,
    max_alt: int = 2,
) -> tuple[RoleFamily, list[RoleFamily]]:
    """Return ``(primary, alternates)``.

    Ties break deterministically on the taxonomy's declaration order so that two
    runs over identical input cannot disagree.

    ``non_engineering`` winning the argmax **stays** primary regardless of what
    else scored, and the alternates list may still carry an engineering family.
    That is deliberate: it is the SDB_10019 (mechanical) and SDB_10023 (HR)
    case, and demoting it to make room for a more flattering engineering family
    would reintroduce exactly the failure this system exists to prevent.
    """
    if not scores:
        return RoleFamily.NON_ENGINEERING, []

    order = list(RoleFamily)
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], order.index(kv[0])))
    primary, top = ranked[0]
    if top <= 0:
        return RoleFamily.NON_ENGINEERING, []

    alts = [fam for fam, score in ranked[1:] if score >= alt_ratio * top][:max_alt]
    return primary, alts


def build_entry_classifications(
    spans: list[Span],
    classifications: list[tuple[RoleFamily | None, float, str, str]],
    attributed_months: dict[int, int],
    as_of: date,
) -> list[EntryClassification]:
    out: list[EntryClassification] = []
    by_index = {span.index: span for span in spans}
    for span_index, (family, confidence, source, evidence) in enumerate(classifications):
        span = by_index.get(span_index)
        if span is None:
            continue
        out.append(
            EntryClassification(
                index=span_index,
                family=family,
                confidence=confidence,
                source=source,
                evidence=evidence,
                months=attributed_months.get(span_index, span.months),
                age_months=age_months(span, as_of),
            )
        )
    return out
