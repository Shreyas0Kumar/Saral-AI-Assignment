"""Derive seniority. Never trust the headline.

Order of evidence:

1. **Title override on the current role.** "Engineering Manager", "Head of",
   "Director", "VP" -> manager. "Staff", "Principal", "Architect", "SDE-3" ->
   staff+. These are level statements a company made, not self-description.
2. **Years, banded.** Otherwise seniority follows `years_relevant`.
3. `staff+` is reachable only through (1). Nobody becomes a staff engineer by
   accumulating years.
4. If the *headline* claims a level above the derived one, the derived value
   wins and `headline_seniority_inflated` is emitted. The claim becomes a
   reason code rather than a score.
"""

from __future__ import annotations

import re

from saral.contracts.taxonomy import SENIORITY_ORDER, Seniority
from saral.core.normalize import norm_text

#: (pattern, level) checked against the normalised title text.
_HEADLINE_LEVELS: tuple[tuple[str, Seniority], ...] = (
    (r"\b(chief technology officer|cto|vp of engineering|vice president)\b", Seniority.MANAGER),
    (r"\b(engineering manager|head of|director|em)\b", Seniority.MANAGER),
    (r"\b(staff|principal|distinguished|fellow|architect)\b", Seniority.STAFF_PLUS),
    (r"\bsde[-\s]?(3|iii)\b", Seniority.STAFF_PLUS),
    (r"\b(senior|sr\.?|lead)\b", Seniority.SENIOR),
    (r"\b(associate|junior|jr\.?)\b", Seniority.JUNIOR),
    (r"\b(intern|trainee|fresher|graduate engineer)\b", Seniority.INTERN),
)

_YEAR_BANDS: tuple[tuple[float, Seniority], ...] = (
    (1.0, Seniority.JUNIOR),   # < 1  -> intern, handled separately
    (3.0, Seniority.JUNIOR),
    (6.0, Seniority.MID),
    (10.0, Seniority.SENIOR),
)


def _title_level(text: str, config) -> Seniority | None:
    """Level implied by an explicit title. ``config`` supplies the override lists."""
    normalized = norm_text(text)
    if not normalized:
        return None
    for phrase in config.manager_titles:
        if phrase and re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", normalized):
            return Seniority.MANAGER
    for phrase in config.staff_titles:
        if phrase and re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", normalized):
            return Seniority.STAFF_PLUS
    return None


def headline_level(headline: str | None) -> Seniority | None:
    """The level the *headline* claims, used only to detect inflation."""
    normalized = norm_text(headline)
    if not normalized:
        return None
    for pattern, level in _HEADLINE_LEVELS:
        if re.search(pattern, normalized):
            return level
    return None


def band_from_years(years: float, is_intern_only: bool) -> Seniority:
    if is_intern_only or years < 1.0:
        return Seniority.INTERN
    if years < 3.0:
        return Seniority.JUNIOR
    if years < 6.0:
        return Seniority.MID
    return Seniority.SENIOR


def derive_seniority(
    current_titles: list[str],
    headline: str | None,
    years_relevant: float,
    is_intern_only: bool,
    config,
) -> tuple[Seniority, list[str]]:
    """Return ``(seniority, reason_codes)``."""
    codes: list[str] = []

    derived: Seniority | None = None
    for title in current_titles:
        level = _title_level(title, config)
        if level is not None:
            derived = level if derived is None else max(
                derived, level, key=lambda s: SENIORITY_ORDER[s]
            )

    if derived is None:
        derived = band_from_years(years_relevant, is_intern_only)

    claimed = headline_level(headline)
    if claimed is not None and SENIORITY_ORDER[claimed] > SENIORITY_ORDER[derived]:
        codes.append("headline_seniority_inflated")

    return derived, codes
