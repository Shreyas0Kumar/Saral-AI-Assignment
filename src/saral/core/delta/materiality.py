"""How much a change is worth.

"materiality is the field that decides whether we spend money. Get it wrong in
one direction and we re-score a million rows because someone added an emoji. Get
it wrong in the other and we miss the one candidate who just became available."

So the two ends of the scale are set by what they cost:

* `is_open_to_work: false -> true` is **high**. The brief calls it the single
  most valuable event in the product.
* Anything whose *normalised* hash is unchanged is **noise**, decided by the
  normaliser rather than by a rule that inspects the text. If a special case
  were needed here, the normaliser would be the thing that is wrong.
"""

from __future__ import annotations

from typing import Any, Literal

Materiality = Literal["high", "medium", "low", "noise"]

_DOWNSTREAM: dict[Materiality, list[str]] = {
    "high": ["re_score", "notify_recruiter"],
    "medium": ["re_score"],
    "low": [],
    "noise": [],
}


def downstream_for(materiality: Materiality) -> list[str]:
    return list(_DOWNSTREAM[materiality])


def classify(
    group: str,
    old_value: Any,
    new_value: Any,
    *,
    experience_change: str | None = None,
) -> tuple[Materiality, str]:
    """Return ``(materiality, note)`` for one applied field change."""
    if group == "is_open_to_work":
        if not old_value and new_value:
            return "high", "became available: the highest-value event in the product"
        if old_value and not new_value:
            return "medium", "no longer marked open to work"
        return "low", "is_open_to_work restated"

    if group == "location":
        return "high", "location change invalidates location_fit on every open job"

    if group == "experience":
        if experience_change == "new_role":
            return "high", "new experience entry: role or company changed"
        if experience_change == "current_role_ended":
            return "high", "the current role acquired an end date"
        return "medium", "existing experience entry updated"

    if group == "skills":
        return "medium", "skills list changed"

    if group == "headline":
        return "medium", "normalised headline changed"

    if group == "education":
        return "medium", "education entry added"

    if group == "about":
        return "low", "about text changed"

    return "low", f"{group} changed"


def deletion_event() -> tuple[Materiality, str]:
    """A field arriving null.

    Policy: **not observed**, so the value is not applied, and a visible event
    is emitted at low materiality.

    Defence. The costs are asymmetric. A wrong delete removes a real candidate
    from search permanently; a missed delete leaves slightly stale data for one
    crawl cycle. Crawler failure is also empirically far more common than a user
    genuinely removing their headline. The event is emitted anyway because
    silently swallowing it is worse than either policy -- nobody could then tell
    "we ignored a null" apart from "we never saw the field".

    The correct production policy is corroboration: require the field to arrive
    null on two consecutive runs, or trust an authoritative `_source` (a direct
    API confirming absence beats a scraped page failing to render). That cannot
    be demonstrated here because only one delta file exists, and saying so is
    better than pretending otherwise.
    """
    return "low", (
        "field arrived null; treated as not-observed and NOT applied "
        "(deletion policy: crawler failure is likelier than genuine removal, "
        "and the costs are asymmetric)"
    )


def stale_event() -> tuple[Materiality, str]:
    return "noise", "observation older than or equal to the stored state for this field"


def noise_event() -> tuple[Materiality, str]:
    return "noise", "normalised forms identical; whitespace, ordering or emoji only"
