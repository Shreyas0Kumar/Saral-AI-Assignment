"""Relevant-years arithmetic.

    years_relevant(target) = sum(months_i * adjacency(family_i, target)) / 12

Called twice with different targets and the two numbers must not be conflated:

* on the ``SignalRecord``, ``target`` is the candidate's own ``role_family`` --
  "how many years has this person spent being what they are";
* at scoring time, ``target`` is the job's implied family, producing
  ``years_relevant_for_job`` -- "how many years count *for this job*".

A mechanical engineer with six years and a Coursera certificate has six relevant
years as a mechanical engineer and zero for an ML role. Both are true, and the
brief asks for the second.

Months are the *de-overlapped* ones, so a freelance contract running alongside a
day job is not counted twice.
"""

from __future__ import annotations

from saral.contracts.taxonomy import Adjacency, RoleFamily
from saral.core.dates import Span, merge_non_overlapping
from saral.core.extract.role_family import EntryClassification


def years_total(spans: list[Span]) -> float:
    return round(merge_non_overlapping(spans) / 12.0, 2)


def years_relevant(
    entries: list[EntryClassification],
    target: RoleFamily,
    adjacency: Adjacency,
    prior_families: set[RoleFamily] | None = None,
) -> float:
    """Adjacency-weighted months in ``target``, in years."""
    priors = prior_families or {e.family for e in entries if e.family is not None}
    months = 0.0
    for entry in entries:
        if entry.family is None:
            continue
        weight = adjacency.score(entry.family, target, priors)
        if weight:
            months += entry.months * weight
    return round(months / 12.0, 2)


def months_in_family(entries: list[EntryClassification], family: RoleFamily) -> int:
    """Unweighted months whose entry classified exactly as ``family``."""
    return sum(e.months for e in entries if e.family == family)
