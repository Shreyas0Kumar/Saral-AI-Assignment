"""Merge policy: how an observed field replaces a stored one.

The asymmetry between `skills` and `experience` is the whole of this file, and
it only falls out if you ask what each field *means*.

**skills replaces wholesale.** A skills list is a snapshot of what someone
currently claims. A shrinking list is information -- removing "PHP" is a signal.
Merging would make every skills list monotonically grow forever, which is both
wrong and a slow data-quality leak.

**experience merges by key** `(normalised_company, start_date)`. Career history
is append-mostly. A partial crawl that saw only the current role would, under
replace semantics, destroy every prior role -- which destroys `years_relevant`,
which destroys `seniority`, which destroys every downstream score. The blast
radius of getting this one wrong is the entire pipeline.

The mirror-image cost is stated rather than hidden: if someone deletes a
fabricated job, this keeps it. Same answer as the null policy -- the cost of
retaining a stale entry is lower than the cost of destroying real history, and
the production fix is corroboration across runs.
"""

from __future__ import annotations

import copy
from typing import Any

from saral.core.normalize import norm_company

#: Fields inside an experience entry that a fresh observation may overwrite.
_ENTRY_FIELDS = (
    "role", "end_date", "is_current", "duration_months", "job_type",
    "work_type", "location", "description", "skills_used",
)


def entry_key(entry: dict) -> tuple[str, str]:
    return (norm_company(entry.get("company_name")), (entry.get("start_date") or "")[:10])


def merge_experience(stored: list[dict], observed: list[dict]) -> list[dict]:
    """Field-merge matching entries, append new ones, never delete unseen ones."""
    merged = [copy.deepcopy(entry) for entry in stored]
    index = {entry_key(entry): position for position, entry in enumerate(merged)}

    for entry in observed:
        key = entry_key(entry)
        position = index.get(key)
        if position is None:
            merged.append(copy.deepcopy(entry))
            index[key] = len(merged) - 1
            continue
        target = merged[position]
        for field in _ENTRY_FIELDS:
            if field not in entry:
                continue
            value = entry[field]
            # A null inside an entry is "not observed", exactly as at the top
            # level. It must not wipe a description the previous crawl saw.
            if value is None:
                continue
            if isinstance(value, list) and not value and target.get(field):
                continue
            target[field] = value

    merged.sort(key=lambda e: (e.get("start_date") or "", e.get("company_name") or ""), reverse=True)
    return merged


def merge_education(stored: list[dict], observed: list[dict]) -> list[dict]:
    merged = [copy.deepcopy(entry) for entry in stored]
    seen = {
        (norm_company(e.get("school_name")), (e.get("degree") or "").casefold()) for e in merged
    }
    for entry in observed:
        key = (norm_company(entry.get("school_name")), (entry.get("degree") or "").casefold())
        if key not in seen:
            merged.append(copy.deepcopy(entry))
            seen.add(key)
    return merged


def apply_field(stored_profile: dict, group: str, value: Any) -> dict:
    """Return a new profile with ``group`` updated under the group's policy."""
    updated = copy.deepcopy(stored_profile)
    if group == "experience":
        updated["experience"] = merge_experience(stored_profile.get("experience") or [], value or [])
    elif group == "education":
        updated["education"] = merge_education(stored_profile.get("education") or [], value or [])
    else:
        updated[group] = value
    return updated
