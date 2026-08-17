"""Content hashing for change detection.

The single most important line in this file is ``_CRAWL_METADATA``. If crawl
metadata sits inside the hash, then every crawl of an unchanged profile produces
a new hash, every profile looks changed, and the entire incremental argument in
Part 3 collapses into a full recompute wearing a disguise.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from saral.core.normalize import norm_company, norm_text

#: Excluded from ``input_hash``: these move on every crawl without the profile changing.
_CRAWL_METADATA = frozenset(
    {"created_at", "updated_at", "_observed_at", "_source", "_ingested_at"}
)

#: Field groups tracked independently in Part 3. Each gets its own hash and its
#: own ``observed_at``, which is what lets a partial record apply two fresh
#: fields while rejecting one stale one.
FIELD_GROUPS: tuple[str, ...] = (
    "headline",
    "about",
    "skills",
    "experience",
    "education",
    "location",
    "is_open_to_work",
)


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no incidental whitespace."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _strip_metadata(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip_metadata(v) for k, v in obj.items() if k not in _CRAWL_METADATA}
    if isinstance(obj, list):
        return [_strip_metadata(v) for v in obj]
    return obj


def input_hash(raw: dict[str, Any]) -> str:
    """``sha256`` over the profile with crawl metadata removed."""
    digest = hashlib.sha256(canonical_json(_strip_metadata(raw)).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _normalize_group_value(group: str, value: Any) -> Any:
    """Reduce a field group to the form that actually matters for change detection."""
    if group in {"headline", "about", "location"}:
        return norm_text(value if isinstance(value, str) else None)

    if group == "skills":
        items = value or []
        if not isinstance(items, list):
            return []
        # Sorted: reordering a skills list is not a change.
        return sorted({norm_text(str(s)) for s in items if s is not None})

    if group == "is_open_to_work":
        return bool(value) if value is not None else None

    if group == "experience":
        items = value or []
        if not isinstance(items, list):
            return []
        out = []
        for entry in items:
            if not isinstance(entry, dict):
                continue
            out.append(
                {
                    "role": norm_text(entry.get("role")),
                    "company": norm_company(entry.get("company_name")),
                    "start_date": (entry.get("start_date") or "")[:10],
                    "end_date": (entry.get("end_date") or "")[:10],
                    "is_current": bool(entry.get("is_current")),
                    "description": norm_text(entry.get("description")),
                    "skills_used": sorted(
                        {norm_text(str(s)) for s in (entry.get("skills_used") or [])}
                    ),
                }
            )
        out.sort(key=lambda e: (e["company"], e["start_date"], e["role"]))
        return out

    if group == "education":
        items = value or []
        if not isinstance(items, list):
            return []
        out = []
        for entry in items:
            if not isinstance(entry, dict):
                continue
            out.append(
                {
                    "school": norm_company(entry.get("school_name")),
                    "degree": norm_text(entry.get("degree")),
                    "field": norm_text(entry.get("field_of_study")),
                    "skills": sorted({norm_text(str(s)) for s in (entry.get("skills") or [])}),
                }
            )
        out.sort(key=lambda e: (e["school"], e["degree"], e["field"]))
        return out

    return value


def field_group_hash(raw: dict[str, Any], group: str) -> str:
    """``sha256`` over the **normalised** value of one field group.

    Hashing the normalised form is what makes a whitespace-or-emoji-only edit
    hash-identical, which is what makes it ``noise`` without a special case.
    """
    key = "location" if group == "location" else group
    normalized = _normalize_group_value(group, raw.get(key))
    digest = hashlib.sha256(canonical_json(normalized).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def file_sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()
