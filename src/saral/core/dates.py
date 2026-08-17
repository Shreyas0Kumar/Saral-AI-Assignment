"""Date handling and non-overlapping tenure arithmetic.

No clock is read here. ``as_of`` is always injected -- see ``DECISIONS.md`` D13
and ``WRITEUP.md`` "Assumptions" for why the corpus snapshot date is derived from
the data rather than taken from ``datetime.now()``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

_FORMATS = ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%Y-%m", "%Y")


def parse_date(value: str | None) -> date | None:
    """Parse an ISO-ish date. Returns ``None`` on failure -- never raises."""
    if not value or not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if "T" in text:
        text = text.split("T", 1)[0]
    from datetime import datetime

    for fmt in _FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def months_between(start: date, end: date) -> int:
    """Whole months from ``start`` to ``end``, floored at 0."""
    months = (end.year - start.year) * 12 + (end.month - start.month)
    if end.day < start.day:
        months -= 1
    return max(0, months)


def add_months(d: date, months: int) -> date:
    total = d.month - 1 + months
    year = d.year + total // 12
    month = total % 12 + 1
    # Clamp the day so 31 Jan + 1 month lands on the last day of February.
    day = min(d.day, [31, 29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28,
                      31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1])
    return date(year, month, day)


@dataclass(frozen=True)
class Span:
    """One resolved experience entry: a date range plus the flags we learned."""

    start: date
    end: date
    months: int
    index: int
    flags: tuple[str, ...] = ()

    @property
    def is_open(self) -> bool:
        return "current" in self.flags


def resolve_span(
    start_raw: str | None,
    end_raw: str | None,
    is_current: bool | None,
    duration_months: int | None,
    as_of: date,
    index: int,
) -> tuple[Span | None, list[str]]:
    """Turn one raw experience entry into a ``Span`` plus extraction flags.

    Policy, in order:

    * an unparseable ``start_date`` makes the entry unusable -> ``None``;
    * a missing ``end_date`` on a current role resolves to ``as_of``;
    * a missing ``end_date`` on a *non*-current role also resolves to ``as_of``
      but is flagged, because a crawler that dropped the field is more likely
      than a role that genuinely never ended;
    * where the stated ``duration_months`` disagrees with the computed span by
      more than 2 months, the **computed span wins** and ``duration_mismatch``
      is flagged. That flag feeds a confidence penalty; it does not change the
      number, because the dates are the more primary evidence.
    """
    flags: list[str] = []
    start = parse_date(start_raw)
    if start is None:
        return None, ["date_parse_failure"]

    end = parse_date(end_raw)
    current = bool(is_current) or end is None
    if end is None:
        if end_raw:  # a value was present but unparseable
            flags.append("date_parse_failure")
        end = as_of
        if not is_current:
            flags.append("open_ended_non_current")
    if end < start:
        flags.append("date_parse_failure")
        end = start

    computed = months_between(start, end)
    if duration_months is not None and abs(duration_months - computed) > 2:
        flags.append("duration_mismatch")

    entry_flags = tuple(flags + (["current"] if current else []))
    return Span(start=start, end=end, months=computed, index=index, flags=entry_flags), flags


def merge_non_overlapping(spans: list[Span]) -> int:
    """Total months covered by the union of ``spans``.

    Concurrent roles (freelance alongside a day job) would otherwise be counted
    twice, inflating ``years_total`` for exactly the candidates whose profiles
    are hardest to read.
    """
    if not spans:
        return 0
    ordered = sorted(spans, key=lambda s: (s.start, s.end))
    merged: list[tuple[date, date]] = []
    for span in ordered:
        if merged and span.start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], span.end))
        else:
            merged.append((span.start, span.end))
    return sum(months_between(a, b) for a, b in merged)


def deoverlap_months(spans: list[Span]) -> dict[int, int]:
    """Attribute months to entries so no calendar month is counted twice.

    Where two entries overlap, the **later-starting** one owns the overlap
    window. Rationale: the more recent role is the one a recruiter would call
    the person's current work, and it is the one recency weighting should credit.
    Returns ``{span.index: attributed_months}``.
    """
    if not spans:
        return {}
    # Latest start first, so earlier roles are trimmed against later ones.
    ordered = sorted(spans, key=lambda s: (s.start, s.end), reverse=True)
    claimed: list[tuple[date, date]] = []
    out: dict[int, int] = {}
    for span in ordered:
        total = span.months
        overlap = 0
        for c_start, c_end in claimed:
            lo = max(span.start, c_start)
            hi = min(span.end, c_end)
            if hi > lo:
                overlap += months_between(lo, hi)
        out[span.index] = max(0, total - overlap)
        claimed.append((span.start, span.end))
    return out


def age_months(span: Span, as_of: date) -> int:
    """Months since the entry ended. 0 for a role that is still running."""
    if span.is_open or span.end >= as_of:
        return 0
    return months_between(span.end, as_of)


def derive_snapshot_date(profiles: list[dict], fallback: date) -> date:
    """Infer the date the corpus was crawled from the data itself.

    The fixture's ``duration_months`` on current roles are all consistent with a
    crawl around mid-2025, while ``created_at`` says 2026-06 and the delta file
    says 2026-08. Trusting the wall clock would make every current role look
    twelve months longer than the platform says it is, firing
    ``duration_mismatch`` on nearly every profile and depressing confidence
    corpus-wide for no real reason. So the snapshot date is derived: for each
    current role take ``start_date + duration_months``, and take the median.

    See ``FAILURE_LOG.md`` FL-002.
    """
    implied: list[date] = []
    for profile in profiles:
        for entry in profile.get("experience") or []:
            if not isinstance(entry, dict) or not entry.get("is_current"):
                continue
            start = parse_date(entry.get("start_date"))
            duration = entry.get("duration_months")
            if start is None or not isinstance(duration, int):
                continue
            implied.append(add_months(start, duration))
    if not implied:
        return fallback
    implied.sort()
    return implied[len(implied) // 2]
