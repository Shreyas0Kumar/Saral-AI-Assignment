"""Tenure stability.

"Three jobs in 26 months is a real signal, in both directions" -- it predicts
availability and it predicts churn, and a recruiter wants to see both.

    avg_tenure_months = mean duration of *completed* roles
                        (falls back to the current role when nothing completed)
    jobs_last_36m     = entries that started within 36 months of `as_of`
    flag              = hopper   if avg < 15 and jobs_last_36m >= 3
                        stable   if avg >= 30
                        moderate otherwise

Completed roles only, because a current role's duration is censored: someone
three months into a job they will hold for five years should not read as a
hopper on that evidence.
"""

from __future__ import annotations

from datetime import date

from saral.contracts.models import TenureStability
from saral.core.dates import Span, months_between

HOPPER_AVG_MONTHS = 15.0
HOPPER_MIN_JOBS = 3
STABLE_AVG_MONTHS = 30.0
RECENT_WINDOW_MONTHS = 36


def compute_tenure(spans: list[Span], as_of: date) -> TenureStability:
    if not spans:
        return TenureStability(avg_tenure_months=0.0, jobs_last_36m=0, flag="moderate")

    completed = [s for s in spans if not s.is_open]
    pool = completed or spans
    avg = sum(s.months for s in pool) / len(pool)

    recent = sum(1 for s in spans if months_between(s.start, as_of) <= RECENT_WINDOW_MONTHS)

    if avg < HOPPER_AVG_MONTHS and recent >= HOPPER_MIN_JOBS:
        flag = "hopper"
    elif avg >= STABLE_AVG_MONTHS:
        flag = "stable"
    else:
        flag = "moderate"

    return TenureStability(
        avg_tenure_months=round(avg, 1), jobs_last_36m=recent, flag=flag
    )
