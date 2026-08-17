"""Switch intent: an additive logit that is deliberately **not** calibrated.

The brief asks for "a calibrated probability from 0 to 1". There is no ground
truth for this field anywhere in the dataset -- no recruiter reply rates, no
accepted-call outcomes, nothing to calibrate *against*. So what ships is a
hand-weighted additive logit: monotone in the things that should raise intent,
interpretable term by term, bounded in (0, 1), and honestly labelled
uncalibrated in `WRITEUP.md` rather than dressed up as a probability that has
been fitted to something.

What would actually calibrate it: recruiter reply / accept outcomes as the
label, then Platt scaling or isotonic regression on a held-out slice, with
reliability curves reported by decile.

`is_open_to_work` is an **input**, not the answer -- the brief says so
explicitly, and training on it would just relearn the flag.
"""

from __future__ import annotations

import math
import re

from saral.core.normalize import norm_text

BASE = -0.85

WEIGHTS: dict[str, float] = {
    "open_to_work": 1.30,
    "seeking_language": 0.50,
    "career_break": 0.45,
    "overdue_for_a_move": 0.40,
    "hopper": 0.30,
    "freelance_or_contract": 0.25,
    "just_started": -0.60,
    "founder_or_self_employed": -0.30,
}

_SEEKING = re.compile(
    r"\b(looking for|open to|seeking|available for|actively (looking|seeking)|"
    r"interested in new|new opportunities|hire me|immediate joiner)\b"
)
_BREAK = re.compile(r"\b(career break|returning to work|sabbatical|on a break|relaunch)\b")
_FREELANCE = re.compile(r"\b(freelance|freelancer|contract|contractor|consultant)\b")
_FOUNDER = re.compile(r"\b(founder|co-founder|cofounder|self-employed|self employed|ceo)\b")


def switch_intent(
    *,
    is_open_to_work: bool | None,
    headline: str | None,
    about: str | None,
    current_tenure_months: float,
    avg_tenure_months: float,
    tenure_flag: str,
    current_job_type: str | None,
    current_role_title: str | None,
    current_company: str | None,
) -> tuple[float, str, dict[str, float]]:
    """Return ``(probability, band, contributing_terms)``."""
    text = f"{norm_text(headline)} {norm_text(about)}"
    role_text = f"{norm_text(current_role_title)} {norm_text(current_company)} {norm_text(current_job_type)}"

    terms: dict[str, float] = {}

    if is_open_to_work:
        terms["open_to_work"] = WEIGHTS["open_to_work"]
    if _SEEKING.search(text):
        terms["seeking_language"] = WEIGHTS["seeking_language"]
    if _BREAK.search(text):
        terms["career_break"] = WEIGHTS["career_break"]
    if (
        avg_tenure_months > 0
        and current_tenure_months > 1.5 * avg_tenure_months
        and current_tenure_months > 36
    ):
        terms["overdue_for_a_move"] = WEIGHTS["overdue_for_a_move"]
    if tenure_flag == "hopper":
        terms["hopper"] = WEIGHTS["hopper"]
    if _FREELANCE.search(role_text):
        terms["freelance_or_contract"] = WEIGHTS["freelance_or_contract"]
    if 0 < current_tenure_months < 12:
        terms["just_started"] = WEIGHTS["just_started"]
    if _FOUNDER.search(role_text):
        terms["founder_or_self_employed"] = WEIGHTS["founder_or_self_employed"]

    z = BASE + sum(terms.values())
    p = 1.0 / (1.0 + math.exp(-z))
    band = "low" if p < 0.35 else ("medium" if p < 0.65 else "high")
    return round(p, 3), band, terms
