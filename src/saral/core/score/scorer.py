"""Fit scoring: signals + a job -> a 0-100 score and the reasons for it.

Every component is bounded, every component is reported in `score_breakdown`,
and every point movement has a reason code. That is not decoration: the brief
requires a recruiter and a support engineer to be able to read *why*, and a
score nobody can decompose is a score nobody can dispute.

Two design choices worth defending up front.

**Nothing is ever filtered.** Missing must-haves cost points; they never remove
a candidate. A dropped candidate emits no reason codes, which would make
`missing_must_haves` dead code and the explainability claim hollow.

**A soft cap replaces a hard gate.** When `role_match` scores zero the total is
capped (default 35). This is the most questionable line in the codebase and it
is deliberately visible in the breakdown rather than hidden in a filter. The
alternative -- letting adjacency smoothly carry an HR executive to 55 -- ranks
worse and explains worse. See DECISIONS.md D8.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from saral.contracts.models import RankingRecord, SignalRecord
from saral.core.extract.years import years_relevant
from saral.core.normalize import norm_text
from saral.core.score.features import ParsedJob, Requirement

#: Verbs that describe having owned an outcome rather than having been present.
_OWNERSHIP_VERBS = (
    "own", "owns", "owned", "ownership", "shipped", "launched", "built",
    "rewrote", "migrated", "migration", "moved", "cut", "reduced", "scaled",
    "led", "drove", "delivered", "maintained", "trained", "designed",
    "implemented", "quantized", "repartitioning",
)
_NUMBER = re.compile(r"\d")


@dataclass
class ScoringFlags:
    """Feature switches for the ablation ladder.

    The ladder exists to turn "my system is better" into "the evidence-versus-
    claim rule is worth X NDCG, and here is the run that shows it". Each rung is
    a flag, not a separate code path, so no rung can quietly differ in something
    other than the feature it is meant to isolate.
    """

    role_prefilter: bool = True
    must_have_scoring: bool = True
    evidence_tiering: bool = True
    seniority_band: bool = True
    evidence_of_shipping: bool = True
    location_fit: bool = True
    switch_intent: bool = True
    tenure_stability: bool = True
    soft_cap: bool = True

    @staticmethod
    def rung(level: int) -> "ScoringFlags":
        """Ablation rungs 1-4, coarsest first."""
        if level <= 1:
            return ScoringFlags(
                role_prefilter=True, must_have_scoring=False, evidence_tiering=False,
                seniority_band=False, evidence_of_shipping=False, location_fit=False,
                switch_intent=False, tenure_stability=False, soft_cap=False,
            )
        if level == 2:
            return ScoringFlags(
                must_have_scoring=True, evidence_tiering=False, seniority_band=False,
                evidence_of_shipping=False, location_fit=False, switch_intent=False,
                tenure_stability=False, soft_cap=False,
            )
        if level == 3:
            return ScoringFlags(
                evidence_tiering=True, seniority_band=False, evidence_of_shipping=False,
                location_fit=False, switch_intent=False, tenure_stability=False,
                soft_cap=False,
            )
        return ScoringFlags()


@dataclass
class ComponentResult:
    points: float
    reason_codes: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)


def _production_evidence(descriptions: list[str], keywords: tuple[str, ...]) -> str | None:
    """Return the snippet evidencing production ownership, or None.

    Two independent tests, either sufficient: an ownership verb ("Own the dbt
    project"), or a quantified outcome ("40M events/day"). A description that
    has neither is someone listing that they were present.
    """
    for description in descriptions:
        normalized = norm_text(description)
        if not normalized:
            continue
        if any(re.search(rf"(?<!\w){verb}(?!\w)", normalized) for verb in _OWNERSHIP_VERBS):
            return description.strip()[:80]
        if _NUMBER.search(normalized) and any(
            re.search(rf"(?<!\w){re.escape(k)}", normalized) for k in keywords
        ):
            return description.strip()[:80]
        if _NUMBER.search(normalized):
            return description.strip()[:80]
    return None


def _requirement_met(
    requirement: Requirement,
    core_skills: set[str],
    claimed_skills: set[str],
    descriptions: list[str],
    keywords: tuple[str, ...],
    domain_relevant: bool,
    evidence_tiering: bool,
) -> tuple[float, str | None]:
    """Return ``(coverage in [0,1], evidence snippet)`` for one requirement."""
    if requirement.kind == "evidence":
        snippet = _production_evidence(descriptions, keywords)
        if snippet is None:
            return 0.0, None
        # Production evidence alone is not enough for a *domain* requirement:
        # a backend engineer's "12k rps" does not evidence "shipped a
        # customer-facing UI". Either the candidate works in the job's domain,
        # or the requirement's own words appear in the history.
        haystack = " ".join(norm_text(d) for d in descriptions)
        term_hit = any(term in haystack for term in requirement.evidence_terms)
        if domain_relevant or term_hit:
            return 1.0, snippet
        return 0.0, None

    if any(alt in core_skills for alt in requirement.alternatives):
        return 1.0, None
    if any(alt in claimed_skills for alt in requirement.alternatives):
        # Satisfied, but only by a declaration. The caller decides what that is
        # worth: full credit with `evidence_tiering` off, `claimed_only_credit`
        # with it on. Keeping the discount in one place is what makes the
        # ablation rung isolate exactly that rule and nothing else.
        return 1.0, None
    return 0.0, None


def score_candidate(
    signal: SignalRecord,
    trace,
    job: ParsedJob,
    cfg,
    weights,
    flags: ScoringFlags | None = None,
) -> tuple[float, dict[str, float], list[str], list[str]]:
    """Return ``(total, breakdown, reason_codes, missing_must_haves)``."""
    flags = flags or ScoringFlags()
    codes: list[str] = []
    breakdown: dict[str, float] = {}
    missing: list[str] = []

    core_skills = set(signal.core_skills)
    claimed_skills = set(signal.claimed_skills_unverified)
    descriptions = list(trace.descriptions)
    prior_families = set(trace.prior_families)

    # -- role_match --------------------------------------------------------
    direct = cfg.adjacency.score(signal.role_family, job.family, prior_families)
    alt_best = max(
        (cfg.adjacency.score(alt, job.family, prior_families) for alt in signal.role_family_alt),
        default=0.0,
    )
    affinity = max(direct, 0.6 * alt_best) if flags.role_prefilter else 1.0
    role_points = weights.components["role_match"] * affinity
    breakdown["role_match"] = round(role_points, 2)

    if direct >= 1.0:
        codes.append(f"role_match:{job.family.value}")
    elif affinity > 0:
        codes.append(
            f"role_adjacent:{signal.role_family.value}~{job.family.value}={affinity:.2f}"
        )
    else:
        codes.append(f"role_mismatch:{signal.role_family.value}!={job.family.value}")

    domain_relevant = affinity >= 0.5

    # -- skill_overlap -----------------------------------------------------
    must_points = good_points = 0.0
    if flags.must_have_scoring:
        must_cov: list[float] = []
        for requirement in job.must_have:
            coverage, snippet = _requirement_met(
                requirement, core_skills, claimed_skills, descriptions,
                weights.shipping_keywords, domain_relevant, flags.evidence_tiering,
            )
            claimed_only = (
                coverage == 1.0
                and requirement.kind == "skill"
                and not any(a in core_skills for a in requirement.alternatives)
            )
            if claimed_only and flags.evidence_tiering:
                coverage = weights.claimed_only_credit
                codes.append(f"must_have_claimed_only:{requirement.label}")
            elif coverage >= 1.0:
                codes.append(
                    f"must_have_met:{requirement.label}|evidence={snippet}"
                    if snippet
                    else f"must_have_met:{requirement.label}"
                )
            else:
                codes.append(f"must_have_missing:{requirement.label}")
                missing.append(requirement.raw)
            must_cov.append(coverage)

        good_cov: list[float] = []
        for requirement in job.good_to_have:
            coverage, _ = _requirement_met(
                requirement, core_skills, claimed_skills, descriptions,
                weights.shipping_keywords, domain_relevant, flags.evidence_tiering,
            )
            if coverage >= 1.0:
                codes.append(f"good_to_have_met:{requirement.label}")
            good_cov.append(coverage)

        must_points = weights.must_have_points * (sum(must_cov) / len(must_cov) if must_cov else 0.0)
        good_points = weights.good_to_have_points * (
            sum(good_cov) / len(good_cov) if good_cov else 0.0
        )
    breakdown["skill_overlap"] = round(must_points + good_points, 2)

    # -- seniority_fit -----------------------------------------------------
    years_for_job = years_relevant(trace.entries, job.family, cfg.adjacency, prior_families)
    seniority_points = 0.0
    if flags.seniority_band:
        low, high = job.spec.min_years, job.spec.max_years
        near = weights.seniority.get("near_band_years", 1.0)
        decay = weights.seniority.get("decay_years", 3.0)
        in_band = weights.seniority.get("in_band", 15.0)
        near_points = weights.seniority.get("near_band", 9.0)
        if low <= years_for_job <= high:
            seniority_points = in_band
            codes.append(f"years_in_band:{years_for_job:.1f} in [{low:g},{high:g}]")
        else:
            distance = low - years_for_job if years_for_job < low else years_for_job - high
            if distance <= near:
                seniority_points = near_points
            else:
                extra = distance - near
                seniority_points = max(0.0, near_points * (1 - extra / decay))
            if years_for_job < low:
                codes.append(f"years_below_min:{years_for_job:.1f}<{low:g}")
            else:
                codes.append(f"years_above_max:{years_for_job:.1f}>{high:g}")
    breakdown["seniority_fit"] = round(seniority_points, 2)

    # -- evidence_of_shipping ---------------------------------------------
    shipping_points = 0.0
    if flags.evidence_of_shipping:
        snippet = _production_evidence(descriptions, weights.shipping_keywords)
        if snippet:
            shipping_points = weights.shipping_points
            codes.append(f"evidence_of_scale:{snippet}")
        else:
            codes.append("no_production_ownership")
    breakdown["evidence_of_shipping"] = round(shipping_points, 2)

    # -- location_fit ------------------------------------------------------
    location_points = 0.0
    if flags.location_fit:
        candidate_city = trace.current_location
        if job.is_remote:
            location_points = weights.location.get("remote_job", 8.0)
            codes.append("location_remote_ok")
        elif job.city and job.city in candidate_city:
            location_points = weights.location.get("exact_city", 8.0)
            codes.append(f"location_exact:{job.city}")
        elif _same_metro(job.city, candidate_city, weights.metros):
            location_points = weights.location.get("same_metro", 5.0)
            codes.append(f"location_exact:{job.city}")
        else:
            location_points = weights.location.get("different_city", 2.0)
            short = candidate_city.split(",")[0].strip() or "unknown"
            codes.append(f"location_mismatch:{short}vs{job.city or 'unknown'}")
    breakdown["location_fit"] = round(location_points, 2)

    # -- switch_intent -----------------------------------------------------
    intent_points = (
        weights.components["switch_intent"] * signal.switch_intent if flags.switch_intent else 0.0
    )
    breakdown["switch_intent"] = round(intent_points, 2)

    # -- tenure_stability --------------------------------------------------
    tenure_points = (
        weights.components["tenure_stability"] * weights.tenure.get(signal.tenure_stability.flag, 0.6)
        if flags.tenure_stability
        else 0.0
    )
    breakdown["tenure_stability"] = round(tenure_points, 2)

    # -- total, penalties, cap --------------------------------------------
    total = sum(breakdown.values())
    total -= weights.missing_must_have_penalty * len(missing)
    if flags.soft_cap and role_points == 0:
        cap = weights.role_mismatch_cap
        if total > cap:
            codes.append(f"score_capped_role_mismatch:{int(cap)}")
            total = cap

    total = max(0.0, min(100.0, total))
    return total, breakdown, codes, missing


def _same_metro(job_city: str, candidate_city: str, metros: dict[str, frozenset[str]]) -> bool:
    if not job_city or not candidate_city:
        return False
    for cities in metros.values():
        if any(c in job_city for c in cities) and any(c in candidate_city for c in cities):
            return True
    return False


def rank_candidates(
    signals: list[SignalRecord],
    traces: dict,
    job: ParsedJob,
    cfg,
    weights,
    flags: ScoringFlags | None = None,
) -> list[RankingRecord]:
    """Score every candidate against one job and rank them.

    Ties break on `confidence` then `candidate_id`, so two runs over identical
    input produce identical ranks. A ranking that reorders on dict iteration
    order would make every metric irreproducible.
    """
    scored = []
    for signal in signals:
        total, breakdown, codes, missing = score_candidate(
            signal, traces[signal.candidate_id], job, cfg, weights, flags
        )
        scored.append((total, signal, breakdown, codes, missing))

    scored.sort(key=lambda item: (-item[0], -item[1].confidence, item[1].candidate_id))

    return [
        RankingRecord(
            job_id=job.spec.job_id,
            candidate_id=signal.candidate_id,
            rank=index,
            fit_score=int(round(total)),
            score_breakdown=breakdown,
            reason_codes=codes + [c for c in signal.reason_codes if _carry_forward(c)],
            missing_must_haves=missing,
            confidence=signal.confidence,
        )
        for index, (total, signal, breakdown, codes, missing) in enumerate(scored, start=1)
    ]


#: Signal-level codes worth repeating on the ranking record, because a recruiter
#: reading one row should not have to go and fetch the signal record too.
_CARRY = (
    "open_to_work:", "switch_intent:", "tenure_flag:", "headline_declares_transition",
    "headline_seniority_inflated", "skills_not_evidenced_in_experience",
    "zero_role_experience_despite_skill_claim", "low_confidence_extraction",
    "career_break",
)


def _carry_forward(code: str) -> bool:
    return code.startswith(_CARRY)
