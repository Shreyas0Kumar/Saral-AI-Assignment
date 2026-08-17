"""`extract(raw, cfg, classifier, computed_at, as_of) -> SignalRecord`.

Pure and deterministic. Both time values are injected; nothing here reads a
clock. Identical input therefore produces byte-identical output, which is what
makes the Part 3 idempotency test a plain equality assertion rather than a diff
with an exclusion list papering over a timestamp.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

from saral.contracts.models import RawProfile, SignalRecord, TenureStability
from saral.contracts.taxonomy import RoleFamily, Seniority
from saral.contracts.versions import SIGNALS_VERSION
from saral.core.dates import Span, age_months, deoverlap_months, resolve_span
from saral.core.extract.confidence import compute_confidence, noise_penalty_scale
from saral.core.extract.role_family import (
    EntryClassification,
    decay,
    family_scores,
    resolve_family,
)
from saral.core.extract.seniority import derive_seniority
from saral.core.extract.skills import SkillProfile, build_skill_profile
from saral.core.extract.tenure import compute_tenure
from saral.core.extract.switch_intent import switch_intent
from saral.core.extract.years import years_relevant, years_total
from saral.core.hashing import input_hash
from saral.core.normalize import norm_text
from saral.core.extract.title_classifier.base import ChainClassifier

_TRANSITION_MARKERS = (
    "transitioning to", "transition to", "aspiring", "switching to", "pivoting to",
    "career change", "moving into", "breaking into", "want to become",
)


@dataclass
class ExtractionTrace:
    """Everything the scorer or a debugger needs that is not in the contract."""

    spans: list[Span] = field(default_factory=list)
    entries: list[EntryClassification] = field(default_factory=list)
    skills: SkillProfile | None = None
    prior_families: set[RoleFamily] = field(default_factory=set)
    descriptions: list[str] = field(default_factory=list)
    current_location: str = ""
    switch_terms: dict[str, float] = field(default_factory=dict)
    confidence_penalties: dict[str, float] = field(default_factory=dict)


def _headline_family(headline: str | None, classifier, aliases) -> RoleFamily | None:
    result = classifier.classify(headline or "", "")
    return result.family


def extract(
    raw: RawProfile | dict,
    cfg,
    classifier: ChainClassifier,
    computed_at: datetime,
    as_of: date,
) -> tuple[SignalRecord, ExtractionTrace]:
    profile = raw if isinstance(raw, RawProfile) else RawProfile.model_validate(raw)
    raw_dict = raw if isinstance(raw, dict) else profile.model_dump()

    flags: set[str] = set()
    trace = ExtractionTrace()

    # -- 1. resolve experience entries into date spans --------------------
    spans: list[Span] = []
    entries_raw: list[dict] = []
    for index, entry in enumerate(profile.experience):
        entry_dict = entry.model_dump()
        entries_raw.append(entry_dict)
        span, entry_flags = resolve_span(
            entry.start_date,
            entry.end_date,
            entry.is_current,
            entry.duration_months,
            as_of,
            index,
        )
        flags.update(f for f in entry_flags if f in {"date_parse_failure", "duration_mismatch"})
        if span is not None:
            spans.append(span)

    attributed = deoverlap_months(spans)
    trace.spans = spans

    # -- 2. classify each entry independently -----------------------------
    classifier.reset_counters()
    classifications: list[tuple[RoleFamily | None, float, str, str]] = []
    by_index = {s.index: s for s in spans}
    for index, entry_dict in enumerate(entries_raw):
        if index not in by_index:
            classifications.append((None, 0.0, "", ""))
            continue
        context = " ".join(
            [entry_dict.get("description") or "", " ".join(entry_dict.get("skills_used") or [])]
        )
        result = classifier.classify(entry_dict.get("role") or "", context)
        classifications.append(
            (result.family, result.confidence, result.source, result.evidence)
        )

    entry_classifications = [
        EntryClassification(
            index=idx,
            family=fam,
            confidence=conf,
            source=src,
            evidence=ev,
            months=attributed.get(idx, by_index[idx].months) if idx in by_index else 0,
            age_months=age_months(by_index[idx], as_of) if idx in by_index else 0,
        )
        for idx, (fam, conf, src, ev) in enumerate(classifications)
        if idx in by_index
    ]
    trace.entries = entry_classifications

    if classifier.fallback_invocations():
        flags.add("fallback_classifier_used")
    if any(e.family is None for e in entry_classifications):
        flags.add("unclassified_entries")

    # -- 3. role family ---------------------------------------------------
    scores = family_scores(entry_classifications, cfg.recency_half_life_months)
    primary, alts = resolve_family(scores, cfg.alt_family_ratio, cfg.max_alt_families)
    trace.prior_families = {e.family for e in entry_classifications if e.family is not None}

    # -- 4. years ---------------------------------------------------------
    total_years = years_total(spans)
    relevant_years = years_relevant(
        entry_classifications, primary, cfg.adjacency, trace.prior_families
    )

    # -- 5. tenure --------------------------------------------------------
    tenure = compute_tenure(spans, as_of)
    current_spans = [s for s in spans if s.is_open]
    current_tenure_months = max((s.months for s in current_spans), default=0)

    # -- 6. seniority -----------------------------------------------------
    current_titles = [
        entries_raw[s.index].get("role") or "" for s in current_spans if s.index < len(entries_raw)
    ]
    if not current_titles and entries_raw:
        latest = max(spans, key=lambda s: s.end, default=None)
        if latest is not None:
            current_titles = [entries_raw[latest.index].get("role") or ""]
    intern_only = bool(entries_raw) and all(
        norm_text(e.get("job_type")) in {"internship", "intern"} for e in entries_raw
    )
    seniority, seniority_codes = derive_seniority(
        current_titles, profile.headline, relevant_years, intern_only, cfg.lexicon
    )
    if "headline_seniority_inflated" in seniority_codes:
        flags.add("headline_seniority_inflated")

    # -- 7. skills --------------------------------------------------------
    entry_months = {i: attributed.get(i, 0) for i in range(len(entries_raw))}
    entry_ages = {
        i: age_months(by_index[i], as_of) for i in range(len(entries_raw)) if i in by_index
    }
    skill_profile = build_skill_profile(
        profile.skills,
        entries_raw,
        [e.model_dump() for e in profile.education],
        cfg.skills.aliases,
        cfg.skills.off_domain,
        entry_months,
        entry_ages,
        cfg.recency_half_life_months,
    )
    trace.skills = skill_profile
    if skill_profile.skill_noise_ratio > 0.5:
        flags.add("high_skill_noise")
    confidence_scales = {
        "high_skill_noise": noise_penalty_scale(skill_profile.skill_noise_ratio)
    }

    descriptions = [e.get("description") or "" for e in entries_raw]
    trace.descriptions = descriptions
    if entries_raw and not any(d.strip() for d in descriptions):
        flags.add("no_experience_descriptions")
    if len(entries_raw) < 2 and sum(s.months for s in spans) < 12:
        flags.add("thin_history")

    # -- 8. switch intent -------------------------------------------------
    current_entry = entries_raw[current_spans[0].index] if current_spans else {}
    intent, band, terms = switch_intent(
        is_open_to_work=profile.is_open_to_work,
        headline=profile.headline,
        about=profile.about,
        current_tenure_months=current_tenure_months,
        avg_tenure_months=tenure.avg_tenure_months,
        tenure_flag=tenure.flag,
        current_job_type=current_entry.get("job_type"),
        current_role_title=current_entry.get("role"),
        current_company=current_entry.get("company_name"),
    )
    trace.switch_terms = terms
    trace.current_location = norm_text(profile.location)

    # -- 9. headline agreement -------------------------------------------
    headline_fam = _headline_family(profile.headline, classifier, cfg.skills.aliases)
    reason_codes: list[str] = []
    normalized_headline = norm_text(profile.headline)
    if any(marker in normalized_headline for marker in _TRANSITION_MARKERS):
        reason_codes.append("headline_declares_transition")
    if headline_fam is not None and headline_fam != primary:
        flags.add("headline_family_mismatch")
        reason_codes.append(f"headline_family_mismatch:{headline_fam.value}!={primary.value}")

    # -- 10. confidence ---------------------------------------------------
    confidence = compute_confidence(flags, confidence_scales)
    trace.confidence_penalties = confidence.applied

    # -- 11. reason codes -------------------------------------------------
    reason_codes.extend(seniority_codes)
    if skill_profile.claimed_unverified:
        reason_codes.append("skills_not_evidenced_in_experience")
    # Claims skills for a family they have never actually worked in.
    claimed_families = _families_implied_by_skills(skill_profile.claimed_unverified, cfg)
    # "Zero experience" has to mean zero, not "zero under this exact label". A
    # platform engineer claiming Postgres and gRPC has adjacent experience; a
    # mechanical engineer claiming PyTorch has none. Adjacency is what tells
    # them apart, so the code fires only when every claimed family is
    # unreachable from anything the person has actually done.
    unreachable = {
        family
        for family in claimed_families
        if not any(
            cfg.adjacency.score(prior, family, trace.prior_families) > 0
            for prior in (trace.prior_families or {primary})
        )
    }
    if unreachable:
        reason_codes.append("zero_role_experience_despite_skill_claim")
    if skill_profile.off_domain:
        reason_codes.append(f"off_domain_skills:{len(skill_profile.off_domain)}")
    if profile.is_open_to_work is not None:
        reason_codes.append(f"open_to_work:{str(bool(profile.is_open_to_work)).lower()}")
    reason_codes.append(f"switch_intent:{band}")
    reason_codes.append(f"tenure_flag:{tenure.flag}")
    if "career_break" in terms:
        reason_codes.append("career_break")
    if "fallback_classifier_used" in flags:
        fired = [n for n in classifier.invocations if n != classifier.primary_name]
        reason_codes.append(f"fallback_classifier_used:{fired[0] if fired else 'unknown'}")
    if "no_experience_descriptions" in flags:
        reason_codes.append("no_experience_descriptions")
    if "date_parse_failure" in flags:
        reason_codes.append("date_parse_failure")
    if "duration_mismatch" in flags:
        reason_codes.append("duration_mismatch")
    if confidence.is_low:
        reason_codes.append("low_confidence_extraction")

    record = SignalRecord(
        candidate_id=profile.id,
        role_family=primary,
        role_family_alt=alts,
        seniority=seniority,
        years_total=total_years,
        years_relevant=relevant_years,
        core_skills=skill_profile.core_skills,
        claimed_skills_unverified=skill_profile.claimed_unverified,
        skill_noise_ratio=skill_profile.skill_noise_ratio,
        tenure_stability=tenure,
        switch_intent=intent,
        confidence=confidence.value,
        reason_codes=_dedupe(reason_codes),
        signals_version=SIGNALS_VERSION,
        computed_at=computed_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        input_hash=input_hash(raw_dict),
        role_family_scores={f.value: round(v, 2) for f, v in sorted(
            scores.items(), key=lambda kv: -kv[1]
        )},
        extraction_flags=sorted(flags),
    )
    return record, trace


def _families_implied_by_skills(skills: list[str], cfg) -> set[RoleFamily]:
    """Which engineering families a set of *unverified* skills is pointing at.

    ``non_engineering`` is excluded: claiming Excel while never having worked in
    a data role is not an ambition, it is just a skills list.
    """
    counts: dict[RoleFamily, int] = {}
    for skill in skills:
        family = cfg.skills.skill_families.get(skill)
        if family is None or family is RoleFamily.NON_ENGINEERING:
            continue
        counts[family] = counts.get(family, 0) + 1
    threshold = cfg.skills.claimed_family_min_skills
    return {family for family, count in counts.items() if count >= threshold}


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out
