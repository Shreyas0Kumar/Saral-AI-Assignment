"""Extraction behaviour, pinned against named profiles in the real corpus.

Each test here names the candidate it protects and says what would break if the
assertion flipped. A test that only says `assert x == y` tells a reviewer
nothing about why `y`.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from saral.contracts.reason_vocab import validate
from saral.contracts.taxonomy import RoleFamily, Seniority
from saral.core.extract.pipeline import extract
from saral.core.hashing import field_group_hash, input_hash


# --------------------------------------------------------------------------
# Determinism -- the property the whole of Part 3 rests on
# --------------------------------------------------------------------------
def test_extract_is_deterministic(profile_by_id, cfg, classifier, computed_at, as_of):
    raw = profile_by_id["SDB_10002"]
    first, _ = extract(raw, cfg, classifier, computed_at, as_of)
    second, _ = extract(raw, cfg, classifier, computed_at, as_of)
    assert first.model_dump() == second.model_dump()


def test_computed_at_is_injected_not_read(profile_by_id, cfg, classifier, as_of):
    """Two different injected clocks differ *only* in `computed_at`."""
    raw = profile_by_id["SDB_10002"]
    a, _ = extract(raw, cfg, classifier, datetime(2026, 1, 1, tzinfo=timezone.utc), as_of)
    b, _ = extract(raw, cfg, classifier, datetime(2027, 6, 6, tzinfo=timezone.utc), as_of)
    da, db = a.model_dump(), b.model_dump()
    assert da.pop("computed_at") != db.pop("computed_at")
    assert da == db


# --------------------------------------------------------------------------
# Role family
# --------------------------------------------------------------------------
def test_sdb_10020_is_engineering_manager(signals):
    """Recency decay, not duration, decides.

    SDB_10020 is 12 years: Yahoo SWE 46m, Uber Senior ML 45m, Atlassian EM 55m
    and current. Weighted by duration alone the two IC roles outweigh the EM
    role and this person reads as an ML engineer. The 36-month half-life makes
    the current role dominate, which is what a recruiter would say.
    """
    records, _ = signals
    assert records["SDB_10020"].role_family is RoleFamily.ENGINEERING_MANAGER
    assert records["SDB_10020"].seniority is Seniority.MANAGER


def test_sdb_10019_is_not_a_data_role(signals):
    """The headline says Data Science. The work history says six years of AutoCAD.

    This is the case Appendix A calls out. If this ever flips to a data family,
    the per-entry classification has collapsed back into blob classification.
    """
    record = records_for(signals, "SDB_10019")
    assert record.role_family is RoleFamily.NON_ENGINEERING
    assert record.years_relevant == pytest.approx(record.years_total, abs=0.01)
    assert "headline_declares_transition" in record.reason_codes
    assert "zero_role_experience_despite_skill_claim" in record.reason_codes


def test_sdb_10023_hr_person_is_non_engineering(signals):
    """An HR executive who lists Excel must not be scoreable as an engineer."""
    assert records_for(signals, "SDB_10023").role_family is RoleFamily.NON_ENGINEERING


def test_sdb_10009_platform_work_is_devops(signals):
    """"Staff Engineer" is ambiguous; "internal developer platform" is not."""
    record = records_for(signals, "SDB_10009")
    assert record.role_family is RoleFamily.DEVOPS_SRE
    assert record.seniority is Seniority.STAFF_PLUS


def test_ambiguous_titles_abstain_without_context(cfg):
    """A bare ambiguous title with no context must abstain, never guess."""
    from saral.core.extract.title_classifier.lexicon import LexiconClassifier

    lexicon = LexiconClassifier(cfg.lexicon)
    for title in ["Engineer", "Developer", "Consultant", "Member of Technical Staff",
                  "Design Engineer", "SDE"]:
        assert lexicon.classify(title, "").abstained, title


def test_design_engineer_with_cad_context_is_non_engineering(cfg):
    from saral.core.extract.title_classifier.lexicon import LexiconClassifier

    lexicon = LexiconClassifier(cfg.lexicon)
    result = lexicon.classify("Design Engineer", "AutoCAD SolidWorks tolerance stack")
    assert result.family is RoleFamily.NON_ENGINEERING


# --------------------------------------------------------------------------
# Tenure, skills, seniority
# --------------------------------------------------------------------------
def test_sdb_10017_is_a_hopper(signals):
    """Three jobs of 6, 9 and 11 months. If this reads `stable`, tenure is broken."""
    tenure = records_for(signals, "SDB_10017").tenure_stability
    assert tenure.flag == "hopper"
    assert tenure.jobs_last_36m >= 3
    assert tenure.avg_tenure_months < 15


def test_staff_plus_is_unreachable_by_years_alone(signals):
    """`staff+` requires title evidence. Nobody accumulates their way into it."""
    records, _ = signals
    for record in records.values():
        if record.seniority is Seniority.STAFF_PLUS:
            assert record.years_relevant > 0
    # SDB_10011 has 7 relevant years and no staff title -> senior, not staff+.
    assert records["SDB_10011"].seniority is Seniority.SENIOR


def test_skill_evidence_tiering_separates_claim_from_history(signals):
    """The mechanical engineer's data skills are claimed, not evidenced."""
    record = records_for(signals, "SDB_10019")
    assert "autocad" in record.core_skills
    for claimed in ("python", "machine learning", "pandas", "matplotlib"):
        assert claimed in record.claimed_skills_unverified, claimed


def test_skill_noise_ratio_is_claimed_over_declared(signals):
    """Definition check.

    NOTE: this yields 0.857 for SDB_10019, not the 0.57 printed in Appendix A.
    The brief states the appendix values are "illustrative, not a target to
    reproduce"; reaching 0.57 requires crediting `solidworks` and `excel` as
    evidenced when neither appears anywhere in the work history. See
    FAILURE_LOG.md FL-003 -- the divergence is reported rather than fitted away.
    """
    record = records_for(signals, "SDB_10019")
    declared = len(record.core_skills) + len(record.claimed_skills_unverified)
    assert declared == 7
    assert record.skill_noise_ratio == pytest.approx(
        len(record.claimed_skills_unverified) / declared, abs=1e-3
    )


def test_freelancer_overlap_is_not_double_counted(signals):
    """`years_total` is over the union of spans, never the sum of durations."""
    record = records_for(signals, "SDB_10021")
    assert record.years_total <= (35 + 27) / 12 + 0.01


# --------------------------------------------------------------------------
# Contract compliance
# --------------------------------------------------------------------------
def test_all_signal_reason_codes_are_in_vocabulary(signals):
    records, _ = signals
    offenders = {
        cid: validate(r.reason_codes) for cid, r in records.items() if validate(r.reason_codes)
    }
    assert not offenders, offenders


def test_confidence_is_bounded_and_varies(signals):
    records, _ = signals
    values = [r.confidence for r in records.values()]
    assert all(0.05 <= v <= 0.99 for v in values)
    # A confidence signal that is constant is not a signal.
    assert max(values) - min(values) > 0.2


def test_every_candidate_produces_exactly_one_record(signals, profiles):
    records, _ = signals
    assert len(records) == len(profiles) == 25


# --------------------------------------------------------------------------
# Hashing
# --------------------------------------------------------------------------
def test_input_hash_ignores_crawl_metadata(profile_by_id):
    """If crawl metadata is inside the hash, every crawl looks like a change."""
    import copy

    original = profile_by_id["SDB_10001"]
    recrawled = copy.deepcopy(original)
    recrawled["updated_at"] = "2027-01-01T00:00:00Z"
    recrawled["created_at"] = "2027-01-01T00:00:00Z"
    assert input_hash(original) == input_hash(recrawled)


def test_input_hash_detects_a_real_change(profile_by_id):
    import copy

    original = profile_by_id["SDB_10001"]
    changed = copy.deepcopy(original)
    changed["headline"] = "Staff Engineer @ Razorpay"
    assert input_hash(original) != input_hash(changed)


def test_emoji_and_whitespace_edit_is_hash_identical():
    """Appendix A.3's rocket-emoji case, resolved by the normaliser alone.

    If this ever needs a special case in the delta engine, the normaliser is
    wrong and the delta engine is compensating for it.
    """
    before = {"headline": "Machine Learning Engineer | Recommender Systems | Serving at scale"}
    after = {"headline": "Machine Learning Engineer  | Recommender Systems | Serving at scale \U0001F680"}
    assert field_group_hash(before, "headline") == field_group_hash(after, "headline")


def test_skill_reordering_is_not_a_change():
    a = {"skills": ["Python", "Kafka", "Redis"]}
    b = {"skills": ["Redis", "Python", "Kafka"]}
    assert field_group_hash(a, "skills") == field_group_hash(b, "skills")


def test_skill_removal_is_a_change():
    a = {"skills": ["Python", "Kafka", "Redis"]}
    b = {"skills": ["Python", "Kafka"]}
    assert field_group_hash(a, "skills") != field_group_hash(b, "skills")


# --------------------------------------------------------------------------
# Tolerance: a bad row must not kill the batch
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "mutation",
    [
        {"experience": None},
        {"skills": None},
        {"total_experience_months": "72"},
        {"is_open_to_work": "true"},
        {"experience": [{"role": "Engineer", "start_date": "not-a-date"}]},
        {"experience": [{"role": None, "start_date": "2020-01-01", "duration_months": "36"}]},
    ],
    ids=["null-exp", "null-skills", "str-int", "str-bool", "bad-date", "null-role"],
)
def test_malformed_rows_coerce_rather_than_raise(
    profile_by_id, cfg, classifier, computed_at, as_of, mutation
):
    import copy

    raw = copy.deepcopy(profile_by_id["SDB_10001"])
    raw.update(mutation)
    record, _ = extract(raw, cfg, classifier, computed_at, as_of)
    assert record.candidate_id == "SDB_10001"


def records_for(signals, candidate_id):
    records, _ = signals
    return records[candidate_id]
