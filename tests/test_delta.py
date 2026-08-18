"""Part 3: idempotency, materiality, and the traps.

The brief says the delta file "contains a handful of situations designed to
break a naive implementation" and will not say which. The answer to that is to
enumerate what was defended against, so the coverage is legible regardless of
what the file actually turned out to contain.

`test_trap_matrix` is table-driven, one row per scenario, each with its expected
event. Six of the ten scenarios occur in the real feed; four do not, and are
tested against synthetic records so the defence is demonstrated rather than
asserted.
"""

from __future__ import annotations

import copy
import json

import pytest

from saral.core.delta.apply import DeltaState, apply_delta
from saral.core.delta.dependency import DEPENDS, affected_signals, requires_recompute
from saral.core.delta.merge import merge_experience
from saral.pipeline import io

BASE_OBSERVED_AT = "2025-08-01T00:00:00Z"
DELTA_OBSERVED_AT = "2026-08-16T04:00:00Z"


@pytest.fixture
def base_state(profiles):
    return DeltaState.from_profiles(profiles, BASE_OBSERVED_AT)


@pytest.fixture(scope="session")
def delta_records():
    return io.load_delta()


def _events_for(result, candidate_id, field=None):
    return [
        e
        for e in result.events
        if e.candidate_id == candidate_id and (field is None or e.field == field)
    ]


# --------------------------------------------------------------------------
# Idempotency -- the property the brief asks to be proven in a test
# --------------------------------------------------------------------------
def test_applying_twice_produces_identical_state(base_state, delta_records):
    """Applying the same feed twice must leave the same state as applying it once."""
    first = copy.deepcopy(base_state)
    apply_delta(first, delta_records)
    snapshot_one = json.dumps(first.profiles, sort_keys=True, default=str)

    apply_delta(first, delta_records)
    snapshot_two = json.dumps(first.profiles, sort_keys=True, default=str)

    assert snapshot_one == snapshot_two


def test_second_application_emits_no_material_events(base_state, delta_records):
    """Not just the same state -- nothing that would cost money to act on."""
    apply_delta(base_state, delta_records)
    second = apply_delta(base_state, delta_records)

    material = [e for e in second.events if e.materiality != "noise"]
    assert not material, [
        (e.candidate_id, e.field, e.change_type, e.materiality) for e in material
    ]
    assert not second.dirty, "nothing should be dirty on a re-application"


def test_field_state_advances_so_replays_are_stale(base_state, delta_records):
    apply_delta(base_state, delta_records)
    second = apply_delta(base_state, delta_records)
    assert all(e.change_type == "stale_skip" for e in second.events)


# --------------------------------------------------------------------------
# The traps
# --------------------------------------------------------------------------
def test_trap_1_out_of_order_observation_is_rejected_per_field(base_state):
    """An older observation must not overwrite newer state -- at field granularity."""
    fresh = {
        "id": "SDB_10001",
        "_observed_at": "2026-08-16T04:00:00Z",
        "headline": "Staff Engineer @ Razorpay",
    }
    stale = {
        "id": "SDB_10001",
        "_observed_at": "2024-01-01T00:00:00Z",
        "headline": "Junior Engineer @ Nowhere",
        "location": "Pune, Maharashtra, India",
    }
    apply_delta(base_state, [fresh])
    result = apply_delta(base_state, [stale])

    assert base_state.profiles["SDB_10001"]["headline"] == "Staff Engineer @ Razorpay"
    headline_events = _events_for(result, "SDB_10001", "headline")
    assert headline_events[0].change_type == "stale_skip"


def test_trap_1b_partial_record_applies_fresh_fields_and_rejects_stale_ones(base_state):
    """The reason `observed_at` is per field group and not per record.

    Record-level rejection would discard the two fresh fields along with the
    stale one. That is the case a naive implementation gets wrong, and it is why
    `field_state` is keyed by (candidate, field group) rather than by candidate.
    """
    apply_delta(
        base_state,
        [{"id": "SDB_10001", "_observed_at": "2026-08-16T04:00:00Z", "headline": "Staff Engineer"}],
    )
    mixed = {
        "id": "SDB_10001",
        "_observed_at": "2026-08-10T00:00:00Z",  # older than the headline we hold
        "headline": "Should Not Apply",
        "location": "Chennai, Tamil Nadu, India",  # but newer than the base load
        "is_open_to_work": True,
    }
    result = apply_delta(base_state, [mixed])

    profile = base_state.profiles["SDB_10001"]
    assert profile["headline"] == "Staff Engineer", "stale field must be rejected"
    assert profile["location"] == "Chennai, Tamil Nadu, India", "fresh field must apply"
    assert profile["is_open_to_work"] is True, "fresh field must apply"

    kinds = {e.field: e.change_type for e in result.events}
    assert kinds["headline"] == "stale_skip"
    assert kinds["location"] == "field_update"


def test_trap_2_duplicate_candidate_lines_in_one_file(base_state, delta_records):
    """SDB_10009 appears twice in the real feed with identical content and timestamp."""
    ids = [r["id"] for r in delta_records]
    assert ids.count("SDB_10009") == 2, "fixture assumption: the feed duplicates SDB_10009"

    result = apply_delta(base_state, delta_records)
    events = _events_for(result, "SDB_10009", "is_open_to_work")
    assert [e.change_type for e in events] == ["field_update", "stale_skip"]
    assert [e.materiality for e in events] == ["high", "noise"]


def test_trap_3_unknown_candidate_is_created_not_dropped(base_state):
    """Dropping the row would silently lose a real person."""
    result = apply_delta(
        base_state,
        [
            {
                "id": "SDB_99999",
                "_observed_at": DELTA_OBSERVED_AT,
                "headline": "Backend Engineer @ Somewhere",
            }
        ],
    )
    assert result.unknown_candidates == ["SDB_99999"]
    assert "SDB_99999" in base_state.profiles
    assert any(e.change_type == "candidate_created" for e in result.events)


def test_trap_4_null_field_is_not_applied_and_is_visible(base_state, delta_records):
    """A null is 'not observed'. The value survives and an event records the decision."""
    before = base_state.profiles["SDB_10007"]["about"]
    result = apply_delta(base_state, delta_records)

    assert base_state.profiles["SDB_10007"]["about"] == before
    events = _events_for(result, "SDB_10007", "about")
    assert events[0].change_type == "suspected_deletion"
    assert events[0].materiality == "low"
    assert events[0].signals_recomputed is False


def test_trap_5_emoji_and_whitespace_edit_is_noise(base_state, delta_records):
    """Appendix A.3's case, and it must cost nothing.

    Resolved by the normaliser, with no branch in the delta engine. If this ever
    needs a special case, the normaliser is broken.
    """
    result = apply_delta(base_state, delta_records)
    events = _events_for(result, "SDB_10015", "headline")
    assert len(events) == 1
    assert events[0].materiality == "noise"
    assert events[0].signals_recomputed is False
    assert events[0].downstream == []
    assert "SDB_10015" not in result.dirty


def test_trap_6_partial_experience_list_preserves_unseen_entries(base_state, delta_records):
    """The highest-blast-radius trap.

    SDB_10001's delta carries only the two Razorpay roles. Replacing wholesale
    would destroy the Freshworks entry, which destroys years_relevant, which
    destroys seniority, which destroys every downstream score.
    """
    apply_delta(base_state, delta_records)
    companies = [e["company_name"] for e in base_state.profiles["SDB_10001"]["experience"]]
    assert "Freshworks" in companies
    assert len(companies) == 3

    # SDB_10002's Mu Sigma role is likewise absent from the feed.
    assert "Mu Sigma" in [
        e["company_name"] for e in base_state.profiles["SDB_10002"]["experience"]
    ]


def test_trap_7_end_date_filled_in_marks_the_role_ended(base_state, delta_records):
    result = apply_delta(base_state, delta_records)
    razorpay = [
        e
        for e in base_state.profiles["SDB_10001"]["experience"]
        if e["company_name"] == "Razorpay" and e["start_date"] == "2022-04-01"
    ][0]
    assert razorpay["end_date"] == "2026-07-31"
    assert razorpay["is_current"] is False
    # And the description the delta did not carry survived.
    assert "payments ledger" in razorpay["description"]

    events = _events_for(result, "SDB_10001", "experience")
    assert events[0].materiality == "high"


def test_trap_8_open_to_work_flips_in_both_directions(base_state, delta_records):
    result = apply_delta(base_state, delta_records)
    becoming_available = _events_for(result, "SDB_10009", "is_open_to_work")[0]
    going_away = _events_for(result, "SDB_10002", "is_open_to_work")[0]

    assert becoming_available.materiality == "high"
    assert "notify_recruiter" in becoming_available.downstream
    assert going_away.materiality == "medium"
    assert "notify_recruiter" not in going_away.downstream


def test_trap_9_record_with_no_observed_at_cannot_overwrite(base_state):
    before = base_state.profiles["SDB_10001"]["headline"]
    result = apply_delta(
        base_state, [{"id": "SDB_10001", "headline": "Chief Everything Officer"}]
    )
    assert base_state.profiles["SDB_10001"]["headline"] == before
    assert any(e.change_type == "malformed_record" for e in result.events)


def test_trap_10_numeric_fields_arriving_as_strings(base_state, cfg, classifier, computed_at, as_of):
    """Coercion happens in the contract, so the delta engine never sees the mess."""
    from saral.core.extract.pipeline import extract

    apply_delta(
        base_state,
        [
            {
                "id": "SDB_10001",
                "_observed_at": DELTA_OBSERVED_AT,
                "experience": [
                    {
                        "role": "Staff Engineer",
                        "company_name": "Razorpay",
                        "start_date": "2026-08-01",
                        "end_date": None,
                        "is_current": "true",
                        "duration_months": "1",
                    }
                ],
            }
        ],
    )
    record, _ = extract(
        base_state.profiles["SDB_10001"], cfg, classifier, computed_at, as_of
    )
    assert record.candidate_id == "SDB_10001"


# --------------------------------------------------------------------------
# Materiality and the dependency map
# --------------------------------------------------------------------------
def test_location_change_is_material_but_recomputes_no_signal(base_state, delta_records):
    """The dependency map doing real work.

    SDB_10024 moved to Berlin. That invalidates location_fit on every open job,
    so it is high materiality -- but location is a scoring input and never
    enters a SignalRecord, so no signal is recomputed. Recomputing signals here
    would be spending money for nothing.
    """
    result = apply_delta(base_state, delta_records)
    event = _events_for(result, "SDB_10024", "location")[0]
    assert event.materiality == "high"
    assert event.signals_recomputed is False
    assert event.affected_signals == []
    assert DEPENDS["location"] == frozenset()


def test_only_changed_candidates_are_recomputed(base_state, delta_records):
    result = apply_delta(base_state, delta_records)
    assert set(result.dirty) == {
        "SDB_10001", "SDB_10002", "SDB_10003", "SDB_10009", "SDB_10010",
        "SDB_10024", "SDB_10025",
    }
    # 25 candidates in the corpus, and the feed touched 9 of them.
    assert len(result.dirty) < 25


def test_dependency_map_covers_every_tracked_field_group():
    from saral.core.hashing import FIELD_GROUPS

    assert set(DEPENDS) == set(FIELD_GROUPS)


@pytest.mark.parametrize(
    "group,expected",
    [
        ("is_open_to_work", True),
        ("headline", True),
        ("experience", True),
        ("location", False),
    ],
)
def test_recompute_is_triggered_only_by_signal_bearing_fields(group, expected):
    assert requires_recompute({group}) is expected


def test_experience_merge_never_deletes():
    stored = [
        {"company_name": "A Ltd", "start_date": "2020-01-01", "role": "Engineer",
         "description": "did things", "is_current": False},
        {"company_name": "B Inc", "start_date": "2022-01-01", "role": "Senior Engineer",
         "description": "did more things", "is_current": True},
    ]
    observed = [
        {"company_name": "B Inc", "start_date": "2022-01-01", "role": "Senior Engineer",
         "end_date": "2026-01-01", "is_current": False, "description": None},
    ]
    merged = merge_experience(stored, observed)
    assert len(merged) == 2
    b = [e for e in merged if e["company_name"] == "B Inc"][0]
    assert b["end_date"] == "2026-01-01"
    assert b["is_current"] is False
    assert b["description"] == "did more things", "a null must not wipe a stored value"


def test_experience_merge_key_normalises_company_names():
    stored = [{"company_name": "Zeta Pvt Ltd", "start_date": "2022-01-01", "role": "Engineer"}]
    observed = [{"company_name": "Zeta", "start_date": "2022-01-01", "role": "Senior Engineer"}]
    merged = merge_experience(stored, observed)
    assert len(merged) == 1, "Zeta and Zeta Pvt Ltd are one company"
    assert merged[0]["role"] == "Senior Engineer"


def test_affected_signals_reports_what_the_map_says():
    assert "switch_intent" in affected_signals({"is_open_to_work"})
    assert "years_relevant" in affected_signals({"experience"})
    assert affected_signals({"location"}) == set()


# --------------------------------------------------------------------------
# Reproducibility of the emitted artefact
# --------------------------------------------------------------------------
def test_event_ids_are_content_addressed_not_random(base_state, delta_records):
    """Two runs over identical input must emit byte-identical events.

    A `uuid4` here would break the reproducibility claim for
    `out/change_events.jsonl` and, worse, defeat idempotency at the storage
    layer: re-applying a feed would insert the same logical event under a new
    primary key, so `INSERT OR IGNORE` could never dedupe it.
    """
    import copy as _copy

    first = apply_delta(_copy.deepcopy(base_state), delta_records)
    second = apply_delta(_copy.deepcopy(base_state), delta_records)

    assert [e.event_id for e in first.events] == [e.event_id for e in second.events]
    assert [e.model_dump() for e in first.events] == [e.model_dump() for e in second.events]


def test_event_ids_are_unique_within_a_run(base_state, delta_records):
    result = apply_delta(base_state, delta_records)
    ids = [e.event_id for e in result.events]
    assert len(ids) == len(set(ids))


def test_reinserting_the_same_events_is_a_storage_noop(tmp_path, base_state, delta_records):
    """The property content-addressed ids exist to give."""
    from saral.adapters.store.sqlite_repo import SqliteRepo

    repo = SqliteRepo(tmp_path / "t.db")
    result = apply_delta(base_state, delta_records)
    with repo.transaction():
        repo.record_events(result.events)
    once = repo.event_count()
    with repo.transaction():
        repo.record_events(result.events)
    assert repo.event_count() == once
    repo.close()
