"""Apply a change feed to stored state, idempotently.

The algorithm, and why each line is there:

    group records by candidate_id, sort each group by _observed_at ascending
    for each record:
      for each field group present in the record:
        if _observed_at <= field_state.observed_at:  -> stale_skip, continue
        if value is null:                            -> suspected_deletion, continue
        if normalised hash == stored hash:           -> noise, continue
        apply the group's merge policy
        update field_state (hash and observed_at)
        mark the candidate dirty via the dependency map
      if dirty: recompute this candidate's signals only

**Idempotency is a property of the `<=` comparison, not a feature bolted on.**
Re-applying a feed whose timestamps equal the stored state skips every field,
emits zero events, and leaves a byte-identical dump. That also handles the
duplicate-line and out-of-order traps with the same line of code.

**`observed_at` is per field group, not per record.** A partial record can carry
one stale field and two fresh ones; record-level rejection would discard the
fresh two. See DECISIONS.md D11.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable

from saral.contracts.models import ChangeEvent
from saral.core.delta.dependency import DEPENDS, affected_signals, requires_recompute
from saral.core.delta.materiality import (
    classify,
    deletion_event,
    downstream_for,
    noise_event,
    stale_event,
)
from saral.core.delta.merge import apply_field, entry_key
from saral.core.hashing import FIELD_GROUPS, canonical_json, field_group_hash

#: Keys the crawler adds that are not profile content.
_META = {"_observed_at", "_source", "id"}


@dataclass
class FieldState:
    normalized_hash: str
    observed_at: str


@dataclass
class DeltaState:
    """Everything the apply loop reads and writes. Pure data, no I/O."""

    profiles: dict[str, dict]
    field_state: dict[tuple[str, str], FieldState] = field(default_factory=dict)

    @classmethod
    def from_profiles(cls, profiles: list[dict], observed_at: str) -> "DeltaState":
        state = cls(profiles={p["id"]: copy.deepcopy(p) for p in profiles})
        for profile in state.profiles.values():
            for group in FIELD_GROUPS:
                state.field_state[(profile["id"], group)] = FieldState(
                    normalized_hash=field_group_hash(profile, group),
                    observed_at=observed_at,
                )
        return state


@dataclass
class DeltaResult:
    events: list[ChangeEvent]
    dirty: dict[str, set[str]]
    unknown_candidates: list[str]
    records_seen: int
    fields_seen: int


def _event_id(
    candidate_id: str, observed_at: str, change_type: str, field_name: str, new_value: Any
) -> str:
    """Content-addressed, not random.

    A `uuid4` here would make two runs over identical input produce different
    bytes, which breaks the reproducibility claim for `out/change_events.jsonl`.
    Worse, it would defeat idempotency at the *storage* layer: re-applying a feed
    would insert the same logical event again under a new primary key, so
    `INSERT OR IGNORE` could never dedupe it. Deriving the id from the event's
    own content makes re-insertion a genuine no-op.
    """
    digest = hashlib.sha256(
        canonical_json([candidate_id, observed_at, change_type, field_name, new_value]).encode()
    ).hexdigest()
    return f"evt_{digest[:22]}"


def _event(
    candidate_id: str,
    observed_at: str,
    source: str | None,
    change_type: str,
    field_name: str,
    old_value: Any,
    new_value: Any,
    materiality: str,
    note: str,
    recomputed: bool = False,
    affected: set[str] | None = None,
) -> ChangeEvent:
    return ChangeEvent(
        event_id=_event_id(candidate_id, observed_at, change_type, field_name, new_value),
        candidate_id=candidate_id,
        observed_at=observed_at,
        source=source,
        change_type=change_type,
        field=field_name,
        old_value=old_value,
        new_value=new_value,
        materiality=materiality,  # type: ignore[arg-type]
        downstream=downstream_for(materiality),  # type: ignore[arg-type]
        signals_recomputed=recomputed,
        note=note,
        affected_signals=sorted(affected or set()),
    )


def _experience_change_kind(stored: list[dict], observed: list[dict]) -> str:
    stored_keys = {entry_key(e) for e in stored}
    for entry in observed:
        if entry_key(entry) not in stored_keys:
            return "new_role"
    for entry in observed:
        key = entry_key(entry)
        for existing in stored:
            if entry_key(existing) == key:
                if existing.get("is_current") and entry.get("end_date"):
                    return "current_role_ended"
    return "entry_updated"


def _summarise(group: str, value: Any) -> Any:
    """Keep event payloads readable. A 40-entry experience list is not a value."""
    if group == "experience" and isinstance(value, list):
        return [
            f"{e.get('role')} @ {e.get('company_name')} "
            f"({e.get('start_date')}..{e.get('end_date') or 'present'})"
            for e in value
        ]
    if group == "education" and isinstance(value, list):
        return [f"{e.get('degree')} @ {e.get('school_name')}" for e in value]
    return value


def _observed_at_of(record: dict) -> str:
    """A record with no `_observed_at` is treated as the oldest possible.

    That way it can never overwrite something we already know, but it is still
    visible as an event rather than silently dropped.
    """
    return record.get("_observed_at") or ""


def apply_delta(
    state: DeltaState,
    delta_records: list[dict],
    recompute: Callable[[str, dict], None] | None = None,
) -> DeltaResult:
    """Apply `delta_records` to `state` in place. Returns the events emitted."""
    events: list[ChangeEvent] = []
    dirty: dict[str, set[str]] = {}
    unknown: list[str] = []
    fields_seen = 0

    by_candidate: dict[str, list[dict]] = {}
    for record in delta_records:
        candidate_id = record.get("id")
        if not candidate_id:
            continue
        by_candidate.setdefault(candidate_id, []).append(record)

    for candidate_id in sorted(by_candidate):
        records = sorted(by_candidate[candidate_id], key=_observed_at_of)

        if candidate_id not in state.profiles:
            # A candidate we have never seen. Treat the partial record as a new
            # profile rather than dropping it -- dropping would silently lose a
            # real person, and a thin profile is visible in `confidence`.
            unknown.append(candidate_id)
            state.profiles[candidate_id] = {"id": candidate_id}
            events.append(
                _event(
                    candidate_id, _observed_at_of(records[0]), records[0].get("_source"),
                    "candidate_created", "_record", None, None, "medium",
                    "candidate_id not present in the base load; created from the partial record",
                )
            )

        for record in records:
            observed_at = _observed_at_of(record)
            source = record.get("_source")
            if not observed_at:
                events.append(
                    _event(
                        candidate_id, "", source, "malformed_record", "_observed_at",
                        None, None, "low",
                        "record carries no _observed_at; treated as oldest possible so it "
                        "cannot overwrite known state",
                    )
                )

            for group in [g for g in FIELD_GROUPS if g in record]:
                fields_seen += 1
                key = (candidate_id, group)
                stored_state = state.field_state.get(key)
                stored_profile = state.profiles[candidate_id]
                old_value = stored_profile.get(group)
                new_value = record[group]

                # 1. stale, or a re-application of what we already have
                if stored_state and observed_at <= stored_state.observed_at:
                    materiality, note = stale_event()
                    events.append(
                        _event(
                            candidate_id, observed_at, source, "stale_skip", group,
                            _summarise(group, old_value), _summarise(group, new_value),
                            materiality, note,
                        )
                    )
                    continue

                # 2. null means not observed
                if new_value is None:
                    materiality, note = deletion_event()
                    events.append(
                        _event(
                            candidate_id, observed_at, source, "suspected_deletion", group,
                            _summarise(group, old_value), None, materiality, note,
                        )
                    )
                    # Advance `observed_at` while leaving the hash alone. We did
                    # observe this field at this time; we decided not to apply
                    # what we saw. Recording that is what makes a re-run a
                    # stale_skip instead of re-emitting the same deletion event,
                    # and it is also the state a corroboration policy would need
                    # in order to notice the null arriving twice.
                    if stored_state is not None:
                        state.field_state[key] = FieldState(
                            stored_state.normalized_hash, observed_at
                        )
                    continue

                # 3. normalised-identical: noise, by construction
                candidate_view = dict(stored_profile)
                candidate_view[group] = new_value
                new_hash = field_group_hash(candidate_view, group)
                if stored_state and new_hash == stored_state.normalized_hash:
                    materiality, note = noise_event()
                    events.append(
                        _event(
                            candidate_id, observed_at, source, "field_update", group,
                            _summarise(group, old_value), _summarise(group, new_value),
                            materiality, note,
                        )
                    )
                    # The observation is still fresh; record that we have seen it
                    # so a re-run does not re-emit the same noise event.
                    state.field_state[key] = FieldState(new_hash, observed_at)
                    continue

                # 4. a real change: apply the group's merge policy
                experience_kind = (
                    _experience_change_kind(stored_profile.get("experience") or [], new_value)
                    if group == "experience"
                    else None
                )
                updated = apply_field(stored_profile, group, new_value)
                state.profiles[candidate_id] = updated
                state.field_state[key] = FieldState(
                    field_group_hash(updated, group), observed_at
                )

                materiality, note = classify(
                    group, old_value, new_value, experience_change=experience_kind
                )
                affected = DEPENDS.get(group, frozenset())
                will_recompute = bool(affected)
                if will_recompute:
                    dirty.setdefault(candidate_id, set()).add(group)

                events.append(
                    _event(
                        candidate_id, observed_at, source, "field_update", group,
                        _summarise(group, old_value),
                        _summarise(group, updated.get(group)),
                        materiality, note, will_recompute, set(affected),
                    )
                )

    # 5. recompute dirty candidates only
    if recompute is not None:
        for candidate_id in sorted(dirty):
            recompute(candidate_id, state.profiles[candidate_id])

    return DeltaResult(
        events=events,
        dirty=dirty,
        unknown_candidates=unknown,
        records_seen=len(delta_records),
        fields_seen=fields_seen,
    )


def summarise_dirty(dirty: dict[str, set[str]]) -> dict[str, list[str]]:
    return {
        candidate_id: sorted(affected_signals(groups))
        for candidate_id, groups in sorted(dirty.items())
    }
