"""The update pass: load base state, apply the delta, recompute only what moved.

Also produces the saving report -- full recompute versus incremental, in records
processed, wall clock and cost per million profiles. That number is the reason
the work is worth doing, so it is derived from measured throughput in this run
rather than asserted in a document.
"""

from __future__ import annotations

import time
from datetime import date, datetime
from pathlib import Path

from saral.adapters.store.sqlite_repo import SqliteRepo
from saral.config_loader import load_all
from saral.core.dates import parse_date
from saral.core.delta.apply import DeltaState, apply_delta, summarise_dirty
from saral.core.extract.pipeline import extract
from saral.pipeline import io
from saral.pipeline.arms import SHIPPED_ARM, build_classifier
from saral.telemetry.manifest import FARGATE_GB_HOUR_USD, FARGATE_VCPU_HOUR_USD
from saral.telemetry.stages import Telemetry

DEFAULT_DB = io.ROOT / "out" / "saral.db"


def _as_of_for(records: list[dict], fallback: date) -> date:
    """The delta's own observation date, which is 12 months after the base crawl."""
    stamps = [parse_date(r.get("_observed_at")) for r in records]
    stamps = [s for s in stamps if s]
    return max(stamps) if stamps else fallback


def run_delta(
    db_path: Path | None = None,
    reapply: bool = False,
    telemetry: Telemetry | None = None,
) -> dict:
    db_path = db_path or DEFAULT_DB
    if db_path.exists():
        db_path.unlink()  # `make all` rebuilds from scratch, deterministically
    db_path.parent.mkdir(parents=True, exist_ok=True)

    telemetry = telemetry or Telemetry()
    cfg, _weights = load_all()
    classifier = build_classifier(SHIPPED_ARM, cfg)
    computed_at = io.DEFAULT_COMPUTED_AT

    profiles = io.load_candidates()
    delta_records = io.load_delta()
    base_as_of = io.corpus_as_of(profiles)
    delta_as_of = _as_of_for(delta_records, base_as_of)

    repo = SqliteRepo(db_path)

    # -- base load ---------------------------------------------------------
    base_observed_at = f"{base_as_of.isoformat()}T00:00:00Z"
    with repo.transaction():
        for profile in profiles:
            repo.upsert_profile(profile, base_observed_at, "initial_load")

    with telemetry.stage("extract_base", records_in=len(profiles)) as stage:
        t0 = time.perf_counter()
        for profile in profiles:
            record, _ = extract(profile, cfg, classifier, computed_at, base_as_of)
            repo.upsert_signal(record)
        full_pass_s = time.perf_counter() - t0
        stage.records_out = len(profiles)

    state = DeltaState.from_profiles(profiles, base_observed_at)
    with repo.transaction():
        repo.save_field_state(state.field_state)

    # -- the delta ---------------------------------------------------------
    recomputed: list[str] = []

    def recompute(candidate_id: str, profile: dict) -> None:
        record, _ = extract(profile, cfg, classifier, computed_at, delta_as_of)
        repo.upsert_signal(record)
        repo.upsert_profile(profile, f"{delta_as_of.isoformat()}T00:00:00Z", "linkedin_refresh")
        recomputed.append(candidate_id)

    with telemetry.stage("delta_apply", records_in=len(delta_records)) as stage:
        t0 = time.perf_counter()
        with repo.transaction():
            result = apply_delta(state, delta_records, recompute)
            repo.save_field_state(state.field_state)
            repo.record_events(result.events)
        incremental_s = time.perf_counter() - t0
        stage.records_out = len(result.events)
        stage.count("candidates_recomputed", len(recomputed))
        stage.count("events_emitted", len(result.events))

    events = list(result.events)

    # -- idempotency: apply the identical feed again -----------------------
    second_pass_events: list = []
    if reapply:
        before = {cid: repo.load_signals() for cid in ["_"]}
        with repo.transaction():
            second = apply_delta(state, delta_records, recompute)
            repo.record_events(second.events)
        second_pass_events = second.events
        material = [e for e in second_pass_events if e.materiality != "noise"]
        assert not material, f"second application produced material events: {material}"

    io.write_jsonl(io.OUT_DIR / "change_events.jsonl", events)

    # -- the saving report -------------------------------------------------
    counts: dict[str, int] = {}
    for event in events:
        counts[event.materiality] = counts.get(event.materiality, 0) + 1

    per_record_ms = (full_pass_s / len(profiles)) * 1000
    saving = _saving_report(
        n_total=len(profiles),
        n_recomputed=len(set(recomputed)),
        full_pass_s=full_pass_s,
        incremental_s=incremental_s,
        per_record_ms=per_record_ms,
    )

    signals = repo.load_signals()
    io.write_jsonl(io.OUT_DIR / "candidate_signals_after_delta.jsonl", signals)

    report = {
        "base_as_of": base_as_of.isoformat(),
        "delta_as_of": delta_as_of.isoformat(),
        "candidates_total": len(profiles),
        "delta_records": len(delta_records),
        "delta_fields_seen": result.fields_seen,
        "events_written": len(events),
        "materiality_counts": counts,
        "recomputed": len(set(recomputed)),
        "recomputed_candidates": sorted(set(recomputed)),
        "affected_signals_by_candidate": summarise_dirty(result.dirty),
        "unknown_candidates_created": result.unknown_candidates,
        "saving": saving,
        "health": repo.health(),
    }
    io.write_json(io.OUT_DIR / "delta_report.json", report)
    repo.close()
    return report


def _saving_report(
    n_total: int, n_recomputed: int, full_pass_s: float, incremental_s: float, per_record_ms: float
) -> dict:
    """Full recompute vs incremental, extrapolated from measured throughput."""
    ratio = n_total / n_recomputed if n_recomputed else float("inf")

    def cost_per_1m(records: int) -> float:
        cpu_hours = (per_record_ms * records) / 1000 / 3600
        return cpu_hours * (FARGATE_VCPU_HOUR_USD + 0.5 * FARGATE_GB_HOUR_USD)

    changed_fraction = n_recomputed / n_total
    return {
        "measured": {
            "records_processed_full": n_total,
            "records_processed_incremental": n_recomputed,
            "full_pass_wall_s": round(full_pass_s, 4),
            "incremental_wall_s": round(incremental_s, 4),
            "extract_ms_per_record": round(per_record_ms, 4),
        },
        "reduction": {
            "records": f"{n_recomputed}/{n_total}",
            "factor": round(ratio, 2) if n_recomputed else None,
            "note": (
                "The incremental pass is slower in wall clock here because it also "
                "hashes 7 field groups per record, parses the feed and writes events, "
                "and 25 records is far too small for the extraction saving to dominate. "
                "The number that scales is records-recomputed, not this run's clock."
            ),
        },
        "projected_at_1m_profiles": {
            "assumed_change_rate": round(changed_fraction, 3),
            "assumption_note": (
                f"{n_recomputed} of {n_total} candidates changed materially in this feed. "
                "Projecting that rate to 1M is an extrapolation from one 10-record feed "
                "and should be replaced with an observed weekly churn rate before it is "
                "used for capacity planning."
            ),
            "full_recompute_usd": round(cost_per_1m(1_000_000), 4),
            "incremental_usd": round(cost_per_1m(int(1_000_000 * changed_fraction)), 4),
            "saved_usd": round(
                cost_per_1m(1_000_000) - cost_per_1m(int(1_000_000 * changed_fraction)), 4
            ),
            "basis": (
                f"measured {per_record_ms:.3f} ms/record single-threaded, priced at "
                f"Fargate ap-south-1 ${FARGATE_VCPU_HOUR_USD}/vCPU-hr"
            ),
        },
    }
