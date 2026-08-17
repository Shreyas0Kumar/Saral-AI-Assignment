"""The create pass: raw profiles -> out/candidate_signals.jsonl."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any

from saral.config_loader import load_all
from saral.contracts.models import SignalRecord
from saral.core.extract.pipeline import ExtractionTrace, extract
from saral.pipeline.arms import SHIPPED_ARM, build_classifier
from saral.pipeline.io import DEFAULT_COMPUTED_AT, OUT_DIR, corpus_as_of, write_jsonl
from saral.telemetry.stages import Telemetry


def run_extract(
    profiles: list[dict[str, Any]],
    arm: str = SHIPPED_ARM,
    computed_at: datetime | None = None,
    as_of: date | None = None,
    telemetry: Telemetry | None = None,
    config_dir: str | None = None,
) -> tuple[list[SignalRecord], dict[str, ExtractionTrace], Telemetry]:
    cfg, _weights = load_all(config_dir)
    classifier = build_classifier(arm, cfg)
    computed_at = computed_at or DEFAULT_COMPUTED_AT
    as_of = as_of or corpus_as_of(profiles)
    telemetry = telemetry or Telemetry()

    records: list[SignalRecord] = []
    traces: dict[str, ExtractionTrace] = {}

    with telemetry.stage("extract", records_in=len(profiles)) as stage:
        fallback_total = 0
        abstain_total = 0
        titles_total = 0
        for profile in profiles:
            record, trace = extract(profile, cfg, classifier, computed_at, as_of)
            records.append(record)
            traces[record.candidate_id] = trace
            fallback_total += classifier.fallback_invocations()
            abstain_total += classifier.abstentions
            titles_total += sum(classifier.invocations.values()) + classifier.abstentions
        # Accumulated to match `wall_ms`, which also accumulates. Dividing an
        # accumulated wall time by a non-accumulated record count is how a
        # cost-per-million figure silently doubles.
        stage.records_out += len(records)
        # Assigned, not accumulated. These describe one pass over the corpus,
        # and `run_full_evaluation` extracts again inside the same telemetry
        # object -- accumulating would silently double every reported rate's
        # numerator and denominator.
        stage.counters["titles_classified"] = titles_total
        stage.counters["fallback_invocations"] = fallback_total
        stage.counters["lexicon_abstentions"] = abstain_total
        stage.counters["fallback_rate"] = (
            round(fallback_total / titles_total, 4) if titles_total else 0.0
        )
        stage.counters["lexicon_abstention_rate"] = (
            round(abstain_total / titles_total, 4) if titles_total else 0.0
        )

    return records, traces, telemetry


def write_signals(records: list[SignalRecord], path: Path | None = None) -> int:
    return write_jsonl(path or OUT_DIR / "candidate_signals.jsonl", records)
