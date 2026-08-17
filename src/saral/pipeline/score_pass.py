"""The scoring pass: signals + jobs -> out/rankings.jsonl."""

from __future__ import annotations

from pathlib import Path

from saral.config_loader import load_all
from saral.contracts.models import JobSpec, RankingRecord
from saral.core.score.features import ParsedJob, parse_job
from saral.core.score.scorer import ScoringFlags, rank_candidates
from saral.pipeline.arms import SHIPPED_ARM, build_classifier
from saral.pipeline.io import OUT_DIR, write_jsonl
from saral.telemetry.stages import Telemetry


def parse_jobs(jobs: list[JobSpec], config_dir: str | None = None) -> list[ParsedJob]:
    cfg, _ = load_all(config_dir)
    classifier = build_classifier(SHIPPED_ARM, cfg)
    return [parse_job(job, cfg, classifier) for job in jobs]


def run_scoring(
    signals: list,
    traces: dict,
    jobs: list[JobSpec],
    flags: ScoringFlags | None = None,
    telemetry: Telemetry | None = None,
    config_dir: str | None = None,
) -> dict[str, list[RankingRecord]]:
    cfg, weights = load_all(config_dir)
    telemetry = telemetry or Telemetry()
    parsed = parse_jobs(jobs, config_dir)

    rankings: dict[str, list[RankingRecord]] = {}
    with telemetry.stage("score", records_in=len(signals) * len(jobs)) as stage:
        for job in parsed:
            rankings[job.spec.job_id] = rank_candidates(
                signals, traces, job, cfg, weights, flags
            )
        stage.records_out += sum(len(r) for r in rankings.values())
    return rankings


def write_rankings(
    rankings: dict[str, list[RankingRecord]], path: Path | None = None
) -> int:
    ordered = [record for job_id in sorted(rankings) for record in rankings[job_id]]
    return write_jsonl(path or OUT_DIR / "rankings.jsonl", ordered)
