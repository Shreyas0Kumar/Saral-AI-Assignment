"""Assemble out/metrics.json.

Ordering of this file mirrors how much each block should be trusted: headline
numbers first, then what explains them, then how uncertain they are, then where
the system is wrong. The uncertainty block is not optional and it is not last
because it is least important -- it is last because it is the conclusion.
"""

from __future__ import annotations

import json

from saral.contracts.versions import SCORING_VERSION, SIGNALS_VERSION
from saral.core.score.scorer import ScoringFlags
from saral.pipeline import io
from saral.pipeline.arms import ARMS, SHIPPED_ARM
from saral.pipeline.eval_pass import (
    ABLATION_RUNGS,
    evaluate_rankings,
    labels_by_job,
    signals_rankings,
    uncertainty_block,
    worst_misses,
)
from saral.pipeline.extract_pass import run_extract
from saral.pipeline.score_pass import run_scoring
from saral.telemetry.manifest import machine_info
from saral.telemetry.stages import Telemetry


def _latency_block(profiles: list[dict], arm: str) -> dict:
    """Measured extract latency for batch sizes 1 and 100. Warmed up first."""
    import time

    from saral.config_loader import load_all
    from saral.core.extract.pipeline import extract
    from saral.pipeline.arms import build_classifier

    cfg, _ = load_all()
    classifier = build_classifier(arm, cfg)
    computed_at = io.DEFAULT_COMPUTED_AT
    as_of = io.corpus_as_of(profiles)

    for profile in profiles[:5]:  # warm up: import paths, regex compilation, caches
        extract(profile, cfg, classifier, computed_at, as_of)

    def measure(batch_size: int, iterations: int) -> dict:
        timings: list[float] = []
        for i in range(iterations):
            batch = [profiles[(i + j) % len(profiles)] for j in range(batch_size)]
            t0 = time.perf_counter()
            for profile in batch:
                extract(profile, cfg, classifier, computed_at, as_of)
            timings.append((time.perf_counter() - t0) * 1000)
        timings.sort()
        return {
            "p50": round(timings[len(timings) // 2], 3),
            "p95": round(timings[min(len(timings) - 1, int(len(timings) * 0.95))], 3),
            "iterations": iterations,
        }

    per_record = measure(1, 500)
    return {
        "extract": {
            "batch_1": per_record,
            "batch_100": measure(100, 30),
            "unit": "milliseconds for the whole batch",
            "note": (
                "batch_1 is per-record. The distribution is bimodal by design: the "
                "lexicon path is microseconds, and the distilled-LR fallback loads "
                "lazily on first miss, so p95 exposes the fallback cost rather than "
                "hiding it in a flat mean."
            ),
        },
        "machine": machine_info(),
    }


def run_full_evaluation(
    skip_baseline: bool = False, telemetry: Telemetry | None = None
) -> dict:
    telemetry = telemetry or Telemetry()
    profiles = io.load_candidates()
    jobs = io.load_jobs()
    labels = io.load_labels()
    grouped = labels_by_job(labels)

    records, traces, telemetry = run_extract(profiles, arm=SHIPPED_ARM, telemetry=telemetry)
    signals = {r.candidate_id: r for r in records}

    systems: dict[str, dict] = {}
    arms: dict[str, object] = {}

    # -- baseline ----------------------------------------------------------
    baseline_result = None
    if not skip_baseline:
        try:
            from saral.adapters.embed.baseline import MODEL_NAME, rank_by_cosine

            with telemetry.stage("baseline_embed", records_in=len(profiles)) as stage:
                baseline_rankings = rank_by_cosine(jobs, profiles, io.OUT_DIR)
                stage.records_out = len(profiles)
            baseline_result = evaluate_rankings("baseline_cosine", baseline_rankings, grouped)
            arms["baseline_cosine"] = baseline_result
            systems["baseline_cosine"] = {
                "model": MODEL_NAME,
                "description": ARMS["baseline_cosine"].description,
                **_arm_payload(baseline_result),
            }
        except Exception as exc:  # pragma: no cover - reviewer without torch
            systems["baseline_cosine"] = {
                "error": f"{type(exc).__name__}: {exc}",
                "note": "install the eval extra (pip install -e '.[eval]') to run the baseline",
            }

    # -- the shipped system ------------------------------------------------
    ranked_records = run_scoring(records, traces, jobs, telemetry=telemetry)
    system_rankings = {
        job_id: [r.candidate_id for r in rows] for job_id, rows in ranked_records.items()
    }
    system_result = evaluate_rankings(SHIPPED_ARM, system_rankings, grouped)
    arms[SHIPPED_ARM] = system_result
    systems[SHIPPED_ARM] = {
        "description": ARMS[SHIPPED_ARM].description,
        "shipped": True,
        **_arm_payload(system_result),
    }

    # -- ablation ladder ---------------------------------------------------
    ablation = {}
    for level, name, description in ABLATION_RUNGS:
        rankings = signals_rankings(records, traces, jobs, ScoringFlags.rung(level))
        result = evaluate_rankings(name, rankings, grouped)
        ablation[f"{level}_{name}"] = {
            "description": description,
            "ndcg@10": result.means["ndcg@10"],
            "precision@5": result.means["precision@5"],
            "per_job_ndcg@10": {j: m["ndcg@10"] for j, m in result.per_job.items()},
        }
    rungs = list(ablation.values())
    for previous, current in zip(rungs, rungs[1:]):
        current["delta_ndcg@10_vs_previous_rung"] = round(
            current["ndcg@10"] - previous["ndcg@10"], 4
        )

    payload: dict = {
        "run_id": io.DEFAULT_COMPUTED_AT.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "versions": {"signals": SIGNALS_VERSION, "scoring": SCORING_VERSION},
        "labelled_pairs": len(labels),
        "jobs": len(jobs),
        "candidates_ranked_per_job": len(profiles),
        "list_conventions": {
            "primary": "condensed -- rank all 25, then keep labelled candidates in order",
            "also_reported": "zero_fill -- unlabelled candidates scored as relevance 0",
            "why": (
                "Zero-fill penalises a system for surfacing a good candidate the "
                "recruiter never graded, which is exactly what this system is built to "
                "do. Condensed lists flatter every system equally, so they compare "
                "ordering quality but cannot be used to claim a win. Both are shown so "
                "the reader can check the ranking does not flip between conventions."
            ),
        },
        "metric_ceilings": {
            job_id: {
                "n_labelled": len(labels_map),
                "n_relevant_at_threshold_2": sum(1 for r in labels_map.values() if r >= 2),
                "precision@5_ceiling": round(
                    min(sum(1 for r in labels_map.values() if r >= 2), 5) / 5, 2
                ),
            }
            for job_id, labels_map in sorted(grouped.items())
        },
        "ceiling_note": (
            "JD-004 has only 2 candidates graded >= 2, so its Precision@5 cannot exceed "
            "0.40 for any system. Averaging Precision@5 across jobs without disclosing "
            "that is quietly misleading, which is why the per-job ceilings are printed "
            "next to the per-job values."
        ),
        "metric_saturation": {
            "note": (
                "NDCG@10 barely discriminates on this dataset. Each job has 10-16 "
                "labelled candidates, so a condensed @10 cut covers most or all of the "
                "list and mostly measures whether the tail is ordered. MRR is 1.00 for "
                "both the baseline and the shipped system -- both put a relevant "
                "candidate first in all four jobs -- so it carries no information here "
                "and is reported only to show that it is saturated rather than omitted. "
                "The metrics that actually separate the systems on this data are "
                "NDCG@5 and Precision@5."
            ),
            "mrr_is_saturated": True,
        },
        "systems": systems,
        "ablation_ladder": ablation,
        "ablation_note": (
            "Rung 3 (the evidence-vs-claim discount) moves NDCG@10 by 0.000. That is "
            "reported as zero rather than dropped. The rule changes scores and reason "
            "codes on the profiles it targets (SDB_10010, SDB_10019, SDB_10023), but "
            "those candidates are already ranked low by role adjacency at rung 1, so on "
            "these four jobs it has nothing left to fix. Its value is a claim about "
            "corpora containing many such profiles, and this corpus contains three."
        ),
    }

    if baseline_result is not None:
        payload["uncertainty"] = uncertainty_block(baseline_result, system_result, grouped)
    else:
        payload["uncertainty"] = {
            "verdict": "baseline not run, so no comparison was made",
        }

    payload["error_analysis"] = {
        "method": (
            "The three labelled pairs giving up the most discounted gain against the "
            "ideal ordering. Nothing was tuned after reading this section -- identified "
            "fixes are listed as future work in WRITEUP.md."
        ),
        "worst_misses": worst_misses(system_result, grouped, ranked_records, signals),
    }

    payload["latency_ms"] = _latency_block(profiles, SHIPPED_ARM)
    payload["telemetry"] = telemetry.to_dict()

    fallback_report = io.OUT_DIR / "fallback_comparison.json"
    if fallback_report.exists():
        payload["fallback_comparison"] = json.loads(fallback_report.read_text(encoding="utf-8"))
    cost_arm = io.OUT_DIR / "llm_cost_arm.json"
    if cost_arm.exists():
        payload["llm_per_row_cost_arm"] = json.loads(cost_arm.read_text(encoding="utf-8"))

    return payload


def _arm_payload(result) -> dict:
    return {
        "ndcg@10": {"mean": result.means["ndcg@10"], "per_job": {j: m["ndcg@10"] for j, m in result.per_job.items()}},
        "ndcg@5": {"mean": result.means["ndcg@5"], "per_job": {j: m["ndcg@5"] for j, m in result.per_job.items()}},
        "precision@5": {"mean": result.means["precision@5"], "per_job": {j: m["precision@5"] for j, m in result.per_job.items()}},
        "mrr": {"mean": result.means["mrr"]},
        "recall@10": {"mean": result.means["recall@10"]},
        "zero_fill": {
            "ndcg@10": {"mean": result.means_zero_fill["ndcg@10"], "per_job": {j: m["ndcg@10"] for j, m in result.per_job_zero_fill.items()}},
            "precision@5": {"mean": result.means_zero_fill["precision@5"]},
        },
    }
