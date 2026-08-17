"""The evaluation pass: everything that ends up in out/metrics.json.

Produces, in order of how much they should be trusted:

1. per-job metrics for every arm, both list conventions, with ceilings stated;
2. the ablation ladder, which attributes the difference to specific rules;
3. uncertainty -- bootstrap CI, per-job deltas shown individually, permutation
   test, and a plain-language verdict;
4. error analysis of the three worst misses.

Nothing here is allowed to change a weight or a threshold. This module reads
the system; it does not get to argue with it.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from saral.contracts.models import JobSpec, SignalRecord
from saral.core.evaluation.bootstrap import bootstrap_delta, paired_permutation, verdict
from saral.core.evaluation.metrics import aggregate, evaluate_job, ndcg_at_k
from saral.core.score.scorer import ScoringFlags
from saral.pipeline.score_pass import run_scoring

ABLATION_RUNGS = [
    (1, "role_family_prefilter_only", "role adjacency alone, nothing else"),
    (2, "plus_must_have_scoring", "+ must-have and good-to-have coverage, claims counted at face value"),
    (3, "plus_evidence_tiering", "+ the evidence-vs-claim discount on must-haves"),
    (4, "full_system", "+ seniority band, production evidence, location, intent, tenure, soft cap"),
]


@dataclass
class ArmResult:
    name: str
    rankings: dict[str, list[str]]
    per_job: dict
    means: dict
    per_job_zero_fill: dict
    means_zero_fill: dict


def labels_by_job(labels: list[dict]) -> dict[str, dict[str, int]]:
    grouped: dict[str, dict[str, int]] = defaultdict(dict)
    for row in labels:
        grouped[row["job_id"]][row["candidate_id"]] = row["relevance"]
    return dict(grouped)


def evaluate_rankings(
    name: str, rankings: dict[str, list[str]], grouped: dict[str, dict[str, int]]
) -> ArmResult:
    condensed = [evaluate_job(j, rankings[j], grouped[j], condensed=True) for j in sorted(grouped)]
    zero_fill = [evaluate_job(j, rankings[j], grouped[j], condensed=False) for j in sorted(grouped)]
    return ArmResult(
        name=name,
        rankings=rankings,
        per_job={m.job_id: m.to_dict() for m in condensed},
        means=aggregate(condensed),
        per_job_zero_fill={m.job_id: m.to_dict() for m in zero_fill},
        means_zero_fill=aggregate(zero_fill),
    )


def signals_rankings(
    signals: list[SignalRecord],
    traces: dict,
    jobs: list[JobSpec],
    flags: ScoringFlags | None = None,
) -> dict[str, list[str]]:
    ranked = run_scoring(signals, traces, jobs, flags)
    return {job_id: [r.candidate_id for r in rows] for job_id, rows in ranked.items()}


# --------------------------------------------------------------------------
# uncertainty
# --------------------------------------------------------------------------
def _payload(
    rankings_a: dict[str, list[str]],
    rankings_b: dict[str, list[str]],
    grouped: dict[str, dict[str, int]],
) -> dict[str, list[tuple[int, int, int]]]:
    """``{job: [(rank_in_a, rank_in_b, relevance), ...]}`` for labelled candidates."""
    out: dict[str, list[tuple[int, int, int]]] = {}
    for job_id, labels in grouped.items():
        pos_a = {cid: i for i, cid in enumerate(rankings_a[job_id]) if cid in labels}
        pos_b = {cid: i for i, cid in enumerate(rankings_b[job_id]) if cid in labels}
        out[job_id] = [
            (pos_a[cid], pos_b[cid], rel) for cid, rel in labels.items() if cid in pos_a
        ]
    return out


def _ndcg_delta(sample: dict[str, list[tuple[int, int, int]]]) -> float:
    """Mean over jobs of NDCG@10(b) - NDCG@10(a) on a resampled pair set."""
    deltas = []
    for rows in sample.values():
        if not rows:
            continue
        by_a = [rel for _, _, rel in sorted(rows, key=lambda r: r[0])]
        by_b = [rel for _, _, rel in sorted(rows, key=lambda r: r[1])]
        deltas.append(ndcg_at_k(by_b, 10) - ndcg_at_k(by_a, 10))
    return sum(deltas) / len(deltas) if deltas else 0.0


def uncertainty_block(
    baseline: ArmResult, system: ArmResult, grouped: dict[str, dict[str, int]]
) -> dict:
    payload = _payload(baseline.rankings, system.rankings, grouped)
    n_pairs = sum(len(rows) for rows in payload.values())
    interval = bootstrap_delta(payload, _ndcg_delta)

    per_job_deltas = {
        job_id: round(
            system.per_job[job_id]["ndcg@10"] - baseline.per_job[job_id]["ndcg@10"], 4
        )
        for job_id in sorted(grouped)
    }
    deltas = list(per_job_deltas.values())

    return {
        "n_labelled_pairs": n_pairs,
        "n_jobs": len(grouped),
        "ndcg@10_delta": interval.to_dict(),
        "per_job_ndcg@10_delta": per_job_deltas,
        "independence_caveat": (
            "Pairs within a job share a query, a grader and a grading session, so the "
            "pair-level interval above is optimistically narrow. The true independent "
            "unit is the job, of which there are 4. The per-job deltas are listed "
            "individually so the reader can see whether the effect is consistent or is "
            "one job carrying the mean."
        ),
        "paired_permutation_over_jobs": paired_permutation(deltas),
        "verdict": verdict(interval, deltas, n_pairs),
    }


# --------------------------------------------------------------------------
# error analysis
# --------------------------------------------------------------------------
def worst_misses(
    system: ArmResult,
    grouped: dict[str, dict[str, int]],
    ranked_records: dict[str, list],
    signals: dict[str, SignalRecord],
    limit: int = 3,
) -> list[dict]:
    """The pairs costing the most NDCG, with the component that drove each.

    Cost is the discounted-gain a pair gives up versus the position it would
    hold in the ideal ordering -- so a relevance-3 candidate ranked 6th is a
    bigger miss than a relevance-1 candidate ranked 9th.
    """
    import math

    candidates = []
    for job_id, labels in grouped.items():
        condensed_order = [cid for cid in system.rankings[job_id] if cid in labels]
        ideal = sorted(labels.items(), key=lambda kv: -kv[1])
        ideal_position = {cid: i for i, (cid, _) in enumerate(ideal)}
        by_id = {r.candidate_id: r for r in ranked_records[job_id]}

        for position, cid in enumerate(condensed_order):
            relevance = labels[cid]
            gain = 2 ** relevance - 1
            loss = gain * (
                1 / math.log2(ideal_position[cid] + 2) - 1 / math.log2(position + 2)
            )
            if loss <= 0:
                continue
            record = by_id[cid]
            candidates.append(
                {
                    "job_id": job_id,
                    "candidate_id": cid,
                    "label": relevance,
                    "rank_given_condensed": position + 1,
                    "rank_ideal": ideal_position[cid] + 1,
                    "ndcg_loss": round(loss, 4),
                    "fit_score": record.fit_score,
                    "score_breakdown": record.score_breakdown,
                    "driving_component": min(
                        record.score_breakdown.items(), key=lambda kv: kv[1]
                    )[0],
                    "missing_must_haves": record.missing_must_haves,
                    "role_family": signals[cid].role_family.value,
                    "years_relevant": signals[cid].years_relevant,
                    "reason_codes": record.reason_codes[:8],
                }
            )

    candidates.sort(key=lambda c: -c["ndcg_loss"])
    return candidates[:limit]
