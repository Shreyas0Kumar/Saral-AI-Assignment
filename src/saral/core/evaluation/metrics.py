"""Ranking metrics.

NDCG uses the **exponential** gain, `2**rel - 1`. With graded relevance 0-3 the
linear form treats the gap between "worth a call" and "would interview today"
as identical to the gap between "irrelevant" and "weak stretch", which is not
what the grader meant. The two forms differ on this data, so the choice is not
cosmetic; `test_ndcg_matches_hand_computation` pins it against an example
computed by hand.

Two list conventions are computed for every metric, and both are reported:

* **condensed** -- rank all 25, then keep only the labelled candidates,
  preserving order. This is the primary number.
* **zero-filled** -- unlabelled candidates are treated as relevance 0.

Neither is neutral. Zero-fill punishes the system for surfacing a good
candidate the recruiter never graded, which is precisely the case this system is
built to produce. Condensed lists flatter every system equally, so they cannot
be used to claim a win -- only to compare ordering quality. Reporting one alone
would look like picking the flattering convention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def dcg(relevances: list[int], k: int) -> float:
    return sum(
        (2 ** rel - 1) / math.log2(index + 2) for index, rel in enumerate(relevances[:k])
    )


def ndcg_at_k(ranked_relevances: list[int], k: int) -> float:
    ideal = sorted(ranked_relevances, reverse=True)
    denominator = dcg(ideal, k)
    if denominator == 0:
        return 0.0
    return dcg(ranked_relevances, k) / denominator


def precision_at_k(ranked_relevances: list[int], k: int, threshold: int = 2) -> float:
    if k <= 0:
        return 0.0
    top = ranked_relevances[:k]
    return sum(1 for rel in top if rel >= threshold) / k


def mrr(ranked_relevances: list[int], threshold: int = 2) -> float:
    for index, rel in enumerate(ranked_relevances, start=1):
        if rel >= threshold:
            return 1.0 / index
    return 0.0


def recall_at_k(ranked_relevances: list[int], k: int, threshold: int = 2) -> float:
    total = sum(1 for rel in ranked_relevances if rel >= threshold)
    if total == 0:
        return 0.0
    return sum(1 for rel in ranked_relevances[:k] if rel >= threshold) / total


@dataclass(frozen=True)
class JobMetrics:
    job_id: str
    n_ranked: int
    n_labelled: int
    n_relevant: int
    ndcg_at_10: float
    ndcg_at_5: float
    precision_at_5: float
    mrr: float
    recall_at_10: float
    #: the highest Precision@5 this job can attain, given how many relevant
    #: candidates exist at all
    precision_at_5_ceiling: float

    def to_dict(self) -> dict:
        return {
            "n_ranked": self.n_ranked,
            "n_labelled": self.n_labelled,
            "n_relevant_at_threshold_2": self.n_relevant,
            "ndcg@10": round(self.ndcg_at_10, 4),
            "ndcg@5": round(self.ndcg_at_5, 4),
            "precision@5": round(self.precision_at_5, 4),
            "precision@5_ceiling": round(self.precision_at_5_ceiling, 4),
            "mrr": round(self.mrr, 4),
            "recall@10": round(self.recall_at_10, 4),
        }


def evaluate_job(
    job_id: str,
    ranked_candidate_ids: list[str],
    labels: dict[str, int],
    condensed: bool = True,
) -> JobMetrics:
    """Score one job's ranked list against its labels."""
    if condensed:
        relevances = [labels[cid] for cid in ranked_candidate_ids if cid in labels]
    else:
        relevances = [labels.get(cid, 0) for cid in ranked_candidate_ids]

    n_relevant = sum(1 for rel in labels.values() if rel >= 2)
    return JobMetrics(
        job_id=job_id,
        n_ranked=len(ranked_candidate_ids),
        n_labelled=len(labels),
        n_relevant=n_relevant,
        ndcg_at_10=ndcg_at_k(relevances, 10),
        ndcg_at_5=ndcg_at_k(relevances, 5),
        precision_at_5=precision_at_k(relevances, 5),
        mrr=mrr(relevances),
        recall_at_10=recall_at_k(relevances, 10),
        precision_at_5_ceiling=min(n_relevant, 5) / 5,
    )


def aggregate(per_job: list[JobMetrics]) -> dict[str, float]:
    """Unweighted mean across jobs -- the job is the independent unit, not the pair."""
    if not per_job:
        return {}
    n = len(per_job)
    return {
        "ndcg@10": round(sum(m.ndcg_at_10 for m in per_job) / n, 4),
        "ndcg@5": round(sum(m.ndcg_at_5 for m in per_job) / n, 4),
        "precision@5": round(sum(m.precision_at_5 for m in per_job) / n, 4),
        "precision@5_ceiling": round(sum(m.precision_at_5_ceiling for m in per_job) / n, 4),
        "mrr": round(sum(m.mrr for m in per_job) / n, 4),
        "recall@10": round(sum(m.recall_at_10 for m in per_job) / n, 4),
    }
