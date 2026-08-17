"""Uncertainty.

The headline interval is a bootstrap over labelled pairs, resampled **within**
job, 1000 resamples, fixed seed. It is reported together with the caveat that
makes it readable rather than misleading:

    Pairs inside a job are not independent -- they share a query, a grader and a
    grading session. The true independent unit is the **job**, and there are
    four. So the pair-level interval is optimistically narrow, and the four
    per-job deltas are shown individually so a reader can see whether a win is
    consistent or is one job carrying the mean.

A paired permutation test over the four per-job deltas is also reported. With
n=4 it can only produce p >= 0.0625 (2^-4), which is stated rather than
presented as a near-miss.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Interval:
    point: float
    low: float
    high: float
    method: str

    def to_dict(self) -> dict:
        return {
            "point": round(self.point, 4),
            "ci95": [round(self.low, 4), round(self.high, 4)],
            "method": self.method,
        }


def bootstrap_delta(
    per_job_pairs: dict[str, list[tuple[int, int]]],
    metric_fn,
    resamples: int = 1000,
    seed: int = 20260817,
) -> Interval:
    """Bootstrap the difference between two systems.

    ``per_job_pairs`` maps job_id to a list of ``(rank_a, rank_b)`` positions
    for each labelled candidate, plus its relevance -- supplied by the caller as
    an opaque payload that ``metric_fn`` knows how to score. Resampling happens
    *within* each job so the job composition of the sample is preserved.
    """
    rng = random.Random(seed)
    point = metric_fn(per_job_pairs)

    deltas: list[float] = []
    for _ in range(resamples):
        sample = {
            job_id: [rng.choice(rows) for _ in rows] if rows else []
            for job_id, rows in per_job_pairs.items()
        }
        deltas.append(metric_fn(sample))

    deltas.sort()
    low = deltas[int(0.025 * len(deltas))]
    high = deltas[min(len(deltas) - 1, int(0.975 * len(deltas)))]
    return Interval(
        point=point,
        low=low,
        high=high,
        method=f"bootstrap over labelled pairs resampled within job, {resamples} resamples, seed {seed}",
    )


def paired_permutation(per_job_deltas: list[float]) -> dict:
    """Exact sign-flip test over per-job deltas.

    With four jobs there are 2^4 = 16 sign assignments, so the smallest
    attainable two-sided p-value is 0.125 and the smallest one-sided p-value is
    0.0625. Reported explicitly, because "p = 0.0625" looks like a near-miss
    until you know it is the floor.
    """
    n = len(per_job_deltas)
    if n == 0:
        return {"p_one_sided": None, "note": "no jobs"}

    observed = sum(per_job_deltas) / n
    extreme = 0
    for mask in range(1 << n):
        flipped = [
            delta if (mask >> i) & 1 == 0 else -delta
            for i, delta in enumerate(per_job_deltas)
        ]
        if sum(flipped) / n >= observed:
            extreme += 1
    p = extreme / (1 << n)
    return {
        "p_one_sided": round(p, 4),
        "n_jobs": n,
        "minimum_attainable_p": round(1 / (1 << n), 4),
        "note": (
            f"exact sign-flip test over {n} per-job deltas; with n={n} the smallest "
            f"attainable one-sided p is {1 / (1 << n):.4f}, so this test cannot "
            "establish significance at 0.05 regardless of effect size"
        ),
    }


def verdict(interval: Interval, per_job_deltas: list[float], n_pairs: int) -> str:
    """Plain language, no hedging, on whether the difference is real."""
    crosses_zero = interval.low <= 0 <= interval.high
    consistent = all(d > 0 for d in per_job_deltas) or all(d < 0 for d in per_job_deltas)
    direction = "improvement" if interval.point > 0 else "regression"

    if crosses_zero:
        base = (
            f"The {direction} of {interval.point:+.3f} NDCG@10 is NOT distinguishable "
            f"from noise at n={n_pairs} labelled pairs: the 95% interval "
            f"[{interval.low:.3f}, {interval.high:.3f}] contains zero."
        )
    else:
        base = (
            f"The pair-level 95% interval [{interval.low:.3f}, {interval.high:.3f}] "
            f"excludes zero, but this interval is optimistically narrow because pairs "
            f"within a job share a query and a grader and are not independent."
        )

    if consistent:
        base += (
            f" The sign is at least consistent across all {len(per_job_deltas)} jobs, "
            "which is weak but real corroboration."
        )
    else:
        signs = ", ".join(f"{d:+.3f}" for d in per_job_deltas)
        base += (
            f" The per-job deltas do not agree in sign ({signs}), so the mean is "
            "being carried by a subset of jobs rather than reflecting a consistent effect."
        )

    base += (
        " With 4 jobs as the independent unit, no experiment on this dataset can "
        "establish significance; the honest reading is that these numbers rank the "
        "approaches for further testing, not that they settle the question."
    )
    return base
