"""The cosine-similarity baseline -- roughly what SARAL does today.

Deliberately **not** sandbagged. It gets the full `text_context_full` as
written, the full `raw_query` as written, and a standard off-the-shelf model
with no preprocessing tricks that would handicap it. A baseline you beat by
crippling it tells you nothing, and a reviewer can spot it in one diff.

Embeddings are cached to `out/baseline_embeddings.npz` so `make all` reproduces
the same numbers without re-running the model, and so the evaluation is not
gated on having torch installed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _cache_path(out_dir: Path) -> Path:
    return out_dir / "baseline_embeddings.npz"


def rank_by_cosine(
    jobs: list,
    profiles: list[dict],
    out_dir: Path,
    use_cache: bool = True,
) -> dict[str, list[str]]:
    """Return ``{job_id: [candidate_id ordered by descending cosine]}``."""
    candidate_ids = [p["id"] for p in profiles]
    job_ids = [j.job_id for j in jobs]
    cache = _cache_path(out_dir)

    payload = None
    if use_cache and cache.exists():
        stored = np.load(cache, allow_pickle=True)
        if (
            list(stored["candidate_ids"]) == candidate_ids
            and list(stored["job_ids"]) == job_ids
            and str(stored["model_name"]) == MODEL_NAME
        ):
            payload = stored

    if payload is None:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(MODEL_NAME)
        candidate_texts = [
            p.get("text_context_full") or _fallback_text(p) for p in profiles
        ]
        job_texts = [j.raw_query or j.title for j in jobs]
        candidate_vectors = model.encode(
            candidate_texts, normalize_embeddings=True, show_progress_bar=False
        )
        job_vectors = model.encode(
            job_texts, normalize_embeddings=True, show_progress_bar=False
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache,
            candidate_vectors=candidate_vectors,
            job_vectors=job_vectors,
            candidate_ids=np.array(candidate_ids),
            job_ids=np.array(job_ids),
            model_name=np.array(MODEL_NAME),
        )
        payload = np.load(cache, allow_pickle=True)

    candidate_vectors = payload["candidate_vectors"]
    job_vectors = payload["job_vectors"]

    rankings: dict[str, list[str]] = {}
    for index, job_id in enumerate(job_ids):
        similarities = candidate_vectors @ job_vectors[index]
        # Ties break on candidate_id so the baseline is reproducible too.
        order = sorted(
            range(len(candidate_ids)),
            key=lambda i: (-float(similarities[i]), candidate_ids[i]),
        )
        rankings[job_id] = [candidate_ids[i] for i in order]
    return rankings


def _fallback_text(profile: dict) -> str:
    parts = [
        profile.get("headline") or "",
        profile.get("about") or "",
        " ".join(profile.get("skills") or []),
        " ".join(
            f"{e.get('role')} at {e.get('company_name')}"
            for e in (profile.get("experience") or [])
        ),
    ]
    return " | ".join(p for p in parts if p)
