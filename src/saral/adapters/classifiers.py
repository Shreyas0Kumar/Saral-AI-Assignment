"""Loaders for the two fallback classifiers.

Both return ``None`` when their artifact is missing, so a clean clone with no
model files still runs -- it just runs the lexicon-only arm and says so. Nothing
here is on the default hot path until the lexicon actually abstains.
"""

from __future__ import annotations

import functools
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"
LR_ARTIFACT = CONFIG_DIR / "distilled_lr.joblib"


@functools.lru_cache(maxsize=1)
def load_distilled_lr():
    """The shipped fallback: TF-IDF + multinomial logistic regression, ~KB not MB.

    Loaded lazily on first use rather than at import, which is what makes the
    served latency distribution bimodal -- p50 is the lexicon path, p95 exposes
    this. See DECISIONS.md D18.
    """
    if not LR_ARTIFACT.exists():
        return None
    import joblib

    from saral.core.extract.title_classifier.distilled import DistilledLRClassifier

    bundle = joblib.load(LR_ARTIFACT)
    return DistilledLRClassifier(
        vectorizer=bundle["vectorizer"],
        model=bundle["model"],
        classes=bundle["classes"],
        min_confidence=bundle.get("min_confidence", 0.45),
    )


@functools.lru_cache(maxsize=1)
def load_st_centroid():
    """The compared-but-not-shipped fallback. Drags torch in, hence not shipped."""
    centroids_path = CONFIG_DIR / "st_centroids.npz"
    if not centroids_path.exists():
        return None
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None

    from saral.core.extract.title_classifier.embed_centroid import CentroidClassifier

    payload = np.load(centroids_path, allow_pickle=True)
    model = SentenceTransformer(str(payload["model_name"]))

    def encoder(texts: list[str]):
        return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    return CentroidClassifier(
        encoder=encoder,
        centroids=payload["centroids"],
        classes=[str(c) for c in payload["classes"]],
        min_confidence=float(payload.get("min_confidence", 0.35)),
    )
