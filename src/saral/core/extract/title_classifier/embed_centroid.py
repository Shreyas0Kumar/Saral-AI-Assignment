"""MiniLM nearest-centroid fallback -- compared, not shipped.

Lives under ``core/`` but needs sentence embeddings. Rather than adding
``sentence_transformers`` to the purity test's allow-list (which would make the
guarantee a lie), the encoder is **injected**: a
``Callable[[list[str]], np.ndarray]`` constructed in ``adapters/embed/``. This
module imports numpy and nothing else, so the purity test passes honestly.

Its advantage over the distilled LR is semantic generalisation to titles unlike
anything in the synthetic corpus -- real, but unmeasurable at n=25. Its cost is
an 800MB torch dependency in the served image, which is very measurable. See
DECISIONS.md D2.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from saral.contracts.taxonomy import RoleFamily
from saral.core.extract.title_classifier.base import ClassifyResult
from saral.core.normalize import norm_title


class CentroidClassifier:
    name = "minilm_centroid"

    def __init__(
        self,
        encoder: Callable[[list[str]], "np.ndarray"],
        centroids: "np.ndarray",
        classes: list[str],
        min_confidence: float = 0.35,
    ) -> None:
        self.encoder = encoder
        self.centroids = centroids
        self.classes = classes
        self.min_confidence = min_confidence

    def classify(self, title: str, context: str = "") -> ClassifyResult:
        core, _ = norm_title(title)
        if not core:
            return ClassifyResult.abstain()
        vector = np.asarray(self.encoder([core]))[0]
        norm = np.linalg.norm(vector)
        if norm == 0:
            return ClassifyResult.abstain()
        similarities = self.centroids @ (vector / norm)
        best = int(similarities.argmax())
        score = float(similarities[best])
        if score < self.min_confidence:
            return ClassifyResult.abstain()
        try:
            family = RoleFamily(self.classes[best])
        except ValueError:  # pragma: no cover
            return ClassifyResult.abstain()
        return ClassifyResult(family, round(score * 0.9, 3), self.name, f"cos={score:.2f}")
