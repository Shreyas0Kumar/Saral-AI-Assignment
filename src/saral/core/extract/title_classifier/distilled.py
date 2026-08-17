"""The shipped fallback: a distilled multinomial logistic regression.

Trained offline on an LLM-generated synthetic title corpus (see
``scripts/generate_llm_artifacts.py``) and validated on hand-labelled real
experience entries that were never in training. The LLM's knowledge about what
job titles mean is transferred into a ~100KB joblib artifact, and the hot path
becomes one sparse matrix multiply: no torch, no network, no per-row model call.

It abstains below ``min_confidence`` rather than forcing a guess, because a
low-probability guess on an unfamiliar title is exactly the input that produces
a confidently wrong role family.

This module imports numpy only. The vectorizer and model are injected by
``adapters/classifiers.py``, which keeps ``core`` free of joblib and sklearn.
"""

from __future__ import annotations

from saral.contracts.taxonomy import RoleFamily
from saral.core.extract.title_classifier.base import ClassifyResult
from saral.core.normalize import norm_title


class DistilledLRClassifier:
    name = "distilled_lr"

    def __init__(self, vectorizer, model, classes: list[str], min_confidence: float = 0.45) -> None:
        self.vectorizer = vectorizer
        self.model = model
        self.classes = classes
        self.min_confidence = min_confidence

    def classify(self, title: str, context: str = "") -> ClassifyResult:
        core, _ = norm_title(title)
        if not core:
            return ClassifyResult.abstain()
        features = self.vectorizer.transform([core])
        probabilities = self.model.predict_proba(features)[0]
        best_index = int(probabilities.argmax())
        confidence = float(probabilities[best_index])
        if confidence < self.min_confidence:
            return ClassifyResult.abstain()
        try:
            family = RoleFamily(self.classes[best_index])
        except ValueError:  # pragma: no cover - artifact/taxonomy drift
            return ClassifyResult.abstain()
        # Scaled below the lexicon's floor so an LR hit never looks as certain
        # as a deterministic pattern match.
        return ClassifyResult(family, round(confidence * 0.9, 3), self.name, f"lr:p={confidence:.2f}")
