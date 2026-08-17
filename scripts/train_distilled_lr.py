"""Distil the synthetic title corpus into a sparse logistic regression.

Trains on `config/synthetic_titles.jsonl` (LLM-generated, offline) and validates
on `config/title_labels_holdout.jsonl` (45 real experience entries, hand
labelled, never trained on).

The output is a joblib bundle of roughly 100KB that classifies a title with one
sparse matrix multiply. No torch, no network, no per-row model call -- which is
the entire point: the LLM's knowledge is paid for once, offline, and the hot
path is free.

Run: `make regenerate-llm-artifacts` (never on the default path).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import FeatureUnion

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from saral.core.normalize import norm_title  # noqa: E402

CORPUS = ROOT / "config" / "synthetic_titles.jsonl"
HOLDOUT = ROOT / "config" / "title_labels_holdout.jsonl"
ARTIFACT = ROOT / "config" / "distilled_lr.joblib"
REPORT = ROOT / "out" / "fallback_comparison.json"

SEED = 20260817
MIN_CONFIDENCE = 0.45


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalise(title: str) -> str:
    core, _ = norm_title(title)
    return core


def build_vectorizer() -> FeatureUnion:
    """Char n-grams catch "sde3"/"sde-3"; word n-grams catch "senior data engineer"."""
    return FeatureUnion(
        [
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True)),
            ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ]
    )


def main() -> None:
    corpus = read_jsonl(CORPUS)
    X_raw = [normalise(r["title"]) for r in corpus]
    y = np.array([r["role_family"] for r in corpus])
    print(f"training corpus: {len(corpus)} titles, {len(set(y))} classes")

    vectorizer = build_vectorizer()
    X = vectorizer.fit_transform(X_raw)
    model = LogisticRegression(
        max_iter=2000, C=4.0, class_weight="balanced", random_state=SEED
    )

    # In-corpus generalisation, before the real held-out number.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_pred = cross_val_predict(model, X, y, cv=cv)
    cv_acc = float((cv_pred == y).mean())
    print(f"synthetic 5-fold CV accuracy: {cv_acc:.3f}")

    model.fit(X, y)
    classes = [str(c) for c in model.classes_]

    # -- the number that matters: real entries, never trained on -----------
    holdout = [r for r in read_jsonl(HOLDOUT) if r["role_family"]]
    unlabelled = len(read_jsonl(HOLDOUT)) - len(holdout)
    H = vectorizer.transform([normalise(r["title"]) for r in holdout])
    probabilities = model.predict_proba(H)
    predictions = [classes[int(i)] for i in probabilities.argmax(axis=1)]
    confidences = probabilities.max(axis=1)
    truth = [r["role_family"] for r in holdout]

    correct = sum(p == t for p, t in zip(predictions, truth))
    answered = [(p, t, c) for p, t, c in zip(predictions, truth, confidences) if c >= MIN_CONFIDENCE]
    answered_correct = sum(p == t for p, t, _ in answered)

    print(f"held-out real entries: {len(holdout)} labelled ({unlabelled} deliberately unlabelled)")
    print(f"  accuracy (forced choice): {correct}/{len(holdout)} = {correct/len(holdout):.3f}")
    print(
        f"  accuracy (abstain below {MIN_CONFIDENCE}): "
        f"{answered_correct}/{len(answered)} = "
        f"{answered_correct/len(answered):.3f} at {len(answered)/len(holdout):.0%} coverage"
    )

    errors = [
        {"title": h["title"], "context": h["context"][:60], "true": t, "pred": p, "p": round(float(c), 3)}
        for h, p, t, c in zip(holdout, predictions, truth, confidences)
        if p != t
    ]
    print("  misclassifications:")
    for e in errors:
        print(f"    {e['title']!r:38s} true={e['true']:20s} pred={e['pred']:20s} p={e['p']}")

    # -- latency, measured not estimated ----------------------------------
    samples = [normalise(r["title"]) for r in holdout] * 20
    timings = []
    for title in samples:
        t0 = time.perf_counter()
        model.predict_proba(vectorizer.transform([title]))
        timings.append((time.perf_counter() - t0) * 1000)
    timings.sort()

    joblib.dump(
        {
            "vectorizer": vectorizer,
            "model": model,
            "classes": classes,
            "min_confidence": MIN_CONFIDENCE,
            "trained_on": CORPUS.name,
            "seed": SEED,
        },
        ARTIFACT,
        compress=3,
    )
    size_kb = ARTIFACT.stat().st_size / 1024
    print(f"wrote {ARTIFACT.relative_to(ROOT)} ({size_kb:.0f} KB)")

    report = {
        "distilled_lr": {
            "train_corpus": {"titles": len(corpus), "source": corpus[0]["source"]},
            "synthetic_cv_accuracy": round(cv_acc, 4),
            "holdout_real_entries": len(holdout),
            "holdout_unlabelled_excluded": unlabelled,
            "holdout_accuracy_forced": round(correct / len(holdout), 4),
            "holdout_accuracy_when_answering": round(answered_correct / len(answered), 4),
            "holdout_coverage": round(len(answered) / len(holdout), 4),
            "errors": errors,
            "artifact_kb": round(size_kb, 1),
            "latency_ms": {
                "p50": round(timings[len(timings) // 2], 4),
                "p95": round(timings[int(len(timings) * 0.95)], 4),
                "n": len(timings),
            },
            "runtime_deps": ["scikit-learn", "scipy", "numpy"],
        }
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(REPORT.read_text(encoding="utf-8")) if REPORT.exists() else {}
    existing.update(report)
    REPORT.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"wrote {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
