"""Compare the two fallback classifiers on their own task.

Why not on NDCG
---------------
Across the 25 profiles the lexicon abstains on a handful of titles. The fallback
therefore decides a small number of entries, and a change there cannot move a
ranking metric computed over 55 labels. Reporting "the two arms tie on NDCG@10"
without saying that reads as a null result rather than a correctly scoped one.

So the fallbacks are measured on the task they actually do:

* accuracy on the 45 hand-labelled real entries, **split by whether the title
  appears verbatim in the training corpus** -- without that split the headline
  accuracy measures memorisation and calls it generalisation;
* accuracy restricted to the entries where the lexicon genuinely abstains,
  which is the only population the fallback ever sees in production;
* latency p50/p95 on CPU;
* artifact size, cold start, and the dependency each one drags into the image.

Also builds `config/st_centroids.npz` for the MiniLM arm.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from saral.config_loader import load_all  # noqa: E402
from saral.core.extract.title_classifier.lexicon import LexiconClassifier  # noqa: E402
from saral.core.normalize import norm_title  # noqa: E402

CORPUS = ROOT / "config" / "synthetic_titles.jsonl"
HOLDOUT = ROOT / "config" / "title_labels_holdout.jsonl"
CENTROIDS = ROOT / "config" / "st_centroids.npz"
REPORT = ROOT / "out" / "fallback_comparison.json"
ST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ST_MIN_CONFIDENCE = 0.35


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def build_centroids() -> None:
    """Mean-pool the synthetic titles per family into 12 unit centroids."""
    from sentence_transformers import SentenceTransformer

    corpus = read_jsonl(CORPUS)
    model = SentenceTransformer(ST_MODEL)
    families = sorted({r["role_family"] for r in corpus})
    vectors = model.encode(
        [norm_title(r["title"])[0] for r in corpus],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    centroids = []
    for family in families:
        mask = np.array([r["role_family"] == family for r in corpus])
        centroid = vectors[mask].mean(axis=0)
        centroids.append(centroid / np.linalg.norm(centroid))
    np.savez(
        CENTROIDS,
        centroids=np.vstack(centroids).astype(np.float32),
        classes=np.array(families),
        model_name=np.array(ST_MODEL),
        min_confidence=np.array(ST_MIN_CONFIDENCE),
    )
    print(f"wrote {CENTROIDS.relative_to(ROOT)} ({CENTROIDS.stat().st_size/1024:.0f} KB)")


def evaluate(classifier, holdout: list[dict], seen: set[str], abstained: set[str]) -> dict:
    total_correct = answered = answered_correct = 0
    novel_total = novel_answered = novel_correct = 0
    lex_total = lex_answered = lex_correct = 0
    errors = []
    timings: list[float] = []

    for row in holdout:
        core = norm_title(row["title"])[0]
        t0 = time.perf_counter()
        result = classifier.classify(row["title"], "")
        timings.append((time.perf_counter() - t0) * 1000)

        predicted = result.family.value if result.family else None
        correct = predicted == row["role_family"]
        is_novel = core not in seen
        in_lexicon_gap = core in abstained

        total_correct += correct
        if predicted is not None:
            answered += 1
            answered_correct += correct
        if is_novel:
            novel_total += 1
            if predicted is not None:
                novel_answered += 1
                novel_correct += correct
        if in_lexicon_gap:
            lex_total += 1
            if predicted is not None:
                lex_answered += 1
                lex_correct += correct
        if predicted is not None and not correct:
            errors.append({"title": row["title"], "true": row["role_family"], "pred": predicted})

    timings.sort()
    return {
        "n": len(holdout),
        "accuracy_forced": round(total_correct / len(holdout), 4),
        "coverage": round(answered / len(holdout), 4),
        "accuracy_when_answering": round(answered_correct / answered, 4) if answered else None,
        "seen_verbatim_in_training": {
            "n": len(holdout) - novel_total,
        },
        "novel_titles": {
            "n": novel_total,
            "answered": novel_answered,
            "correct": novel_correct,
            "accuracy_when_answering": round(novel_correct / novel_answered, 4) if novel_answered else None,
        },
        "titles_the_lexicon_abstains_on": {
            "n": lex_total,
            "answered": lex_answered,
            "correct": lex_correct,
            "accuracy_when_answering": round(lex_correct / lex_answered, 4) if lex_answered else None,
        },
        "errors_when_answering": errors,
        "latency_ms": {
            "p50": round(timings[len(timings) // 2], 4),
            "p95": round(timings[int(len(timings) * 0.95)], 4),
        },
    }


def main() -> None:
    if not CENTROIDS.exists():
        build_centroids()

    cfg, _ = load_all()
    corpus = read_jsonl(CORPUS)
    seen = {norm_title(r["title"])[0] for r in corpus}
    holdout = [r for r in read_jsonl(HOLDOUT) if r["role_family"]]

    lexicon = LexiconClassifier(cfg.lexicon)
    abstained = {
        norm_title(r["title"])[0]
        for r in holdout
        if lexicon.classify(r["title"], r["context"]).abstained
    }

    report: dict = {
        "method": {
            "training_corpus": {"titles": len(corpus), "source": corpus[0]["source"]},
            "holdout": {
                "labelled_entries": len(holdout),
                "titles_appearing_verbatim_in_training": len(
                    [r for r in holdout if norm_title(r["title"])[0] in seen]
                ),
                "provenance": (
                    "hand-labelled by the author after writing the lexicon but before "
                    "either fallback existed: independent of the models, not of the author"
                ),
            },
            "why_not_ndcg": (
                "the lexicon abstains on a handful of the 70 titles in the corpus, so the "
                "fallback cannot move a ranking metric over 55 labels; selection is made on "
                "these metrics and on cost instead"
            ),
        }
    }

    from saral.adapters.classifiers import load_distilled_lr, load_st_centroid

    lr = load_distilled_lr()
    if lr is not None:
        report["distilled_lr_eval"] = evaluate(lr, holdout, seen, abstained)
        report["distilled_lr_eval"]["artifact_kb"] = round(
            (ROOT / "config" / "distilled_lr.joblib").stat().st_size / 1024, 1
        )
        report["distilled_lr_eval"]["image_cost"] = "scikit-learn + scipy, ~90MB"

    st = load_st_centroid()
    if st is not None:
        t0 = time.perf_counter()
        st.classify("warmup engineer", "")
        report["minilm_centroid_eval"] = evaluate(st, holdout, seen, abstained)
        report["minilm_centroid_eval"]["artifact_kb"] = round(
            CENTROIDS.stat().st_size / 1024, 1
        )
        report["minilm_centroid_eval"]["cold_start_s"] = round(time.perf_counter() - t0, 2)
        report["minilm_centroid_eval"]["image_cost"] = (
            "torch + sentence-transformers + the 90MB model, ~800MB"
        )

    existing = json.loads(REPORT.read_text(encoding="utf-8")) if REPORT.exists() else {}
    existing.update(report)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nwrote {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
