"""Arm registry.

Adding an arm is one dict entry. Nothing in the evaluation loop branches on an
arm name -- that is what keeps the four-way comparison honest, because there is
nowhere for a special case that flatters one arm to hide.

| arm                        | hot path                     | role                    |
|----------------------------|------------------------------|-------------------------|
| `baseline_cosine`          | MiniLM embed + cosine        | required by the brief   |
| `signals_v1_lexicon_lr`    | rules + distilled LR         | **shipped**             |
| `signals_v1_lexicon_st`    | rules + MiniLM centroid      | compared, not shipped   |
| `signals_v1_lexicon_only`  | rules, abstain on a miss     | isolates the fallback   |
| `llm_per_row`              | a local LLM per profile      | measured cost ceiling   |
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from saral.core.extract.title_classifier.base import ChainClassifier, TitleClassifier


@dataclass(frozen=True)
class Arm:
    name: str
    description: str
    #: builds the fallback chain member list beyond the lexicon; may be empty
    build_fallbacks: Callable[[], list[TitleClassifier]]
    #: True for arms that rank by embedding similarity rather than signals
    is_embedding_baseline: bool = False
    shipped: bool = False


def _no_fallback() -> list[TitleClassifier]:
    return []


def _distilled_lr() -> list[TitleClassifier]:
    from saral.adapters.classifiers import load_distilled_lr

    clf = load_distilled_lr()
    return [clf] if clf is not None else []


def _minilm_centroid() -> list[TitleClassifier]:
    from saral.adapters.classifiers import load_st_centroid

    clf = load_st_centroid()
    return [clf] if clf is not None else []


ARMS: dict[str, Arm] = {
    "baseline_cosine": Arm(
        name="baseline_cosine",
        description="all-MiniLM-L6-v2 cosine over raw_query vs text_context_full",
        build_fallbacks=_no_fallback,
        is_embedding_baseline=True,
    ),
    "signals_v1_lexicon_only": Arm(
        name="signals_v1_lexicon_only",
        description="rules only; abstains when the lexicon misses",
        build_fallbacks=_no_fallback,
    ),
    "signals_v1_lexicon_lr": Arm(
        name="signals_v1_lexicon_lr",
        description="rules + distilled logistic regression fallback",
        build_fallbacks=_distilled_lr,
        shipped=True,
    ),
    "signals_v1_lexicon_st": Arm(
        name="signals_v1_lexicon_st",
        description="rules + MiniLM nearest-centroid fallback",
        build_fallbacks=_minilm_centroid,
    ),
    "llm_per_row": Arm(
        name="llm_per_row",
        description="a local instruct model called once per profile (cost ceiling)",
        build_fallbacks=_no_fallback,
    ),
}

SHIPPED_ARM = "signals_v1_lexicon_lr"


def build_classifier(arm_name: str, cfg) -> ChainClassifier:
    """Lexicon first, then the arm's fallbacks."""
    from saral.core.extract.title_classifier.lexicon import LexiconClassifier

    arm = ARMS[arm_name]
    members: list[TitleClassifier] = [LexiconClassifier(cfg.lexicon)]
    members.extend(arm.build_fallbacks())
    return ChainClassifier(members)
