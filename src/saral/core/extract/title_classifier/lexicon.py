"""Deterministic lexicon classifier -- the hot path.

Matching order:

1. exact match on the normalised title core            -> 1.00
2. longest token-boundary substring match              -> 0.90
3. the title is on the `ambiguous` list -> route to the context vote  -> 0.80
4. single distinctive token match                      -> 0.75
5. abstain

Step 3 is the one that earns its keep. "Design Engineer", "SDE", "Member of
Technical Staff" and "Consultant" carry no family information in the Indian
market; guessing on them is how a mechanical engineer with AutoCAD ends up in a
data role. Routing them through the entry's own description and `skills_used`
resolves most of them and abstains honestly on the rest.
"""

from __future__ import annotations

import re

from saral.contracts.taxonomy import RoleFamily
from saral.core.extract.title_classifier.base import ClassifyResult
from saral.core.normalize import norm_text, norm_title


def _contains_phrase(haystack: str, needle: str) -> bool:
    """Token-boundary substring test: 'qa' must not match 'aqua'."""
    return re.search(rf"(?<![\w]){re.escape(needle)}(?![\w])", haystack) is not None


class LexiconClassifier:
    name = "lexicon"

    def __init__(self, config) -> None:  # LexiconConfig, kept untyped to stay pure
        self.config = config
        # Longest patterns first so "senior data engineer" beats "data engineer",
        # and "computer vision engineer" beats "engineer".
        self._by_length = sorted(
            config.patterns.items(), key=lambda kv: len(kv[0]), reverse=True
        )
        self._ambiguous = config.ambiguous

    # -- context vote -----------------------------------------------------
    def _context_vote(self, context: str) -> tuple[RoleFamily | None, float, str]:
        text = norm_text(context)
        if not text:
            return None, 0.0, ""
        scores: dict[RoleFamily, float] = {}
        hits: dict[RoleFamily, list[str]] = {}
        for family, terms in self.config.context.items():
            for term, weight in terms.items():
                if term and _contains_phrase(text, term):
                    scores[family] = scores.get(family, 0.0) + weight
                    hits.setdefault(family, []).append(term)
        if not scores:
            return None, 0.0, ""
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best, best_score = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
        if best_score < self.config.context_min_score:
            return None, 0.0, ""
        if best_score - runner_up < self.config.context_min_margin:
            # Two families tied on context: abstaining is the honest answer.
            return None, 0.0, ""
        return best, best_score, ",".join(sorted(hits[best])[:3])

    # -- main entry point -------------------------------------------------
    def classify(self, title: str, context: str = "") -> ClassifyResult:
        core, _prefixes = norm_title(title)
        if not core:
            return ClassifyResult.abstain()

        # 1. exact
        family = self.config.patterns.get(core)
        if family is not None:
            return ClassifyResult(family, 1.0, self.name, f"exact:{core}")

        # 2. longest substring, but never over an ambiguous title -- "software
        #    engineer" contains "engineer", and resolving it that way would
        #    defeat the entire point of the ambiguous list.
        if core not in self._ambiguous:
            for pattern, fam in self._by_length:
                if " " in pattern and _contains_phrase(core, pattern):
                    return ClassifyResult(fam, 0.90, self.name, f"phrase:{pattern}")

        # 3. ambiguous -> context vote
        if core in self._ambiguous or any(
            _contains_phrase(core, a) for a in self._ambiguous if " " in a
        ):
            fam, score, evidence = self._context_vote(context)
            if fam is not None:
                # Context evidence is graded. A single weak term ("Software
                # Engineer" whose only clue is one entry in skills_used) is a
                # real signal but a thin one, and weighting it the same as a
                # description full of corroborating terms lets one stray token
                # outvote a well-evidenced adjacent role.
                strength = 0.80 if score >= 2.0 else 0.60
                return ClassifyResult(fam, strength, self.name, f"context:{evidence}")
            return ClassifyResult.abstain()

        # 4. single distinctive token
        for pattern, fam in self._by_length:
            if " " not in pattern and _contains_phrase(core, pattern):
                return ClassifyResult(fam, 0.75, self.name, f"token:{pattern}")

        # 5. unknown title -- hand it to the fallback
        return ClassifyResult.abstain()
