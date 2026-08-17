"""The title classifier protocol.

The important design point is **abstention**. A classifier that must always
return a family will guess on "Design Engineer" and land a mechanical engineer
in a data role -- the exact failure Appendix A calls out. So `classify` may
return ``None``, and a chain tries the next member.

Which member fired is recorded, not inferred. That is how the fallback hit rate
gets reported in Phase 3 without extra plumbing, and how a single profile's
`confidence` knows to take the fallback penalty.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from saral.contracts.taxonomy import RoleFamily


@dataclass(frozen=True)
class ClassifyResult:
    family: RoleFamily | None
    confidence: float
    #: name of the classifier that decided, or "" when every member abstained
    source: str = ""
    #: short human-readable trace, e.g. the lexicon pattern that matched
    evidence: str = ""

    @property
    def abstained(self) -> bool:
        return self.family is None

    @staticmethod
    def abstain() -> "ClassifyResult":
        return ClassifyResult(family=None, confidence=0.0)


@runtime_checkable
class TitleClassifier(Protocol):
    name: str

    def classify(self, title: str, context: str) -> ClassifyResult: ...


class ChainClassifier:
    """Try each member in order; the first non-abstaining result wins."""

    name = "chain"

    def __init__(self, members: list[TitleClassifier]) -> None:
        if not members:
            raise ValueError("ChainClassifier needs at least one member")
        self.members = members
        #: classifier name -> number of titles it decided
        self.invocations: dict[str, int] = {}
        self.abstentions = 0

    def classify(self, title: str, context: str) -> ClassifyResult:
        for member in self.members:
            result = member.classify(title, context)
            if not result.abstained:
                self.invocations[member.name] = self.invocations.get(member.name, 0) + 1
                return result
        self.abstentions += 1
        return ClassifyResult.abstain()

    @property
    def primary_name(self) -> str:
        return self.members[0].name

    def fallback_invocations(self) -> int:
        """How many titles were decided by something other than the first member."""
        return sum(
            count for name, count in self.invocations.items() if name != self.primary_name
        )

    def reset_counters(self) -> None:
        self.invocations = {}
        self.abstentions = 0
