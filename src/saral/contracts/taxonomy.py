"""Role family / seniority taxonomy and the adjacency lookup.

The 12 ``RoleFamily`` values are fixed by the brief. ``Seniority`` likewise.
Adjacency values themselves live in ``config/adjacency.yaml`` -- this module only
provides the lookup semantics, including the one asymmetric case
(``engineering_manager``), which cannot be expressed as a static cell.
"""

from __future__ import annotations

from enum import Enum
from typing import Iterable, Mapping


class RoleFamily(str, Enum):
    BACKEND = "backend"
    FRONTEND = "frontend"
    FULLSTACK = "fullstack"
    MOBILE = "mobile"
    ML_ENGINEER = "ml_engineer"
    DATA_ENGINEER = "data_engineer"
    DATA_ANALYST = "data_analyst"
    DATA_SCIENTIST = "data_scientist"
    DEVOPS_SRE = "devops_sre"
    QA = "qa"
    ENGINEERING_MANAGER = "engineering_manager"
    NON_ENGINEERING = "non_engineering"


class Seniority(str, Enum):
    INTERN = "intern"
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"
    STAFF_PLUS = "staff+"
    MANAGER = "manager"


#: Ordering used when comparing a headline-implied level against the derived one.
SENIORITY_ORDER: dict[Seniority, int] = {
    Seniority.INTERN: 0,
    Seniority.JUNIOR: 1,
    Seniority.MID: 2,
    Seniority.SENIOR: 3,
    Seniority.STAFF_PLUS: 4,
    Seniority.MANAGER: 4,  # a manager is not "above" a staff engineer, it is sideways
}


class Adjacency:
    """Symmetric adjacency with a runtime rule for ``engineering_manager``.

    An engineering manager's adjacency to a target family is not a fixed number:
    an EM who spent eight years in ML is adjacent to ``ml_engineer``, an EM who
    came up through frontend is not. So the EM row is resolved against the
    families that actually appear in that candidate's own history.
    """

    def __init__(
        self,
        identity: float,
        pairs: Mapping[str, float],
        em_prior_ic: float,
        default: float,
    ) -> None:
        self.identity = identity
        self.default = default
        self.em_prior_ic = em_prior_ic
        self._pairs: dict[frozenset[str], float] = {}
        for key, value in pairs.items():
            a, _, b = key.partition("|")
            self._pairs[frozenset((a.strip(), b.strip()))] = float(value)

    def score(
        self,
        candidate_family: RoleFamily,
        target_family: RoleFamily,
        prior_families: Iterable[RoleFamily] = (),
    ) -> float:
        """Return adjacency in [0, 1] between a candidate's family and a target."""
        if candidate_family == target_family:
            return self.identity

        if candidate_family == RoleFamily.ENGINEERING_MANAGER:
            # Resolved at runtime against the EM's own IC history.
            if target_family in set(prior_families):
                return self.em_prior_ic
            return self.default

        direct = self._pairs.get(frozenset((candidate_family.value, target_family.value)))
        if direct is not None:
            return direct
        return self.default
