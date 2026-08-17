"""Pydantic contracts.

Two rules govern this file.

1. ``RawProfile`` and ``ExperienceEntry`` are **tolerant**. Missing ``end_date``,
   a ``duration_months`` that disagrees with the date span, a null ``experience``
   list, numbers arriving as strings: everything coerces and records a flag. A
   parse failure on row 7 of a million must not kill the batch, and Part 3 sends
   partial records by design.
2. ``SignalRecord`` / ``RankingRecord`` / ``ChangeEvent`` match Appendix A field
   names exactly. The only additions are ``role_family_scores`` and
   ``extraction_flags`` on the signal record, which are purely additive.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from saral.contracts.taxonomy import RoleFamily, Seniority


def _coerce_optional_int(value: Any) -> int | None:
    """``"42"`` -> 42, ``""`` -> None, ``"junk"`` -> None. Never raises."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        text = value.strip().replace(",", "")
        if not text:
            return None
        try:
            return int(float(text))
        except ValueError:
            return None
    return None


def _coerce_optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().casefold()
        if text in {"true", "yes", "y", "1"}:
            return True
        if text in {"false", "no", "n", "0"}:
            return False
    return None


def _coerce_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value if v is not None and str(v).strip()]
    return []


class ExperienceEntry(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str | None = None
    company_name: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    is_current: bool | None = None
    duration_months: int | None = None
    job_type: str | None = None
    work_type: str | None = None
    location: str | None = None
    description: str | None = None
    skills_used: list[str] = Field(default_factory=list)

    @field_validator("duration_months", mode="before")
    @classmethod
    def _dm(cls, v: Any) -> int | None:
        return _coerce_optional_int(v)

    @field_validator("is_current", mode="before")
    @classmethod
    def _cur(cls, v: Any) -> bool | None:
        return _coerce_optional_bool(v)

    @field_validator("skills_used", mode="before")
    @classmethod
    def _su(cls, v: Any) -> list[str]:
        return _coerce_str_list(v)

    @field_validator("role", "company_name", "job_type", "work_type", "location", "description", mode="before")
    @classmethod
    def _str(cls, v: Any) -> str | None:
        if v is None:
            return None
        return str(v)


class EducationEntry(BaseModel):
    model_config = ConfigDict(extra="allow")

    degree: str | None = None
    field_of_study: str | None = None
    school_name: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    period: str | None = None
    skills: list[str] = Field(default_factory=list)

    @field_validator("skills", mode="before")
    @classmethod
    def _sk(cls, v: Any) -> list[str]:
        return _coerce_str_list(v)


class RawProfile(BaseModel):
    """A candidate profile exactly as it arrives from the crawler."""

    model_config = ConfigDict(extra="allow")

    id: str
    name: str | None = None
    location: str | None = None
    linkedin_url: str | None = None
    headline: str | None = None
    about: str | None = None
    is_open_to_work: bool | None = None
    skills: list[str] = Field(default_factory=list)
    experience: list[ExperienceEntry] = Field(default_factory=list)
    education: list[EducationEntry] = Field(default_factory=list)
    total_experience_months: int | None = None
    text_context_full: str | None = None
    created_at: str | None = None
    updated_at: str | None = None

    @field_validator("skills", mode="before")
    @classmethod
    def _skills(cls, v: Any) -> list[str]:
        return _coerce_str_list(v)

    @field_validator("experience", "education", mode="before")
    @classmethod
    def _lists(cls, v: Any) -> list[Any]:
        if v is None:
            return []
        if isinstance(v, list):
            return [x for x in v if isinstance(x, dict)]
        return []

    @field_validator("total_experience_months", mode="before")
    @classmethod
    def _tem(cls, v: Any) -> int | None:
        return _coerce_optional_int(v)

    @field_validator("is_open_to_work", mode="before")
    @classmethod
    def _otw(cls, v: Any) -> bool | None:
        return _coerce_optional_bool(v)


class TenureStability(BaseModel):
    avg_tenure_months: float
    jobs_last_36m: int
    flag: Literal["hopper", "moderate", "stable"]


class SignalRecord(BaseModel):
    """Appendix A.1. Field names are the contract -- do not rename."""

    candidate_id: str
    role_family: RoleFamily
    role_family_alt: list[RoleFamily] = Field(default_factory=list)
    seniority: Seniority
    years_total: float
    years_relevant: float
    core_skills: list[str] = Field(default_factory=list)
    claimed_skills_unverified: list[str] = Field(default_factory=list)
    skill_noise_ratio: float
    tenure_stability: TenureStability
    switch_intent: float
    confidence: float
    reason_codes: list[str] = Field(default_factory=list)
    signals_version: str
    computed_at: str
    input_hash: str

    # --- additive, not part of the Appendix A contract -------------------
    role_family_scores: dict[str, float] = Field(default_factory=dict)
    extraction_flags: list[str] = Field(default_factory=list)


class JobSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    job_id: str
    title: str
    company: str | None = None
    location: str | None = None
    min_years: float
    max_years: float
    budget_lpa: list[float] = Field(default_factory=list)
    must_have: list[str] = Field(default_factory=list)
    good_to_have: list[str] = Field(default_factory=list)
    raw_query: str = ""


class RankingRecord(BaseModel):
    """Appendix A.2."""

    job_id: str
    candidate_id: str
    rank: int
    fit_score: int
    score_breakdown: dict[str, float]
    reason_codes: list[str] = Field(default_factory=list)
    missing_must_haves: list[str] = Field(default_factory=list)
    confidence: float


class ChangeEvent(BaseModel):
    """Appendix A.3."""

    event_id: str
    candidate_id: str
    observed_at: str
    source: str | None = None
    change_type: str
    field: str
    old_value: Any = None
    new_value: Any = None
    materiality: Literal["high", "medium", "low", "noise"]
    downstream: list[str] = Field(default_factory=list)
    signals_recomputed: bool = False
    note: str | None = None
    affected_signals: list[str] = Field(default_factory=list)
