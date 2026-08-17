"""Parse a job spec into scoreable requirements.

The job specs are written the way recruiters write them, not the way a scorer
would like them:

    "FastAPI or Django"                          -> alternatives
    "embeddings / retrieval / ranking"           -> alternatives
    "offline eval (NDCG, recall@k)"              -> a concept plus its examples
    "Airflow or equivalent orchestrator"         -> a named tool plus a concept
    "production ownership of a high-traffic service"  -> not a skill at all

The last one matters most. Three of the four jobs have a must-have that is an
*evidence* requirement rather than a skill: something the person must have
**done**, which no skills list can satisfy. Treating it as a skill would let
anyone who lists "Python" collect the point. So requirements are typed, and an
evidence requirement is checked against experience descriptions only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from saral.contracts.models import JobSpec
from saral.contracts.taxonomy import RoleFamily
from saral.core.normalize import norm_skill, norm_text

RequirementKind = Literal["skill", "evidence"]

#: Phrases that make a requirement about demonstrated work rather than a tool.
_EVIDENCE_MARKERS = (
    "ownership", "owned", "own ", "shipped", "shipping", "production",
    "in prod", "at scale", "experience", "has built", "built and",
    "end to end", "end-to-end", "customer-facing", "customer facing",
)

#: Concept -> the tokens that satisfy it, for requirements naming a category
#: rather than a product ("equivalent orchestrator", "cloud warehouse").
_CONCEPT_SYNONYMS: dict[str, tuple[str, ...]] = {
    "orchestrator": ("airflow", "prefect", "dagster", "luigi", "oozie", "step functions", "adf", "azure data factory"),
    "cloud warehouse": ("snowflake", "bigquery", "redshift", "databricks", "synapse"),
    "vector db": ("faiss", "pgvector", "pinecone", "weaviate", "milvus", "qdrant", "chroma", "vector databases"),
    "offline eval": ("ndcg", "recall@k", "mrr", "map@k", "auc", "offline eval", "evaluation", "llm evaluation"),
    "embeddings": ("embeddings", "sentence-transformers", "faiss", "transformers", "pgvector"),
    "retrieval": ("retrieval", "rag", "faiss", "search", "vector databases"),
    "ranking": ("ranking", "reranker", "ndcg", "learning to rank", "recommender"),
    "mlops": ("mlflow", "mlops", "kubeflow", "sagemaker", "feature store"),
    "inference latency optimisation": ("triton inference server", "onnx", "tensorrt", "quantization", "latency"),
    "performance work": ("lcp", "latency", "p95", "p99", "performance", "webpack", "bundle"),
    "design system experience": ("design system", "storybook", "component library", "accessibility"),
}

_SPLIT = re.compile(r"\s+or\s+|\s*/\s*|\s*,\s*|\s*\|\s*")
_PAREN = re.compile(r"\(([^)]*)\)")


@dataclass(frozen=True)
class Requirement:
    raw: str
    kind: RequirementKind
    #: normalised alternatives; satisfying any one satisfies the requirement
    alternatives: tuple[str, ...]
    #: for evidence requirements, the phrases to look for in a description
    evidence_terms: tuple[str, ...] = ()

    @property
    def label(self) -> str:
        """Stable, readable identifier used inside reason codes."""
        return re.sub(r"[^a-z0-9]+", "_", norm_text(self.raw)).strip("_")[:48]


@dataclass(frozen=True)
class ParsedJob:
    spec: JobSpec
    family: RoleFamily
    must_have: tuple[Requirement, ...]
    good_to_have: tuple[Requirement, ...]
    city: str
    is_remote: bool
    family_evidence: str = ""
    extra: dict = field(default_factory=dict)


def _expand(text: str, aliases: dict[str, str]) -> tuple[str, ...]:
    """Split a requirement string into normalised alternatives."""
    parenthetical = " ".join(_PAREN.findall(text))
    stem = _PAREN.sub(" ", text)
    parts = [p for p in _SPLIT.split(f"{stem} , {parenthetical}") if p.strip()]

    out: list[str] = []
    for part in parts:
        cleaned = norm_text(part)
        cleaned = re.sub(r"\b(equivalent|any|preferred|experience with|experience in|a|an|the)\b", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if not cleaned:
            continue
        canonical = norm_skill(cleaned, aliases)
        if canonical:
            out.append(canonical)
        for concept, synonyms in _CONCEPT_SYNONYMS.items():
            if concept in cleaned:
                out.extend(norm_skill(s, aliases) for s in synonyms)
    # Preserve order, drop duplicates.
    seen: set[str] = set()
    return tuple(o for o in out if o and not (o in seen or seen.add(o)))


def _is_evidence(text: str) -> bool:
    normalized = norm_text(text)
    return any(marker in normalized for marker in _EVIDENCE_MARKERS)


def _evidence_terms(text: str) -> tuple[str, ...]:
    """Content words a description must contain to satisfy an evidence requirement."""
    normalized = norm_text(text)
    stop = {
        "of", "a", "an", "the", "and", "or", "to", "in", "on", "with", "for",
        "has", "have", "been", "at", "experience", "must",
    }
    words = [w for w in re.findall(r"[a-z0-9-]+", normalized) if w not in stop and len(w) > 2]
    return tuple(words)


def parse_requirement(text: str, aliases: dict[str, str]) -> Requirement:
    if _is_evidence(text):
        return Requirement(
            raw=text,
            kind="evidence",
            alternatives=_expand(text, aliases),
            evidence_terms=_evidence_terms(text),
        )
    return Requirement(raw=text, kind="skill", alternatives=_expand(text, aliases))


def _job_city(location: str | None) -> tuple[str, bool]:
    normalized = norm_text(location)
    remote = "remote" in normalized
    city = _PAREN.sub(" ", normalized)
    city = re.sub(r"\b(remote|hybrid|on-site|onsite|india)\b", " ", city)
    city = re.sub(r"[^a-z ]", " ", city)
    return re.sub(r"\s+", " ", city).strip(), remote


def parse_job(spec: JobSpec, cfg, classifier) -> ParsedJob:
    """Resolve the job's implied role family from its title, then its query."""
    result = classifier.classify(spec.title, spec.raw_query)
    family = result.family
    evidence = result.evidence
    if family is None:
        result = classifier.classify(spec.raw_query, spec.raw_query)
        family, evidence = result.family, f"raw_query:{result.evidence}"
    if family is None:
        # Never silently default to a family -- an unresolved job would score
        # every candidate as a mismatch and nobody would know why.
        raise ValueError(
            f"could not resolve a role family for {spec.job_id} ({spec.title!r}); "
            "add the title to config/lexicon.yaml"
        )

    city, remote = _job_city(spec.location)
    aliases = cfg.skills.aliases
    return ParsedJob(
        spec=spec,
        family=family,
        must_have=tuple(parse_requirement(m, aliases) for m in spec.must_have),
        good_to_have=tuple(parse_requirement(g, aliases) for g in spec.good_to_have),
        city=city,
        is_remote=remote,
        family_evidence=evidence,
    )
