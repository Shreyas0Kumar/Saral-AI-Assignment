"""The service. Not a prototype -- the thing that could sit behind a gateway.

Three endpoints and a health check that means something.

Notes on what is deliberately *not* here:

* No torch. The served image installs `requirements.txt` only; the embedding
  baseline and the MiniLM arm live in `requirements-eval.txt` and stay out of
  production. That is roughly 800MB of image.
* No model load at startup. The distilled LR loads lazily on first cache miss,
  which is what makes the latency distribution bimodal and legible rather than
  a flat average hiding its structure. The production answer is a startup
  warm-up, which is a deliberate tradeoff rather than an oversight: lazy
  loading is what makes the fallback's cost visible in p95.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from saral.adapters.store.sqlite_repo import SqliteRepo
from saral.config_loader import load_all
from saral.contracts.models import JobSpec, RankingRecord, SignalRecord
from saral.contracts.versions import SCORING_VERSION, SIGNALS_VERSION
from saral.core.extract.pipeline import extract
from saral.core.score.features import parse_job
from saral.core.score.scorer import rank_candidates
from saral.pipeline import io
from saral.pipeline.arms import SHIPPED_ARM, build_classifier

DB_PATH = Path(io.ROOT / "out" / "saral.db")

app = FastAPI(
    title="SARAL candidate signals",
    version=SIGNALS_VERSION,
    description=(
        "Structured signal extraction and fit scoring. The hot path makes no LLM "
        "call: a deterministic lexicon over normalised titles, with a distilled "
        "logistic regression that abstains rather than guesses."
    ),
)

_state: dict[str, Any] = {}


def _cfg():
    if "cfg" not in _state:
        cfg, weights = load_all()
        _state["cfg"] = cfg
        _state["weights"] = weights
        _state["classifier"] = build_classifier(SHIPPED_ARM, cfg)
    return _state["cfg"], _state["weights"], _state["classifier"]


def _repo() -> SqliteRepo | None:
    if "repo" not in _state:
        _state["repo"] = SqliteRepo(DB_PATH) if DB_PATH.exists() else None
    return _state["repo"]


# --------------------------------------------------------------------------
# request / response models
# --------------------------------------------------------------------------
class ExtractRequest(BaseModel):
    profiles: list[dict[str, Any]] = Field(..., min_length=1)
    #: Injected, never read from the clock -- identical input, identical output.
    as_of: str | None = Field(
        None, description="YYYY-MM-DD. Defaults to the corpus snapshot date derived from the batch."
    )
    computed_at: str | None = None


class ExtractResponse(BaseModel):
    signals: list[SignalRecord]
    count: int
    elapsed_ms: float
    signals_version: str
    fallback_invocations: int


class ScoreRequest(BaseModel):
    job: JobSpec
    candidate_ids: list[str] | None = Field(
        None, description="Defaults to every candidate with stored signals."
    )
    profiles: list[dict[str, Any]] | None = Field(
        None, description="Score profiles supplied inline instead of reading the store."
    )
    top_k: int | None = None


class ScoreResponse(BaseModel):
    job_id: str
    job_family: str
    results: list[RankingRecord]
    count: int
    elapsed_ms: float
    scoring_version: str


# --------------------------------------------------------------------------
# endpoints
# --------------------------------------------------------------------------
@app.post("/signals/extract", response_model=ExtractResponse)
def signals_extract(request: ExtractRequest) -> ExtractResponse:
    cfg, _weights, classifier = _cfg()
    try:
        as_of = (
            datetime.strptime(request.as_of, "%Y-%m-%d").date()
            if request.as_of
            else io.corpus_as_of(request.profiles)
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"bad as_of: {exc}") from exc
    computed_at = (
        datetime.fromisoformat(request.computed_at.replace("Z", "+00:00"))
        if request.computed_at
        else datetime.now(timezone.utc)
    )

    started = time.perf_counter()
    records: list[SignalRecord] = []
    fallbacks = 0
    for raw in request.profiles:
        if "id" not in raw:
            raise HTTPException(status_code=422, detail="every profile needs an 'id'")
        record, _trace = extract(raw, cfg, classifier, computed_at, as_of)
        fallbacks += classifier.fallback_invocations()
        records.append(record)

    return ExtractResponse(
        signals=records,
        count=len(records),
        elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
        signals_version=SIGNALS_VERSION,
        fallback_invocations=fallbacks,
    )


@app.post("/score", response_model=ScoreResponse)
def score(request: ScoreRequest) -> ScoreResponse:
    cfg, weights, classifier = _cfg()
    started = time.perf_counter()

    if request.profiles:
        as_of = io.corpus_as_of(request.profiles)
        computed_at = datetime.now(timezone.utc)
        pairs = [extract(p, cfg, classifier, computed_at, as_of) for p in request.profiles]
        signals = [record for record, _ in pairs]
        traces = {record.candidate_id: trace for record, trace in pairs}
    else:
        repo = _repo()
        if repo is None:
            raise HTTPException(
                status_code=503,
                detail="no signal store; run `saral all` first or pass profiles inline",
            )
        profiles = {p["id"]: p for p in repo.load_profiles()}
        wanted = request.candidate_ids or sorted(profiles)
        missing = [cid for cid in wanted if cid not in profiles]
        if missing:
            raise HTTPException(status_code=404, detail=f"unknown candidate_ids: {missing}")
        as_of = io.corpus_as_of(list(profiles.values()))
        computed_at = io.DEFAULT_COMPUTED_AT
        pairs = [extract(profiles[cid], cfg, classifier, computed_at, as_of) for cid in wanted]
        signals = [record for record, _ in pairs]
        traces = {record.candidate_id: trace for record, trace in pairs}

    try:
        parsed = parse_job(request.job, cfg, classifier)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    ranked = rank_candidates(signals, traces, parsed, cfg, weights)
    if request.top_k:
        ranked = ranked[: request.top_k]

    return ScoreResponse(
        job_id=parsed.spec.job_id,
        job_family=parsed.family.value,
        results=ranked,
        count=len(ranked),
        elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
        scoring_version=SCORING_VERSION,
    )


@app.get("/health")
def health() -> dict[str, Any]:
    """Checks something real.

    Reads the signals table, requires a non-zero row count, and requires those
    rows to have been written by the `signals_version` the running code
    declares. Returns 503 when unhealthy so a load balancer acts on it -- a 200
    that is returned unconditionally is a lie with a status code.
    """
    payload: dict[str, Any] = {
        "signals_version": SIGNALS_VERSION,
        "scoring_version": SCORING_VERSION,
    }
    try:
        cfg, _weights, _classifier = _cfg()
        payload["config_hashes"] = cfg.config_hashes
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"config unreadable: {exc}") from exc

    repo = _repo()
    if repo is None:
        payload.update(
            status="degraded",
            reason="no signal store on disk; extraction works, /score needs inline profiles",
        )
        return payload

    payload.update(repo.health())
    if payload.get("status") == "unhealthy":
        raise HTTPException(status_code=503, detail=payload)
    return payload


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> str:
    from saral.api.dashboard import render_dashboard

    return render_dashboard()
