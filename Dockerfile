# Served image: no torch, no transformers, no model download at build time.
# The distilled classifier is a ~180KB joblib artifact committed to the repo.
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first, so a source change does not invalidate the wheel layer.
COPY requirements.txt ./
RUN pip install --no-cache-dir --timeout 120 --retries 5 -r requirements.txt

COPY pyproject.toml ./
COPY src/ ./src/
COPY config/ ./config/
COPY data/ ./data/
# The cached baseline embeddings (47KB). With these present the cosine baseline
# is a numpy matmul, so the image reproduces the full metrics comparison --
# including the per-job deltas the dashboard renders -- without torch.
COPY out/baseline_embeddings.npz ./out/baseline_embeddings.npz
# Measured evidence produced by the offline scripts (the LLM-per-row cost arm
# and the fallback comparison). The pipeline folds these into metrics.json, so
# without them the container's dashboard silently omits its most important
# comparison -- the shipped extractor against an LLM per row.
COPY out/llm_cost_arm.json out/fallback_comparison.json ./out/
RUN pip install --no-cache-dir --no-deps -e .

# Build the SQLite store and every out/ artefact at image build time, so
# /health is meaningful and /dashboard is populated the moment the container
# starts rather than reporting "degraded" until someone runs the pipeline.
RUN python -m saral.cli --arm signals_v1_lexicon_lr all

RUN useradd --create-home --uid 10001 saral && chown -R saral:saral /app
USER saral

EXPOSE 8000

# Exercises the same code path a load balancer would.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

CMD ["uvicorn", "saral.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
