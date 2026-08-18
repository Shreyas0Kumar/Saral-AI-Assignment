# Everything a reviewer needs. `make all` works from a clean clone with no
# network, no Ollama, no GPU and no model download: the LLM-derived artifacts in
# config/ are committed, and the only step that needs a model download is the
# embedding baseline, which degrades to a clear message rather than a crash.

PY ?= python
PIP ?= $(PY) -m pip

.PHONY: help install install-dev install-eval all extract score evaluate delta test serve \
        docker-build docker-run clean regenerate-llm-artifacts cost-arm

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-26s\033[0m %s\n", $$1, $$2}'

install:  ## install the served dependencies only (no torch)
	$(PIP) install -r requirements.txt
	$(PIP) install -e . --no-deps

install-dev:  ## the above plus pytest and httpx, enough to run `make test` (still no torch)
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -e . --no-deps

install-eval:  ## install everything, including torch, for the baseline and offline scripts
	$(PIP) install -r requirements-eval.txt
	$(PIP) install -e . --no-deps

all:  ## extract -> score -> evaluate -> delta, writing every file in out/
	$(PY) -m saral.cli all

extract:  ## out/candidate_signals.jsonl
	$(PY) -m saral.cli extract

score:  ## out/rankings.jsonl
	$(PY) -m saral.cli score

evaluate:  ## out/metrics.json
	$(PY) -m saral.cli evaluate

delta:  ## out/change_events.jsonl, applying the feed twice to prove idempotency
	$(PY) -m saral.cli delta --reapply

test:  ## the full suite
	$(PY) -m pytest tests/ -q

serve:  ## run the API on :8000
	$(PY) -m uvicorn saral.api.app:app --host 0.0.0.0 --port 8000

docker-build:  ## build the served image
	docker build -t saral-signals:1.0.0 .

docker-run:  ## run it
	docker run --rm -p 8000:8000 saral-signals:1.0.0

# ---------------------------------------------------------------------------
# Offline artifact regeneration. NEVER on the default path -- `make all` uses
# the committed artifacts so a clean clone reproduces identical output.
# ---------------------------------------------------------------------------
regenerate-llm-artifacts:  ## rebuild the synthetic corpus, distilled LR and centroids
	$(PY) scripts/build_holdout_labels.py
	$(PY) scripts/generate_llm_artifacts.py --backend ollama --model qwen2.5:3b-instruct
	$(PY) scripts/train_distilled_lr.py
	$(PY) scripts/compare_fallbacks.py

# `--backend` must be stated: it defaults to ollama, and a HuggingFace repo id
# is not an ollama tag. Without it this target exits on "ollama does not have
# HuggingFaceTB/SmolLM2-135M-Instruct" before measuring anything.
cost-arm:  ## measure a real LLM per row on CPU (slow, ~25 min)
	$(PY) scripts/run_llm_cost_arm.py --backend transformers --model HuggingFaceTB/SmolLM2-135M-Instruct

clean:
	rm -rf out/saral.db out/*.npz .pytest_cache
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
