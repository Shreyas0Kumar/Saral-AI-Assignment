# SARAL candidate signals and fit scoring

A structured signal layer over raw candidate profiles, a fit scorer that
explains itself, an offline evaluation harness, and an incremental update pass.

**The hot path makes zero LLM calls.** Extraction is a deterministic lexicon
over normalised job titles, with a distilled logistic regression that abstains
rather than guesses. Measured at **1.0 ms p50 per profile** on the machine
stated below.

---

## Run it

Requires Python 3.11+. No network, no GPU, no Ollama, no model download.

```bash
pip install -r requirements.txt
pip install -e . --no-deps

make all          # extract -> score -> evaluate -> delta, writes everything in out/
make test         # 118 tests
```

`make all` prints:

```
[1/4] extract: 25 signals
[2/4] score: 100 rankings
[3/4] evaluate: out/metrics.json
[4/4] delta: 19 change events
```

### The embedding baseline

The required cosine baseline needs `sentence-transformers`, which pulls in
torch. It is **not** in `requirements.txt`, because it is not in the served
image either.

```bash
pip install -r requirements-eval.txt   # adds torch + sentence-transformers
make all
```

Without it, `make all` still completes: the baseline block in `metrics.json`
carries an explicit error and instructions rather than a crash or a silent zero.
Embeddings are cached to `out/baseline_embeddings.npz` (committed), so a second
run needs no download.

### The service

```bash
make serve                        # http://127.0.0.1:8000
curl -s localhost:8000/health | python -m json.tool
```

```bash
# one or more raw profiles -> signal records
curl -s -X POST localhost:8000/signals/extract \
  -H 'content-type: application/json' \
  -d "{\"profiles\": [$(head -1 data/candidates.jsonl)]}" | python -m json.tool

# a job + candidates -> ranked results with scores and reason codes
curl -s -X POST localhost:8000/score \
  -H 'content-type: application/json' \
  -d "{\"job\": $(python -c "import json;print(json.dumps(json.load(open('data/jobs.json'))[1]))"), \"top_k\": 5}" \
  | python -m json.tool
```

`GET /dashboard` renders the metrics, the per-job deltas, the cost table and a
per-job candidate drill-down with reason codes. Served from the same process;
no second port, no Streamlit.

### Docker

```bash
make docker-build          # or: docker build -t saral-signals:1.0.0 .
make docker-run            # or: docker run --rm -p 8000:8000 saral-signals:1.0.0
curl -s localhost:8000/health
```

The image installs `requirements.txt` only, so it contains no torch,
transformers or sentence-transformers — verified, not assumed:

```
$ docker exec <container> python -c "import torch"
ModuleNotFoundError: No module named 'torch'
```

**439 MB**, against roughly 1.2 GB with the deep-learning stack. It runs the
whole pipeline at build time, so `/health` returns `healthy` with 25 signals and
19 change events the moment the container starts, rather than `degraded` until
someone remembers to run the pipeline.

---

## What is in `out/`

| file | what |
|---|---|
| `candidate_signals.jsonl` | Appendix A.1, one record per candidate |
| `rankings.jsonl` | Appendix A.2, one record per (job, candidate) -- all 25 ranked for all 4 jobs |
| `change_events.jsonl` | Appendix A.3, Part 3 |
| `metrics.json` | Appendix A.4, plus ablation, per-job deltas, error analysis, latency |
| `delta_report.json` | incremental vs full recompute, and the saving |
| `fallback_comparison.json` | the two fallback classifiers measured on their own task |
| `llm_cost_arm.json` | a real LLM per row, measured on CPU |
| `run_manifest.json` | git sha, machine, config hashes, per-stage throughput, derived cost |

---

## Headline numbers

Measured on **AMD Zen 3, 8 physical / 16 logical cores, 42 GB RAM, Windows 11,
Python 3.11.9, single-threaded**.

| | baseline (MiniLM cosine) | signals_v1 | delta |
|---|---|---|---|
| NDCG@10 | 0.878 | 0.970 | +0.092 |
| NDCG@5 | 0.893 | 0.944 | +0.050 |
| Precision@5 | 0.700 | 0.750 | +0.050 |

**The improvement is not distinguishable from noise.** The bootstrap 95%
interval on the NDCG@10 delta is `[-0.005, 0.203]` and contains zero. The four
per-job deltas are `+0.294, +0.110, -0.029, -0.007` -- they do not agree in
sign, so the mean is carried by JD-001. With four jobs as the independent unit,
no experiment on this dataset can establish significance. This is stated at
length in `metrics.json → uncertainty.verdict` and in `WRITEUP.md`, and it is
the number I would most want discussed.

Latency, warmed up, 500 iterations:

| batch | p50 | p95 |
|---|---|---|
| 1 profile | 1.0 ms | 4.3 ms |
| 100 profiles | 181 ms | 211 ms |

Cost of one full pass over 1M profiles, derived from measured throughput:

| arm | per profile | CPU-hours / 1M | Fargate ap-south-1 |
|---|---|---|---|
| signals_v1 (shipped) | 2.2 ms | 0.61 | **$0.03** |
| `llm_per_row`, SmolLM2-135M on CPU | 40,430 ms | 11,231 | **$523** |

The LLM arm is ~18,400x more expensive **and** got 1 of 25 role families right.
Working in `INFRA.md`.

---

## Layout

```
src/saral/
  contracts/   pydantic schemas, taxonomy, adjacency, closed reason vocabulary
  core/        extract, score, delta, evaluation      <- PURE: no I/O, no clock
  adapters/    sqlite store, embedding baseline, classifier loaders
  pipeline/    orchestration, arm registry, run manifest
  api/         FastAPI app + dashboard
config/        lexicon, aliases, adjacency, weights (YAML, hashed into the manifest)
               + committed LLM-derived artifacts
scripts/       offline artifact generation. Never on the default path.
tests/         118 tests
```

`core/` imports no `sqlite3`, `requests`, `fastapi`, `torch`, `yaml` or `os`,
and never reads a clock -- `computed_at` and `as_of` are injected. This is
enforced by an AST-walking test (`tests/test_core_purity.py`), not by
discipline, and it is why Part 3 idempotency is a plain equality assertion
rather than a diff with a timestamp exclusion list.

---

## What was cut, and why

* **A second fallback arm shipped.** The MiniLM nearest-centroid classifier is
  built and measured (`out/fallback_comparison.json`) but not shipped. It gets
  0 of 5 right on the titles the lexicon actually abstains on, because it has no
  way to abstain. Details in `WRITEUP.md`.
* **A large generation model for the offline corpus.** `Phi-3.5-mini` is not
  fully present in the local model cache and would need a 4.9 GB download, which
  breaks the no-network constraint. The committed synthetic corpus was generated
  by Claude Code instead (`AI_LOG.md` AI-005); `scripts/generate_llm_artifacts.py
  --backend local` reproduces the pipeline with any local model.
* **Partial signal recomputation.** The dependency map exists and reports which
  signals a change affects, but a dirty candidate has their whole `SignalRecord`
  recomputed. Extraction is ~2 ms; the saving that matters is candidate-level
  (7 of 25, not 25 of 25). Claiming an optimisation I did not build would be
  worse than saying this.
* **Calibrated switch intent.** There is no outcome label anywhere in the
  dataset to calibrate against. The logit is monotone and interpretable and is
  labelled uncalibrated rather than dressed up.

## Files worth reading first

1. `WRITEUP.md` -- what was built, what failed, where it fails silently, AI usage.
2. `FAILURE_LOG.md` -- six entries with the hypotheses that were abandoned.
3. `INFRA.md` -- one page, AWS, with the arithmetic visible.
