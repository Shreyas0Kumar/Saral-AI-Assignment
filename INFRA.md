# INFRA.md — running this for real

AWS **ap-south-1** on-demand, August 2026. Every throughput figure is measured on
this repo's `out/run_manifest.json` (AMD Zen 3, 8 cores, single-threaded Python
3.11), not estimated. **All five questions are answered above the rule; the
arithmetic behind the answers is below it.**

## Where it runs

| Component | Service | Why |
|---|---|---|
| Extract / score workers | **ECS Fargate**, 1 vCPU / 2 GB | CPU-bound, stateless, sub-5 ms, bursty with the crawl. Per-second billing, no cluster. |
| Profiles, signals, `field_state`, events | **Aurora Postgres** | A delta failing halfway must not leave half-applied state. Needs transactions. |
| Vector search | **Aurora pgvector** | The whole point: `role_family`, `seniority`, `years_relevant` become **SQL predicates**, so search runs pre-filtered. Vectors must sit beside the filters. |
| Raw crawl payloads | **S3**, Glacier after 90d | Replay and audit, off the hot path. |
| Signal cache | **ElastiCache**, keyed by `input_hash` | `input_hash` already *is* a cache key. |
| Change feed | **SQS** off the events writer | `high` fans out to recruiter alerts; `noise` is dropped at the producer. |

All already in the stack. No inference endpoint, no model server, no GPU —
nothing on the hot path is a neural network. **Caveat:** this repo ships SQLite,
so Aurora is argued, not validated.

## What it costs at 1M profiles, re-scored weekly

Measured **1.31 ms/profile** single-threaded (A1). Everything below is per
month, at 4.33 weekly passes:

```
compute   0.36 vCPU-h/pass -> $0.017 vCPU + $0.001 mem = $0.018/pass  ≈ $0.08
Aurora Serverless v2, 2 ACU baseline                                 ≈ $175
Aurora storage, 40 GB @ $0.11/GB                                     ≈ $4.40
S3 8 GB + ElastiCache t4g.micro + SQS 300k msgs                      ≈ $12.32
                                                                       -------
                                                                       ≈ $192
```

**The signal layer is ~0.1% of that bill** — 0.36 CPU-hours of string handling
per million rows, no model inference in it. So the reason to build it is not its
own cost: `role_family = 'backend' AND years_relevant BETWEEN 5 AND 9` cuts the
pgvector search space ~10x, shrinking the ACU baseline, and takes the LLM
validation step off the query path entirely.

**Incremental (Part 3)** saves $0.05 a month, **and I will not pretend that is
the argument.** 7 of 25 changed materially, so 72% of profiles never re-rank,
re-embed, invalidate a cache or fire an alert — and 2 of 19 events were
emoji-only noise, which extrapolates to 110,000 avoided re-scores per
million-profile refresh. That 28% comes from one 10-record fixture and is the
number to re-derive with real churn data.

## CPU or GPU

**CPU, and it is not close.** The hot path is a dict lookup, a regex, and — on
the ~16% of titles the lexicon abstains on — one sparse matrix multiply. A GPU
cannot accelerate a hash lookup.

Against the alternative, an LLM per row, measured four ways, all
schema-constrained so every error is a reasoning error rather than a parse error:

| arm | s/profile | `role_family` correct | cost / 1M |
|---|---|---|---|
| **signals_v1 (shipped)** | **0.0013** | **25/25** | **$0.018** |
| gemini-3.5-flash-lite | 1.25 | 24/25 | ~$36 (per token) |
| gemma3:1b (local) | 2.68 | 17/25 | $35 (744 CPU-h) |
| qwen2.5:3b (local) | 5.03 | 20/25 | $65 (1,397 CPU-h) |

**Gemini gets 24 of 25, so the accuracy argument is weaker than I expected.** The
real case is ~2,000x cost, 1.25 s against 1.3 ms (which alone rules it out of a
synchronous path), rate limits (HTTP 429 after 14 calls), and a third-party
dependency on the field that gates search. Its one error is SDB_10019, the
profile Appendix A uses to define the problem, and **every model tested gets him
wrong** (A4).

The `g4dn.xlarge` already being paid for serves the *validation* LLM on the query
path. This component's job is to shrink what reaches it: 10x fewer survivors and
that GPU takes 10x the queries or gets downsized. That is where the GPU spend
moves. For this component the answer never flips (A5).

## How I know it broke

Every alarm is a **distribution over the corpus**, not a model internal. "Model
drift" is not observable; "17% of profiles became `non_engineering` overnight" is.

| Alarm | Threshold | Now | A breach means |
|---|---|---|---|
| `non_engineering` share | > 3 SD w/w | 12% | The lexicon stopped matching engineering titles — or started matching everything. |
| Lexicon abstention rate | > 25% of titles | 16% | Title drift. The **leading** indicator. |
| Fallback invocation rate | > 15% | 0% | The same, lagging. |
| Mean `confidence` | drops > 0.05 w/w | 0.87 | Thinner profiles: a crawler regression, not a model one. |
| Delta noise-event ratio | falls below 40% | 11%, on a feed built to be eventful — a steady-state crawl should be mostly noise | The normaliser broke, and every emoji is about to cost a re-score. |
| `suspected_deletion` rate | > 5% of fields | — | The crawler failing on a page template, not users deleting headlines. Protects the null policy. |
| p95 extract latency | > 8 ms | 2.9 ms | The fallback is firing far more than expected. The threshold sits well above the 2.8–3.3 ms spread measured over six quiesced runs, not on one reading. |
| `/health` degraded | any | healthy | A deploy shipped code against signals nobody recomputed. |

## What breaks first

1. **The lexicon, silently.** 145 patterns written against 25 profiles. At 1M
   rows abstention climbs, and because an abstained entry contributes *nothing*
   rather than something wrong, profiles get quietly thinner rather than visibly
   wrong. Fix: sample abstained titles weekly, batch through an offline LLM, fold
   confirmed ones in. **That loop is the actual product, and it is not built.**
2. **The distilled LR generalises to nothing** — 0 of 10 novel titles (FL-006).
   Safe because it abstains; not yet a fallback.
3. **Write contention on `field_state`** — 7M rows and 7M potential updates per
   refresh. Fine at 1M, not 20M. Fix: one upsert per candidate, or DynamoDB.
4. **`skill_aliases.yaml`.** Matching is exact-after-alias and a missed skill
   looks identical to an absent one, with no abstention signal to alarm on. Fix:
   offline embedding similarity over the skill vocabulary, hot path unchanged.

---
---

# Appendix — the working

Not part of the one page. Here because the brief asks for visible arithmetic.

## A1. Where 1.31 ms comes from, and why the spread matters

`run_manifest.json → derived.cost_per_1m_profiles`, the committed run, including
cold start; six consecutive runs on a quiesced machine spanned 1.300–1.366 ms.
Warm steady-state p50 is 0.595 ms and p95 2.859 ms (`metrics.json →
latency_ms.extract.batch_1`).

An earlier set of runs, taken while a local LLM saturated the same 8 cores,
reported **2.2–5.0 ms for the identical code**. Any per-profile number measured
on a loaded desktop is a number about the desktop, so every figure above comes
from an idle machine and states its spread.

An earlier draft of this file said 4.13 ms and $0.06 — wrong by exactly 2x,
because the telemetry accumulated wall time across two extraction passes while
*assigning* rather than accumulating the record count. Caught by checking the
document against the data it claims to summarise (`FAILURE_LOG.md` FL-007).

## A2. Full cost breakdown

```
Full pass over 1M profiles
  1,000,000 × 1.31 ms          = 1,313 s   = 0.365 vCPU-hours
  0.365 vCPU-h × $0.04656      = $0.0170
  0.182 GB-h   × $0.00511      = $0.0009   (0.5 GB task)
                                 --------
  per full pass                  $0.0179
  × 4.33 weekly passes/month   = $0.078 / month

Storage and I/O — the part that actually costs money
  Aurora Serverless v2, 2 ACU baseline           ≈ $175.00 / month
  Aurora storage, 1M profiles ≈ 40 GB @ $0.11/GB ≈   $4.40
  pgvector index, 1M × 384-dim fp32 ≈ 1.5 GB     — fits the 2 ACU
  S3 raw payloads, 1M × ~8 KB = 8 GB @ $0.025/GB ≈   $0.20
  ElastiCache cache.t4g.micro                    ≈  $12.00
  Fargate, extract + score, ~2 vCPU-h/month      ≈   $0.15
  SQS, ~300k material events/month               ≈   $0.12
                                                   --------
  total                                          ≈ $192 / month

Incremental vs full recompute
  (compute only, same 1.31 ms basis as above -- excludes the $0.001/pass memory)
  full recompute weekly  0.36 vCPU-h/pass × 4.33 = 1.58 vCPU-h/mo = $0.073
  incremental at 28%     0.10 vCPU-h/pass × 4.33 = 0.44 vCPU-h/mo = $0.021
  saved                                                            $0.05 / month
```

## A3. LLM-per-row arms, priced two ways

Local arms are priced in CPU-seconds, the hosted arm in tokens. Different cost
models, so they are not folded into one number.

| arm | s/profile | valid JSON | `role_family` correct | cost model |
|---|---|---|---|---|
| **signals_v1 (shipped)** | **0.0013** | n/a | **25/25** | CPU-seconds |
| gemini-3.5-flash-lite | 1.25 | 25/25 | 24/25 (96%) | per token |
| gemma3:1b (local, Ollama) | 2.68 | 25/25 | 17/25 (68%) | CPU-seconds |
| qwen2.5:3b (local, Ollama) | 5.03 | 25/25 | 20/25 (80%) | CPU-seconds |
| SmolLM2-135M (unconstrained harness) | 42.7 | 11/25 | 1/25 (4%) | CPU-seconds |

```
Self-hosted, in CPU-seconds
  gemma3:1b    1M × 2.68 s = 744 vCPU-h   × $0.04656 = $35/pass = $150 /month
  qwen2.5:3b   1M × 5.03 s = 1,397 vCPU-h × $0.04656 = $65/pass = $282 /month
  signals_v1   1M × 0.0013 s = 0.36 vCPU-h           = $0.018/pass = $0.08 /month
                                                       ~2,000x and ~3,600x

Hosted, in tokens
  measured     232.5 input + 32.9 output tokens per profile (mean over 25)
  1M profiles  232.5M input + 32.9M output
  at $0.10/M input, $0.40/M output:
               232.5 × $0.10 = $23.25
                32.9 × $0.40 = $13.16
                               ------
               per full pass   $36.41  = $158 /month at weekly re-scoring
```

The token counts are measured; **the rates are an assumption and should be
replaced with current published pricing before anyone budgets on this.** Latency
is irrelevant to a hosted bill — only prompt size is — which is why the fastest
arm in the table is not the cheapest.

## A4. SDB_10019, head to head

| system | verdict |
|---|---|
| hand label | `non_engineering` |
| **signals_v1 (shipped)** | **`non_engineering`** |
| gemini-3.5-flash-lite | `data_scientist` |
| qwen2.5:3b-instruct | `ml_engineer` |
| gemma3:1b | `data_engineer` |

Not fixed by scale: 135M, 1B, 3B and a hosted model all make it. Scope: n=25 on
my own labels — it shows the failure exists and survives scaling; it does not
quantify its rate.

## A5. If a transformer ever lands on the ingest path

The crossover is roughly where GPU-hours × $0.526 beats CPU-hours × $0.04656 —
about **11x throughput**, which a T4 clears comfortably for batched encoder
inference. That is the moment to revisit, not before. A `g4dn.xlarge` left
running is $4,600/year.
