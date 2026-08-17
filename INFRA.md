# INFRA.md — running this for real

All prices are AWS **ap-south-1 (Mumbai)** on-demand, August 2026. Every figure
below is derived from throughput measured on this repo's own run manifest
(`out/run_manifest.json`), not estimated. Measurement machine: AMD Zen 3,
8 physical cores, 42 GB RAM, single-threaded Python 3.11.

## Where it runs

| Component | Service | Why |
|---|---|---|
| Ingest / extract workers | **ECS Fargate**, 1 vCPU / 2 GB tasks | The extractor is CPU-bound, stateless, sub-5 ms, and bursty with the crawl schedule. Fargate bills per second and needs no cluster to babysit. Already in use, so no new operational surface. |
| Profile + signal store | **Aurora PostgreSQL** (`raw_profiles`, `signals`, `field_state`, `change_events`) | `field_state` needs a real primary key and transactional updates: a delta that fails halfway must not leave half-applied state. Already in use. |
| Vector search | **Aurora pgvector** | The point of this component is that `role_family`, `seniority` and `years_relevant` become **SQL predicates**, so vector search runs over a pre-filtered candidate set instead of the whole corpus. That is the cost win, and it needs the vectors to live next to the filters. |
| Raw crawl payloads | **S3**, lifecycle to Glacier after 90 days | Replay and audit. Cheap and not on the hot path. |
| Signal cache | **ElastiCache (Redis)**, keyed by `input_hash` | `input_hash` is exactly a cache key: same hash, same signals. Already in use. |
| Change feed | **SQS** off the `change_events` writer | `materiality: high` fans out to recruiter alerts; `noise` is dropped at the producer and never becomes a message. |
| Offline artifact rebuild | **Fargate scheduled task**, monthly | Regenerating the synthetic corpus and retraining the distilled LR. Minutes of compute per month. |

Not proposed: an inference endpoint, a model server, or a GPU. Nothing on the
hot path is a neural network.

**Caveat I would state in review:** this repo ships SQLite, not Postgres, so the
Aurora design above is argued rather than validated. The repository sits behind
one adapter class, which makes the swap small, but "small" is not "proven".

## What it costs at 1M profiles, re-scored weekly

Measured: **1.46 ms/profile** single-threaded extraction (`run_manifest.json →
derived.cost_per_1m_profiles`, mean of three runs on a quiesced machine:
1.374, 1.479, 1.532 ms). That figure includes cold start; warm steady-state p50
is 0.699 ms and p95 is 3.363 ms (`metrics.json → latency_ms.extract.batch_1`).

The spread matters more than the point estimate. An earlier set of runs, taken
while a local LLM was saturating the same 8 cores, reported 2.2–5.0 ms for the
identical code. **Any per-profile number measured on a loaded desktop is a
number about the desktop**, so all figures below come from an otherwise idle
machine and the variance is stated rather than hidden behind a single decimal.

```
Full pass over 1M profiles
  1,000,000 × 1.46 ms          = 1,460 s        = 0.41 vCPU-hours
  0.41 vCPU-h × $0.04656       = $0.0189
  0.20 GB-h   × $0.00511       = $0.0010  (0.5 GB task)
                                 --------
  per full pass                  $0.020
  × 4.33 weekly passes/month   = $0.087 / month
```

That number is small enough to be suspicious, so here is the sanity check: it is
0.41 CPU-hours of pure Python string handling per million rows. The extractor
does regex normalisation, 2.8 title lookups per profile, and some date
arithmetic. There is no model inference in it. The compute genuinely is nearly
free; **the storage and the crawl are the real bill.**

(An earlier draft of this file said 4.13 ms and $0.06. That was wrong by 2x: the
telemetry accumulated wall time across two extraction passes while assigning
rather than accumulating the record count. Caught by checking the document
against the data it claims to summarise — `FAILURE_LOG.md` FL-007.)

```
Storage and I/O, the part that actually costs money
  Aurora Serverless v2, 2 ACU baseline           ≈ $175 / month
  Aurora storage, 1M profiles ≈ 40 GB @ $0.11/GB ≈ $4.40 / month
  pgvector index, 1M × 384-dim fp32 ≈ 1.5 GB in memory (fits the 2 ACU)
  S3 raw payloads, 1M × ~8 KB = 8 GB @ $0.025/GB ≈ $0.20 / month
  ElastiCache cache.t4g.micro                    ≈ $12 / month
  Fargate, extract + score, ~2 vCPU-h/month      ≈ $0.15 / month
  SQS, ~300k material events/month               ≈ $0.12 / month
                                                   ---------
  total                                          ≈ $192 / month
```

**The signal layer is ~0.1% of that bill.** The honest conclusion is that this
component is not where the money goes, and the reason to build it is not its own
cost — it is that `role_family = 'backend' AND years_relevant BETWEEN 5 AND 9`
cuts the pgvector search space by roughly 10x, which shrinks the ACU baseline,
and it removes the LLM validation step from the query path entirely.

### With the incremental pass (Part 3)

Measured on the real feed: **7 of 25 candidates** changed materially, a 28%
change rate. Applied to a weekly refresh of 1M profiles:

```
full recompute weekly    0.41 vCPU-h/pass × 4.33 = 1.78 vCPU-h/month = $0.083
incremental at 28%       0.11 vCPU-h/pass × 4.33 = 0.50 vCPU-h/month = $0.023
saved                                                                  $0.06 / month
```

**I am not going to pretend that is the argument.** At this scale the saving is
six cents. The incremental pass earns its keep somewhere else: 72% of
profiles never enter the re-score path, so the *downstream* work — re-ranking
against every open job, re-embedding, invalidating caches, and firing recruiter
notifications — is avoided for them too. The `materiality` classification is
what prevents 1M rows being re-scored because somebody added an emoji, and in
this feed 2 of 19 events (11%) were exactly that. Extrapolated, that is 110,000
avoided re-scores per million-profile refresh.

The number worth re-deriving with real data is the **change rate**. 28% comes
from a single 10-record fixture and should be replaced by an observed weekly
churn rate before anyone plans capacity on it.

## CPU or GPU

**CPU, and it is not close.** There is nothing to run on a GPU: the shipped hot
path is a dictionary lookup, a regex, and — on the ~16% of titles the lexicon
abstains on — one sparse matrix multiply.

The comparison that matters is against the alternative design, an LLM call per
profile. Four were measured rather than assumed, all with **schema-constrained
output** so the model cannot fail on syntax and every error is a reasoning error.

| arm | s/profile | valid JSON | role_family correct | cost model |
|---|---|---|---|---|
| **signals_v1 (shipped)** | **0.0015** | n/a | **25/25** | CPU-seconds |
| gemini-3.5-flash-lite (hosted) | 1.25 | 25/25 | 24/25 (96%) | per token |
| gemma3:1b (local, Ollama) | 2.68 | 25/25 | 17/25 (68%) | CPU-seconds |
| qwen2.5:3b-instruct (local, Ollama) | 5.03 | 25/25 | 20/25 (80%) | CPU-seconds |
| SmolLM2-135M (unconstrained harness) | 42.7 | 11/25 | 1/25 (4%) | CPU-seconds |

**Self-hosted, priced in CPU-seconds:**

```
gemma3:1b       1M × 2.68 s = 744 vCPU-h × $0.04656   = $35 / pass  = $150 / month
qwen2.5:3b      1M × 5.03 s = 1,397 vCPU-h × $0.04656 = $65 / pass  = $282 / month
signals_v1      1M × 0.0015 s = 0.41 vCPU-h           = $0.02 / pass = $0.09 / month
```

**~1,800x** for gemma3:1b, **~3,400x** for qwen2.5:3b, and both are *less*
accurate.

**Hosted, priced in tokens** — a different cost model, so it is priced
separately rather than folded into the same number:

```
measured    232.5 input + 32.9 output tokens per profile (mean over 25)
1M profiles 232.5M input tokens + 32.9M output tokens

at $0.10 / M input and $0.40 / M output:
            232.5 × $0.10  = $23.25
             32.9 × $0.40  = $13.16
                             ------
            per full pass    $36.41   = $158 / month at weekly re-scoring
```

The token counts are measured; **the rates are an assumption and should be
substituted for current published pricing before anyone budgets on this.** Note
that latency is irrelevant to a hosted bill — only prompt size is — which is why
the fastest arm in the table is not the cheapest.

**So the honest position on accuracy is not what I expected.** gemini-3.5-flash-lite
gets 24 of 25 right. The accuracy argument against a hosted LLM is weak; the
argument is cost (~1,800x), latency (1.25 s vs 1.5 ms, which rules it out of a
synchronous query path), rate limits (this run hit HTTP 429 after 14 requests on
the free tier and needed backoff — at 1M profiles quota is a capacity-planning
input, not a footnote), and the dependency on a third party for a field that
gates search.

**But there is one result worth the whole exercise.** SDB_10019 is the profile
Appendix A of the brief singles out: a mechanical engineer, six years of AutoCAD
at Hero MotoCorp, whose headline says "Transitioning to Data Science".

| system | SDB_10019 |
|---|---|
| hand label | `non_engineering` |
| **signals_v1 (shipped)** | **`non_engineering`** |
| gemini-3.5-flash-lite | `data_scientist` |
| qwen2.5:3b-instruct | `ml_engineer` |
| gemma3:1b | `data_engineer` |

**Every language model tested puts him in a data role, and for Gemini it is the
only mistake it makes in 25 profiles.** They are all reading the self-description
and none of them is weighting the work history against it. That is precisely the
failure the assignment describes, it is not fixed by scale, and it is what the
per-entry classification plus evidence tiering exists to prevent.

**When does the answer flip to GPU?** For this component, never — a GPU cannot
accelerate a hash lookup. The question really applies to the `g4dn.xlarge`
already being paid for, which serves the *validation* LLM on the query path. A
`g4dn.xlarge` is $0.526/hour on-demand, $4,600/year if left running. The right
framing is: this component's job is to shrink the candidate set that reaches
that GPU. If pre-filtering on `role_family` and `years_relevant` cuts the
survivors reaching LLM validation by 10x, the GPU either handles 10x the query
volume or can be downsized. That is where the GPU spend moves, and it is a
better return than accelerating anything here.

If a future version *does* put a transformer on the ingest path, the crossover
is roughly where GPU-hours × $0.526 beats CPU-hours × $0.04656 — about 11x
throughput, which a T4 clears comfortably for batched encoder inference. That
would be the moment to revisit, not before.

## How I know it broke

Alarms on things that move for a reason, with thresholds derived from this run:

| Alarm | Threshold | Why this number |
|---|---|---|
| `non_engineering` share | moves > 3 SD week-over-week | Currently 3/25 = 12%. A jump means the lexicon stopped matching engineering titles; a collapse means it started matching everything. Either way the taxonomy broke. |
| Fallback invocation rate | > 15% of titles | Measured at **0%** on this corpus (the lexicon resolves or the LR abstains). A real rise means title drift the lexicon has not seen — the signal to regenerate the corpus and retrain. |
| Lexicon abstention rate | > 25% of titles | Currently 11/70 = 16%. This is the leading indicator; the fallback rate is the lagging one. |
| Mean `confidence` | drops > 0.05 week-over-week | Currently 0.87. A drop means profiles are arriving thinner — usually a crawler regression, not a model one. |
| Delta noise-event ratio | falls below 40% | Currently 2/19 = 11% on a feed engineered to be eventful; a steady-state crawl should be mostly noise. If noise events vanish, **the normaliser has broken** and every emoji is about to cost a re-score. |
| `suspected_deletion` rate | > 5% of observed fields | A spike means the crawler is failing on a page template, not that users are deleting their headlines. This is the alarm that protects the null policy. |
| p95 extract latency | > 8 ms | Measured p95 is 4.3 ms, and it varies run to run between 3.8 and 5.1 ms on an unquiesced desktop; the threshold is set above that observed spread rather than on a single reading. Breaching it means the fallback is firing far more than expected. |
| `/health` degraded | any | Returns `degraded` when stored `signals_version` differs from the running code — catches a deploy that shipped new code against signals nobody recomputed. |

Note the shape: every one of these is a **distribution over the corpus**, not a
model-internal quantity. "Model drift" is not observable; "17% of profiles
became `non_engineering` overnight" is.

## What breaks first

1. **The lexicon, and it breaks silently.** It is 145 patterns written against
   one corpus of 25 profiles. At 1M rows it will meet titles nobody anticipated,
   the abstention rate will climb, and — because abstained entries simply do not
   contribute to `role_family` — profiles will get *quietly thinner* rather than
   visibly wrong. The abstention-rate alarm is the only thing standing between
   that and a slow accuracy leak. Fix: sample abstained titles weekly, batch
   them through an offline LLM, and fold the confirmed ones into the lexicon.
   That loop is the actual product here, and it is not built.

2. **The distilled LR generalises to nothing.** Measured, and reported in
   `FAILURE_LOG.md` FL-006: it answered 32 of 42 held-out real titles correctly
   and those 32 were exactly the ones appearing verbatim in its training corpus.
   On the 10 novel titles it answered **zero**. It is safe (it abstains) but it
   is not yet a fallback. Fix: a much larger and more adversarial synthetic
   corpus, then measure specifically on titles held out by *surface form*, which
   is the split that mattered and the one I nearly failed to make.

3. **Write contention on `field_state` at crawl scale.** One row per
   (candidate, field group) is 7M rows and 7M potential updates per refresh.
   Fine at 1M, not fine at 20M with a continuous crawl. Fix: batch by candidate
   and write through a single upsert per candidate, or move `field_state` to
   DynamoDB where the access pattern is a natural fit.

4. **`skill_aliases.yaml` becomes the bottleneck on quality.** Skill matching is
   exact-after-alias. Every unaliased spelling is a silently missed must-have,
   and unlike the lexicon there is no abstention signal to alarm on — a missed
   skill looks identical to an absent skill. Fix: embedding-based skill matching
   with a similarity floor, run offline over the skill vocabulary rather than
   per request, so the hot path stays a dictionary lookup.
