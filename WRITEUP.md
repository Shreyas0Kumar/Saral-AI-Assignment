# WRITEUP.md

## What I built

A structured signal layer that reads a raw profile and emits a scoreable record,
a fit scorer whose every point is traceable to a reason code, an offline
evaluation harness, and an incremental update pass.

The hot path makes **no LLM call**. Extraction is a deterministic lexicon over
normalised job titles; where a title is genuinely ambiguous the entry's own
description and `skills_used` cast a weighted vote; where that fails it abstains
and hands the title to a distilled logistic regression, which also abstains
below a confidence floor. Measured at 0.88 ms p50 per profile.

Three decisions carry most of the design.

**Classification is per experience entry, never over the profile blob.**
SDB_10019 is the proof: the headline says "Transitioning to Data Science", the
skills list says "Machine Learning", and the work history says six years of
AutoCAD at Hero MotoCorp. A blob classifier — which is what a cosine baseline
effectively is — lands him in a data role. Per-entry classification, weighted by
duration and decayed with a 36-month half-life, lands him in `non_engineering`
where the work actually is. The same mechanism resolves SDB_10020 (Yahoo SWE
46m, Uber Senior ML 45m, Atlassian EM 55m and current) to `engineering_manager`
rather than `ml_engineer`, which is what a recruiter would say.

**A declared skill is not an evidenced skill.** A skill counts as `core` only if
it appears in an experience entry's `skills_used`, as a token in a description,
or in an education record. Otherwise it is `claimed_skills_unverified` and earns
30% credit toward a must-have instead of 100%. This is the one thing cosine
similarity structurally cannot do: an embedding of `text_context_full` sees
"Machine Learning" in a list and moves the vector, with no notion of whether the
work history corroborates it.

**Nothing is ever filtered out.** Missing must-haves cost 8 points each; a
candidate whose `role_match` scores zero has their total capped at 35. Both are
visible in `score_breakdown`. A dropped candidate emits no reason codes, and a
recruiter who cannot see *why* somebody was excluded cannot trust the ones who
were included.

## The evaluation, and what it actually says

| | baseline (MiniLM cosine) | signals_v1 | delta |
|---|---|---|---|
| NDCG@10 | 0.878 | 0.970 | +0.092 |
| NDCG@5 | 0.893 | 0.944 | +0.050 |
| Precision@5 | 0.700 | 0.750 | +0.050 |

**This improvement is not distinguishable from noise, and I would not defend it
as real.** The bootstrap 95% interval on the NDCG@10 delta is `[-0.005, 0.203]`
and contains zero. More tellingly, the four per-job deltas are `+0.294`,
`+0.110`, `-0.029`, `-0.007`: they do not agree in sign. The mean is carried by
JD-001, where the baseline ranked a Java/Spring engineer and an ML engineer over
two of the three strongest backend candidates. On the other two jobs my system
is marginally worse. A paired sign-flip test over the four per-job deltas gives
p = 0.25, and with n = 4 the smallest p it could possibly return is 0.0625 — so
that test cannot establish significance at any effect size, which is worth
saying out loud rather than reporting as a near miss.

Three further things about these numbers that I would rather point out than have
found:

* **NDCG@10 barely discriminates here.** Each job has 10–16 labelled
  candidates, so a condensed @10 cut covers most of the list. MRR is 1.00 for
  *both* systems — both put a relevant candidate first in all four jobs — so it
  carries no information and is reported only to show that it is saturated.
* **JD-004's Precision@5 cannot exceed 0.40 for any system**, because only 2 of
  its 10 labelled candidates are graded ≥ 2. Averaging P@5 across jobs with
  different ceilings without disclosing it is quietly misleading. Both systems
  hit 0.40 on JD-004, i.e. both are already at ceiling.
* **Both list conventions are reported.** Condensed (rank 25, keep the labelled
  ones in order) is primary; zero-fill is alongside. Zero-fill punishes a system
  for surfacing a good candidate the recruiter never graded, which is exactly
  what this system is built to produce; condensed lists flatter everything
  equally. The ranking does not flip between them.

### The ablation ladder, including the rung that did nothing

| rung | NDCG@10 | Δ |
|---|---|---|
| 1. role adjacency only | 0.9417 | — |
| 2. + must-have coverage | 0.9572 | +0.0155 |
| 3. + evidence-vs-claim discount | 0.9572 | **+0.0000** |
| 4. + seniority band, production evidence, location, intent, tenure, cap | 0.9701 | +0.0129 |

Rung 3 is the rule I called the core thesis, and on this data it is worth
exactly zero NDCG. I am reporting it as zero. The reason is not that the rule is
inert — it changes scores and reason codes on SDB_10010, SDB_10019 and
SDB_10023 — but that role adjacency at rung 1 has *already* pushed those three
to the bottom, so there is nothing left for evidence tiering to fix. Its value
is a claim about corpora containing many such profiles, and this corpus contains
three. That is a claim I can only support by argument, not by this measurement.

Note also that rung 1 alone reaches 0.9417 against the baseline's 0.878. Most of
the measured gain is `role_family` — the single field the brief identifies as
most valuable — and comparatively little is the rest of the scorer.

### The three worst misses

Ranked by discounted gain given up against the ideal ordering. **Nothing was
tuned after reading this**; the fixes are future work.

1. **JD-002 / SDB_10002, graded 3, ranked 2nd** (NDCG loss 2.58). Marked as
   missing the must-have "offline eval (NDCG, recall@k)" — while his experience
   description literally reads *"Trained cross-encoder reranker, +6% NDCG@10"*.
   The cause is a real design gap: skill-type requirements are matched against
   the declared skills list, and evidence-type requirements against
   descriptions, and nothing checks a skill requirement against a description.
   The system penalised him for the evidence being in the *wrong field*, which
   is the exact inversion of the evidence-tiering principle everything else here
   is built on. Fix: match skill requirements against the union of skills and
   description tokens, tiered — description-only evidence should score *above*
   an unevidenced claim, not below a declared one.

2. **JD-003 / SDB_10011, graded 3, ranked 4th** (loss 1.40). Zero missing
   must-haves and still ranked below a candidate graded 2. Two causes: 7.0
   relevant years against a `[2,6]` band costs 6 of 15 seniority points, and the
   good-to-have coverage is thin because "Databricks/Delta Lake" is not aliased
   to "cloud warehouse". The seniority penalty is arguably correct behaviour and
   arguably a labelling disagreement — the recruiter did not treat one year over
   the band as a negative. This is the clearest case where I would want the
   grader in the room.

3. **JD-001 / SDB_10022, graded 3, ranked 3rd** (loss 0.92). Missing "FastAPI or
   Django" and "PostgreSQL", and he genuinely lists neither — he is an Amazon
   SDE-3 on serverless and DynamoDB. Two missing must-haves cost 16 points. The
   recruiter graded him 3 anyway, valuing the seniority and the scale evidence
   over the exact stack. This is not a bug in the extractor; it is a statement
   that my `missing_must_have` penalty of 8 points is too steep relative to how
   this recruiter actually weighs a stack mismatch for a senior hire. I would
   not change it on n=1.

## What I tried and dropped

**A MiniLM nearest-centroid fallback** (`FL-006`). Built, measured, cut. The
important part is *how* it was cut. The distilled LR initially reported 100%
accuracy on everything it answered over the hand-labelled held-out entries,
which is a reason to look harder rather than to celebrate. Splitting the holdout
by whether the title appears verbatim in the training corpus showed that the 32
titles it answered were exactly the 32 it had seen, and that of the 10 novel
titles it answered **zero**. Its measured generalisation is not high — it is
undefined. I then checked whether the semantic fallback rescued that case, since
generalisation is the entire argument for keeping MiniLM: it answers all 10
novel titles and gets **0 of 10** right ("Software Engineer" → `ml_engineer`,
"SDE II" → `devops_sre`, "Design Engineer" → `engineering_manager`).

Narrowed to the only population a fallback ever sees in production — the titles
where the lexicon genuinely abstains, of which there are 5 — the LR answers 0
and MiniLM answers 5 and gets 5 wrong.

So the measured contribution of the fallback layer on this corpus is **zero**,
and I report it as zero. The LR ships not because it classifies better but
because it *abstains correctly*: it can only fill a gap, never override a
decision the lexicon got right. That safety property is the only one that was
actually measured. Those five ambiguous titles are resolved correctly by the
lexicon's own context vote — the architecture was already covering the case the
fallback was hired for.

**An LLM call per profile** (`llm_per_row`). Not rejected on principle;
measured four ways, all on **schema-constrained output** so no model can fail on
syntax and every error is a reasoning error.

| arm | s/profile | valid JSON | role_family correct | cost / 1M |
|---|---|---|---|---|
| **signals_v1 (shipped)** | **0.0015** | n/a | **25/25** | **$0.02** |
| gemini-3.5-flash-lite (hosted) | 1.25 | 25/25 | 24/25 | ~$36, per token |
| gemma3:1b (local, Ollama) | 2.68 | 25/25 | 17/25 | $35, 744 CPU-h |
| qwen2.5:3b-instruct (local, Ollama) | 5.03 | 25/25 | 20/25 | $65, 1,397 CPU-h |

**This went differently from how I expected and I am reporting the version that
is worse for my argument.** I assumed a per-row LLM would lose on accuracy.
Against a decent hosted model it does not — Gemini gets 24 of 25. So "cheap
*and* more accurate" collapses to "cheap", and the real case against a per-row
LLM is: ~1,800x the cost, 1.25 s against 1.5 ms of latency (which alone rules it
out of a synchronous query path), free-tier rate limits that returned HTTP 429
after 14 consecutive calls and required backoff, and a third-party dependency on
the one field that gates search.

**But the single best result in this submission came out of that comparison.**
SDB_10019 is the profile Appendix A holds up as the canonical failure —
mechanical engineer, six years of AutoCAD at Hero MotoCorp, headline reads
"Transitioning to Data Science":

| system | verdict on SDB_10019 |
|---|---|
| hand label | `non_engineering` |
| **signals_v1 (shipped)** | **`non_engineering`** |
| gemini-3.5-flash-lite | `data_scientist` |
| qwen2.5:3b-instruct | `ml_engineer` |
| gemma3:1b | `data_engineer` |

**Every language model tested gets him wrong, and for Gemini it is the only
mistake it makes in 25 profiles.** All of them read the self-description; none
weights six years of work history against one line of aspiration. The failure is
not fixed by scale — 135M, 1B, 3B and a hosted frontier-lite model all make it.
That is a far stronger claim than "rules are cheaper", and it is exactly what
per-entry classification and evidence tiering exist to prevent. (Honest scope:
n=25, my own labels, one profile deep. It shows the failure exists and survives
scaling; it does not quantify its rate on a real corpus.)

Two further things the LLM arms exposed:

* **A schema constrains shape, not meaning.** Both local models returned
  `years_relevant` in *months* for 22 of 25 — copied off `duration_months` —
  and every value passed validation. A pre-filter reading
  `years_relevant BETWEEN 5 AND 9` against that field would select the wrong
  candidates forever with no error anywhere.
* **Constrained decoding degrades with output length.** The same mechanism that
  gave 25/25 valid objects in the cost arm produced no parseable array for 10 of
  12 families when asked for 45 titles at once (`FL-010`).

I also got the first version of this comparison wrong in my own favour
(`FL-009`). My initial harness ran SmolLM2-135M through a naive `transformers`
loop with no output constraint and reported 40.4 s/profile and 4% accuracy — 14
of its 25 "errors" were unparseable JSON rather than wrong answers. I had
written that using a large model to prove LLMs are expensive would be
sandbagging, then sandbagged in the other direction by giving a small model a
harness that could not succeed. That run is kept in `out/llm_cost_arm.json`
labelled as unconstrained, because the gap between it and the fair test is the
lesson.

**Fitting `skill_noise_ratio` to Appendix A's worked example** (`FL-003`,
`AI-002`). The plan I started from asserted the formula was "confirmed" by the
appendix's 0.57 for SDB_10019. It is not: under a strict evidence rule the
answer is 0.857, and reaching 0.57 requires crediting `solidworks` and `excel`
as evidenced when neither appears anywhere in his work history. I checked the
alternative reading that *does* produce 0.57 — domain congruence rather than
evidence — and rejected it, because it hits the printed number by dropping
SDB_10010's noise ratio from 0.97 to 0.29, disabling the signal on the
thirty-one-skill fresher it exists to catch. The brief says the appendix values
are illustrative and not a target to reproduce, so the divergence is reported.

**Trusting the wall clock for `as_of`** (`FL-002`). The corpus turns out to be
snapshotted around 2025-08 while its own `created_at` says 2026-06 and the delta
feed says 2026-08. Using "today" fired `duration_mismatch` on nearly every
current role. `as_of` is now derived from the data — the median of
`start_date + duration_months` over current roles — which is also the right
production behaviour, since a profile crawled six months ago should be aged
against when it was crawled.

## Part 3, briefly

Idempotency falls out of one comparison operator: field state is rejected when
`_observed_at <= stored.observed_at`, so re-applying a feed skips every field,
emits zero material events, and leaves byte-identical state. `observed_at` is
tracked **per field group**, not per record, so a partial record carrying one
stale field and two fresh ones applies the fresh two.

The traps I found and defended against are enumerated as one test each in
`tests/test_delta.py`. In the real feed: a duplicated `SDB_10009` line, a
headline gaining a rocket emoji and a double space (resolved to `noise` by the
normaliser with no branch in the delta engine), two `about` fields arriving
null, a location change to Berlin, and — the highest blast radius —
`SDB_10001`'s partial experience list containing only the two Razorpay roles.
Replacing wholesale there would have destroyed his Freshworks history, which
destroys `years_relevant`, which destroys `seniority`, which destroys every
downstream score. Four further traps that are *not* in the feed (unknown
candidate, missing `_observed_at`, out-of-order timestamps, numbers as strings)
are tested against synthetic records so the coverage is legible either way.

The asymmetry worth defending: **skills replace wholesale, experience merges by
key**. A skills list is a snapshot of a current claim, and a shrinking list is
information. Career history is append-mostly and a partial crawl must never
delete it. The mirror-image cost is that a removed fabricated job is retained
forever; the production fix for both that and the null policy is corroboration
across two consecutive runs, which cannot be demonstrated with one delta file.

Writing the idempotency assertion *before* believing the code caught a real bug:
the null-field path emitted `suspected_deletion` but never advanced
`observed_at`, so every re-run re-emitted it. It now advances the timestamp
while leaving the hash alone — we did observe the field, we chose not to apply
what we saw — which is also the state a corroboration policy would need.

## Where it fails silently

* **Terse profiles are punished.** Evidence tiering's known false negative: a
  real engineer whose descriptions are one line each has most skills land in
  `claimed_skills_unverified`. Mitigated (claims still earn 30% credit, and
  weak evidence lowers `confidence` rather than the score) but real. Median
  `skill_noise_ratio` on this corpus is ~0.7, which tells you how common terse
  descriptions are.
* **Skill matching is exact-after-alias, and a missed skill is invisible.** An
  unaliased spelling looks identical to an absent skill. Unlike the lexicon,
  there is no abstention signal to alarm on. This is the failure I would fix
  first at scale.
* **A skill requirement evidenced only in a description is scored as missing.**
  Worst-miss #1 above. Silent because the score just comes out lower.
* **The lexicon degrades quietly, not loudly.** An abstained entry contributes
  nothing to `role_family` rather than contributing something wrong, so at scale
  profiles get *thinner* rather than visibly incorrect, and aggregate accuracy
  drifts down with no error to catch.
* **`switch_intent` is not calibrated and cannot be.** There is no outcome label
  anywhere in the dataset. It is monotone and interpretable and its 0.71 means
  nothing in probability terms. Calibrating it needs recruiter reply/accept
  outcomes plus Platt scaling on a held-out slice, with reliability curves by
  decile.
* **The 35-point cap is a discontinuity.** Deliberate and visible in the
  breakdown, but it is a discontinuity, and a learned model with enough labels
  should not need it.

## What two more weeks would buy, ranked

1. **A labelled title set from recruiters, then retrain.** Everything upstream
   is bounded by 42 titles I labelled myself. 5,000 recruiter-graded titles
   would turn the distilled LR from a memoriser into a classifier and let the
   holdout be split by surface form honestly. Highest value by a distance.
2. **Close the skill/description matching gap** (worst miss #1), and replace
   exact-after-alias skill matching with offline embedding similarity over the
   skill vocabulary, keeping the hot path a dictionary lookup.
3. **The abstention feedback loop.** Sample abstained titles weekly, batch them
   through an offline LLM, fold confirmed ones into the lexicon. This is the
   actual product and it is not built.
4. **More labels per job, and more jobs.** With 4 jobs, no experiment here can
   establish significance. 20 jobs × 40 labels would make the ablation ladder
   mean something.
5. **Calibrate `switch_intent`** against reply outcomes.
6. **Corroboration-based deletion policy** across two consecutive crawls.

## Assumptions

Every ambiguity I resolved in code, with what I chose:

1. **`as_of` is derived from the corpus, not the clock** — median of
   `start_date + duration_months` over current roles. Base = 2025-08-01,
   post-delta = 2026-08-15. The data's `created_at` disagrees with its own
   content (FL-002).
2. **`computed_at` is a frozen constant** in `make all` so runs are byte-
   reproducible. Overridable via `--computed-at`; the API defaults to now.
3. **A skill is evidenced only by `skills_used`, a description token, or an
   education record.** The headline and `about` are explicitly not evidence —
   they are the self-description this component exists to distrust.
4. **`skill_noise_ratio` = |unevidenced| / |declared|**, giving 0.857 for
   SDB_10019 rather than Appendix A's illustrative 0.57 (FL-003).
5. **`role_family_alt` is experience-derived only.** It is not "families the
   profile gestures at", because it feeds `role_match` at 0.6 weight and a
   skills-derived alt would hand the mechanical engineer 12.6 points on an ML
   job. For 24 of 25 profiles it is correctly empty.
6. **`non_engineering` stays primary when it wins the argmax**, even if an
   engineering family is close behind.
7. **`staff+` is reachable only via title evidence**, never by accumulating
   years.
8. **A job's role family is resolved from its title, then its `raw_query`**, and
   an unresolvable job raises rather than defaulting — a silent default would
   score everyone as a mismatch with no visible cause.
9. **Must-haves are typed.** "production ownership of a high-traffic service" is
   an evidence requirement checked against descriptions, not a skill. Production
   evidence = an ownership verb *or* a quantified outcome.
10. **Production ownership is deliberately double-weighted** — once in
    `skill_overlap` as a must-have, once as the 12-point
    `evidence_of_shipping` component — because both the brief's example
    breakdown and recruiter behaviour treat it as the discriminator.
11. **A field arriving `null` is "not observed"** and is not applied; a
    `suspected_deletion` event is emitted at low materiality (FL-005 rationale
    in `core/delta/materiality.py`).
12. **Skills replace wholesale; experience merges by `(normalised_company,
    start_date)` and never deletes unseen entries.** Nulls inside a merged entry
    also do not overwrite.
13. **An unknown `candidate_id` in the feed creates a thin profile** rather than
    being dropped; thinness shows up in `confidence`.
14. **A record with no `_observed_at` sorts oldest**, so it can never overwrite
    known state, and emits a `malformed_record` event.
15. **`location` is a scoring input, not a signal** — a location change is high
    materiality but recomputes no `SignalRecord`.
16. **Ties break deterministically** (score, then confidence, then
    `candidate_id`) so rankings are reproducible.
17. **Weights, adjacency and the 36-month half-life were frozen before the first
    metric was computed** and were not tuned afterwards.

## AI tool usage

**Claude Code (Opus 5)** was the primary implementation tool, driven from a plan
and a decision log I wrote first. Full log in `AI_LOG.md`.

**Where it saved time.** Roughly 3,000 lines of pydantic contracts,
normalisation, date arithmetic, YAML validation, the SQLite adapter and the
FastAPI layer — mechanical code where the design was already settled. It also
generated the 528-title synthetic corpus that the distilled LR trains on, which
is a legitimate offline use of an LLM under the brief's own terms and is
committed so the default path needs no model.

**Where it was plausibly wrong, and what I did.** The plan asserted:

> "Validation: SDB_10019 has 7 declared skills, 4 unevidenced (python, machine
> learning, pandas, matplotlib), giving 0.571. Appendix A states 0.57. **The
> formula is confirmed against the spec's own worked example.**"

This is plausible because the arithmetic is correct, the claimed split is
exactly the one printed in Appendix A, and reverse-engineering a spec from its
own example is a real technique. It is also wrong. Implementing it and running
it against the actual profile gives **0.857**: his single experience entry lists
only `AutoCAD` in `skills_used`, has an empty `description`, and his education
record has no skills, so `solidworks` and `excel` are corroborated nowhere. The
appendix's split is not derivable from the appendix's own input.

I caught it because I ran the number before writing it down. What I did instead:
kept the strict rule, tested the alternative formulation that *would* hit 0.57
(domain congruence) and rejected it on evidence — it breaks SDB_10010 — and
documented the divergence in a test docstring, `FAILURE_LOG.md` FL-003 and
Assumption 4.

The failure mode is worth naming precisely, because it is the one I now watch
for: it was not a wrong fact, it was a **fabricated confirmation**. The word
"confirmed" was attached to a check that had never been run. Had I trusted it, I
would have tuned a core signal to match an illustrative example and then
reported that fit as validation — which is exactly the measurement mistake this
brief says it would rather see caught than beaten.

Two smaller cases are logged: the plan cited SDB_10021's freelance work as
requiring overlap resolution (no overlap exists anywhere in the corpus —
`AI-003`), and it predicted out-of-order timestamps in the delta feed that are
not there (the duplicate-line trap is what is actually there — `FL-006`
neighbours). In both cases the generated code was structurally sound and
empirically untested; every one of the eleven `FAILURE_LOG.md` entries comes from
running it against the data rather than from reading it.

The last two are worth singling out because they were caught at the very end, by
checking artefacts against each other rather than by reading code. FL-007: the
throughput figure I was about to publish in INFRA.md was exactly 2x too high,
because the telemetry divided an accumulated wall time by a non-accumulated
record count — visible only because the same quantity was measured twice by
different code paths and the two disagreed. FL-008: a byte-level diff between a
clean `git clone` and the source repo showed `change_events.jsonl` differing
only in `event_id`, which was a `uuid4`. Fixing it to be content-addressed
turned out to matter for more than reproducibility: `event_id` is the primary
key of the `change_events` table and inserts use `INSERT OR IGNORE`, so with
random ids a re-applied feed would have written duplicate rows in the database
while my in-memory idempotency test kept passing. I had a green test and a real
bug at the same time, at different layers.
