# WRITEUP.md

> I built a structured signal layer: per-experience-entry role classification
> with recency decay, evidence-tiered skills, and a fit scorer where every point
> traces to a reason code. The hot path makes no LLM call and runs in
> 1.3 ms/profile — a full pass including cold start, or 1.0 ms amortised at
> batch 100.
>
> It scores +0.092 NDCG@10 over a MiniLM cosine baseline, and **that improvement
> is not distinguishable from noise** — the 95% interval contains zero and the
> four per-job deltas disagree in sign. I would not defend it as real.
>
> The result I *would* defend: **every language model I tested — 135M, 1B, 3B and
> a hosted frontier-lite — misclassifies SDB_10019**, the AutoCAD mechanical
> engineer Appendix A uses to define the problem. For the strongest model it is
> its only error in 25. This system gets him right, for ~2,000x less money.

**The writeup ends at the rule.** After it: the evaluation in full, Part 3 in
full, and the assumptions the brief asks be written down.

## What I built, and the three decisions that carry it

**Classification is per experience entry, never over the profile blob.**
SDB_10019 is the proof: the headline says "Transitioning to Data Science", the
skills say "Machine Learning", the work history says six years of AutoCAD. A blob
classifier — which is what a cosine baseline effectively is — lands him in a data
role. Per-entry classification, weighted by duration and decayed with a 36-month
half-life, lands him in `non_engineering`. The same mechanism resolves SDB_10020
(Yahoo SWE 46m, Uber Senior ML 45m, Atlassian EM 55m and current) to
`engineering_manager`, which is what a recruiter would say.

**A declared skill is not an evidenced skill.** A skill is `core` only if it
appears in an entry's `skills_used`, a description token, or an education record;
otherwise it is `claimed_skills_unverified` and earns 30% credit toward a
must-have instead of 100%. This is the one thing cosine similarity structurally
cannot do: an embedding sees "Machine Learning" in a list and moves the vector,
with no notion of whether the history corroborates it.

**Nothing is ever filtered out.** Missing must-haves cost 8 points each; a zero
`role_match` caps the total at 35. Both are visible in `score_breakdown`. A
dropped candidate emits no reason codes, and a recruiter who cannot see *why*
someone was excluded cannot trust the ones who were included.

## What the evaluation says

| | baseline (MiniLM cosine) | signals_v1 | delta |
|---|---|---|---|
| NDCG@10 | 0.878 | 0.970 | +0.092 |
| NDCG@5 | 0.893 | 0.944 | +0.050 |
| Precision@5 | 0.700 | 0.750 | +0.050 |

**Not distinguishable from noise, and I would not defend it as real.** The
bootstrap 95% interval is `[-0.005, 0.203]` and contains zero. More tellingly the
per-job deltas — `+0.294, +0.110, -0.029, -0.007` — do not agree in sign, so the
mean is carried by JD-001. A sign-flip test over the four jobs gives p = 0.25,
and with n = 4 the smallest p it *could* return is 0.0625 — that test cannot
establish significance at any effect size.

Most of the gain is `role_family` alone, and the evidence-tiering rung I called
the core thesis is worth exactly **+0.0000** here — reported as zero (A1, with
the metric ceilings and list conventions in A2).

The three worst misses, by discounted gain given up (full diagnosis in A2;
**nothing was tuned after reading them**):

1. **JD-002 / SDB_10002**, graded 3, ranked 2nd — penalised for missing "offline
   eval" while his description reads *"+6% NDCG@10"*. A real design gap: skill
   requirements are never checked against descriptions.
2. **JD-003 / SDB_10011**, graded 3, ranked 4th — 7.0 years against a `[2,6]`
   band costs 6 of 15 seniority points. Arguably correct, arguably a labelling
   disagreement; the clearest case where I would want the grader in the room.
3. **JD-001 / SDB_10022**, graded 3, ranked 3rd — genuinely lacks two must-haves;
   the recruiter valued seniority over stack. My 8-point penalty is too steep for
   a senior hire. I would not change it on n=1.

## What I tried and dropped

**A MiniLM nearest-centroid fallback** (FL-006). Built, measured, cut — and *how*
is the point. The distilled LR first reported 100% accuracy on everything it
answered, which is a reason to look harder rather than celebrate. Splitting the
holdout by whether a title appears verbatim in the training corpus showed the 32
it answered were exactly the 32 it had seen; of the 10 novel titles it answered
**zero**. Its generalisation is not high, it is undefined. MiniLM, built to fix
exactly that, answers all 10 and gets **0 of 10** right. The fallback layer's
measured contribution is **zero**, reported as zero; the LR ships because it
*abstains correctly*, not because it classifies better.

**An LLM call per profile.** Not rejected on principle; measured four ways, all
schema-constrained. **This went against my argument and I am reporting the
version that hurts:** I assumed a per-row LLM would lose on accuracy, and against
gemini-3.5-flash-lite it does not — 24 of 25 — so "cheap *and* more accurate"
collapses to "cheap". The real case is ~2,000x cost, 1.25 s against 1.3 ms, rate
limits, and a third-party dependency on the field that gates search.

**But the best result here came out of that comparison.** SDB_10019 is classified
`data_scientist` by Gemini, `ml_engineer` by qwen, `data_engineer` by gemma. All
read the self-description; none weighs six years of history against it, and it is
not fixed by scale. That is exactly what per-entry classification and evidence
tiering exist to prevent. (Scope: n=25, my own labels — it shows the failure
exists and survives scaling; it does not quantify its rate.)

Those arms also showed that **a schema constrains shape, not meaning**: both
local models returned `years_relevant` in *months* for 22 of 25, copied off
`duration_months`, and every value passed validation. I got the first version of
that comparison wrong in my own favour too (FL-009). Also dropped: **fitting
`skill_noise_ratio` to Appendix A's worked example** (below), and **trusting the
wall clock for `as_of`** (FL-002) — the corpus is snapshotted around 2025-08
while its own `created_at` says 2026-06.

## Where it fails silently

* **Terse profiles are punished.** An engineer whose descriptions are one line
  each has most skills land in `claimed_skills_unverified`. Mitigated — claims
  earn 30%, and weak evidence lowers `confidence` rather than the score — but
  real. Median `skill_noise_ratio` here is ~0.7.
* **Skill matching is exact-after-alias, and a missed skill is invisible** — an
  unaliased spelling looks identical to an absent skill, and unlike the lexicon
  there is no abstention signal to alarm on. The failure I would fix first.
* **A skill requirement evidenced only in a description scores as missing** —
  worst miss #1. Silent because the score just comes out lower.
* **The lexicon degrades quietly.** Abstained entries contribute nothing rather
  than something wrong, so at scale profiles get *thinner* rather than visibly
  wrong, and accuracy drifts down with no error to catch.
* **`switch_intent` is not calibrated and cannot be** — there is no outcome label
  anywhere in the dataset. Its 0.71 means nothing in probability terms.
* **The 35-point cap is a discontinuity** — deliberate and visible in the
  breakdown, but a learned model with enough labels should not need it.

## What two more weeks would buy, ranked

1. **A labelled title set from recruiters, then retrain.** Everything upstream is
   bounded by 42 titles I labelled myself; 5,000 graded titles would turn the
   distilled LR from a memoriser into a classifier. Highest value by a distance.
2. **Close the skill/description matching gap** (worst miss #1), and replace
   exact-after-alias matching with offline embedding similarity over the skill
   vocabulary, keeping the hot path a dictionary lookup.
3. **The abstention feedback loop** — sample abstained titles weekly, batch them
   through an offline LLM, fold confirmed ones into the lexicon. The actual
   product, and not built.
4. **More labels per job, and more jobs.** With 4 jobs no experiment here can
   establish significance; 20 jobs × 40 labels would make the ablation mean
   something.
5. **Calibrate `switch_intent`** against reply outcomes.
6. **Corroboration-based deletion policy** across two consecutive crawls.

## AI tool usage

**Claude Code (Opus 5)** was the primary implementation tool, driven from a plan
and a decision log I wrote first (full log in `AI_LOG.md`). It saved time on
roughly 3,000 lines of contracts, normalisation, date arithmetic, the SQLite
adapter and the FastAPI layer — mechanical code where the design was settled —
and generated the 528-title synthetic corpus the distilled LR trains on.

**Where it was plausibly wrong, and what I did.** The plan asserted:

> "SDB_10019 has 7 declared skills, 4 unevidenced (python, machine learning,
> pandas, matplotlib), giving 0.571. Appendix A states 0.57. **The formula is
> confirmed against the spec's own worked example.**"

Plausible: the arithmetic is correct, the claimed split is exactly the one
printed in Appendix A, and reverse-engineering a spec from its own example is a
real technique. It is also wrong. Running it against the actual profile gives
**0.857** — his one experience entry lists only `AutoCAD` in `skills_used`, has
an empty description, and his education record has no skills, so `solidworks` and
`excel` are corroborated nowhere. The appendix's split is not derivable from the
appendix's own input.

I caught it because I ran the number before writing it down. I kept the strict
rule, tested the alternative formulation that *would* hit 0.57 (domain
congruence) and rejected it on evidence — it drops SDB_10010's noise ratio from
0.97 to 0.29, disabling the signal on the thirty-one-skill fresher it exists to
catch — and documented the divergence in FL-003 and Assumption 4.

The failure mode is worth naming precisely: it was not a wrong fact, it was a
**fabricated confirmation**. The word "confirmed" was attached to a check that
had never been run. Had I trusted it, I would have tuned a core signal to match
an illustrative example and reported that fit as validation — exactly the
measurement mistake this brief says it would rather see caught than beaten.

Two smaller cases are logged in `AI_LOG.md`. The pattern across all of them: the
generated code was structurally sound and empirically untested, and all twelve
`FAILURE_LOG.md` entries come from running it against the data rather than
reading it. Two surfaced only by checking artefacts against each other — one of
them a case where I had **a green test and a real bug at the same time, at
different layers** (A5).

---
---

# Appendix

## A1. The ablation ladder, including the rung that did nothing

| rung | NDCG@10 | Δ |
|---|---|---|
| 1. role adjacency only | 0.9417 | — |
| 2. + must-have coverage | 0.9572 | +0.0155 |
| 3. + evidence-vs-claim discount | 0.9572 | **+0.0000** |
| 4. + seniority band, production evidence, location, intent, tenure, cap | 0.9701 | +0.0129 |

Rung 3 is the rule I called the core thesis, and on this data it is worth exactly
zero NDCG. The reason is not that the rule is inert — it changes scores and
reason codes on SDB_10010, SDB_10019 and SDB_10023 — but that role adjacency at
rung 1 has *already* pushed those three to the bottom, so there is nothing left
for evidence tiering to fix. Its value is a claim about corpora containing many
such profiles, and this corpus contains three. That is a claim I can support by
argument, not by this measurement.

Each rung is a feature flag on one code path, not a separate implementation, so
no rung can quietly differ in something other than the feature it isolates.

## A2. Metric caveats, and the three worst misses in full

Three things I would rather point out than have found:

* **NDCG@10 barely discriminates.** 10–16 labels per job means a condensed @10
  cut covers most of the list, and MRR is 1.00 for *both* systems — saturated,
  reported only to show it.
* **JD-004's Precision@5 cannot exceed 0.40** for any system, since only 2 of its
  10 labelled candidates are graded ≥ 2. Both systems are already at that ceiling.
* **Both list conventions are reported** — condensed as primary, zero-fill
  alongside — because zero-fill punishes a system for surfacing a good candidate
  the recruiter never graded, which is exactly what this system is built to do.
  The ranking does not flip between them.

**Miss 1 — JD-002 / SDB_10002, graded 3, ranked 2nd (NDCG loss 2.58).** Marked as
missing the must-have "offline eval (NDCG, recall@k)" while his experience
description literally reads *"Trained cross-encoder reranker, +6% NDCG@10"*. The
cause is a real design gap: skill-type requirements are matched against the
declared skills list, evidence-type requirements against descriptions, and
nothing checks a skill requirement against a description. The system penalised
him for the evidence being in the *wrong field* — the exact inversion of the
evidence-tiering principle everything else here is built on. Fix: match skill
requirements against the union of skills and description tokens, tiered, so
description-only evidence scores *above* an unevidenced claim rather than below.

**Miss 2 — JD-003 / SDB_10011, graded 3, ranked 4th (loss 1.40).** Zero missing
must-haves and still ranked below a candidate graded 2. Two causes: 7.0 relevant
years against a `[2,6]` band costs 6 of 15 seniority points, and good-to-have
coverage is thin because "Databricks/Delta Lake" is not aliased to "cloud
warehouse". The seniority penalty is arguably correct behaviour and arguably a
labelling disagreement — the recruiter did not treat one year over the band as a
negative.

**Miss 3 — JD-001 / SDB_10022, graded 3, ranked 3rd (loss 0.92).** Missing
"FastAPI or Django" and "PostgreSQL", and he genuinely lists neither — he is an
Amazon SDE-3 on serverless and DynamoDB. Two missing must-haves cost 16 points.
The recruiter graded him 3 anyway, valuing seniority and scale evidence over the
exact stack. Not a bug in the extractor; a statement that my `missing_must_have`
penalty is too steep relative to how this recruiter weighs a stack mismatch for a
senior hire.

## A3. Part 3 in full

Idempotency falls out of one comparison operator: field state is rejected when
`_observed_at <= stored.observed_at`, so re-applying a feed skips every field,
emits zero material events, and leaves byte-identical state. `observed_at` is
tracked **per field group**, not per record, so a partial record carrying one
stale field and two fresh ones applies the fresh two.

The traps I found are one test each in `tests/test_delta.py`. In the real feed: a
duplicated `SDB_10009` line; a headline gaining a rocket emoji and a double space
(resolved to `noise` by the normaliser, with no branch in the delta engine); two
`about` fields arriving null; a location change to Berlin; and — the highest
blast radius — `SDB_10001`'s partial experience list containing only the two
Razorpay roles. Replacing wholesale there would have destroyed his Freshworks
history, which destroys `years_relevant`, which destroys `seniority`, which
destroys every downstream score. Four further traps *not* in the feed (unknown
candidate, missing `_observed_at`, out-of-order timestamps, numbers as strings)
are tested against synthetic records so the coverage is legible either way.

The asymmetry worth defending: **skills replace wholesale, experience merges by
key.** A skills list is a snapshot of a current claim, and a shrinking list is
information. Career history is append-mostly and a partial crawl must never
delete it. The mirror-image cost is that a removed fabricated job is retained
forever; the production fix for that and for the null policy is corroboration
across two consecutive runs, which cannot be demonstrated with one delta file.

Writing the idempotency assertion *before* believing the code caught a real bug:
the null-field path emitted `suspected_deletion` but never advanced
`observed_at`, so every re-run re-emitted it. It now advances the timestamp while
leaving the hash alone — we did observe the field, we chose not to apply what we
saw — which is also the state a corroboration policy would need.

## A4. The dashboard

`GET /dashboard` renders seven panels from `out/`, served by the same FastAPI
process — inline SVG, no chart library, no CDN, renders offline. It exists to
make the explainability claim concrete: the last panel is a per-job ranked list
where every score decomposes into components and every component leaves a reason
code.

One editorial rule shaped it (FL-012): **a number only earns a panel if it could
have come out differently.** The original hero was "LLM calls on the hot path: 0",
which is not a measurement but the architecture restated. It was cut for the
SDB_10019 head-to-head. The same audit found the page had metrics about the
*ranking* and nothing about the **signal layer** — the thing the brief calls the
deliverable — so role_family, seniority, confidence and the evidenced-versus-
claimed split are now the second panel. Those distributions double as the
monitoring surface `INFRA.md` alarms on.

Rendering the page and looking at it then found five bugs no validator caught: a
literally-printed HTML entity, a wrong count, a legend for an encoding that did
not exist, an unreadable dot cluster, and a severity ramp that made the least
important class the most prominent mark. The colour palette validator passed
while all five were live. Automated checks and looking at the output are not
substitutes for each other.

## A5. Two bugs found by checking artefacts against each other

**FL-007.** The throughput figure I was about to publish in `INFRA.md` was
exactly 2x too high: the telemetry divided an accumulated wall time by a
non-accumulated record count. It was visible only because the same quantity was
measured twice by different code paths and the two disagreed.

**FL-008.** A byte-level diff between a clean `git clone` and the source repo
showed `change_events.jsonl` differing only in `event_id`, which was a
`uuid4`. Making it content-addressed turned out to matter for more than
reproducibility: `event_id` is the primary key of the `change_events` table and
inserts use `INSERT OR IGNORE`, so with random ids a re-applied feed would have
written duplicate rows in the database while my in-memory idempotency test kept
passing. **A green test and a real bug at the same time, at different layers.**

## Assumptions

Every ambiguity I resolved in code, with what I chose:

1. **`as_of` is derived from the corpus, not the clock** — median of `start_date
   + duration_months` over current roles. Base = 2025-08-01, post-delta =
   2026-08-15. The data's `created_at` disagrees with its own content (FL-002).
2. **`computed_at` is a frozen constant** in `make all` so runs are
   byte-reproducible. Overridable via `--computed-at`; the API defaults to now.
3. **A skill is evidenced only by `skills_used`, a description token, or an
   education record.** The headline and `about` are explicitly not evidence —
   they are the self-description this component exists to distrust.
4. **`skill_noise_ratio` = |unevidenced| / |declared|**, giving 0.857 for
   SDB_10019 rather than Appendix A's illustrative 0.57 (FL-003).
5. **`role_family_alt` is experience-derived only.** Not "families the profile
   gestures at", because it feeds `role_match` at 0.6 weight and a skills-derived
   alt would hand the mechanical engineer 12.6 points on an ML job. For 24 of 25
   profiles it is correctly empty.
6. **`non_engineering` stays primary when it wins the argmax**, even if an
   engineering family is close behind.
7. **`staff+` is reachable only via title evidence**, never by accumulating years.
8. **A job's role family is resolved from its title, then its `raw_query`**, and
   an unresolvable job raises rather than defaulting — a silent default would
   score everyone as a mismatch with no visible cause.
9. **Must-haves are typed.** "Production ownership of a high-traffic service" is
   an evidence requirement checked against descriptions, not a skill. Production
   evidence = an ownership verb *or* a quantified outcome.
10. **Production ownership is deliberately double-weighted** — once in
    `skill_overlap` as a must-have, once as the 12-point `evidence_of_shipping`
    component — because both the brief's example breakdown and recruiter
    behaviour treat it as the discriminator.
11. **A field arriving `null` is "not observed"** and is not applied; a
    `suspected_deletion` event is emitted at low materiality (FL-005).
12. **Skills replace wholesale; experience merges by `(normalised_company,
    start_date)`** and never deletes unseen entries. Nulls inside a merged entry
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
