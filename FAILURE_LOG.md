# FAILURE_LOG.md

What went wrong, what I thought was happening, how I checked, and what it cost.
Entries include the hypotheses I abandoned, because the abandoned branches are
the part worth reading.

Format: `Symptom / Hypothesis + Checked by + Result / Resolution / Cost`.
Entries tagged `-> WRITEUP` produced a finding cited in `WRITEUP.md`.

---

## FL-001 | 2026-08-17 | phase 0 | environment
**Symptom.** The plan specified Ollama (`qwen2.5:3b-instruct` for generation,
`gemma3:1b` for the cost arm). Ollama is not installed on this machine and
downloading and warming two models is not a good use of the budget.

**Hypothesis 1.** Install Ollama and pull both models.
  Checked by: estimated download (~2.8GB) plus first-run warmup against the
  remaining time budget.
  Result: rejected. It buys nothing the assignment is graded on, and it adds a
  dependency a reviewer would also have to install.

**Hypothesis 2.** Use a hosted API for the offline generation step.
  Checked by: re-read the submission constraints.
  Result: rejected. `make all` must work from a clean clone with no network. A
  hosted call in the artifact-generation path makes the artifacts
  unreproducible for anyone without my key.

**Resolution.** Local HuggingFace `transformers`, which is already installed,
with two models already in the cache:
`microsoft/Phi-3.5-mini-instruct` (3.8B) for offline synthetic corpus
generation, and `HuggingFaceTB/SmolLM2-135M-Instruct` plus Phi-3.5 for the
`llm_per_row` cost arm. The argument structure is unchanged -- an LLM used
offline, distilled into a cheap artifact, with the per-row alternative measured
rather than assumed. Only the runtime differs. Generated artifacts are
committed, and regeneration sits behind `make regenerate-llm-artifacts`, off
the default path.

**Cost.** 20 minutes, mostly spent confirming what was already in the model
cache instead of assuming.

---

## FL-002 | 2026-08-17 | phase 1 | dates -> WRITEUP
**Symptom.** The plan assumed `as_of` would be roughly "today" (2026-08-17).
Running that against the corpus fired `duration_mismatch` on nearly every
current role, which would have applied a 0.15 confidence penalty to almost all
25 profiles -- a signal that fires on everything and therefore discriminates
nothing.

**Hypothesis 1.** The date parser is off by a year somewhere.
  Checked by: hand-computed `SDB_10001` Razorpay, start `2022-04-01`,
  `duration_months: 40`. 2022-04 + 40 months = 2025-08, not 2026-08.
  Result: rejected. The parser is right; the *data* is snapshotted earlier than
  its own `created_at` field claims.

**Hypothesis 2.** The whole corpus is stale by a fixed offset.
  Checked by: computed `start_date + duration_months` for every current role.
  Base corpus median: **2025-08-01**. Delta file median: **2026-08-15**.
  Completed roles reconcile exactly with their stated durations in both files
  (Freshworks 2019-07 to 2022-03 = 32 months = stated).
  Result: confirmed. `created_at: 2026-06-02` is crawl metadata and disagrees
  with the profile content it wraps.

**Resolution.** `as_of` is derived from the data, not the clock:
`derive_snapshot_date` takes the median of `start_date + duration_months` over
current roles. Base extraction runs at 2025-08-01, the post-delta recompute at
the delta's own `_observed_at`. This is also the right production behaviour --
a profile crawled six months ago should be aged against when it was crawled,
not against now -- and it keeps `core/` clock-free. Recorded under Assumptions.

**Cost.** 35 minutes.

---

## FL-003 | 2026-08-17 | phase 2 | skills -> WRITEUP
**Symptom.** The plan asserted that `skill_noise_ratio = |claimed| / |declared|`
was "confirmed against the spec's own worked example", because Appendix A shows
`0.57` for SDB_10019 and 4/7 = 0.571. My implementation produced **0.857**.

**Hypothesis 1.** My evidence check is missing a source.
  Checked by: dumped every evidence surface for SDB_10019. The single
  experience entry is `Design Engineer @ Hero MotoCorp` with
  `skills_used: ["AutoCAD"]`, an empty `description`, and an education record
  with an empty `skills` list.
  Result: rejected. `autocad` is the only skill corroborated anywhere. To reach
  4/7 I would have to credit `solidworks` and `excel`, which appear in the
  declared list and nowhere else in the profile.

**Hypothesis 2.** The intended rule is domain congruence, not evidence -- a
  skill counts if it belongs to the family the person demonstrably works in.
  Checked by: applied it to SDB_10010 (three months of experience, thirty-one
  declared skills). His derived family is `data_scientist`, so congruence
  verifies nearly every data skill he lists and the ratio falls to about 0.29.
  Result: rejected. It reproduces the appendix number by breaking the signal on
  the profile the signal exists to catch.

**Resolution.** Kept the strict evidence rule and reported the divergence. The
brief states the appendix values are "illustrative, not a target to reproduce",
so fitting the formula to one printed number would have been reverse-engineering
an example rather than designing a signal. The test
`test_skill_noise_ratio_is_claimed_over_declared` records both the definition
and the reason it does not match the appendix. Cited in WRITEUP.md.

**Cost.** 40 minutes. Worth it -- Hypothesis 2 was tempting and would have
quietly disabled the system's main mechanism.

---

## FL-004 | 2026-08-17 | phase 2 | title classifier
**Symptom.** SDB_10017 (headline "Full Stack Developer | MERN") classified as
`backend`, with `fullstack` only as an alternate.

**Hypothesis 1.** The lexicon is missing a fullstack pattern.
  Checked by: `Full Stack Developer` matches `fullstack` exactly at confidence
  1.0. Not the problem.
  Result: rejected.

**Hypothesis 2.** An adjacent entry is outvoting the current role.
  Checked by: printed per-entry contributions. Current `Full Stack Developer`
  6 months x decay 1.0 x conf 1.0 = 6.0. Previous `Software Engineer` at
  Yellow.ai, 9 months, `skills_used: ["Node.js"]` -- an *ambiguous* title
  resolved by a single context term -- scored 9 x 0.908 x 0.8 = 6.54.
  Result: confirmed. One stray token in `skills_used` outweighed an
  unambiguous, current, exactly-matching title.

**Resolution.** Graded the context vote: an ambiguous title resolved by weak
context (total weight < 2.0) now yields entry confidence 0.60 rather than 0.80.
This is a statement about evidence strength, not a fit to labels -- no metric
existed yet when it was made, and it was checked for side effects on SDB_10013
(whose Infosys entry is also weak-context; he stays `backend` at 6.08 relevant
years, so the change does not simply trade one error for another).

**Cost.** 25 minutes.

---

## FL-005 | 2026-08-17 | phase 2 | reason codes
**Symptom.** `zero_role_experience_despite_skill_claim` fired for SDB_10009, a
staff platform engineer. Reading a reason code that says a PhonePe staff
engineer has "zero role experience" would destroy a recruiter's trust in the
whole reason-code surface.

**Hypothesis 1.** His family is misclassified.
  Checked by: `devops_sre` from "internal developer platform" + Kubernetes, and
  a prior SRE role. Correct.
  Result: rejected.

**Hypothesis 2.** The check treats "no experience under this exact label" as
  "no experience at all". His unverified skills include `postgresql` and
  `grpc`, which point at `backend`, a family he has never held a titled role in
  -- but which is adjacent to what he does every day.
  Result: confirmed.

**Resolution.** Gated the code on adjacency: it fires only when *every* family
the claimed skills point at is unreachable (adjacency 0) from every family the
person has actually worked in. It now fires for exactly SDB_10010 and
SDB_10019, which are the two profiles it exists for.

**Cost.** 15 minutes.

---

## FL-006 | 2026-08-17 | phase 3 | fallback classifiers -> WRITEUP
**Symptom.** The distilled LR reported 100% accuracy on everything it answered
(32/32) over the hand-labelled held-out entries. A perfect score on a held-out
set is a reason to look harder, not to celebrate.

**Hypothesis 1.** Label leakage -- the holdout got into training.
  Checked by: the training corpus is 528 synthetic titles; the holdout is 45
  real experience entries from `candidates.jsonl`. Different files, different
  provenance, no shared rows.
  Result: rejected as literal leakage.

**Hypothesis 2.** *Surface-form* leakage -- the holdout titles are common
  enough that the same strings appear in the synthetic corpus.
  Checked by: normalised every holdout title and intersected with the training
  corpus. **32 of 42 (76%) appear verbatim.** Then split accuracy by that
  boundary. The 32 it answered are exactly the 32 it had seen. Of the 10 novel
  titles it answered **zero**.
  Result: confirmed. Measured generalisation is not 100%, it is undefined --
  the classifier answered no novel title at all.

**Hypothesis 3.** Then perhaps the semantic fallback generalises where the
  sparse one cannot -- that is the whole argument for keeping MiniLM.
  Checked by: built the MiniLM nearest-centroid arm and evaluated on the same
  split. It answers all 10 novel titles and gets **0 of 10** right
  ("Software Engineer" -> ml_engineer, "SDE II" -> devops_sre,
  "Design Engineer" -> engineering_manager).
  Result: rejected, and worse than rejected -- it is confidently wrong where
  the LR is silent.

**Resolution.** Narrowed to the only population a fallback ever sees in
production: the titles where the lexicon genuinely abstains. There are 5 in this
corpus. The LR answers **0 of 5**; the MiniLM centroid answers 5 of 5 and gets
**0 of 5** right.

So the measured contribution of the fallback layer on this corpus is zero, and
this is reported as zero. What changed as a result:

* The claim for shipping the LR is no longer "it classifies better". It is that
  it **abstains correctly**, so it can only ever fill a gap and never override
  a decision the lexicon got right. That is a safety property, and it is the
  one property that was actually measured.
* MiniLM is cut on evidence rather than on image size. It has no abstention
  threshold that works here: cosine against a centroid always has an argmax.
* The five ambiguous titles are resolved *correctly* by the lexicon's context
  vote (`SDE II` + `Java, Spring Boot` -> backend), i.e. by the primary path,
  not the fallback. The architecture was already covering the case the fallback
  was hired for.
* `DECISIONS.md` D2 was rewritten. Its original reasoning ("classification
  difference was inside noise while the cost difference was measurable") was
  too generous to my own component.

**Cost.** 50 minutes, and it removed a claim I would otherwise have made in the
writeup and been unable to defend in review.

---

## FL-007 | 2026-08-17 | phase 8 | telemetry -> WRITEUP
**Symptom.** Writing INFRA.md, I went to cite the extraction throughput from
`run_manifest.json` and got 4.13 ms/record. The warm latency benchmark in the
same file says p50 = 1.0 ms. A 4x gap between two measurements of the same
operation is not a cold-start story.

**Hypothesis 1.** The manifest figure legitimately includes cold start
(first-call regex compilation, config parse, lazy imports).
  Checked by: cold start is real but bounded -- the warm benchmark itself warms
  up on 5 profiles and 25 records is not enough for a 4x amortisation gap.
  Result: partially true, insufficient.

**Hypothesis 2.** The denominator is wrong. `cost_per_1m_profiles` divides
  `stage.wall_ms` by `stage.records_out`. `wall_ms` **accumulates** across every
  entry into the stage context manager, but `records_out` was **assigned**
  (`=`), so the last writer won.
  Checked by: printed the stage dict. `records_in: 50, records_out: 25` -- the
  pipeline extracts twice (once for the deliverable, once inside
  `run_full_evaluation`), so the wall time was for 50 records and the count said
  25.
  Result: confirmed. Every cost figure in the manifest was **exactly 2x too
  high**.

**Resolution.** `records_out` now accumulates, matching `wall_ms`. Corrected
figures across three runs: 2.11-2.37 ms/record, $0.03 per million profiles
rather than $0.06. Also fixed the sibling bug this uncovered: the *counters*
(`titles_classified`, `lexicon_abstentions`) had the opposite problem -- they
accumulated and so double-counted, reporting 22 abstentions out of 140 titles
when the corpus has 11 out of 70. The rate was right by luck, because numerator
and denominator doubled together. Counters are now assigned, records are
accumulated, and each is commented with which it is and why.

**What is worth taking from this.** The bug was only visible because the same
quantity was measured twice by different code paths and the two disagreed. A
single measurement of throughput would have been wrong and unfalsifiable, and it
would have gone into INFRA.md as a headline number with visible working -- the
working would have been visible and the input wrong. The instrumentation needed
a second opinion as much as the model did.

**Cost.** 30 minutes. Caught at the last possible moment, by checking a document
against the data it claimed to summarise rather than trusting my own manifest.
