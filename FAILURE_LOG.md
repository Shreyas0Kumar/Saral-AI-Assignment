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

---

## FL-008 | 2026-08-17 | phase 8 | delta -> WRITEUP
**Symptom.** Testing that a clean `git clone` reproduces `out/` exactly,
`candidate_signals.jsonl` and `rankings.jsonl` came back byte-identical and
`change_events.jsonl` did not.

**Hypothesis 1.** A signal genuinely differs, so a downstream event differs.
  Checked by: diffed the two files. Every field of every event matched --
  candidate, timestamp, old value, new value, materiality, affected signals.
  Only `event_id` differed.
  Result: rejected; the content is identical.

**Hypothesis 2.** `event_id` is `uuid4`, so it is random by construction.
  Checked by: read the line. It was.
  Result: confirmed.

**Resolution.** `event_id` is now content-addressed:
`sha256(candidate_id, observed_at, change_type, field, new_value)`. Two things
follow, and the second is the one that matters.

1. `out/change_events.jsonl` is reproducible, so the reproducibility claim in
   the README is true for all three output files rather than two of them.
2. **Idempotency now holds at the storage layer, not only in memory.** The
   in-memory apply loop was already idempotent, and I had a passing test saying
   so. But `change_events` has `event_id` as its primary key and inserts with
   `INSERT OR IGNORE` -- with random ids, re-applying a feed would have written
   the same logical event again under a new key and the ignore clause could
   never have fired. The test I had was measuring the property one layer above
   where it would have failed in production.

Added `test_reinserting_the_same_events_is_a_storage_noop`, which fails against
the old implementation.

**What is worth taking from this.** I had a green idempotency test and a real
idempotency bug at the same time, because the test and the bug were at different
layers. The clean-clone diff found it, and nothing else would have -- which is
an argument for making byte-level reproducibility a routine check rather than a
claim in a README.

**Cost.** 20 minutes.

---

## FL-009 | 2026-08-17 | phase 3 | cost arm -> WRITEUP
**Symptom.** With Ollama available I re-ran the `llm_per_row` arm properly, and
it contradicted my own published number. My earlier harness (HuggingFace
`transformers`, SmolLM2-135M, no output constraint) reported **40.4 s/profile,
11/25 valid JSON, 4% accuracy**. Ollama running **gemma3:1b** — a model 7x
larger — reported **2.68 s/profile, 25/25 valid JSON, 68% accuracy**.

Faster, bigger, and seventeen times more accurate. One of those two runs was
measuring something other than what I claimed.

**Hypothesis 1.** gemma3:1b is simply a much better model than SmolLM2-135M.
  Checked by: true, but it cannot explain the *speed*. A 1B model is not 15x
  faster than a 135M model on the same CPU.
  Result: rejected as the main cause.

**Hypothesis 2.** My transformers harness was the problem, not the model.
  Checked by: compared the two paths. Mine used a naive `model.generate()` loop
  with `max_new_tokens=80` and no KV-cache tuning, in float32-ish bf16 on an
  unoptimised PyTorch CPU path, and **it did not constrain the output at all** —
  I asked for JSON in the prompt and hoped. Ollama uses a llama.cpp backend with
  a quantised model and, critically, accepts a **JSON schema** in its `format`
  parameter, so `role_family` is constrained to the 12-value enum at decode time.
  Result: confirmed. 14 of SmolLM2's 25 "failures" were the model failing to
  emit parseable JSON, which I was scoring as a wrong role family.

**Resolution.** gemma3:1b via Ollama, schema-constrained, temperature 0, fixed
seed, is now the headline `llm_per_row` arm. The cost claim in INFRA.md drops
from "~18,400x" to **"~1,200x"** and the accuracy comparison changes from "4%"
to "68% against my 100%".

**Why this matters more than the correction.** I had written that using a 7B
model to prove LLMs are expensive would be "sandbagging" — and then sandbagged
in the other direction by giving the small model a harness that could not
succeed. A 4% accuracy figure invites exactly the right objection: *"you
measured your own bad prompt."* It would have been the weakest number in the
submission and the easiest to attack. The corrected figure is smaller and much
harder to argue with.

**What the fair test actually shows**, and it is stronger than the unfair one:

* `non_engineering` is predicted **0 times in 25**. The founder, the mechanical
  engineer and the HR executive are all placed in engineering families —
  SDB_10019 (six years of AutoCAD) into `data_engineer`, which is *precisely*
  the failure Appendix A describes. The rules-based extractor gets all three
  right.
* `seniority` is `"mid"` for **23 of 25**. The Atlassian engineering manager,
  the Amazon SDE-3 and the three-month fresher are all "mid". The field carries
  almost no information.
* `years_relevant` is returned in **months for 22 of 25** — it copies
  `duration_months` off the current role. The schema constrained the *shape* to
  a number and could not constrain the *meaning*. This is the sharpest lesson
  in the whole exercise: structured output guarantees parseability, not
  correctness, and a pipeline that trusts a schema-valid field is trusting
  nothing.

The SmolLM2/transformers run is kept in `out/llm_cost_arm.json` under its own
key and labelled as an unconstrained harness, because the difference between the
two is the finding.

**Cost.** 45 minutes, and it cost me a headline number I liked.

---

## FL-010 | 2026-08-18 | phase 3 | corpus provenance
**Decision rule, written down before the measurement so it cannot be chosen
afterwards.** With Ollama available I can regenerate the synthetic title corpus
with `qwen2.5:3b-instruct`, which would make the artifact reproducible by a
reviewer rather than only by me. That is a real provenance improvement and it is
the plan's original design.

The temptation is obvious: train on both, evaluate both on the hand-labelled
holdout, and ship whichever scores higher. That is selection on the held-out set
— the same contamination I refused in FL-003 — and it would quietly invalidate
every number the holdout produces afterwards.

**So the rule, fixed in advance:**

* Ship the **Ollama-generated** corpus, on provenance grounds alone, because a
  corpus a reviewer can rebuild beats one only I can.
* Override that only if it **fails catastrophically**, defined before looking as:
  accuracy-when-answering on the holdout drops below 0.90, or coverage collapses
  below 0.50. Either would mean the artifact is broken, not merely different.
* Report both either way, in `out/fallback_comparison.json`, so the effect of
  corpus provenance on the distilled classifier is visible rather than a choice
  I made quietly.

**Result.** Recorded in the entry below once measured.

---

## FL-010 (result) | 2026-08-18 | phase 3 | corpus provenance
**Outcome: the rule fired, and it fired against the option I wanted.**

`qwen2.5:3b-instruct` produced a usable JSON array for **2 of the 12 families**
(`backend` 46 titles, `mobile` 71). The other ten returned no parseable array at
all despite the `format` schema constraining the response to
`{"type":"array","items":{"type":"string"}}`. Final corpus: **106 titles across
2 families**, against Claude's 528 across 12.

That is catastrophic by the definition I wrote down beforehand — coverage did not
merely drop below 0.50, ten of twelve taxonomy values would have had **zero**
training examples. So the Claude-generated corpus stays, its `source` field still
says so, and the qwen output is kept alongside as
`config/synthetic_titles.qwen-failed.jsonl` rather than deleted.

**Why the schema did not save it.** The same constrained-decoding approach works
perfectly for the cost arm's small fixed object (25/25 valid across three
models) and fails for a 45-element array. A long constrained array runs into
generation-length limits and the model emits an unterminated structure that
satisfies no parse. **Structured output is reliable in proportion to how short
the structure is** — worth knowing before designing a pipeline around it.

**The bug this exposed, which is the real finding.** The script *wrote the
broken corpus over the good one*. Exit code 0, 106 titles, ten classes silently
deleted. Nothing downstream would have failed: the distilled LR would have
trained on 2 classes, loaded fine, predicted fine, and quietly never returned
`ml_engineer` again. I only noticed because I checked the line count.

`generate_llm_artifacts.py` now refuses to write if any family produced no
titles, or if the corpus shrinks by more than 30%, and leaves the existing
artifact untouched. A generation script that can destroy its own committed
output on partial failure is worse than one that fails loudly.

**Cost.** 50 minutes of generation time for a negative result, plus 10 to make
the failure impossible to repeat.

---

## FL-011 | 2026-08-18 | phase 3 | hosted LLM arm -> WRITEUP
**Symptom.** Added `gemini-3.5-flash-lite` through the Google AI Studio API as a
hosted per-row arm, expecting it to lose on accuracy like the local models did.
It scored **24 of 25** — better than qwen2.5:3b (20), gemma3:1b (17), and one
error away from my own extractor.

**What that does to my argument.** It removes accuracy as the reason to prefer
rules. I had been leaning on "cheap *and* more accurate", and against a decent
hosted model only the first half survives. The remaining case is still strong
but it is a different case, and it is now stated as such in INFRA.md: ~1,800x
cost, 1.25 s versus 1.5 ms of latency (which alone disqualifies it from a
synchronous query path), free-tier rate limiting that produced HTTP 429 after 14
consecutive calls and needed backoff, and a third-party dependency on the field
that gates search.

**The finding I did not expect, and the best single result in this submission.**
SDB_10019 is the profile Appendix A of the brief holds up as the canonical
failure: mechanical engineer, six years of AutoCAD at Hero MotoCorp, headline
reads "Transitioning to Data Science".

    hand label             non_engineering
    signals_v1 (shipped)   non_engineering   correct
    gemini-3.5-flash-lite  data_scientist    wrong  <- its ONLY error in 25
    qwen2.5:3b-instruct    ml_engineer       wrong
    gemma3:1b              data_engineer     wrong

**Every language model tested fails on him, and for the strongest one it is the
single thing it gets wrong.** All of them read the self-description; none
weights six years of work history against one line of aspiration. The failure is
not fixed by scale — 135M, 1B, 3B and a hosted frontier-lite model all make it —
which is a much stronger claim than "rules are cheaper", and it is exactly what
per-entry classification plus evidence tiering is built to prevent.

**Method note, so the comparison is not overclaimed.** n=25, my own hand labels,
one profile deep. This shows the failure exists and is consistent across model
scales; it does not quantify how often it happens in a real corpus. That would
need the recruiter-graded set in the "two more weeks" list.

**Cost.** 40 minutes, and it cost me half my accuracy argument while producing
the best evidence in the writeup.
