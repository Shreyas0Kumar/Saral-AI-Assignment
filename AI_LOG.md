# AI_LOG.md

Which AI tools did what, logged as I went rather than reconstructed afterwards.
Part 6 of the brief asks specifically for a case where an AI tool gave a
plausible answer that was wrong; those entries are marked **WRONG ANSWER**.

Tools used: **Claude Code** (Opus 5) as the primary implementation agent, driven
from a plan I wrote first (`build_plan.md`, not submitted) and a decision log I
maintained alongside it.

---

## AI-001 | 2026-08-17 | Claude Code | planning
**Task.** Turn the brief into a phased build plan with an explicit cut line.
**Outcome.** Useful. The four-tier layering (`contracts / core / pipeline /
adapters`) and the purity-test idea came out of that session and survived
contact with the code unchanged.
**Time saved.** Perhaps 90 minutes of structuring, and it front-loaded the
decisions that are expensive to change later (contracts, hashing policy).

---

## AI-002 | 2026-08-17 | Claude Code | **WRONG ANSWER**
**Task.** Establish the definition of `skill_noise_ratio`.
**Output.** "Validation: SDB_10019 has 7 declared skills, 4 unevidenced
(python, machine learning, pandas, matplotlib), giving 0.571. Appendix A states
0.57. **The formula is confirmed against the spec's own worked example.**"
**Plausible because.** The arithmetic is right, the claimed split is exactly
the one printed in Appendix A, and it reads like a genuine act of close reading
-- reverse-engineering a spec from its own example is a real and good technique.
**Caught by.** Implementing it and running it. The real profile yields 0.857,
because SDB_10019's only experience entry lists `AutoCAD` in `skills_used`, has
an empty `description`, and an education record with no skills. `solidworks`
and `excel` are corroborated *nowhere*. There is no evidence-based reading that
produces 4 unevidenced skills; the appendix's own split is not derivable from
the appendix's own input.
**Did instead.** Kept the strict evidence rule and reported the divergence, with
the reasoning in a test docstring and in `WRITEUP.md`. The brief states the
appendix values are "illustrative, not a target to reproduce", which the plan
had not accounted for. I also checked the alternative reading that *would* hit
0.57 (domain congruence rather than evidence) and rejected it, because it
reproduces the printed number by disabling the signal on SDB_10010 -- the
thirty-one-skill fresher the signal exists to catch. See FL-003.
**Why this one matters.** The failure mode was not a wrong fact but a
**fabricated confirmation**: the word "confirmed" attached to a check that was
never run. Had I trusted it, I would have tuned a core signal to match an
illustrative example and reported the fit as validation.

---

## AI-003 | 2026-08-17 | Claude Code | plan assumptions vs data
**Task.** Handle experience-entry overlaps; the plan named SDB_10021's
freelance work as the case requiring overlap resolution.
**Output.** Plausible -- freelancers commonly do overlap.
**Caught by.** Printing the actual spans. SDB_10021's Dunzo role ends
2022-08-31 and the freelance role starts 2022-09-01. No overlap exists anywhere
in the corpus.
**Did instead.** Kept the de-overlap implementation (it is correct, cheap, and
the 1M-row framing makes it necessary) but removed the claim that it fires on
this dataset, and said so rather than citing a candidate it does not apply to.

---

## AI-004 | 2026-08-17 | Claude Code | implementation
**Task.** Write the extraction modules, contracts and config loaders from the
plan.
**Outcome.** This is where the tool paid for itself -- roughly 1500 lines of
pydantic contracts, normalisation, date arithmetic and YAML validation that I
would otherwise have typed by hand.
**What I still had to do.** Every threshold, every adjacency value, the graded
context vote (FL-004) and the adjacency gate on the claim signal (FL-005) were
my decisions after reading the model's output against real profiles. The
generated code was structurally sound and empirically untested; all five
FAILURE_LOG entries so far come from running it against the data, not from
reading it.

---

## AI-005 | 2026-08-17 | Claude Code | offline corpus generation
**Task.** Generate the synthetic job-title corpus that the distilled classifier
trains on -- the "use an LLM offline to bootstrap labels / distil into a smaller
model" path the brief explicitly permits.
**Output.** 528 titles across the 12 taxonomy values, prompted per family for
realistic Indian-market variants (SDE-2, MTS, Associate Consultant, Analytics
Engineer, and the services-company spellings alongside the product-company ones).
Committed at `config/synthetic_titles.jsonl` with the generating prompt kept in
`scripts/generate_llm_artifacts.py`, so a reviewer can see what produced it and
`make all` needs no model.
**Why this backend.** The original plan called for Ollama, which is not
installed; `Phi-3.5-mini` turned out to be only partially present in the local
HuggingFace cache and wanted a 4.9GB download, which breaks the no-network
constraint (FL-001). Claude Code was already the tool in use and its output is
committed, so the artifact is reproducible for someone who has neither.
**The honest limitation.** The corpus inherits an LLM's priors about what job
titles mean, and it was validated against 42 titles I labelled myself. So the
held-out set is independent of the *models* but not of *me*. At real scale the
fix is recruiter-graded titles. This is stated in
`out/fallback_comparison.json → method.holdout.provenance` rather than left for
a reader to work out.

---

## AI-006 | 2026-08-17 | Claude Code | where it did NOT help
**Task.** Everything measured in this submission.
**Observation worth recording.** The generated code was consistently
well-structured and consistently untested against reality. All seven
`FAILURE_LOG.md` entries came from running it, not from reading it:

* the `as_of` assumption collapsed on contact with the data (FL-002),
* the `skill_noise_ratio` "confirmation" was fabricated (FL-003, AI-002),
* one stray token in `skills_used` outvoted an exact current-title match (FL-004),
* a reason code fired on a staff platform engineer and would have destroyed
  recruiter trust in the whole surface (FL-005),
* the distilled classifier's 100% held-out accuracy was memorisation, and the
  semantic fallback I built to fix that got 0 of 10 novel titles right (FL-006),
* and the throughput figure I was about to publish in INFRA.md was 2x too high
  because the telemetry divided an accumulated numerator by a non-accumulated
  denominator (FL-007).

Each of those is the kind of error that survives code review and dies on
contact with data. The division of labour that worked: the model wrote the
structure, and every number in this repo was checked against the corpus by hand
before it was written into a document.
