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
