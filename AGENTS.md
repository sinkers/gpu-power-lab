# AGENTS.md

Instructions for anyone — human or agent — writing code, documentation or
reports in this repository.

## What this project is

A measurement harness. It produces numbers that other people will use to make
decisions about power provisioning, and those decisions are expensive to get
wrong. That constrains how we write about the results.

## Language and tone

Write as an engineer reporting to other engineers. The reader is technically
literate, skeptical, and interested in whether the measurement is sound before
they are interested in whether it is interesting.

**Report findings, do not sell them.**

- State the measurement, the conditions, and the uncertainty. Let the reader
  judge significance.
- No superlatives about our own results. Not "the finding with teeth", not
  "the most consequential result", not "breaks wide open". If a result matters,
  the number and its conditions make that obvious without help.
- No dramatic framing devices: "the trap we nearly fell into", "what we got
  wrong first time" as a headline, "this changes everything". Corrections are
  stated as corrections, in place, without theatre.
- Avoid rhetorical questions as headings. "How much power can one GPU pull?"
  is a marketing headline; "Sustained power draw by workload" is a section
  title.
- No emphasis inflation. Bold is for the load-bearing number in a paragraph,
  not for every clause the author found exciting.

**Be precise about what was measured.**

- Always state the instrument. "NVML `POWER_INSTANT` reported 716.6 W" is a
  measurement; "the GPU drew 716.6 W" is an inference from one, and on this
  project they are not yet the same thing.
- Always state n. Nearly everything here is n=1. Say so.
- Distinguish measured, derived, and expected. A projection from single-GPU
  data to a node is a projection, and gets labelled as one.
- Give conditions with every number: duration, enforced limit, precision,
  temperature where relevant. A power figure without a window length is not
  reproducible.

**Be honest about error.**

- When a previous result is wrong, correct it plainly and say what caused the
  error. Do not bury it, and do not dramatise it either.
- Negative results are results. "No over-limit excursion was observed at the
  default cap" is a finding, written the same way as a positive one.
- Distinguish "we did not measure this" from "this did not happen". A test
  that failed to run is not evidence.

**Structure.**

- Lead with what was measured and under what conditions, then the numbers,
  then the interpretation. Interpretation is clearly separated from data.
- Tables for anything with more than three comparable values.
- Keep the caveats section specific. "Results may vary" is filler; "one GPU,
  no fabric traffic, in-band NVML only, n=1" is useful.

## Worked example

Not this:

> **The headline: the power cap does not hold on transients.** This is the
> finding with teeth — lower the cap and it breaks wide open, **43% above the
> enforced limit**.

This:

> Bursting a tensor workload against a lowered enforced limit produced peaks
> above that limit. Against a 500 W cap, `POWER_INSTANT` reported 716.6 W
> (143% of the cap) during 500 ms bursts, sustained across the measurement
> window. At the 1100 W default no excursion above the limit was observed.
> Single run per configuration.

## Code

- Comments explain why a choice was made, particularly where the obvious
  approach was rejected. They do not narrate what the next line does.
- Where a measurement decision is encoded in code — a sampling rate, a window
  length, a fallback — the comment states the consequence of getting it wrong.
- Fail loudly on unsupported configurations rather than falling back to
  something that produces a number under the wrong label.

## Commits

Describe what changed and why. Include measured values where a commit reports
results, so `git log` is a usable record of what was found when. No
self-congratulation in commit messages.
