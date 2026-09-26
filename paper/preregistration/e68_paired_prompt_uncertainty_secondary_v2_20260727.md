# E68 secondary paired prompt-level uncertainty analysis, source-aligned v2

Frozen 2026-07-27 after the v1 analysis was executed locally but before it was
integrated into the live result report. This document supersedes v1 only for
the secondary uncertainty display. It does not alter E66, E68, their primary
metrics, or any frozen gate.

## Non-outcome reason for v2

The first v1 execution showed that the dedicated sidecar's repeated
temperature-zero K=1 evaluation is not always bitwise outcome-identical to the
earlier primary greedy evaluation saved in
`eval_results/<step>_multi_answer.json`. At MathIR pass 2, the v1 repeated-trace
E68-minus-E66 estimate was `+0.0130208333`, while the already plotted primary
greedy estimate was `+0.015625`. The two calls differ on only a small number of
prompts, but substituting the later call would make the displayed interval
center disagree with the registered paper point.

V2 therefore aligns every point estimator with its primary source:

- `greedy` uses the per-prompt scores from the primary
  `eval_results/<step>_multi_answer.json` file used by the scaling-curve
  parser; and
- `mean8`, `pass8`, and `distinct8` use the four fixed sampled-K sidecar draws,
  as before.

The sidecar's repeated greedy trace is retained as an explicit evaluation
repeatability sensitivity. V2 reports prompt disagreement counts and mean
score shifts between it and the primary greedy call; it does not silently
merge or select between them.

## Evidential role

This remains a post-specified, descriptive uncertainty analysis. It cannot
change, replace, rescue, or fail any primary E68 terminal, AUC, integrity, or
mechanism gate. The primary experimental unit remains the independently
trained seed, of which there are three per arm. Prompt-level resampling does
not create additional training replicates.

## Fixed input and integrity surface

The analysis is evaluated at passes
`0, 1, 2, 3, 4, 5, 6, 8, 10, 12`. A checkpoint is reported only after all
three paired E66/E68 seeds have:

- one primary greedy result file;
- one sidecar repeated greedy trace; and
- fixed sampled-K traces with K=8 and draw indices 0, 1, 2, and 3.

Across arms and sources, prompt count, prompt order, prompt text, prompt
reference, prompt index, evaluation kind, draw index, and fixed evaluation
seed must match. A mismatch is an integrity violation.

## Fixed estimands

The arm difference is always E68 minus E66. For each seed and prompt:

- `greedy` is the primary result file's single saved score;
- `mean8` is the mean per-prompt `mean_at_k` over the four K=8 draws;
- `pass8` is the mean per-prompt `any_correct_at_k` over the four K=8 draws;
- `distinct8` is the mean per-prompt `distinct_correct_modes_at_k` over the
  four K=8 draws.

The four sampled draws are averaged before uncertainty calculation and are
never treated as independent training or prompt replicates. The point estimate
is the mean paired difference over the three training seeds and all prompts.
The three seed-level paired differences are always reported.

## Fixed descriptive interval

For each domain, checkpoint, and metric, use a crossed paired bootstrap:

1. sample three seed indices with replacement from the three paired training
   seeds;
2. independently sample the full number of prompt indices with replacement;
3. apply both resamples to the E68-minus-E66 seed-by-prompt difference matrix;
4. average the selected cells; and
5. repeat 10,000 times.

Report the 2.5th and 97.5th percentiles as a descriptive 95% interval. Seed
each domain/checkpoint/metric independently from the first eight bytes of
SHA-256 of `20260727|<domain>|<pass>|<metric>`, interpreted as an unsigned
big-endian integer.

No p-value, significance label, multiplicity-adjusted claim, checkpoint
selection, or early-stopping decision may be derived from this analysis.
Intermediate checkpoints remain interim. With only three training seeds, even
an interval excluding zero is not a substitute for more independent runs.

## Reproducibility

The v1 method, script, identity, and output remain archived rather than edited.
V2 is executed by
`ops/exp_scaling/analyze_e68_paired_prompt_uncertainty_v2.py` and hash-bound by
`var/artifacts/e68_paired_prompt_uncertainty_secondary_v2_identity.json`.
The executable also verifies the frozen v1 base-script hash on which its common
trace-validation and bootstrap routines depend.
