# E68 secondary paired prompt-level uncertainty analysis

Frozen 2026-07-27T20:27:23Z after the three-seed MathIR results through
pass 2 were visible and before a complete pass-3 MathIR checkpoint or any
non-Math E68 checkpoint was available.

## Evidential role

This is a post-specified, descriptive uncertainty analysis. It cannot change,
replace, rescue, or fail any primary E68 terminal, AUC, integrity, or
mechanism gate. The primary experimental unit remains the independently
trained seed, of which there are three per arm. Prompt-level resampling does
not create additional training replicates.

The purpose of this analysis is narrower: retain the pairing already present
in the frozen evaluation protocol and show how much of each observed
E68-minus-E66 difference is associated with the finite evaluation prompt
sample.

## Fixed input surface

The analysis uses only the raw neutral-evaluation traces already written by
the E66 and E68 jobs:

- the deterministic greedy trace; and
- fixed sampled-K traces with K=8 and draw indices 0, 1, 2, and 3.

It is evaluated at the ten previously registered ModeBench checkpoints:
passes `0, 1, 2, 3, 4, 5, 6, 8, 10, 12`. A checkpoint is reported only after
all three paired E66/E68 seeds have all five traces.

For every paired seed, arm, and checkpoint, the analysis must verify identical
prompt count, prompt order, prompt text, prompt reference, prompt index,
evaluation kind, draw index, and fixed evaluation seed. Any mismatch is an
integrity violation rather than a missing result.

## Fixed estimands

The arm difference is always E68 minus E66. For each seed and prompt:

- `greedy` is the deterministic trace's per-prompt `mean_at_k`;
- `mean8` is the mean per-prompt `mean_at_k` over the four K=8 draws;
- `pass8` is the mean per-prompt `any_correct_at_k` over the four K=8 draws;
- `distinct8` is the mean per-prompt `distinct_correct_modes_at_k` over the
  four K=8 draws.

The four sampled draws are averaged before uncertainty calculation and are
never treated as independent training or prompt replicates. The point estimate
is the mean paired difference over the three training seeds and all evaluation
prompts. The three seed-level paired differences are always reported beside
it.

## Fixed descriptive interval

For each domain, checkpoint, and metric, use a crossed paired bootstrap:

1. sample three seed indices with replacement from the three paired training
   seeds;
2. independently sample the full number of prompt indices with replacement
   from the shared evaluation prompt set;
3. apply both resamples to the E68-minus-E66 seed-by-prompt difference matrix;
4. average the selected cells; and
5. repeat 10,000 times.

Report the 2.5th and 97.5th percentiles as a descriptive 95% crossed-bootstrap
interval. Randomness is fixed independently for every
domain/checkpoint/metric from SHA-256 of the string
`20260727|<domain>|<pass>|<metric>`, interpreted from its first eight bytes as
an unsigned big-endian integer.

No p-value, significance label, multiplicity-adjusted claim, checkpoint
selection, or early-stopping decision may be derived from this analysis.
Intervals at intermediate checkpoints remain interim. Because only three
training seeds exist, even an interval excluding zero is not a substitute for
additional independent training replicates.

## Reproducibility and freezing

The executable is
`ops/exp_scaling/analyze_e68_paired_prompt_uncertainty.py`. The hash-bound
identity is
`var/artifacts/e68_paired_prompt_uncertainty_secondary_identity.json`.
The executable refuses to emit results if its own hash or this document's
hash differs from that identity.
