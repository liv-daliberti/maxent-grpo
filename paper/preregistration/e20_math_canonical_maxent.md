# E20 prospective MATH canonical-MaxEnt extension

**Status: DATA LANDED / CANONICAL POLICY IN DESIGN / NO TRAINING JOBS AUTHORIZED.**

## Question and scope

E20 asks whether the direct canonical-action Standard MaxEnt treatments that
were used for graph coloring and Countdown transfer to competition
mathematics.  It does **not** test aggregation rescaling.  No xDr, fixed-xDr,
feedback-xDr, or xDr Haarnoja arm belongs in this experiment.

A zero-entropy constrained Dr.GRPO arm is nevertheless required as the matched
control.  Historical free-text Dr.GRPO results from the OAT paper are context,
not a matched control for a different action space.

## Source task and leakage boundary

The OAT Dr.GRPO paper did not train on MATH-500.  It trained on MATH
level-3--5 questions and evaluated on MATH-500 (along with four other math
benchmarks).  E20 preserves that boundary:

- training: the released 8,523-row `math_lvl3to5_8k/train` split;
- primary evaluation: the released 500-row `evaluation_suite/math` split,
  conventionally called MATH-500;
- exact train/evaluation problem overlap: zero;
- MATH-500 is never sampled for an optimizer update, controller warmup, dose
  selection, prompt selection, or early stopping.

The byte-exact source is
`https://github.com/sail-sg/understand-r1-zero.git` at the initial paper-code
commit `559bcfd7a50727e7ed97f06a586da2c97236f496`.  The files remain unchanged
at the upstream head audited on 2026-07-19.  `ops/math500/import_oat_math.py`
pins every source-file SHA-256 and writes a local provenance manifest.

The released training artifact contains two rows with blank reference answers
and one repeated problem with the same answer.  The importer records these
facts and does not silently rewrite the paper distribution.  A later launch
protocol must state explicitly whether the two unverifiable rows are retained
as always-zero examples or excluded before the first job is submitted.

## Why the current canonical codec cannot simply be enabled

The landed canonical policies have a fixed, audited action grammar: three
color decisions for graph coloring and three operator/operand decisions for
Countdown.  MATH-500 answers are variable-length symbolic strings.  In the
released splits, answer strings use dozens of characters and range up to 159
characters in training.  Setting `canonical_action_task=countdown` or
`graph_coloring` would therefore change the task rather than extend it.

Three tempting shortcuts are prohibited:

1. exposing a per-question answer list and asking for an index, which turns
   MATH-500 into a multiple-choice benchmark;
2. constructing the action support from the gold answer, which leaks labels;
3. rewarding entropy in ignored padding or post-termination actions, which
   creates exploration that has no semantic effect.

Before any E20 training job is authorized, a codec must demonstrate all of the
following on every train and MATH-500 row:

- label-free action supports determined only by the prompt and public grammar;
- lossless encode/decode for the reference answer representation;
- no gold answer or candidate answer list in the rendered prompt;
- every entropy-bearing action can alter the decoded mathematical answer;
- exact rollout/learner log-probability agreement on the restricted support;
- a defined treatment of variable termination and entropy units;
- grading parity with the released OAT `boxed_reward_fn(..., fast=False)` on
  decoded outputs.

## Planned arms after the codec gate

The first cohort is a small feasibility smoke, followed by a 0.5B matched
comparison only if all codec and evaluator gates pass.

| Arm | Canonical objective |
|---|---|
| C0 | constrained Dr.GRPO, `alpha=0` |
| M-fixed | Standard MaxEnt with fixed `alpha` |
| M-proportional | Standard MaxEnt with proportional entropy control |
| M-dual | Standard MaxEnt with Haarnoja-style dual entropy control |

The fixed dose and controller targets are not copied blindly from E16.  E16's
entropy unit is a three-action sequence with a small finite support; a MATH
codec will have a different maximum entropy and possibly a variable semantic
horizon.  Dose calibration must use the codec's exact canonical entropy unit
and must not inspect MATH-500 outcomes.

## Evaluation and stopping contract

- MATH-500 greedy pass@1 is the primary performance measure.
- Sampled pass@8 and response/action entropy are secondary diagnostics.
- Evaluation is taken at initialization and every quarter pass over the
  training prompt pool, using the same frozen MATH-500 rows each time.
- Three seeds are reported separately and as a seed mean; the plot must never
  present a single seed as the method curve.
- Invalid codes, decode failures, blank decoded answers, nonfinite entropy,
  action-support violations, or rollout/learner probability disagreement are
  fail-closed smoke errors.
- No aggregation-rescaling arm may be added under the E20 identifier.

## Current authorization boundary

The paper artifacts and provenance checks are ready.  Dataset readiness does
not authorize GPU work.  E20 remains in design until the canonical codec is
specified, tested, and frozen in a follow-up protocol amendment.
