# E77: fixed-component necessity screen

**Frozen before submission on 2026-08-04.**

## Question and status

This exploratory screen asks whether any of the three fixed auxiliary terms is
incrementally useful once the one-time discovery reward and every adaptive
coefficient controller have been removed:

- semantic MaxEnt, eta = 0.10;
- verified-mass replay, mu = 0.10; and
- known-mode balance, alpha = 0.10.

This is a screening experiment, not paper evidence. One seed cannot establish
equivalence or necessity. Its sole purpose is to decide which exact-zero arms
deserve a multi-seed terminal rerun.

## Arms

All arms use the same persistent verified bank, deterministic global replay
schedule, replay scoring traversal, and four-pass optimizer budget. A zero
coefficient means an exact-zero derivative, not a small dose.

| arm | eta | mu | alpha | interpretation |
| --- | ---: | ---: | ---: | --- |
| `none` | 0 | 0 | 0 | compute-matched Dr.GRPO |
| `mass_only` | 0 | .10 | 0 | ordinary verified rehearsal |
| `no_semantic` | 0 | .10 | .10 | replay plus known-mode balance |
| `no_mass` | .10 | 0 | .10 | remove verified-mass preservation |
| `no_balance` | .10 | .10 | 0 | remove known-mode balance |
| `full_fixed` | .10 | .10 | .10 | all three fixed terms |

No arm uses novelty/discovery reward, inverse adaptation, EMA reference,
coefficient warmup, counterfactual proposal, singleton escape, token entropy,
or KL regularization.

## Data, model, and budget

- Model: Qwen2.5-0.5B-Instruct from the common pinned initialization.
- Domains: Graph Coloring and PantryPlan. These are deliberately opposite
  historical balance cases: neutral and favorable, respectively.
- Seed: 58 for every arm.
- Data: the already sealed E76 320-row training and 64-row validation splits.
  The reported ModeBench test sets are never passed to these jobs.
- Training: four passes (1,280 optimizer updates), group size 16, one PPO
  epoch, learning rate 2e-7, rollout temperature 1, top-p 1, beta_KL = 0.
- Evaluation: every 80 updates; the registered screen endpoint is update 1,280.

## Readout and continuation rule

Report validation `distinct@8` and `pass@8` for every arm and domain at update
1,280. Do not pool domains. Inspect the paired trajectories for catastrophic
correctness loss, but do not select an earlier checkpoint.

For each component, the primary screening contrast is the full fixed arm minus
its remove-one arm. Verified mass also has the direct `mass_only - none`
contrast; balance also has `no_semantic - mass_only`; semantic MaxEnt also has
`no_balance - mass_only`.

A component advances to a multi-seed terminal confirmation in a domain only if
at least one of its two contrasts improves `distinct@8` by at least 0.15 while
reducing `pass@8` by no more than 0.02. Failure to advance is not evidence of
equivalence or harm; it means the quick screen found no effect large enough to
justify the next allocation.

## Integrity and failure policy

The runtime snapshot and job environment are recorded before submission.
Every live arm must report its registered fixed coefficients, with no adaptive
controller telemetry and no novelty advantage. The `none` arm must report an
exact-zero applied replay gradient. Each remove-one arm must report an
exact-zero gradient for the named component and a nonzero opportunity for each
retained component when eligible. A malformed run, non-finite loss, traceback,
missing terminal evaluation, source mismatch, or coefficient mismatch fails
closed and is not replaced silently.
