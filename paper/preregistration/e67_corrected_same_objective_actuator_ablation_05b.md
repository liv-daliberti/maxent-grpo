# E67: corrected same-objective entropy-gated actuator ablation

Frozen 2026-07-27 after E66 began but before any three-seed-complete
post-training E66 checkpoint existed. E66 checkpoint-0 results and incomplete
training telemetry were visible. No E66 terminal result or registered
post-training three-seed comparison was available.

## Non-outcome reason for correction

The E65R1 protocol described its treatment as literal E58 plus an
entropy-gated, support-only singleton actuator. A direct runtime equivalence
check found that this was false: E65R1's frozen `run_experiment.sh` branch
overwrote `online_canonical_novelty_beta=0.50` to `0.0`, while E58 and E66
retained `0.50`. Countdown seed 43 therefore differed at optimizer update 1,
before the 64-update actuator warmup could complete, despite identical rollout
reward, request seed, discovered outcomes, replay loss, and controller state.

This is an objective mismatch, not an outcome-dependent judgment. E65R1 is
retained as disclosed engineering evidence but is excluded from confirmatory
performance gates. Its jobs are stopped to avoid spending compute on a cohort
that cannot answer the causal question.

## Corrected design

E67 contains 12 independent 0.5B treatment runs:

- Graph coloring, seeds 43, 44, and 45;
- Countdown, seeds 43, 44, and 45;
- Python factors, seeds 43, 44, and 45; and
- MathIR action menu, seeds 43, 44, and 45.

It uses E65R1's frozen model source and E66's data, seed, optimizer, replay,
replicated request-seeding, local actor synchronization, evaluation, and
checkpoint contracts. Relative to E66, only these actuator settings differ:

- `online_canonical_counterfactual_proposals = true`; and
- `online_canonical_counterfactual_singleton_entropy_gate = true`.

Both E66 and E67 must use:

- `online_canonical_bank_alpha = 0`;
- `online_canonical_novelty_beta = 0.50`;
- unbounded, unprojected open-set, verified-mass, and known-mode inverse
  controllers;
- one global verified replay group per optimizer update;
- no proposal row in PPO; and
- no gold support, desired entropy, desired mode count, reference exemplar, or
  evaluation feedback.

The corrected execution surface contains one substantive E65 branch fix:
the repair variant inherits the configured E58 bank alpha and novelty beta
instead of forcing both to zero. A runtime expectation variable must equal
`0.50`; disagreement aborts before training.

## Fixed evidence surface

The ModeBench checkpoints remain
`0, 1, 2, 3, 4, 5, 6, 8, 10, 12` passes. Every paper point requires all three
seeds. Pass 12 and trapezoidal AUC over all ten fixed checkpoints are primary.
Intermediate peaks cannot select checkpoints, exclusions, or hyperparameters.

E61-R1 and E64 remain the historical method and held-out-realism tracks. The
confirmatory campaign continues to contain 54 valid runs:

- 24 E61-R1 ModeBench runs;
- 6 E64 MATH-500 runs;
- 12 E66 same-plumbing actuator-off controls; and
- 12 corrected E67 actuator-on treatments.

The 12 invalid E65R1 runs are not included in that denominator.

## Placement

After a read-only capacity check and before any E67 submission, placement was
fixed to the same paired accelerator families without requiring the same
physical hostname:

- Graph requests A6000 node103;
- Countdown and Python request RTX 3090 node022 or node026; and
- MathIR requests A100 node302.

At freeze time these nodes had enough newly released compatible capacity to
avoid serializing treatment behind control. Actual node and accelerator
identity are reported per seed and cannot select an exclusion.

## Fail-closed equivalence and mechanism audits

The campaign cannot pass unless:

- E66 and E67 runtime logs both report bank alpha `0` and novelty beta `0.5`;
- paired E66/E67 request seeds, rollout rewards, discoveries, replay
  quantities, and non-actuator objective telemetry agree before the first E67
  intervention, allowing only timing fields and ordinary floating-point
  tolerance;
- every E67 proposal admission satisfies the registered warmup,
  below-own-reference, inverse-multiplier, singleton-support, and active-gate
  conjunction;
- every proposal group admits at most one alternate;
- proposal rows sent to PPO remain zero;
- all controller coefficients remain finite and unprojected;
- all 24 E66/E67 runs reach pass 12; and
- all ten registered checkpoints are present in the saved seed-level result
  surface.

## Frozen interpretation

The corrected singleton actuator is causally successful only if, relative to
E66:

- Python terminal three-seed mean pass@8 is higher and its worst-seed pass@8
  is higher;
- Graph, Countdown, and MathIR each lose no more than `0.05` in terminal
  three-seed mean pass@8 and no paired seed loses more than `0.15`;
- at least one fully audited E67 intervention occurs; and
- E66, E67, equivalence, and combined integrity audits pass.

The historical E58 comparisons and invalidated E65R1 trajectories remain
reported with their evidential roles stated explicitly. Neither may replace
the E67-versus-E66 causal gate.
