# E68: separated-support entropy-gated actuator ablation

Frozen 2026-07-27 before any E68 submission. E66 checkpoint-0 results and
incomplete E66 training telemetry were visible. No E66 terminal result or
registered post-training three-seed comparison was available. E67 produced no
optimizer metrics.

## Non-outcome reason for superseding E67

E67 attempted to combine literal E58 novelty
(`online_canonical_novelty_beta = 0.50`) with counterfactual proposal
admissions. The frozen source correctly rejected that configuration before
optimization because both mechanisms used one canonical bank: a proposal
admission would enter the historical count table and could therefore change
the next neutral rollout's novelty advantage. Allowing that path would make
the proposed off-policy support part of PPO's on-policy advantage.

Three E67 MathIR jobs reached the same fail-closed argument validator and
raised `proposal support cannot feed an on-policy canonical-bank advantage`.
The other nine E67 jobs were canceled after the common configuration failure
was established. None of the 12 wrote `train_metrics.jsonl`. E67 is retained
as engineering evidence and excluded from all performance claims.

## Corrected design

E68 contains 12 independent 0.5B treatment runs:

- Graph coloring, seeds 43, 44, and 45;
- Countdown, seeds 43, 44, and 45;
- Python factors, seeds 43, 44, and 45; and
- MathIR action menu, seeds 43, 44, and 45.

E68 preserves E66's data, seeds, model, optimizer, neutral request seeding,
replicated sampling, local actor synchronization, evaluation, and checkpoint
contracts. Relative to E66, the registered actuator surface is:

- `online_canonical_counterfactual_proposals = true`;
- `online_canonical_counterfactual_singleton_entropy_gate = true`; and
- `online_canonical_counterfactual_separate_objective_support = true`.

The third switch is an isolation contract. Proposal-derived outcomes may enter
the replay-exemplar layer, which is the intended actuator, but may not enter
the historical on-policy count table used by canonical entropy or novelty
advantages. A proposed outcome remains novel to the E58 objective until the
neutral policy itself produces and validates it. At that point it receives
the ordinary E58 novelty bonus, graduates into the on-policy count table, and
its neutral exemplar replaces the proposal-only exemplar.

Both E66 and E68 use:

- `online_canonical_bank_alpha = 0`;
- `online_canonical_novelty_beta = 0.50`;
- unbounded, unprojected open-set, verified-mass, and known-mode inverse
  controllers;
- one global verified replay group per optimizer update;
- no proposal row in PPO; and
- no gold support, desired entropy, desired mode count, reference exemplar, or
  evaluation feedback.

## Fixed evidence surface

The ModeBench checkpoints remain
`0, 1, 2, 3, 4, 5, 6, 8, 10, 12` passes. Every paper point requires all three
seeds. Pass 12 and trapezoidal AUC over all ten fixed checkpoints are primary.
Intermediate peaks cannot select checkpoints, exclusions, or hyperparameters.

E61-R1 and E64 remain the historical method and held-out-realism tracks. The
confirmatory campaign contains 54 valid runs:

- 24 E61-R1 ModeBench runs;
- 6 E64 MATH-500 runs;
- 12 E66 same-plumbing actuator-off controls; and
- 12 E68 separated-support actuator-on treatments.

The invalid E65R1 and E67 trajectories are not included in that denominator.

## Placement

Read-only `srun --test-only` probes are recorded before submission. Placement
uses compatible nodes and partitions already advertised for the requested GPU
families:

- Graph requests A6000 node103 or node104 on `pvl-lowprio`;
- Countdown and Python request RTX 3090 nodes
  node020, node021, node022, node024, or node026 on `pvl-lowprio`; and
- MathIR requests A100 node302 on `mltheory`.

All jobs use the `mltheory` account. Actual node and accelerator identity are
reported per seed and cannot select an exclusion.

## Fail-closed equivalence and mechanism audits

The campaign cannot pass unless:

- E66 and E68 runtime logs both report bank alpha `0` and novelty beta `0.5`;
- every E68 runtime reports separated proposal/objective support enabled;
- every proposal admission reports objective-outcome delta exactly zero;
- proposal-only outcomes are absent from the checkpointed on-policy count
  table and present only in the separately labeled replay-exemplar support;
- paired E66/E68 request seeds, rollout rewards, discoveries, replay
  quantities, and non-actuator objective telemetry agree before the first E68
  intervention, allowing only timing fields and registered floating-point
  tolerance;
- every E68 proposal admission satisfies the registered warmup,
  below-own-reference, inverse-multiplier, singleton-support, and active-gate
  conjunction;
- every proposal group admits at most one alternate;
- proposal rows sent to PPO remain zero;
- all controller coefficients remain finite and unprojected;
- all 24 E66/E68 runs reach pass 12; and
- all ten registered checkpoints are present in the saved seed-level result
  surface.

## Frozen interpretation

The separated-support singleton actuator is causally successful only if,
relative to E66:

- Python terminal three-seed mean pass@8 is higher and its worst-seed pass@8
  is higher;
- Graph, Countdown, and MathIR each lose no more than `0.05` in terminal
  three-seed mean pass@8 and no paired seed loses more than `0.15`;
- at least one fully audited E68 intervention occurs; and
- E66, E68, pre-intervention equivalence, and combined integrity audits pass.

Historical E58 comparisons and invalidated E65R1/E67 trajectories remain
reported with their evidential roles stated explicitly. None may replace the
E68-versus-E66 causal gate.
