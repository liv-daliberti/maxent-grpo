# E66: same-plumbing actuator ablation for E65R1

Frozen 2026-07-27 14:00 EDT, before submitting any E66 job and before any
three-seed-complete post-training E65R1 checkpoint existed. E65R1 mechanism
telemetry and checkpoint-0 base-model evaluations were visible; no
three-seed E65R1 performance comparison after training was available.

## Why this cohort is required

The zero-step E65R1 amendment correctly enabled the execution path required by
support-only counterfactual proposals:

- replicated free-form sampling; and
- local one-learner/one-actor weight synchronization.

Inspection after launch established that replicated free-form sampling also
uses an explicit deterministic request seed for every prompt batch, whereas
the historical E61-R1 E58 cohort used the ordinary collector path. Therefore
E65R1 versus historical E58 is not a clean causal actuator ablation even
though both implement the same optimizer objective outside the proposal
mechanism.

E66 closes that gap. It runs literal
`verified_first_global_replay_canonical` from E65R1's already frozen source and
execution snapshots, with replicated free-form sampling and local actor
weight synchronization enabled, but with counterfactual proposals and the
singleton entropy gate disabled. Thus:

- E61-R1 E58 versus Dr.GRPO remains the broad method comparison;
- E66 versus E61-R1 E58 measures the execution-plumbing/request-stream
  sensitivity;
- E65R1 versus E66 is the causal actuator comparison.

No E61-R1, E64, or E65R1 job is stopped, modified, or selected because of this
addition.

## Frozen design

The cohort contains 12 independent 0.5B runs:

- Graph coloring, seeds 43, 44, and 45;
- Countdown, seeds 43, 44, and 45;
- Python factors, seeds 43, 44, and 45; and
- MathIR action menu, seeds 43, 44, and 45.

Every run uses the exact E65R1 frozen source hash
`f6147daacbfdde22e0e9d5fab6fc45b41017d4dbf5e2848f827923ddf7828a7f`
and execution-surface hash
`ff2a4f37d8653ba1a5888538d22743269c73d541b7bc6c8db214cedaec9c5d8f`.
Model, dataset, prompt order, seed, optimizer, semantic coefficient,
unbounded inverse controllers, verified bank, replay schedule, evaluation
seeds, and checkpoint cadence match E65R1. The only algorithmic differences
are:

- `online_canonical_counterfactual_proposals = false`; and
- `online_canonical_counterfactual_singleton_entropy_gate = false`.

Replicated free-form sampling and local actor synchronization are both true.

The fixed ModeBench checkpoints remain
`0, 1, 2, 3, 4, 5, 6, 8, 10, 12` passes. Every reported point requires all
three seeds. Terminal pass 12 and the trapezoidal AUC over all ten registered
checkpoints are primary; intermediate peaks cannot select a result.

## Placement

Placement is fixed before submission:

- all three Graph controls request A6000 node103;
- all six Countdown/Python controls request RTX 3090 node024;
- all three MathIR controls request A100 node302.

This exactly matches the E65R1 accelerator family for 11 of 12 paired runs.
Graph seed 43 is the disclosed exception: E65R1 used A5000 and E66 requests
A6000. Accelerator identity and node are reported per seed and cannot select
checkpoints or exclusions.

## Information firewall and audit

E66 may not read gold support, a desired mode count, a desired entropy,
evaluation feedback, reference exemplars, or a coefficient bound. The audit
must establish:

- the frozen source and execution hashes above;
- literal E58 optimizer and replay configuration;
- deterministic replicated request seeding and local actor synchronization;
- zero counterfactual proposal activity;
- zero proposal rows sent to PPO;
- zero coefficient projection and forbidden feedback;
- finite controller/model telemetry;
- all three seeds at every terminal domain; and
- all ten fixed checkpoints in the saved result surface.

## Frozen interpretation

The E65R1 repair is causally successful only if, relative to E66:

- Python's terminal three-seed mean pass@8 is higher and its worst-seed
  pass@8 is higher;
- Graph, Countdown, and MathIR each lose no more than 0.05 in terminal
  three-seed mean pass@8 and no paired seed loses more than 0.15;
- at least one audited E65R1 entropy-gated singleton intervention occurs; and
- the E65R1, E66, and combined integrity audits pass.

Historical E58 comparisons remain fully reported. E66 versus historical E58
is a plumbing-sensitivity analysis with the same 0.05 mean and 0.15 paired
seed descriptive margins; it is not allowed to replace the E65R1-versus-E66
causal gate.

The held-out MATH-500 realism gate is unchanged and remains a separate
E58-versus-Dr.GRPO external-validity result.
