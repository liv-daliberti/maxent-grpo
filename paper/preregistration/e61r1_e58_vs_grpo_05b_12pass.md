# E61-R1: three-seed E58 versus matched Dr.GRPO over four executable domains

Status: executable protocol, frozen before submission.

## Question

At Qwen2.5-0.5B scale, does the E58 global verified-replay canonical
objective improve and sustain model-generated correct-mode multiplicity over
matched Dr.GRPO on:

1. graph coloring,
2. Countdown,
3. executable Python factors, and
4. executable MathIR action menus?

This is a literal replication of the E58 actuator, not the E60
bootstrap-then-local repair. E60 remains a separate causal pilot. E61-R1
differs from the invalidated E61 attempt only by the admission-safety repair
below.

## Cohort

- Arms: `grpo` and `verified_first_global_replay_canonical`.
- Seeds: 43, 44, and 45 for each arm and domain.
- Model: the locally pinned Qwen2.5-0.5B-Instruct snapshot.
- Rollouts per training prompt: 16.
- Training budget: exactly 12 complete passes through each domain's fixed
  training pool, unless a run fails.
- Total jobs: 4 domains × 2 arms × 3 seeds = 24.
- Evaluation: fixed K=8 stochastic mode-coverage evaluation with four
  replicates, every quarter pass.

The terminal budget is fixed before results are observed. No result-dependent
early stopping or extension is permitted.

## Matched variables

Within a domain, arms share the exact dataset, source snapshot, model,
optimizer, learning rate, prompt formatting, rollout count, training budget,
evaluation cadence, evaluation seed, checkpoint cadence, and recovery rules.
Only the E58 canonical mechanism differs from Dr.GRPO.

## E58 mechanism

- No direct token-entropy objective.
- Empty online canonical bank at initialization.
- Model-verified discoveries are the only entries admitted to the bank.
- Admission is fail-closed at the intersection of executable canonical
  validation and positive rollout task reward. A disagreement is excluded
  from both passive tracking and the active bank and is recorded in telemetry;
  it is not a fatal process assertion.
- One persistent-hash round-robin global verified replay group per optimizer
  update after discovery.
- Split verified-mass and known-mode-balance replay objective.
- Open-set semantic entropy controller, verified-mass controller, and
  known-mode-balance controller use their own fixed self-warmup observations.
- Coefficients are unprojected: there are no minimum or maximum alpha bounds.

## Information firewall

Training and adaptation receive none of the following:

- the enumerated gold support,
- a desired number of modes,
- a desired entropy value derived from gold support,
- evaluation scores,
- held-out examples, or
- a target value for mean distinct correct.

Gold catalogues are used only by held-out evaluation and never enter the
training process, replay bank, controller state, stopping rule, or scheduler.

## Outcomes

The primary descriptive trajectory is mean distinct correct@8 over training
passes. Pass@8, mean correctness@8, and excess multiplicity
(`distinct@8 - pass@8`) are reported alongside it. Plots show all three seeds
and the across-seed mean without hiding incomplete seeds.

Stability is assessed from the observed three-seed trajectories and terminal
windows; no numeric success threshold is supplied to training or chosen as a
controller target.

## Failure policy

CUDA OOM, non-finite loss/coefficient, traceback, malformed checkpoint,
missing expected seed/arm, or failure to reach the fixed terminal budget is a
run failure. Failed jobs may resume only from their own exact, source-bound
checkpoint. Replacement jobs retain the same arm, seed, data, and protocol.

The original E61 cohort is not resumed because its Graph-coloring baseline
seed 44 failed before the first checkpoint under the obsolete unconditional
admission assertion. E61-R1 starts every arm/domain/seed from initialization
under one newly frozen source snapshot; no E61 weights or controller state are
reused.
