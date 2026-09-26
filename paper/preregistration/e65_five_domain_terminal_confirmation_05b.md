# E65: five-domain terminal confirmation and entropy-gated singleton repair

**Status: repair and reporting contract frozen before E65 submission —
2026-07-27.**

## Evidential status

This protocol converts the current E61-R1/E64 live campaign into a
paper-grade result and prospectively tests one repair. E61-R1 ModeBench
trajectories through approximately 4–10 passes and E64 initialization results
were inspected before this document was frozen. They are therefore described
as exploratory trajectories with a subsequently frozen terminal/AUC analysis,
not as untouched preregistered data.

E65 weights, checkpoints, and evaluation results do not yet exist. Its
algorithm, source, seeds, endpoints, and gates below are prospective. E65 is
never relabeled as E58.

## Five environments and arms

The result surface contains:

1. Graph coloring;
2. Countdown;
3. executable Python factors;
4. executable MathIR action menus; and
5. held-out MATH-500 transfer after training on frozen MATH12K-384.

E61-R1 supplies matched Dr.GRPO and literal E58 for the first four domains.
E64 supplies matched Dr.GRPO and literal E58 for the held-out MATH-500 row.
E65 adds `verified_entropy_gated_singleton_escape_canonical` on the four
ModeBench domains only. MATH-500 has one certified correct-answer class, so
the multi-mode singleton actuator is structurally inapplicable and E64
remains the realism result.

All comparisons use Qwen2.5-0.5B-Instruct, seeds 43/44/45, 16 rollouts per
training prompt, the same frozen domain data, learning rate `2e-7`, one PPO
epoch, and exactly 12 training passes.

## Fixed reporting checkpoints

The paper plot uses no more than ten checkpoints per environment:

- ModeBench: passes 0, 1, 2, 3, 4, 5, 6, 8, 10, and 12;
- MATH-500: passes 0, 2, 4, 6, 8, 10, and 12.

Intermediate optimizer telemetry may be retained for safety auditing but is
excluded from the confirmatory plot and AUC. Axes expand as fixed checkpoints
land; they are not truncated at an early pass or padded with future points.

## E65 actuator

E65 is literal E58 plus one support-only escape actuator. The ordinary E58
open-set semantic coefficient, verified-mass coefficient, and known-mode
coefficient remain self-warmup, inverse, unprojected controllers. No
coefficient has a lower or upper bound.

For the current prompt, a counterfactual proposal is eligible only when all
of these conditions hold:

1. the current/prior verified bank contains exactly one outcome;
2. the open-set controller completed its fixed 64-observation warmup;
3. its current normalized predictive-entropy EMA is below the model's own
   warmup-mean reference; and
4. consequently its unprojected inverse multiplier is greater than one.

The proposal starts from the model's own reward-positive, executable response.
Domain-specific validator-preserving transformations are independently
revalidated before and after tokenization. If no transformed alternate is
available, the already isolated fixed-budget original-prompt temperature
sweep may search for one. Proposal rows never enter PPO.

At most one novel outcome can be admitted. A bank with zero outcomes or two
or more outcomes is ineligible. Thus the repair restores actuator
identifiability at singleton support but cannot fill a replay bank, chase a
mode count, or keep synthetically expanding a prompt.

## Information firewall

Training, gating, adaptation, checkpoint selection, and scheduling may not
read:

- gold or exhaustive support;
- a desired mode count or desired entropy;
- evaluation distinctness, pass@K, correctness, or MATH-500 output;
- a coefficient projection or clipping bound;
- reference solutions as replay exemplars; or
- an LLM-derived strategy label.

The singleton test is an actuator-identifiability condition. It is not a
claim that the task has exactly two valid modes.

## Outcomes and estimands

For each ModeBench checkpoint report every seed, the three-seed mean/range,
and paired seed differences for:

- neutral pass@1;
- sampled mean correctness@8;
- sampled pass@8;
- mean distinct correct@8; and
- excess multiplicity, `distinct_correct@8 - pass@8`.

For MATH-500 report greedy pass@1, sampled mean correctness@8, sampled
pass@8, response length, and answer-extraction rate. Distinct surface forms
are not treated as reasoning modes.

The primary summaries are the terminal pass-12 value and trapezoidal AUC over
the fixed checkpoints. Peak performance and best-checkpoint selection are
not primary evidence. With three seeds, uncertainty is shown as all points,
range, and paired differences; asymptotic p-values are not used to imply
precision the cohort does not contain.

## Frozen interpretation gates

Literal E58 is called a cross-domain positive result only if its terminal and
fixed-checkpoint AUC are directionally above matched Dr.GRPO for both
pass@8 and distinct correct@8 in at least three of four ModeBench domains,
with no domain losing more than 0.05 terminal pass@8 in the three-seed mean.
The Python seed dispersion remains reported even if its mean passes.

The E65 repair is called successful only if:

- Python's terminal three-seed mean pass@8 is above E58 and its worst-seed
  pass@8 is above E58's worst seed;
- on each of Graph, Countdown, and MathIR, terminal E65 mean pass@8 is no
  more than 0.05 below E58 and no paired seed loses more than 0.15;
- at least one E65 run records an entropy-gated singleton intervention and
  every intervention admits no more than one alternate before becoming
  ineligible; and
- all forbidden-feedback, projection, proposal-to-PPO, and finite-state
  audits pass.

The realism claim is separate: E58 transfers to held-out MATH-500 only if its
terminal greedy pass@1 and sampled mean@8 are each no more than 0.02 below
matched Dr.GRPO, and at least one is directionally higher. A failed MATH gate
is reported as a boundary of multi-mode transfer, not hidden by ModeBench.

## Completeness and recovery

No aggregate checkpoint is drawn unless all three seeds for that arm/domain
landed it. A terminal claim requires all expected runs at pass 12.

A preemption may resume only from the same job's source-bound checkpoint with
matching model, optimizer, progress counter, data cursor, canonical bank,
three controller states, and deterministic request-stream state. Auditing is
attempt-aware: a recovered, explicitly permitted infrastructure termination
is retained in provenance but is not confused with an unrecovered final
attempt. A non-finite value, semantic-state mismatch, missing terminal
checkpoint, or algorithm exception is a run failure and cannot be erased by
an appended log.

## E65R1 zero-step launch amendment

The original E65 Slurm cohort reached runtime argument validation with zero
optimizer steps and aborted because its proposal arm did not pin the
replicated free-form sampling and local one-GPU weight-sync flags. No outcome,
evaluation, or training trajectory from that cohort existed when this
amendment was made. E65R1 supersedes those job identities, pins both required
execution flags in the arm-level submission contract, and adds a held-job
audit for both flags before release. The model, data, seeds, controller,
actuator, fixed checkpoints, estimands, and interpretation gates above are
unchanged.
