# E101: open-bank MaxEnt replay pilot on Countdown

Date frozen: 2026-08-14

## Question

E72 already compared replay mass alone with replay mass plus direct balance over
known bank modes. E101 does not repeat that as its main novelty. It asks whether
a separate verifier-gated explorer can enlarge the bank, after which the same
direct balance loss can protect the newly admitted mode without adding any
semantic term to the PPO advantage.

The design separates discovery from actuation:

1. ordinary task-only Dr.GRPO samples the on-policy group;
2. verified on-policy outputs populate the neutral bank;
3. only while that bank is a singleton, an isolated proposal channel may find
   one independently validator-positive, canonically new exemplar;
4. the proposal enters replay support only, never PPO and never neutral counts;
5. mass and balance are applied by teacher-forced replay.

An unseen bucket is therefore a sensor, not an actuator. A sequence must exist
before a sequence-level gradient can increase its probability.

## Objective

For prompt-local verified replay bank B and length-normalized sequence scores
s_theta(y), define q_theta(y | B) as the softmax of those scores over B. The
open arm minimizes

    L = L_PPO_task
        + lambda_m [-mean_{y in B} s_theta(y)]
        + lambda_b KL(U_B || q_theta(. | B)).

Both coefficients are fixed at 0.10. The implementation additionally applies
the existing Dr.GRPO/per-rollout scale shared by all three cells. No semantic
Shannon bonus, online canonical-bank advantage, token entropy term, exhaustive
support size, gold mode, or evaluation feedback enters training.

The balance derivative with respect to a bank score is q_i - 1/|B|. It is zero
for a singleton and restorative for a low-score mode after admission. The mass
derivative is -1/|B| for every retained mode.

## Frozen arms

| arm | replay mass | known-bank balance | singleton explorer |
|---|---:|---:|---:|
| mass | yes | no | no |
| balance | yes | yes | no |
| open | yes | yes | yes |

The mass arm uses verified_likelihood_per_rollout. The other arms use
split_mass_balance_per_rollout. The open arm uses separate objective support,
singleton-only activation, one proposal attempt per eligible update, and sends
exactly zero proposal rows to PPO. Deterministic validator-preserving
transformations are disabled for E101: a proposal must be a fresh sample from
the untouched original prompt, then pass the same validator and canonicalizer.
All three arms use replicated free-form sampling with one-GPU local actor weight
synchronization. This plumbing is held matched because the proposal path
requires it; only the open arm enables the proposal actuator.

## Tiny run

- Model: Qwen2.5-0.5B-Instruct.
- Dataset: 32 exact three-number multi-answer Countdown training prompts and 32
  disjoint evaluation prompts, generated with seed 101 and 2--8 canonical modes.
- One matched training seed: 101.
- Four passes, 128 optimizer updates, group size 16, learning rate 2e-7, beta 0.
- Evaluation at steps 0, 64, and 128: greedy plus K=8 sampled coverage, two
  fixed draws, evaluation seed 810101.
- Same A6000 node family for all cells. Each Slurm allocation has a hard
  00:55:00 limit. This is a mechanism screen, not a powered performance claim.

## Readouts and decision rule

Report endpoint and change-from-step-zero for greedy accuracy, pass@8, mean@8,
distinct correct modes@8, and normalized mode coverage@8. Also report replay
actuator groups/modes/tokens, balance loss/entropy, proposal eligibility,
admissions, cumulative new outcomes, and every proposal-to-PPO telemetry field.

Interpret the pilot in this order:

1. No open-arm admission: the bottleneck is discovery; improve the explorer,
   not MaxEnt or advantage mixing.
2. Admission but no subsequent replay actuation: improve scheduling, likely a
   fresh-admission priority queue.
3. Admission and actuation, but balance is no better than mass: improve direct
   bank dose or scheduling before adding components.
4. Open exceeds matched balance on diversity without a clear correctness loss:
   queue a multi-seed scale-up of the simple composition.
5. Open does not exceed balance despite admission and actuation: do not scale
   the combination yet.

No tuning or arm replacement is allowed after observing these three cells. Any
follow-up is a separately named experiment.

## Pre-materialization execution repair

Initial jobs 30579748--30579750 remained pending at elapsed zero and created no
run directories. The E101m startup diagnostic then proved that the open-arm
configuration was missing the replicated free-form sampling and local one-GPU
weight-sync flags required by the frozen argument validator. Releasing the two
controls under different sampling plumbing would not be a matched comparison,
so all three initial jobs were canceled before allocation. The `e101r1` run
stamps add the two required flags to every arm; no data, objective coefficient,
seed, decoding setting, evaluation, or decision rule changes.
