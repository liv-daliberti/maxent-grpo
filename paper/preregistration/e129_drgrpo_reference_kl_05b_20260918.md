# E129: compute-matched Dr.GRPO plus a reference KL, at three coefficients

**Frozen before submission on 2026-09-18.**

## Question

A reference-KL penalty is the regularizer most RLVR recipes carry, and every
arm this project has run so far sets `beta_KL = 0`. Does adding it back retain
verified execution-mode breadth, and does the breadth it retains depend on the
coefficient?

The appendix result this cohort tests is not that reference KL fails to prevent
winner-take-all. In the categorical mean flow it prevents it for every positive
coefficient. The result is that the allocation it holds the policy at is the
*reference's own*, and that this allocation does not move with the coefficient.
Because oat resolves an empty `ref_pretrain` to `pretrain`, the reference here
is exactly the frozen base model whose pass-0 concentration is already
measured, which makes the prediction directly checkable against a quantity the
project has in hand.

## Prespecified predictions

Stated before submission, so that the cohort can disconfirm them.

1. **Anchoring.** Terminal success-conditional breadth (PMD) under each KL arm
   lands nearer the frozen pass-0 model's PMD than the matched Dr.GRPO control
   does, and does not exceed the matched verified-replay arm's PMD.
2. **Coefficient independence.** Terminal PMD does not differ systematically
   across the three coefficients. Pairwise arm differences in terminal PMD are
   smaller than the control-to-replay difference in the same domain.
3. **No restoration.** In domains where the matched control has already
   collapsed to a single verified key by an intermediate checkpoint, the KL
   arms do not recover additional verified keys by pass 8.

Disconfirmation of (1) or (2) --- in particular a coefficient-ordered rise in
breadth --- is a result about the flow's applicability to neural training and
is to be reported as such, not set aside.

## Design

- Model: the pinned Qwen2.5-0.5B-Instruct revision used by the 0.5B main-body
  cohort.
- Domains: Graph Coloring, Countdown, Python Factors, MathIR, and PantryPlan.
- Data: each domain's released 384-prompt training pool and fixed 128-prompt
  evaluation split.
- Seeds: 43, 44, 45, 46, and 47, paired within domain and GPU model.
- Training: exactly eight passes, hence 3,072 optimizer updates per run;
  group size 16, one PPO epoch, learning rate 2e-7, rollout temperature 1,
  and top-p 1.
- Evaluation and resumable model checkpoints: every 192 updates, corresponding
  to passes 0, 0.5, 1.0, ..., 8.0. Pass 8 is the terminal endpoint.
- Placement: every cell inherits `source_node` from the same frozen manifest
  the matched control read, and the launcher refuses to submit a cell whose
  node differs from its control's. Pass-0 values cluster by GPU pool, so a
  cross-pool difference would confound placement with the intervention.

The complete cohort has 5 domains x 3 arms x 5 seeds = 75 runs.

## Arms

No control is re-run. The comparator is the existing compute-matched Dr.GRPO
arm of E78, at the same domains, seeds, and nodes, and the existing verified
replay arm of E78 is the second reference point. Each E129 cell reproduces the
E78 control configuration exactly --- including the passive verified bank,
global round-robin scheduler, replay batch materialization, teacher-forced
score traversal, and exact-zero applied replay derivative --- and moves one
knob.

### `kl0p001`, `kl0p01`, `kl0p04`

Reference-KL coefficient `beta` set to 0.001, 0.01 and 0.04 respectively. The
implementation adds `beta * k3(pi_ref || pi_theta)` to the loss, where `k3` is
the low-variance estimator that is unbiased for `KL(pi_theta || pi_ref)` under
samples from `pi_theta`, aggregated by the same masked aggregator the policy
loss uses. The reference policy is the frozen `pretrain` checkpoint, resident
for the whole run and never updated.

The three coefficients span the range common in published recipes. They are
not tuned, and no coefficient is selected using evaluation behavior.

## Exact exclusions

Every arm hard-disables semantic Shannon shaping, semantic separate advantages,
quality-gated or signed semantic pressure, conditioned-bank balance KL,
canonical-bank entropy shaping, token entropy bonuses, adaptive coefficients,
counterfactual proposals, singleton escape, novelty reward, and the applied
verified-replay derivative. Reference KL is the single live intervention. No
exhaustive support, evaluation outcome, desired mode count, or desired entropy
is available to training, scheduling, stopping, or checkpoint choice.

## Outcomes and estimands

At every registered half-pass checkpoint report, separately by domain:

- greedy pass@1;
- sampled mean correctness@8;
- sampled pass@8;
- mean distinct correct modes@8;
- excess multiplicity, `distinct@8 - pass@8`; and
- success-conditional PMD.

The primary comparison is the paired seed difference `kl_arm - control` at pass
8 for PMD and `pass@8`, against the E78 control at the same domain and seed.
The secondary comparison is the paired difference between KL arms at pass 8,
which is what prediction (2) is read from, and the third is each arm's terminal
PMD against the frozen pass-0 value for prediction (1). Show all five paired
seed differences and their mean and range; do not pool domains into one effect
and do not select a best checkpoint.

Mechanism telemetry reports `reg_loss` and `kl3` on every update, the resident
reference policy's identity, and the exact-zero applied replay gradient. A run
whose `kl3` is identically zero while its coefficient is positive fails closed:
that is the silent no-op this design exists to rule out.

## Integrity and failure policy

All 75 cells are submitted held from one hash-bound runtime snapshot and are
released only after their scheduler environments pass an exact audit that
includes each cell's own coefficient. A malformed environment, source mismatch,
duplicate run directory, non-finite loss, traceback, or missing pass-8 endpoint
fails closed. Infrastructure interruption may resume only from the same run's
hash-bound checkpoint and exact bank, optimizer, data-cursor, and request-stream
state. No failed scientific run is silently replaced or excluded.

Cells inherit a frozen manifest that this launcher verifies to be
reference-free (`beta = 0`) before submitting; the coefficient under study is
applied here and is never inherited.
