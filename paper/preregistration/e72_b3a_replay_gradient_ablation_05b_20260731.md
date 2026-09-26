# E72 B3a: count-based rare-outcome credit without acting replay

**Status: FROZEN BEFORE COHORT SUBMISSION — 2026-07-31**

## Question

Does verified replay contribute to xGRPO's retained support, or do the
open-set rare-mode and first-discovery advantages account for it on their own?

The same cohort answers a reviewer's question about the baseline set: a
count-based rare-outcome bonus with no memory beyond the collection batch is one
of the simpler diversity-preserving alternatives xGRPO must beat.

## Arm

Runtime variant `verified_first_replay_gradient_ablation`. It is the frozen E58
treatment with `online_canonical_replay_compute_only=1`, which replaces the
replay score derivative with exact zeros before the optimizer while the same
verified banks are still retained, scheduled in global round robin,
teacher-forced, and traversed backward. Every other coefficient, the open-set
rare-mode advantage, the first-discovery credit, and their self-referenced
controllers act exactly as in the treatment.

This makes the arm simultaneously:

- a **compute-matched remove-one ablation** of verified replay, with generation,
  scoring, and optimizer work identical to the treatment; and
- a **count-based rare-outcome baseline** that receives discovery credit but has
  no mechanism to act on a discovery after its collection batch.

No coefficient is tuned. Every value is inherited from the paired xGRPO run's
own recorded arguments, and the launcher refuses to submit if any of them
disagrees with the frozen E58 doses (semantic coefficient `.10`, novelty
`.50`, replay and mass alphas `.10`, capacity 16, one global replay group per
update, 64-step warmups, EMA `.9`, surprisal clip 5, pseudocount 1, bank alpha
0, learning rate `2e-7`, 12 epochs, `beta_KL=0`, rollout temperature 1,
$G=16$). Dose screening is deliberately omitted: tuning a coefficient would
change the question from "does this component matter" to "can this baseline be
made to work", and the second question is only worth asking if the first
answers yes.

## Cohort

- Model: pinned Qwen2.5-0.5B-Instruct `7ae557604adf67be50417f59c2c2f167def9a775`,
  trained from initialization exactly as every reported arm.
- Domains: graph coloring, Countdown, Python factors, MathIR, PantryPlan.
- Seeds: 43, 44, 45, 46, 47. Size: 5 domains x 5 seeds = 25 runs.
- Budget: 12 complete passes over the frozen 384-prompt pool, 4,608 optimizer
  updates, evaluated on the same fixed 128-prompt split every quarter pass with
  greedy decoding plus four deterministic temperature-one $K=8$ replicates.
- Data, prompt template, response budget, verifier, evaluation draw seeds, and
  canonical-action interface are inherited per domain from the E70/E71 runs.

**Placement.** Each run is pinned to the GPU model that trained its paired
xGRPO seed. GPU model changes floating-point reduction order and therefore
which tokens are sampled from identical weights under identical draw seeds; a
three-way per-seed comparison on mixed hardware would reintroduce that
confound. This is a departure from, and an improvement on, the E70/E71 design,
where three of twenty-five paired seeds are split across GPU models.

## Reporting

Terminal pass 12, all five seeds, per domain, reported on the common rule
already used for the headline surface: greedy `pass@1`, `mean@8`, `pass@8`, and
`distinct@8`, with paired per-seed deltas against both the matched Dr.GRPO
control and the xGRPO treatment. No best-checkpoint selection, early stopping,
seed substitution, carry-forward, or missing-cell averaging.

The primary quantity is `distinct@8` at pass 12, read two ways:

- **B3a vs xGRPO** isolates the replay gradient. A gap means replay contributes
  beyond discovery credit.
- **B3a vs matched Dr.GRPO** measures what count-based credit alone buys.

## Registered interpretations

Written before any B3a result is observed.

- **R1.** If B3a lands within `.15` `distinct@8` of xGRPO in at least three of
  five domains, verified replay is not the active ingredient. The replay
  component is demoted in Section~4.3 and `tab:component-ablations`, and the
  contribution is restated around open-set discovery credit.
- **R2.** If B3a sits between the control and the treatment in most domains,
  both components act, and the appendix reports the decomposition rather than
  claiming either is sufficient.
- **R3.** If B3a is at or near the matched Dr.GRPO control, discovery credit
  alone does not retain support, and replay carries the effect. This licenses
  the first empirical remove-one row for verified replay, replacing an analytic
  argument.
- **R4.** If B3a exceeds xGRPO anywhere, that is reported as-is; a favorable
  ablation is not selected over an unfavorable one.

The comparison is per domain. Nothing is pooled across domains, and no domain
is dropped for being unfavorable.

## Failure policy

CUDA OOM, non-finite loss or coefficient, traceback, malformed checkpoint,
identity mismatch, a missing seed, or failure to reach the fixed terminal budget
is a run failure. A failed run may resume only from its own source-bound
checkpoint with the same arm, seed, data, and protocol. Runs that do not reach
pass 12 on all five seeds of a domain leave that domain unreported.

An integrity requirement specific to this arm: `canonical_replay_compute_only`
must be 1 and the applied replay score gradient must be identically zero in
telemetry for every logged update. A run with a nonzero applied replay gradient
is not this arm and is discarded rather than reinterpreted.
