# E70 Stage A: clean five-seed verified MaxEnt versus compute-matched Dr.GRPO

**Status: FROZEN BEFORE SUBMISSION — 2026-07-29**

## Scope

This is the prospectively staged launch of the four already-established
ModeBench rows in the clean eight-environment comparison:

1. graph coloring;
2. Countdown;
3. executable Python factors; and
4. executable MathIR action menus.

The user authorized staged submission so these rows do not wait for the new
ConstructiveCode, PantryPlan, PointMaze, and AntMaze admission work. Staging
changes only submission time. The final paper analysis still requires the
complete admitted eight-row manifest and may not treat Stage A as a selected
standalone result.

## Cohort

- Model: pinned Qwen2.5-0.5B-Instruct snapshot
  `7ae557604adf67be50417f59c2c2f167def9a775`.
- Arms: `grpo_compute_matched` and
  `verified_first_global_replay_canonical`.
- Seeds: 43, 44, 45, 46, and 47.
- Rollouts per training prompt: 16.
- Budget: exactly 12 complete passes through each frozen training pool.
- Evaluation: greedy plus four deterministic temperature-one `K=8`
  replicates every quarter pass.
- Stage-A size: 4 environments x 2 arms x 5 seeds = 40 jobs.

No historical checkpoint or result is reused.

## Arms

The treatment is the literal E58-style online verified MaxEnt recipe:
semantic Shannon coefficient 0.10, online canonical novelty beta 0.50,
replay alpha 0.10, replay capacity 16, one persistent-hash global verified
replay group per optimizer update, and 64-step open-set, verified-mass, and
known-mode-balance warmups. There is no coefficient projection, token-entropy
objective, E68 proposal actuator, E69 route-successor replay, gold support,
target entropy, target mode count, reference solution, or evaluation
feedback.

The control is objective-equivalent plain Dr.GRPO with the same passive
verifier-bank bookkeeping, one-group replay scoring, and backward traversal.
`online_canonical_replay_compute_only=1` forces the replay derivative to
exactly zero; entropy, novelty, and balance coefficients are zero.

## Matching and execution

Within each environment, arms share the model, dataset, source and execution
snapshots, optimizer, learning rate, prompt order, rollout count, verifier
calls, token limits, training budget, evaluation draws, checkpoint cadence,
and recovery rules. Jobs run under the `mltheory` account on the `mltheory`
partition, restricted to the A5000/A100 pool.

All 40 jobs are submitted held. Before release, the launcher must verify the
exact job count, seeds, arms, 12-pass budget, evaluation cadence, immutable
source/ops roots, resource request, treatment actuator, and compute-only
control. Any mismatch cancels the still-held cohort.

## Outcome and failure policy

Report terminal pass 12 and trapezoidal AUC over registered anchors for
greedy success, `mean@8`, `pass@8`, and `distinct@8`, retaining all five seed
trajectories and paired deltas. No best-checkpoint selection, early stopping,
seed substitution, carry-forward, result-dependent extension, or
missing-cell averaging is allowed.

CUDA OOM, non-finite loss/coefficient, traceback, malformed checkpoint,
identity mismatch, missing expected seed/arm, or failure to reach the fixed
terminal budget is a run failure. A failed job may resume only from its own
source-bound checkpoint with the same arm, seed, data, and protocol.
