# E79-PM: prospective Falcon3-1B PointMaze extension

**Frozen before Falcon warm-start training or E79-PM job submission on
2026-08-05.**

## Status and question

E79-PM is a separately identified sixth-domain extension to E79, not a
retroactive change to E79's frozen five-domain estimator.  It asks only whether
verified replay improves Falcon3-1B on PointMaze relative to a compute-matched
control.  It may be displayed beside E79 and E78-PM only with model/interface
strata identified.

## Shared environment and data

Use the exact certified `point-waypoint-v1` task, pinned MuJoCo runtime, 384
training maps, 64 development maps, and 128 evaluation maps materialized for
E78-PM from seed `88104`.  The ten E79-PM cells do not load the development
split.  The policy observes only the public maze, current and previous cells,
goal cell, position, velocity, remaining waypoint budget, and adjacent free
moves.  It selects one legal label in `A B C D`; a deterministic hash-bound PD
adapter executes that local edge.  No planner, shortest-path signal, route
identity, or verifier feedback enters the policy support.

## Falcon initialization

All cells start from one shared Falcon-specific weak warm start.  Train it from
the exact E75R3 train-only demonstration file used for Qwen's PointMaze warm
start, with no development/evaluation rows or online rewards.  Preserve its
semantic prompt and route-length-neutral dynamic-support loss, but serialize it
with Falcon3-1B-Instruct's published no-tools chat surface.  Freeze:

- base model: `tiiuae/Falcon3-1B-Instruct`, revision
  `28ba2251970a01dd1edc7ba7dad2eb71216ccfdf`;
- SFT seed `88404`, one epoch capped at exactly 72 AdamW updates;
- batch size 4, accumulation 8, learning rate `2e-5`, weight decay `0.01`;
- ten warm-up updates, linear decay, gradient norm 1, BF16, length 1536.

The labels `A`, `B`, `C`, and `D` must each be a distinct exact single Falcon
token.  Online jobs depend on a successful warm-start receipt and on the shared
E78-PM data-certification job.

## Paired online design

- Arms: compute-matched Dr.GRPO (`control`) and verified replay (`replay`).
- Seeds: 55, 56, 57, 58, and 59, paired across arms.
- Training: exactly eight ordered passes over 384 maps (3,072 updates).
- Rollouts: 16 per update; fixed interactive horizon 64.
- Optimizer: AdamW, learning rate `2e-7`, betas `(0.9, 0.999)`, epsilon
  `1e-8`, weight decay `0`, constant schedule, gradient norm 1.
- Evaluation: all 128 fixed evaluation maps at passes 0, 0.5, ..., 8.0.
- Checkpoint: one rolling resumable model/optimizer/replay-bank checkpoint
  every 192 updates, exactly every 0.5 pass.
- Prompt: identical public task content in Falcon's published chat surface.

The map order, rollout seeds, simulator request schedule, replay traversal, and
compute envelope are common across arms.

## Only scientific difference

The replay arm applies the same fixed uniform verified-likelihood rehearsal
loss as E78-PM, with coefficient `0.10`.  The control materializes and scores
the identical replay slots but applies an exact-zero replay derivative.
Singleton verified banks remain eligible.

Both arms hard-disable semantic MaxEnt, semantic/canonical novelty advantages,
balance KL, adaptive coefficients, counterfactual proposals, singleton escape,
token entropy, reference KL, goal-directed masks, and planner feedback.
Verified replay is the only auxiliary derivative.

## Reporting and failures

Report every seed and arm at every registered half pass: mean correctness@8,
pass@8, mean distinct verified routes@8, and excess route multiplicity.  The
primary paired endpoints are replay-minus-control distinct@8 and pass@8 at
pass 8; trapezoidal AUC over the complete half-pass grid is secondary.  Do not
replace, exclude, or pool failed cells silently.

Any source/data/model drift, split leakage, action-token failure, support
escape, nonfinite loss, nonzero disabled-mechanism telemetry, nonzero control
replay derivative, missing registered evaluation, or mismatched resume state
fails closed.
