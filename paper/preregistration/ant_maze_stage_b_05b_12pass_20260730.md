# AntMaze Stage B: clean 0.5B verified MaxEnt versus compute-matched Dr.GRPO

**Status: FROZEN BEFORE THE V13 VIABILITY AND PAIRED-SMOKE OUTCOMES AND BEFORE STAGE-B SUBMISSION — 2026-07-30**

## Conditional authorization

This protocol may launch only if AntMaze v13 passes its frozen viability gate and the resulting immutable paired online smoke reports `status=pass` and `decision=eligible_for_ten_ant_maze_stage_b_jobs`. Dependency job 30202318 is already bound to v13 job 30202183 and can launch that development-only smoke only after the exact preregistered v13 pass. A failed or incomplete gate or smoke stops AntMaze; no final cell is submitted. The smoke seed and weights are never reused.

## Frozen Cartesian product

- Arms: compute-matched plain Dr.GRPO and `verified_first_global_replay_canonical`.
- Seeds: 43, 44, 45, 46, and 47.
- Shared initial checkpoint: the byte-identical Qwen2.5-0.5B-Instruct AntMaze v13 train-only constrained-action warm start.
- Training pool: all four frozen AntMaze v12 train maps in stored order.
- Schedule: exactly 12 complete passes, 48 optimizer updates, 16 interactive rollouts per update, maximum 16 language-policy decisions per rollout, action repeat 400.
- No resume, checkpoint selection, seed replacement, post-outcome map substitution, extra pass, or result-dependent extension.

## Policy, controller, objective, and compute match

The language model observes only the public maze, Ant position, planar velocity, goal, remaining high-level horizon, and fixed eight-action compass menu. It emits one exact capital action token at each decision. Each token targets one adjacent grid cell through the fixed v11 low-level controller and unchanged v12 cumulative-targeting rule. The controller is trusted environment machinery shared byte-for-byte across arms; its reward, route, and diagnostic state never enters a prompt.

The model never sees a certified route, planner output, directed gate, route key, reward, checker detail, development row, or evaluation outcome in its prompt. Optimizer and online objectives are exactly those qualified by the paired smoke: AdamW at `2e-7`, 16-rollout Dr.GRPO task centering, semantic Shannon coefficient 0.10, novelty beta 0.50, verified replay mass and balance coefficients 0.10, capacity 16, and 64-step warmups. Treatment applies detached exploration and replay derivatives. Control traverses the same passive banks, fixed action forwards, retained replay selection, replay scoring, and backward graph while applying exact-zero semantic, novelty, mass, and balance derivatives.

Training request seeds are deterministic functions of final seed, arm namespace, update, episode, and decision round. Fixed terminal padding preserves identical 16 × 16 policy slots and 16 × 16 replay slots per update across arms; it cannot call the simulator or enter any verified bank.

## Frozen evaluation

The four frozen v12 `multi_answer` evaluation maps are loaded only by the final paper jobs. Evaluation occurs before training and after every update, giving all 49 quarter-pass coordinates from pass 0 through pass 12.

At every coordinate, each map receives one greedy trajectory plus four deterministic temperature-one replicates of K=8 trajectories. All 132 trajectories at a coordinate are evaluated in a common feedback-free batched state machine through the same fixed v11 controller. Evaluation never updates the optimizer, semantic tracker, canonical bank, replay bank, prompt order, or training schedule.

Report greedy success, mean@8, pass@8, and the number of verifier-distinct successful route keys@8. Registered anchors are passes 0, 1, 2, 3, 4, 5, 6, 8, 10, and 12; terminal pass 12 and trapezoidal AUC over those anchors are primary.

## Fail-closed terminal audit

The independent audit requires the exact ten-cell manifest; 48 ordered updates and 49 evaluation coordinates per cell; four complete K=8 draws per map and coordinate; finite losses, log probabilities, gradients, and metrics; exact eight-token action-mask containment; exact replay of every stored public-state, compass-action, fixed-controller transition; matched per-seed policy/replay traversal; exact-zero applied control exploration/replay derivatives; nonzero raw control telemetry when eligible; and applied treatment telemetry when eligible.

It also binds the v13 checkpoint, paired-smoke qualification, source, operations, model, v12 data, simulator, protocol, manifest, scheduler, metrics, and state-replay identities. Missing or failed cells remain visible and prevent AntMaze from becoming a terminal paper row.
