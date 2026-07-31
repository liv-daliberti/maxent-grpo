# PointMaze Stage B: clean 0.5B verified MaxEnt versus compute-matched Dr.GRPO

**Status: FROZEN BEFORE THE PAIRED-SMOKE OUTCOME AND BEFORE STAGE-B SUBMISSION — 2026-07-30**

## Conditional authorization

This protocol may launch only if `point_maze_interactive_paired_smoke_v1_audit.json` reports `status=pass` and `decision=eligible_for_ten_point_maze_stage_b_jobs` for the immutable jobs 30201821 and 30201822. A failed or incomplete paired smoke stops PointMaze; no final cell is submitted. The smoke seed and weights are never reused.

## Frozen Cartesian product

- Arms: compute-matched plain Dr.GRPO and `verified_first_global_replay_canonical`.
- Seeds: 43, 44, 45, 46, and 47.
- Shared initial checkpoint: the byte-identical Qwen2.5-0.5B-Instruct PointMaze v3 train-only warm start.
- Training pool: all eight frozen PointMaze v1 train maps in stored order, two maps from each of `bar7`, `block9`, `bar9`, and `asymmetric_block9`.
- Schedule: exactly 12 complete passes, 96 optimizer updates, 16 interactive rollouts per update, maximum 96 language-policy decisions per rollout, action repeat 5.
- No resume, checkpoint selection, seed replacement, post-outcome map substitution, extra pass, or result-dependent extension.

## Policy, objective, and compute match

The language model observes only the public maze, position, velocity, goal, and fixed nine-action compass/coast menu. It emits one exact capital action token at each decision. It never sees a certified route, planner output, directed gate, route key, reward, checker detail, development row, or evaluation outcome in its prompt.

Optimizer and online objectives are exactly those qualified by the paired smoke: AdamW at `2e-7`, 16-rollout Dr.GRPO task centering, semantic Shannon coefficient 0.10, novelty beta 0.50, verified replay mass and balance coefficients 0.10, capacity 16, and 64-step warmups. Treatment applies the detached exploration and replay derivatives. Control traverses the same passive banks, fixed action forwards, retained replay selection, replay scoring, and backward graph while applying exact-zero semantic, novelty, mass, and balance derivatives.

Training request seeds are deterministic functions of final seed, arm namespace, update, episode, and decision round. Fixed terminal padding preserves identical 16 × 96 policy slots and 16 × 96 replay slots per update across arms; it cannot call the simulator or enter any verified bank.

## Frozen evaluation

The four frozen `multi_answer` evaluation maps—one per map family—are loaded only by the final paper jobs. Evaluation occurs before training and every two updates, giving all 49 quarter-pass coordinates from pass 0 through pass 12.

At every coordinate, each map receives one greedy trajectory plus four deterministic temperature-one replicates of K=8 trajectories. All 132 trajectories at a coordinate are evaluated in a common feedback-free batched state machine. Evaluation never updates the optimizer, semantic tracker, canonical bank, replay bank, prompt order, or training schedule.

Report greedy success, mean@8, pass@8, and the number of verifier-distinct successful route keys@8. Registered anchors are passes 0, 1, 2, 3, 4, 5, 6, 8, 10, and 12; terminal pass 12 and trapezoidal AUC over those anchors are primary.

## Fail-closed terminal audit

The independent audit requires the exact ten-cell manifest; 96 ordered updates and 49 evaluation coordinates per cell; four complete K=8 draws per map and coordinate; finite losses, log probabilities, gradients, and metrics; exact action-mask containment; exact simulator transition replay; matched per-seed policy/replay traversal; exact-zero applied control exploration/replay derivatives; nonzero raw control replay telemetry when eligible; and applied treatment exploration/replay telemetry when eligible.

It also binds the qualification, v3 checkpoint, source, operations, model, data, simulator, protocol, manifest, scheduler, metrics, and state-replay identities. Missing or failed cells remain visible and prevent PointMaze from becoming a terminal paper row.
