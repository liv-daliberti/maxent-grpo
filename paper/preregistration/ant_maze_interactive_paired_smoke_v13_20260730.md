# AntMaze v13 constrained-policy paired online-training smoke

**Status: FROZEN BEFORE THE V13 VIABILITY OUTCOME AND BEFORE ONLINE SAMPLING — 2026-07-30**

## Conditional authorization

This development-only pair launches only if job 30202183 completes and `ant_maze_interactive_05b_viability_v13.json` reports `status=pass` with decision `eligible_for_ant_v13_paired_online_smoke`, prefix success on at least two of four unchanged development maps, and multiple verified route keys on at least one map. Failure stops AntMaze before online training. These two jobs are not paper seeds.

## Frozen pair

- arms: compute-matched plain Dr.GRPO and `verified_first_global_replay_canonical`;
- shared initial model: the exact completed v13 restricted-action SFT checkpoint, loaded independently by both arms;
- common seed: 76313, with deterministic arm-separated training request namespaces;
- training rows: all four v12 train maps in stored order, one pass, 16 rollouts per map, four optimizer updates per arm;
- public interface: map, Ant position, planar velocity, goal, and remaining high-level horizon; one of eight exact compass tokens per decision;
- horizon: at most 16 high-level language decisions, each targeting one adjacent grid cell through the fixed v11 low-level controller and unchanged v12 cumulative-targeting rule; and
- optimizer: AdamW, learning rate 2e-7, betas `(0.9,0.999)`, epsilon 1e-8, weight decay 0, clip epsilon 0.2, gradient norm cap 1.

Every update executes a fixed 16 × 16 policy-forward budget and a fixed 16 × 16 replay-decision budget. Real terminal episodes use public zero-masked padding for the remaining slots. Padding never calls the environment, enters a verified bank, or changes reward.

## Frozen objectives

The treatment uses success-conditioned signed semantic Shannon coefficient 0.10, novelty beta 0.50, verified replay mass and balance coefficients 0.10, capacity 16, one persistent-hash round-robin replay prompt per update, and 64-observation warmups. Detached semantic/novelty advantages enter only outside task-reward centering. Only terminal verifier-positive route keys and their retained public state/action episodes enter the banks.

Control performs identical worker calls, canonicalization, passive bank updates, retained-mode selection, replay scoring, and backward traversal while applying exact-zero semantic, novelty, mass, and balance derivatives. The frozen controller is identical across arms and receives no reward or route feedback.

## Information boundary and pass criterion

No certified route, planner, target sequence, canonical key, reward, directed gate, controller diagnostic, development row, evaluation row, or future state appears in a nonterminal prompt. The low-level controller remains trusted environment machinery, not a learned comparison arm.

The independent audit requires both jobs to complete all four updates from one byte-identical checkpoint; at least one verified rollout and one verifier-distinct multimode training map per arm; fixed action support; complete simulator transition replay; finite losses and gradients; exact-zero applied control exploration/replay derivatives with raw compute telemetry; applied treatment exploration/replay derivatives whenever eligible; and exact policy/replay traversal equality. A pass authorizes a separately frozen ten-cell AntMaze Stage-B protocol. No threshold, map, seed, checkpoint, or interface change is allowed after this outcome.
