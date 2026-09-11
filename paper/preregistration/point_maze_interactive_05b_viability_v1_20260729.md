# PointMaze closed-loop 0.5B viability v1

Frozen before model sampling on 2026-07-29.

## Purpose

Test whether Qwen2.5-0.5B-Instruct has nonzero PointMaze capability when it
controls the admitted simulator step by step through the public nine-action
alphabet. This is a prospective development gate, not a confirmatory MaxEnt
comparison and not a paper result.

The failed one-shot PointMaze v1 and prompt-repaired v2 probes remain failed.
This protocol does not replace or reinterpret them.

## Frozen inputs

- Model: immutable local Qwen2.5-0.5B-Instruct snapshot
  `7ae557604adf67be50417f59c2c2f167def9a775`.
- Data: `var/data/point_maze_modebench_v1/dev`, split `multi_answer`.
- Prompt count: 4.
- Sampling: 64 rollouts per prompt, temperature 1.0, top-p 1.0.
- Prefix decision: first 16 rollouts per prompt.
- Seed: 75101.
- Evaluation prompts and certified route programs are not loaded.

## Interface

Each rollout owns one persistent episode in the pinned networkless PointMaze
runtime. At every decision:

1. the model receives the original printed map, current continuous position,
   goal, remaining action horizon, and recent actions;
2. it emits one single-token label constrained to the nine frozen actions; and
3. the worker applies that action for the frozen five simulator steps.

There is no planner, demonstration, route catalogue, collision oracle, or
intermediate endpoint reward. Success and semantic identity are computed by the
existing executable verifier from the full terminal trajectory. The maximum
horizon remains 96 actions.

## Frozen decision

Pass only if both hold:

- at least 2 of 4 prompts have one verified route in the first 16 rollouts; and
- at least 1 of 4 prompts has two distinct verified route modes across all 64
  rollouts.

On pass, freeze the receipt and design a shared warm-start/matched-training
smoke. On fail, proceed prospectively to train-only planner imitation; do not
repair this probe, change the four development maps, or relax its thresholds.
