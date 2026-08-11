# AntMaze v15 / continuing-task controller-v19 admission

**Status:** FROZEN WHILE CONTROLLER JOB 30259111 WAS RUNNING  
**Date:** 2026-08-04  
**Role:** Secondary post-outcome engineering repair; excluded from the original
80-cell estimator and the completed AntMaze Stage-B estimator.

## Purpose and outcome firewall

This gate asks whether prospectively trained controller v19 can execute the
already frozen intermediate AntMaze v15 route slate. The v19 controller
outcome, v15/v19 route outcome, and any downstream language-model outcome are
unavailable at freeze time. An exact failed or missing v19 gate stops before
route materialization. No map, route, seed, split, controller checkpoint, or
threshold may then be substituted.

## Immutable v15 route contract

- Environment: `AntMaze_UMaze-v5`.
- Map size: 13 by 13.
- Central obstacle: rows 5 through 7 and columns 5 through 7.
- Reset cell: `(6, 4)`; goal cell: `(6, 8)`.
- Upper route: `N N E E E E S S`.
- Lower route: `S S E E E E N N`.
- Twelve maps, four per train/dev/eval split.
- Reset seed base: 108500.
- Action repeat: 400.
- Minimum/maximum commands: 8/20.
- The v15 peripheral-cell sequence and split assignment are unchanged.

These are exact copies of the v15 contract frozen for controller v17 and later
bound to failed controller v18. No v19 route is sampled before this protocol.

## V19 controller boundary

The executor is `ant_maze_worker_v19.py`, reached only through isolated
`maze_modebench_worker_v19.py`. Route execution is permitted only if the v19
receipt says `pass` with decision
`admitted_to_fresh_maze_route_gate_v19`, seed 73019, 6,000,000 transitions,
eight workers, 96 fresh gate sequences, every check true, zero maze-task
termination events, and matching model, training, and identity hashes.

Every waypoint requires distance at most 0.45 and planar speed at most 1.0.
The route-generation identity binds the receipt, model, v19 training identity,
worker source, targeting version, waypoint distance, position threshold,
speed threshold, continuing-task controller gate, and explicit Ant-health
gate.

## Admission decision

Admission requires all 24 official routes to replay successfully, two distinct
topology keys per map, 2,400 stable perturbation replays, the existing
execution-throughput floor, disjoint split fingerprints, and exact
source/identity hashes.

Only a passing audit with decision
`admitted_to_ant_v15_v19_frozen_model_viability_gate` may authorize a
separately frozen, fresh 0.5B development viability sample. This admission
gate samples no language model and authorizes no paired or five-seed cohort.
