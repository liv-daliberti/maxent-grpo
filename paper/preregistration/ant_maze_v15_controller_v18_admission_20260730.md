# AntMaze v15 / stable-handoff controller v18 admission protocol

**Status:** FROZEN WHILE CONTROLLER JOB 30205570 WAS RUNNING  
**Date:** 2026-07-30  
**Role:** Secondary post-outcome engineering repair; excluded from the original
80-cell estimator.

## Purpose and outcome firewall

This gate asks whether the prospectively trained stable-handoff v18 controller
can execute the already frozen intermediate AntMaze v15 route slate. The
controller outcome, v15 execution outcome, and any language-model outcome were
unavailable when this binding was frozen. Failure of the controller gate stops
before route materialization; no map, route, seed, or split may then be
substituted.

## Immutable v15 route contract

- Environment: `AntMaze_UMaze-v5`.
- Map size: 13 by 13.
- Central obstacle: rows 5 through 7 and columns 5 through 7.
- Reset cell: `(6, 4)`.
- Goal cell: `(6, 8)`.
- Upper route: `N N E E E E S S`.
- Lower route: `S S E E E E N N`.
- Twelve maps, four per train/dev/eval split.
- Reset seed base: 108500.
- Action repeat: 400.
- Minimum/maximum commands: 8/20.
- The v15 peripheral-cell sequence and split assignment are unchanged.

These are exact copies of the v15 contract used for controller v17. The sealed
v15 first-route failure and v17 controller failure do not create a new
maze-selection opportunity.

## Stable-handoff controller boundary

The executor is `ant_maze_worker_v18.py`, reached only through the isolated
`maze_modebench_worker_v18.py` process. Route execution is allowed only if the
v18 receipt says `pass` with decision
`admitted_to_fresh_maze_route_gate_v18`, seed 73018, 6,000,000 transitions,
eight workers, 96 held-out sequences, all checks true, and matching model and
training hashes.

Every intermediate and final waypoint requires both distance at most 0.45 and
planar speed at most 1.0 before it is accepted. A position-only goal hit is not
a verified route. The route-generation identity binds the receipt, model,
training identity, worker source, targeting version, waypoint distance,
position threshold, and speed threshold.

## Admission decision

Admission requires all 24 official routes to replay successfully, two distinct
topology keys per map, 2,400 stable perturbation replays, the existing
execution-throughput floor, disjoint split fingerprints, and exact
source/identity hashes.

Only a passing audit with decision
`admitted_to_ant_v15_v18_frozen_model_viability_gate` permits a fresh 0.5B
language-model viability sample. No language model is sampled in this gate.
