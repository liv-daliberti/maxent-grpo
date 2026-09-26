# AntMaze v15 / controller v17-r1 admission protocol

Frozen on 2026-07-30 before the v17 controller outcome was available and
before any route was executed with that controller.

## Purpose

This gate asks whether the prospectively trained v17 controller can execute
the already frozen intermediate AntMaze v15 route slate. It is an engineering
repair of the controller only, not a new maze-selection opportunity.

## Immutable route contract

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

These values are copied exactly from
`make_ant_maze_mode_data_v15_intermediate.py`. The v15 first-route failure,
the v16 fresh-sequence failure, and the v17 training outcome may not be used
to substitute a map, route, reset seed, or split.

## Controller boundary

The executor is `ant_maze_worker_v17_r1.py`, reached through the isolated
`maze_modebench_worker_v17_r1.py` process. Route execution is allowed only if
the v17 receipt says `pass` with decision
`admitted_to_fresh_maze_route_gate_v17`, seed 73017, 4,000,000 timesteps,
96 held-out episodes, all receipt checks true, and matching model hash.

The route-generation identity binds the controller receipt, model, training
identity, worker source, targeting version, waypoint distance, and success
threshold. A failed controller receipt stops the gate before route
materialization.

## Admission decision

Admission requires all 24 official routes to replay successfully, two
distinct topology keys per map, 2,400 stable perturbation replays, the
existing execution-throughput floor, disjoint split fingerprints, and exact
source/identity hashes.

Only a passing audit with decision
`admitted_to_ant_v15_v17_frozen_model_viability_gate` permits a fresh 0.5B
language-model viability sample. No language model is sampled in this gate.
