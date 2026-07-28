# E69 Gate 2 compute-matched screen

Status: **in_progress**.

MATH-500 remains sealed. MATH E66/E68/E69 are one physical endpoint-only run with three reporting aliases.

| Domain | Arm | Pass | Greedy | Mean@8 | Pass@8 | Distinct@8 |
|---|---|---:|---:|---:|---:|---:|
| mathir | grpo | 0 | 0.0625 | 0.0391 | 0.2812 | 0.2969 |
| mathir | verified_first_global_replay_canonical | 0 | 0.0625 | 0.0391 | 0.2812 | 0.2969 |
| mathir | verified_entropy_gated_singleton_escape_canonical | 0 | 0.0625 | 0.0391 | 0.2812 | 0.2969 |
| mathir | verified_route_successor | 0 | 0.0625 | 0.0391 | 0.2812 | 0.2969 |

## Pre-optimizer infrastructure repair

- Excluded job 30159730 and used exact replacement job 30160101 under the prospectively recorded startup-repair identity.
- Replaced 12 never-started pending executable-domain jobs under the prospectively recorded matched-placement identity.

## Pending

- countdown/grpo/job30160185: latest step -1/2304
- countdown/grpo/job30160185: missing evaluation pass 0
- countdown/grpo/job30160185: missing evaluation pass 1
- countdown/grpo/job30160185: missing evaluation pass 2
- countdown/grpo/job30160185: missing evaluation pass 3
- countdown/grpo/job30160185: missing evaluation pass 4
- countdown/grpo/job30160185: missing evaluation pass 5
- countdown/grpo/job30160185: missing evaluation pass 6
- countdown/verified_entropy_gated_singleton_escape_canonical/job30160187: latest step -1/2304
- countdown/verified_entropy_gated_singleton_escape_canonical/job30160187: missing evaluation pass 0
- countdown/verified_entropy_gated_singleton_escape_canonical/job30160187: missing evaluation pass 1
- countdown/verified_entropy_gated_singleton_escape_canonical/job30160187: missing evaluation pass 2
- countdown/verified_entropy_gated_singleton_escape_canonical/job30160187: missing evaluation pass 3
- countdown/verified_entropy_gated_singleton_escape_canonical/job30160187: missing evaluation pass 4
- countdown/verified_entropy_gated_singleton_escape_canonical/job30160187: missing evaluation pass 5
- countdown/verified_entropy_gated_singleton_escape_canonical/job30160187: missing evaluation pass 6
- countdown/verified_first_global_replay_canonical/job30160186: latest step -1/2304
- countdown/verified_first_global_replay_canonical/job30160186: missing evaluation pass 0
- countdown/verified_first_global_replay_canonical/job30160186: missing evaluation pass 1
- countdown/verified_first_global_replay_canonical/job30160186: missing evaluation pass 2
- countdown/verified_first_global_replay_canonical/job30160186: missing evaluation pass 3
- countdown/verified_first_global_replay_canonical/job30160186: missing evaluation pass 4
- countdown/verified_first_global_replay_canonical/job30160186: missing evaluation pass 5
- countdown/verified_first_global_replay_canonical/job30160186: missing evaluation pass 6
- countdown/verified_route_successor/job30160188: latest step -1/2304
- countdown/verified_route_successor/job30160188: missing evaluation pass 0
- countdown/verified_route_successor/job30160188: missing evaluation pass 1
- countdown/verified_route_successor/job30160188: missing evaluation pass 2
- countdown/verified_route_successor/job30160188: missing evaluation pass 3
- countdown/verified_route_successor/job30160188: missing evaluation pass 4
- countdown/verified_route_successor/job30160188: missing evaluation pass 5
- countdown/verified_route_successor/job30160188: missing evaluation pass 6
- graph_coloring/grpo/job30160181: run directory absent
- graph_coloring/verified_entropy_gated_singleton_escape_canonical/job30160183: run directory absent
- graph_coloring/verified_first_global_replay_canonical/job30160182: run directory absent
- graph_coloring/verified_route_successor/job30160184: run directory absent
- math_dev/grpo/job30159729: latest step 75/2304
- math_dev/grpo/job30159729: missing evaluation pass 0
- math_dev/grpo/job30159729: missing evaluation pass 1
- math_dev/grpo/job30159729: missing evaluation pass 2
- math_dev/grpo/job30159729: missing evaluation pass 3
- math_dev/grpo/job30159729: missing evaluation pass 4
- math_dev/grpo/job30159729: missing evaluation pass 5
- math_dev/grpo/job30159729: missing evaluation pass 6
- math_dev/verified_first_global_replay_canonical/job30160101: latest step 17/2304
- math_dev/verified_first_global_replay_canonical/job30160101: missing evaluation pass 0
- math_dev/verified_first_global_replay_canonical/job30160101: missing evaluation pass 1
- math_dev/verified_first_global_replay_canonical/job30160101: missing evaluation pass 2
- math_dev/verified_first_global_replay_canonical/job30160101: missing evaluation pass 3
- math_dev/verified_first_global_replay_canonical/job30160101: missing evaluation pass 4
- math_dev/verified_first_global_replay_canonical/job30160101: missing evaluation pass 5
- math_dev/verified_first_global_replay_canonical/job30160101: missing evaluation pass 6
- mathir/grpo/job30159725: latest step 273/2304
- mathir/grpo/job30159725: missing evaluation pass 1
- mathir/grpo/job30159725: missing evaluation pass 2
- mathir/grpo/job30159725: missing evaluation pass 3
- mathir/grpo/job30159725: missing evaluation pass 4
- mathir/grpo/job30159725: missing evaluation pass 5
- mathir/grpo/job30159725: missing evaluation pass 6
- mathir/verified_entropy_gated_singleton_escape_canonical/job30159727: latest step 270/2304
- mathir/verified_entropy_gated_singleton_escape_canonical/job30159727: missing evaluation pass 1
- mathir/verified_entropy_gated_singleton_escape_canonical/job30159727: missing evaluation pass 2
- mathir/verified_entropy_gated_singleton_escape_canonical/job30159727: missing evaluation pass 3
- mathir/verified_entropy_gated_singleton_escape_canonical/job30159727: missing evaluation pass 4
- mathir/verified_entropy_gated_singleton_escape_canonical/job30159727: missing evaluation pass 5
- mathir/verified_entropy_gated_singleton_escape_canonical/job30159727: missing evaluation pass 6
- mathir/verified_first_global_replay_canonical/job30159726: latest step 266/2304
- mathir/verified_first_global_replay_canonical/job30159726: missing evaluation pass 1
- mathir/verified_first_global_replay_canonical/job30159726: missing evaluation pass 2
- mathir/verified_first_global_replay_canonical/job30159726: missing evaluation pass 3
- mathir/verified_first_global_replay_canonical/job30159726: missing evaluation pass 4
- mathir/verified_first_global_replay_canonical/job30159726: missing evaluation pass 5
- mathir/verified_first_global_replay_canonical/job30159726: missing evaluation pass 6
- mathir/verified_route_successor/job30159728: latest step 273/2304
- mathir/verified_route_successor/job30159728: missing evaluation pass 1
- mathir/verified_route_successor/job30159728: missing evaluation pass 2
- mathir/verified_route_successor/job30159728: missing evaluation pass 3
- mathir/verified_route_successor/job30159728: missing evaluation pass 4
- mathir/verified_route_successor/job30159728: missing evaluation pass 5
- mathir/verified_route_successor/job30159728: missing evaluation pass 6
- python_factor/grpo/job30160202: latest step -1/2304
- python_factor/grpo/job30160202: missing evaluation pass 0
- python_factor/grpo/job30160202: missing evaluation pass 1
- python_factor/grpo/job30160202: missing evaluation pass 2
- python_factor/grpo/job30160202: missing evaluation pass 3
- python_factor/grpo/job30160202: missing evaluation pass 4
- python_factor/grpo/job30160202: missing evaluation pass 5
- python_factor/grpo/job30160202: missing evaluation pass 6
- python_factor/verified_entropy_gated_singleton_escape_canonical/job30160204: latest step -1/2304
- python_factor/verified_entropy_gated_singleton_escape_canonical/job30160204: missing evaluation pass 0
- python_factor/verified_entropy_gated_singleton_escape_canonical/job30160204: missing evaluation pass 1
- python_factor/verified_entropy_gated_singleton_escape_canonical/job30160204: missing evaluation pass 2
- python_factor/verified_entropy_gated_singleton_escape_canonical/job30160204: missing evaluation pass 3
- python_factor/verified_entropy_gated_singleton_escape_canonical/job30160204: missing evaluation pass 4
- python_factor/verified_entropy_gated_singleton_escape_canonical/job30160204: missing evaluation pass 5
- python_factor/verified_entropy_gated_singleton_escape_canonical/job30160204: missing evaluation pass 6
- python_factor/verified_first_global_replay_canonical/job30160203: latest step -1/2304
- python_factor/verified_first_global_replay_canonical/job30160203: missing evaluation pass 0
- python_factor/verified_first_global_replay_canonical/job30160203: missing evaluation pass 1
- python_factor/verified_first_global_replay_canonical/job30160203: missing evaluation pass 2
- python_factor/verified_first_global_replay_canonical/job30160203: missing evaluation pass 3
- python_factor/verified_first_global_replay_canonical/job30160203: missing evaluation pass 4
- python_factor/verified_first_global_replay_canonical/job30160203: missing evaluation pass 5
- python_factor/verified_first_global_replay_canonical/job30160203: missing evaluation pass 6
- python_factor/verified_route_successor/job30160205: run directory absent
