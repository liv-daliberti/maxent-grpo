# PointMaze-GeometryShift paired online smoke v1

Status: prospective; frozen on 2026-07-30 after the development viability
receipt passed and before any model sampling or optimization on the shifted
training maps.

This smoke is authorized by the frozen PointMaze-GeometryShift viability gate:
10 verified routes, two of four prefix-success maps, two multimode maps, 256
terminal trajectories, and no hard violation. It uses the unchanged
`point_maze_interactive_warmstart_v3` 0.5B checkpoint and the already audited
closed-loop PointMaze policy, simulator, verifier, online MaxEnt objective, and
compute-matched Dr.GRPO control.

The two arms use fresh seed 75304 and shifted training rows `0,2,4,6`, one per
family. Each arm performs exactly four ordered optimizer updates with 16
rollouts per row, a 96-decision horizon, learning rate 2e-7, and policy/replay
microbatch 16. The control computes but zeros every MaxEnt and verified replay
derivative; the treatment applies both. Fixed policy and replay traversal must
match exactly between arms. Development and evaluation rows are not loaded.

An independent simulator replay audit must require terminal artifacts from
both scheduler jobs, 64 terminal episodes per arm, at least one verified route
and one verifier-distinct multimode update per arm, exact prompt/action and
transition-hash replay, byte-identical initial checkpoints, zero escaped
control derivative, eligible treatment derivatives, finite telemetry, and
behavior/live maximum absolute log-probability error at most 1e-4.

A pass authorizes only five seeds by two arms for 12 passes on this registered
configuration-level replacement. These smoke weights are discarded. This row
must not be described as an independent semantic domain or a ConstructiveCode
result.
