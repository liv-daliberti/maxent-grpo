# E75F1: PointMaze waypoint 0.5B paper-scale cohort

**Frozen:** 2026-08-04, after the terminal E75R3 pilot and before E75F1 data
materialization, development evaluation, or online training.

## Purpose and prior evidence

E75R3 established that the `point-waypoint-v1` interface is executable by
Qwen2.5-0.5B, that its route verifier distinguishes three to five certified
routes per map, and that the compute-matched replay path applies nonzero
derivatives while the control applies exact zero. Its one-pass, one-seed
untouched evaluation did not show a diversity gain: current minus Dr.GRPO was
-0.03125 distinct routes@8 and delayed minus Dr.GRPO was 0. E75R3 therefore
qualifies plumbing and initialization only; it is not a paper endpoint.

E75F1 asks the paper estimand: whether the canonical x-Mode treatment differs
from matched Dr.GRPO after twelve passes over a 384-prompt pool in five paired
Qwen2.5-0.5B seeds. Outcomes are reported regardless of sign. There is no
efficacy gate, arm selection, delayed-replay arm, or result-dependent extension.

## Frozen model and domain

The shared initial checkpoint is the byte-identical E75R3 train-only waypoint
warm start at `var/models/point_maze_waypoint_warmstart_e75r3`, derived from
`Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775`.
Its receipt and tree hash are bound by the E75F1 launch identity. E75F1 does
not tune, continue, or replace this checkpoint.

The language policy sees the complete public maze, current and previous cells,
continuous position and velocity, goal, remaining horizon, and adjacent free
cells. It emits one exact capital label selecting one adjacent free cell. The
legal mask removes walls only; it never uses a certified route, goal distance,
route identity, shortest-path calculation, or revisit history. A deterministic
hash-bound PD controller executes each waypoint in MuJoCo. Success and the
canonical route key are derived from the same continuous trajectory; a valid
route must reach the goal while crossing exactly one directed corridor gate.

## Fresh paper-shaped data firewall

Data seed `88104` materializes exactly:

- 384 train maps;
- 64 development maps; and
- 128 evaluation maps.

Each split is rotation-balanced and contains three-, four-, and five-corridor
families. Every map has three to five independently executed successful route
programs with different canonical gate keys. All 576 geometry fingerprints
must be mutually disjoint and must also be disjoint from every map in E75,
E75R1, E75R2, and E75R3 (640 predecessor fingerprints). Certified programs
are stored for audit but never enter model context or online replay.

The shared warm start is evaluated once on all 64 new development maps with
eight trajectories per map. The ten paper jobs launch only if:

- at least 48/64 maps have pass@8;
- at least 32/64 maps expose two or more verified routes@8;
- every corridor family has a successful map;
- aggregate mean@8 is in [0.10, 0.70]; and
- the evaluation is optimizer-free and all input/output hashes agree.

This gate is a scale-transfer check for the already frozen initialization. A
failure stops E75F1. It does not authorize new SFT, a threshold change, map
replacement, or another E75F1 attempt.

## Frozen Cartesian product

- Model family: Qwen2.5-0.5B-Instruct via the frozen E75R3 checkpoint.
- Arms: `grpo` and `verified_first_global_replay_canonical`.
- Paired seeds: 43, 44, 45, 46, and 47.
- Training: 12 ordered passes over all 384 train maps, exactly 4,608 optimizer
  updates per cell.
- Per update: 16 live trajectories, fixed 64-decision horizon, replay capacity
  16, learning rate 2e-7, policy/replay microbatch 16, context cap 1536.
- Evaluation: all 128 evaluation maps at update zero and every 96 updates,
  yielding 49 coordinates (four per pass plus initialization), with K=8 and
  arm-paired deterministic sampling schedules.

Both arms execute identical live-policy and replay-decision forward slots.
Both compute semantic/canonical discovery and replay diagnostics. Dr.GRPO
applies exact-zero exploration and replay derivatives. Treatment applies the
canonical verified discovery, verified-mass, and balance derivatives already
qualified in E75R3. Only online-discovered, verifier-accepted route keys enter
the banks. Evaluation never changes the optimizer, banks, prompt order, or
training schedule.

The registered reporting anchors are passes 0, 1, 2, 3, 4, 5, 6, 8, 10, and
12; the paper's main Figure 4 view uses passes 0--4 and the endpoint table uses
pass 12. Primary metrics are greedy success when available, mean@8, pass@8,
and distinct verified routes@8. Modes per successful sample is descriptive.

## Fail-closed audit and paper admission

The dependent terminal audit requires all ten cells, exact arm/seed membership,
4,608 ordered updates and 49 fixed evaluation coordinates per cell, all 384
train rows repeated exactly 12 times in order, all 128 evaluation maps at every
coordinate, finite values, zero support escapes, exact-zero applied control
derivatives, nonzero raw control telemetry when eligible, applied treatment
telemetry when eligible, immutable data/model/source/operations hashes, and
valid receipt, metrics, replay, and terminal model hashes.

No efficacy threshold is part of validity. If the audit passes, E75F1 is a
valid full PointMaze domain result and is added to the paper as a sixth domain
with its observed direction and uncertainty. If it fails, PointMaze remains
developmental and is not silently inserted into `main.pdf`. Failed or missing
jobs remain visible; recovery must preserve the frozen identity and is governed
by a separately recorded operations-only amendment.
