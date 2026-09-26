# E78-PM: prospective PointMaze verified-replay-only extension

**Frozen before data materialization or job submission on 2026-08-04.**

## Status relative to E78

This is a separately identified sixth-domain extension, not a retroactive
change to E78's frozen 50-cell estimator.  Its ten cells may be displayed
beside E78 because they ask the same control-versus-verified-replay question,
but reporting must identify PointMaze as a prospective post-E78 extension.
The completed E75--E75R3 development ladder fixed the interface and the shared
warm start; none of its online estimates enters this comparison.

## Environment and policy boundary

The environment is the pinned MuJoCo `PointMaze_UMaze-v3` runtime used by the
audited `point-waypoint-v1` interface.  At each decision the 0.5B language
policy sees the maze, current cell, previous cell, goal cell, remaining
waypoints, and adjacent free moves.  It selects one adjacent prompt-visible
free cell.  A hash-bound deterministic PD adapter applies continuous force to
reach that cell.  The legal mask removes walls only: it does not use goal
distance, shortest paths, route identity, revisit history, or verifier
feedback.

The simulator supplies terminal success.  A successful trajectory's canonical
mode is the single directed barrier corridor it crossed.  Invalid, timed-out,
or unsuccessful trajectories have no mode key.  The language model does not
emit MuJoCo forces, and PointMaze must be labeled accordingly in the paper.

## Frozen fresh data

- Generator: the audited deterministic E75R3 PointMaze generator.
- Data seed: `88104`.
- Splits: 384 train, 64 development, and 128 evaluation maps, row-matched to
  the five static-domain comparisons on the scientific train/evaluation rows.
- Each map has three, four, or five separated barrier corridors.
- Rotations are balanced within each split.
- Every fingerprint must be disjoint from E75, E75R1, E75R2, and E75R3.
- Every advertised corridor route must execute in the pinned MuJoCo runtime.
- Certified route programs never enter model prompts, training, or replay.

There is no outcome-dependent viability gate.  All ten cells run and all
outcomes are reported, including zero-success or saturated outcomes.

## Shared initialization and paired design

All cells start from the already frozen E75R3 72-update weak warm start,
`var/models/point_maze_waypoint_warmstart_e75r3`.  That checkpoint was trained
only on the disjoint E75R3 training split and was fixed before E78-PM's seed,
maps, or outcomes existed.

- Arms: compute-matched Dr.GRPO (`control`) and verified replay (`replay`).
- Seeds: 43, 44, 45, 46, and 47, paired across arms.
- Training maps: 384.
- Training: exactly eight ordered passes, or 3,072 optimizer updates.
- Rollouts per update: 16.
- Fixed maximum interactive horizon: 64 decisions.
- Optimizer: AdamW, one update per rollout group, learning rate `2e-7`.
- Evaluation: all 128 fixed evaluation maps at passes 0, 0.5, 1.0, ..., 8.0.
- A rolling resumable model/optimizer/bank checkpoint is written every 192
  updates, exactly every half pass.

The two arms use common map order and sampling seeds.  Each on-policy group and
each evaluation coordinate executes the same simulator request schedule across
arms.  The replay bank, scheduler, replay materialization, teacher-forced score
traversal, and fixed compute envelope are also shared.

## Only scientific difference

For the one globally round-robin prompt bank scheduled at an update, retain one
deterministically selected validator-positive episode per observed route, up
to 16 routes.  Let

    score_theta(b | x)

be the mean current selected-action log probability along stored interactive
episode `b`, evaluated at the exact public states and legal supports originally
observed.  The replay arm minimizes

    0.10 * mean_{b in B_x} -score_theta(b | x),

with E78's `(N-1)/N` Dr.GRPO scale and `1/N` per-rollout normalization for
`N=16`.  Singleton banks are eligible.  The control performs the identical
score traversal with an exact-zero applied replay derivative.

Both arms hard-disable semantic MaxEnt, semantic or canonical novelty
advantages, known-bank balance KL, adaptive coefficients, counterfactual
proposals, singleton escape, token entropy, reference KL, goal-directed action
masking, and planner feedback.  Thus verified replay is the only auxiliary
derivative.

## Outcomes

At every registered half pass report, separately by seed and arm:

- sampled mean correctness@8;
- sampled pass@8;
- mean distinct verified routes@8; and
- excess route multiplicity, `distinct@8 - pass@8`.

PointMaze has no meaningful deterministic greedy action-program endpoint under
the interactive common-random-number protocol, so it does not fabricate a
greedy pass@1 column.  The primary PointMaze comparisons are paired seed
differences `replay - control` at pass 8 for distinct routes@8 and pass@8.  The
secondary trajectory summary is trapezoidal AUC over the complete half-pass
grid.  Do not select a best checkpoint or pool PointMaze with the five static
response domains without an explicit interface stratum.

## Failure and disclosure policy

Missing simulator identities, map overlap, route-certification failure,
source drift, action-support escape, non-finite loss, nonzero semantic or
balance telemetry, nonzero control replay gradient, missing half-pass
evaluation, or missing terminal output fails closed.  Infrastructure
interruption may resume only from the same cell's rolling hash-bound model,
optimizer, replay bank, artifact offsets, and update number.  No failed
scientific cell, map, or seed is silently replaced or excluded.
