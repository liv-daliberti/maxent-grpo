# AntMaze v12-r3 route admission: dispatcher-order repair

Status: **FROZEN AFTER THE TERMINAL V12-R2 PRE-EXECUTION FAILURE AND BEFORE R3**  
Frozen: 2026-07-30

## Antecedent

V12-r2 job `30200147` used the unchanged v11 map/route slate, admitted v11
controller, and anchored v12 executor. It exited after 13 seconds on the first
fixture with no data root and no admission audit. Exact replay through its
sealed dispatcher reproduced the cause before the v12 simulator was entered:
the dispatcher evaluated `ant_v9_receipt_sha256()` before the v12 identity and
the older validator raised `RuntimeError: Ant v9 controller is not admitted`.

Direct invocation of the same sealed v12 executor on the same first map, seed,
route `N E E S`, controller, and threshold passed the complete validator:

- canonical key `ant_maze_action_program:maze-action-ant-v5:ant_v12_admission_train_00:upper+`;
- final goal distance `0.44959770417218775` versus the frozen `0.5` threshold;
- `571` simulator steps; and
- directed route `upper+`.

Thus r2 did not expose a controller or route outcome. It exposed a dispatcher
identity-order defect.

## Sole repair

R3 moves the v12 controller-identity branch ahead of the older v9, v10, and
v11 receipt validators. The v12 worker itself is byte-unchanged. The regression
test requires the order `v12 < v9 < v10 < v11`.

There is no map, seed, route, controller, waypoint, threshold, horizon,
action-space, fixture, or outcome substitution. R3 reuses the exact 12-map
slate and runs the same admission and perturbation audit. A route or audit
failure after the corrected dispatcher is a scientific failure and stops this
line.
