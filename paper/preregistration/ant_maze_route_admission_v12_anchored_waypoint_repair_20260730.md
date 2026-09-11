# AntMaze route admission v12: anchored-waypoint repair

**Status: FROZEN AFTER THE TERMINAL V11 ROUTE FAILURE AND DIAGNOSTIC, BEFORE V12 IMPLEMENTATION OR EXECUTION — 2026-07-30**

## Antecedent

The v11 local controller completed job `30198291` and passed all frozen checks:
89/96 local edges, minimum heading success 9/12, minimum map success 21/24,
zero unhealthy terminations, and median successful duration 154 steps. The
separately frozen route job `30199417` then failed on the first unchanged
fixture, `ant_v11_admission_train_00` upper route `N E E S`. No data root or
route audit was materialized, the lower fixture on that map and all other 11
maps remained unexecuted, and no language model was sampled.

A labeled replay of only that already-failed fixture under the identical
v11 worker found that all four relative waypoints executed in 662 simulator
steps, but the final position `[4.2890063, 0.4781209]` was 0.5586808 from the
goal, just outside the frozen 0.5 success radius. The trajectory reached the
intended obstacle side (`y` maximum 6.4202), so the failure is accumulated
sub-cell waypoint error: every command targeted `current + 4 * heading`,
making the next nominal grid cell relative to the controller's imperfect
previous endpoint.

V11 route admission remains failed. Its identity, scheduler record, logs, and
diagnostic are antecedents, not paper results.

## Authorized executor repair

V12 keeps the exact v11 controller model and passing receipt. It changes only
waypoint target construction. At reset it records the initial physical
`xy`. For command `t`, the nominal target is

`initial_xy + 4 * cumulative_sum(heading(command_0), ..., heading(command_t))`.

Thus every token still denotes one four-unit logical grid edge, but subsequent
targets remain anchored to the original grid frame and correct rather than
compound previous endpoint error. The same deterministic v11 policy,
observation, clipped relative target input, 0.45 local-waypoint threshold,
400-step per-command limit, environment, and terminal goal check remain
unchanged. There is no planner, goal-distance reward, trajectory feedback, or
route-specific correction.

## Frozen route slate and interpretation

V12 reuses the exact 12 11x11 maps, static/peripheral walls, reset/goal cells,
reset seeds `107300` through `107311`, upper/lower programs, four-to-16 token
range, action repeat 400, route gates, perturbations, and split assignment
from v11. There is no map, seed, route, threshold, or outcome substitution.

The first upper fixture is now a regression row because its v11 outcome is
known. It cannot count as fresh evidence. The other 23 simple route fixtures
and every perturbation replay were not reached by v11 and remain the fresh
admission evidence. The unchanged audit must still pass all rows, including
the known regression, with two distinct certified route keys per map, exact
hash and split identities, zero invalid/isolation/timeout outcomes, and the
frozen perturbation robustness thresholds. A pass authorizes only the
separately frozen cross-node determinism and 0.5B viability gates. A failure
stops v12 without any route or map replacement.
