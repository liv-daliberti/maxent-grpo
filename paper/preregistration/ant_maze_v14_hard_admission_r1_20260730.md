# AntMaze v14-hard admission r1: protocol-identity isolation repair

**Status: FROZEN AFTER JOB 30204495 FAILED AND BEFORE ANY ROUTE EXECUTION OR
MODEL SAMPLING — 2026-07-30**

Job 30204495 failed in three seconds while importing the controller receipt.
The v12 controller loader interpreted the new v14 protocol-identity path as a
v12 controller identity and raised `Ant v12 route identity schema mismatch`.
No route, map, simulator episode, or language model was executed.

R1 changes only the controller-receipt lookup boundary: it temporarily removes
`OAT_ZERO_PROTOCOL_IDENTITY` while loading the already frozen v12 controller
receipt, then restores it before data generation. Map geometry, route
programs, seeds, controller weights, simulator, horizons, splits, admission
criteria, and the no-substitution rule are unchanged from v14-hard.

