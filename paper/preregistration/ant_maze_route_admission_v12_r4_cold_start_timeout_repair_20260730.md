# AntMaze v12-r4 route admission: cold-start timeout repair

Status: **FROZEN AFTER TERMINAL V12-R3 AND BEFORE R4**  
Frozen: 2026-07-30

## Antecedent

The dispatcher-only r3 repair passed its four focused tests and job `30200503`
reached the exact first v12 request. It returned no execution record after 27
seconds: the outer verifier boundary was 25 seconds and the worker alarm was
20 seconds. The batch log again reported only the materializer's fail-closed
`ant_v12_admission_train_00 upper fixture failed` message, and produced neither
a data root nor an admission audit.

The exact r3 source, identity, first map, seed `107300`, controller, and route
`N E E S` pass the complete dispatcher and validator on the login host in
`5.81105083390139` seconds: canonical key `...:upper+`, goal distance
`0.44959770417218775`, and `571` simulator steps. Together with the job's
27-second wall time, this identifies cold-node import/model initialization at
the process timeout boundary, before any trusted r3 execution record.

## Sole repair

- The external verifier wait becomes 90 seconds for v12 materialization.
- The isolated worker alarm is exported as 80 seconds for this job.
- The isolated worker is launched with Python `-B -I`, so imports cannot write
  bytecode into either sealed snapshot (`-I` ignores environment-only flags).

The simulator budget remains exactly `len(tokens) * action_repeat`; route
programs still use `action_repeat=400`. The map, route slate, reset seeds,
controller, waypoint targets, success threshold, trajectory checks, and audit
are unchanged. Timeout extension cannot create simulator steps or change an
execution result; it only permits cold process initialization to finish.

Any returned route failure, or any failure after the complete admission audit,
is a scientific failure and stops this line.
