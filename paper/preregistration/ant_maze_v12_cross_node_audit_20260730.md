# AntMaze v12 cross-node determinism audit

Status: **FROZEN AFTER V12 ROUTE ADMISSION AND BEFORE CROSS-NODE EXECUTION**  
Frozen: 2026-07-30

V12-r4 job `30200585` passed all 12 maps, 24 real route executions, 2,400
validator perturbation replays, and the 0.15 executions/second floor. The audit
decision is `admitted_to_v12_cross_node_route_determinism_gate`.

This gate exports those exact 12 specs and two certified programs per map. On
three distinct Slurm nodes, each program is executed three times through the
same sealed v12 worker: 72 executions per node and 216 total. Every execution
must succeed, retain its exact canonical key and directed gate, and bind the
same controller, environment, spec, source, execution, and protocol hashes.

There is no model sampling, map replacement, route replacement, seed change,
or threshold change. A pass authorizes only the frozen development-only 0.5B
AntMaze viability gate. Any failure stops v12 before model sampling.
