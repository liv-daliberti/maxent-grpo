# AntMaze v12 cross-node audit r1: executor-identity binding repair

**Status: FROZEN AFTER JOB 30200738 FAILED BEFORE SIMULATION AND BEFORE THE R1 JOB — 2026-07-30**

Job 30200738 launched the frozen three-node gate but every replica stopped in
the pre-execution controller check. The v12 exporter inherited the v10 field
name and exported the SHA-256 of the controller evaluation receipt, while each
materialized v12 spec and the admitted route identity bind the canonical
executor identity (worker source, receipt, model, training identity, targeting
rule, waypoint distance, and success threshold). No MuJoCo environment was
created and zero route replays occurred.

R1 changes only that export binding: v12 reads
`executor_identity_sha256`; v10 continues to read
`controller_receipt_sha256`. R1 also invokes the replica shell without a login
profile. The exact admitted 12 rows, two routes per row, controller, source
snapshot, reset seeds, three distinct allocated nodes, three repetitions per
route per node, and 216 total real-simulator executions are unchanged. There
is no language-model sample, map substitution, route substitution, threshold
change, or scientific change.

All 216 executions must validate, reproduce their exact canonical topology
keys and directed gates, bind the same source/execution/protocol/spec hashes,
and come from three distinct hostnames. A pass alone authorizes the frozen
Qwen2.5-0.5B AntMaze viability gate. Failure stops AntMaze unless it is another
demonstrated pre-execution infrastructure defect addressed by a separately
frozen repair.
