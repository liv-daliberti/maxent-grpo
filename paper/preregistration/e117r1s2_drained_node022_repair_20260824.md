# E117-R1-S2 drained-node022 placement repair

Frozen: 2026-08-24 while all 12 E117-R1 jobs are pending at zero runtime and
before any run directory or endpoint exists. PointMaze remains excluded.

After E117-R1 reached a satisfiable `lowprio` partition, node022 entered
`MIXED+COMPLETING+DRAIN`. A drained node cannot accept the six registered Qwen
Python and Falcon MathIR C/P/F jobs. Waiting does not resolve this constraint.
Node021 remains healthy, so Countdown and Graph are unchanged.

Contemporaneous read-only inspection found compatible capacity on lowprio:

- node101: A40, used for all three Qwen Python arms;
- node203: A5000, used for all three Falcon MathIR arms.

Both are Ampere-class and support the frozen bfloat16/vLLM execution path.
Every causal C/P/F block remains on one exact node and GPU type. Hardware is a
block, not an efficacy selection variable.

Transactionally user-hold jobs 30873701--30873706, verify zero runtime, zero
restarts, absent run directories, byte-exact scientific exports, lowprio
partition, mltheory account, and generic one-GPU/8-CPU/64-GiB/eight-hour
resources. Change only the required node list:

- 30873701--30873703: node022 to node101;
- 30873704--30873706: node022 to node203.

Audit all six while held, record before/held/after rows and environment hashes,
then release them together. On failure restore node022 and release the user
holds. The training snapshot, run paths, treatments, seed, data, optimizer,
request streams, and audit dependency remain unchanged. No outcome is read.
