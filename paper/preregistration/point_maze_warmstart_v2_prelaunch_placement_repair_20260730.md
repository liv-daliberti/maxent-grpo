# PointMaze v2 prelaunch placement-only repair

**Status: RECORDED BEFORE ANY V2 MODEL UPDATE OR DEVELOPMENT SAMPLE — 2026-07-30**

The first held submission, job `30197923`, was canceled by the launcher's
fail-closed audit at runtime `00:00:00`. Although the frozen Slurm script
declared partition `all`, the submission environment reported partition
`mltheory`. The job remained user-held, received no node or GPU allocation,
and produced no stdout, model artifact, SFT receipt, or development receipt.

The retry adds explicit command-line placement arguments
`--partition=all --account=mltheory`. It preserves the exact v2 protocol,
source and execution snapshots, compact train-only examples, base model,
seeds, 184 optimizer steps, resources, sampling slate, and scientific gates.
The canceled identity is retained as
`point_maze_interactive_warmstart_v2_prelaunch_canceled_30197923.json`.

Slurm also reported the explicit-flag retry, job `30197998`, in `mltheory`.
The fail-closed launcher again canceled it at runtime `00:00:00`, before any
allocation or scientific execution. Its identity is retained under the same
job-specific archival convention. The final placement procedure therefore
submits the job held, verifies all scientific resources, applies
`scontrol update Partition=all` while still held, verifies the resulting
partition, and only then disables requeue and releases it. This is the same
placement-only mechanism already used successfully for PointMaze v1.
