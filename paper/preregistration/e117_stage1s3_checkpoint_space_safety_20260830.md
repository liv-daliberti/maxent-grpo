# E117 Stage 1-S3 checkpoint-space safety amendment

Frozen: 2026-08-30 after jobs 30980336--30980372 were placed under user hold
and canceled, and before any Stage 1 allocation, run directory, response, or
optimizer update existed.

The released S2 ledger is preserved as
`var/artifacts/e117_stage1s3_superseded_storage_unsafe_jobs.json`, SHA-256
`bc4e276f9215000ad3f16cbe88593aa311a396fb98db09186c33dfe35fa9845e`.
Its audit-job receipt is preserved beside it with SHA-256
`64882cab729180034b888418c128d8cc3ca7018a3f25209f167963a94eb3933c`.
Accounting verifies all 37 jobs canceled, zero elapsed runtime, and no assigned
node.

## Newly quantified storage constraint

The shared filesystem had 207 GiB free (98% used). Existing checkpoints from
the same runtime measure approximately 7.1 GiB for Qwen-0.5B and 26 GiB for
Falcon-1B. The S2 design allowed two retained rolling checkpoints per job and
up to the physical-node concurrency limit. Its ordinary steady-state demand
could exceed available space; atomic rotation temporarily retains the old and
new checkpoint, making the peak still larger. Launching that plan would risk
the same space-related failures already observed elsewhere in the campaign.

## Prospective repair

The replacement keeps exactly one rolling resume checkpoint. OAT's atomic
rotation may temporarily occupy two checkpoints for one job. Jobs form two
completion-serialized lanes, one for node202 and one for node203. Every job on
a lane depends `afterok` on the preceding job, so at most one Stage 1 job runs
per physical node and at most two run campaign-wide. In the measured worst case
of simultaneous Falcon atomic rotations, Stage 1 checkpoint demand is bounded
at approximately 104 GiB; steady retained demand is approximately 52 GiB.

The lane order retains each context/seed block contiguously and enforces its
registered C-P-F, P-F-C, or F-C-P order. The two registered physical nodes can
still run in parallel. A predecessor failure stops its lane and the terminal
`afterany` audit fails closed; no downstream cell silently substitutes for a
failed registered cell.

This amendment changes only retention count and scheduler concurrency. Resume
interval remains 64 updates; evaluation remains every 192 updates; successful
jobs still prune recovery state; terminal model export remains disabled. It
changes no source, training/evaluation data, update count, request seed,
treatment, endpoint, uncertainty, advancement gate, causal contrast, node
block, resource envelope per running job, or confirmation boundary.
