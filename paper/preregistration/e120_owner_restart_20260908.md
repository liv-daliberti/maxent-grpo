# E120 owner capacity restart, September 8, 2026

The user explicitly requested repairing unfinished E118/E119/E120 cells, getting
them running, using node302 and node105, and finishing E120. This supersedes the
September 4 E118 400k priority holds and supplies the previously missing E120
replacement authorization recorded in the September 6 automatic approval rejection.
The ten Qwen-3B holds were resource priority decisions, not scientific gates.

Keep all ten Qwen-3B scheduler IDs and the immutable 45-cell primary ledger.
Validate six full model/optimizer checkpoints, and restart the remaining four
cells from initialization because no durable checkpoint exists. Pantry seeds
71–73 lost their pre-checkpoint work in the already recorded September 4 pause;
seed 74 never started. Preserve recipe, data, seed, eight-pass horizon, run path,
frozen repaired runtime, automatic resume, 128 GiB, 16 CPUs and one A100 per job.
Add the existing explicit excluded node list and release two afterany chains on
node302/mltheory, ordered by durable progress. At most two Qwen allocations run
simultaneously, leaving a third 128-GiB owner slot for E118/E119 recovery.

Move the two never-started Falcon Pantry seeds 55–56 from the priority-blocked
cs A6000 pool to node105/mltheory A5000. Same-treatment seed 57 completed on this
hardware. Preserve scientific/runtime exports byte for byte, wrapper, output
identity, 64 GiB, 8 CPUs, 72-hour walltime, Nice100 and automatic resume. Submit
held replacements, audit, register the two continuations, retire only the old
pending allocations, then release. Both may run concurrently. Preserve all
other jobs. Check the repaired runtime fingerprints before applying and verify
actual current-attempt optimizer progress after release.

All before/after scheduler state, checkpoint counters, ledger hashes and action
receipts are recorded in var/artifacts/e120_owner_restart_20260908/.
