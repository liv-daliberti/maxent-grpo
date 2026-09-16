# E118 Countdown replay seed-74 capacity placement, 2026-09-08

The user authorized additional existing E118/E119/E120 jobs on available
capacity. The reviewed target is the existing Qwen-3B Countdown ReplayMaxRL
seed-74 cell, current pending job 31048126. Its run directory, run stamp,
treatment, seed, data, training target, evaluation, and checkpoint settings
remain unchanged. No new scientific cell is created and no evaluation
outcome is used to choose the target.

Place this one cell on fixed node203 using allcs/lowprio, one A5000, 16 CPUs,
116 GiB host memory, and a 72-hour walltime, with requeue enabled and nice
200. Check current scheduler memory, CPU, and typed GPU capacity immediately
before preparing and applying the transaction. The 2026-09-08 review chose
node203 after node202's available memory fell below 116 GiB. If node203 no
longer has capacity, stop for placement reconciliation; do not silently
change the selected node in an existing audit.

Apply the already validated A5000 runtime allocation amendment:
`OAT_ZERO_VLLM_GPU_RATIO` changes from 0.25 to 0.40. This is the only non-root
runtime export change. The frozen launcher and all scientific exports,
evaluation settings, save/resume settings, run directory, and run stamp
remain identical. Keep explicit repository-root exports and the existing
PVL exclusion `node[004-008,020-026,101,103-104,403,805-808,901-902,906-909,911-914]`.

Read-only target review found job 31048126 pending, with no incoming or
outgoing dependencies, no other active writer, no training metrics, no
checkpoint, and no completion receipt. The shared controller must recheck
these conditions at admission. The two MathIR seed-74 cells, jobs 31073903
and 31073904, already completed successfully and are excluded from this
placement. Their completion receipts report step 3073.

Support for 116 GiB and the 0.40 A5000 ratio includes the current node202
Countdown MaxRL seed-74 job and MathIR seed-71 pair: all have verified fresh
optimizer progress, positive sleep/wake timings, no fatal errors, and zero
memory pressure/OOM events at approximately 84 GiB noncache usage. Evidence
is recorded in `var/artifacts/e118_owner_backfill_20260908/node202_first_optimizer_verified.json`.
The shared controller's fresh live node105/node302 admission gates remain
mandatory.

Use `ops/exp_scaling/backfill_e118_countdown_replay_s74_20260908.py`, which
defaults to read-only preflight. Its `--apply` transaction imports the shared
controller and preserves its held replacement audit, source-ledger promotion,
aggregate rebuild, old-job cancellation, and replacement release sequence.
Hold `var/artifacts/e118_ledger_promotion.lock` continuously across both
prepare and apply. Record the wrapper path and SHA-256 in transaction
`var/artifacts/e118_owner_backfill_20260908/31048126.json` before apply;
validate that hash and the fixed profile on any resume. Do not edit the
shared controller or base helper.

After release, independently verify the current attempt's StartTime and
metrics mtime, optimizer progress above its checkpoint baseline, positive
sleep/wake timings, and absence of fatal errors or cgroup high/OOM events.
Record the current job ID and verification evidence. Training continues
toward the original target; the first scheduled checkpoint is not required
to establish successful startup.
