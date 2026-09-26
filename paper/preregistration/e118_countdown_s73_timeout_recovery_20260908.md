# E118 Countdown MaxRL seed 73 timeout continuation — September 8, 2026

The user requested fixing the single failed E118 cell. Its authoritative job
31048123 is the existing Qwen2.5-3B Countdown MaxRL seed-73 cell. The allocation
reached its 12-hour limit on node205 and ended TIMEOUT. It has no terminal
completion receipt and has a valid model/optimizer checkpoint at step 1920.
This is an operational continuation of that registered cell; no evaluation
outcomes are used to select its configuration or checkpoint.

Resume on the same node205 A6000 hardware, with one GPU, 16 CPUs, 128 GiB host
memory, a 72-hour walltime and nice 200 on `allcs/lowprio`. Preserve requeue
eligibility and the exact existing PVL exclusion. Check live unallocated CPU,
GPU and host-memory capacity before submission. The longer scheduler limit
retains the same eight-pass, 3072-step training target.

Keep the original frozen launcher and its SHA-256, source snapshot, model,
seed, data, objective, optimizer, evaluation/checkpoint cadence, sampling,
offloading, `SAVE_PATH`, `RUN_STAMP`, and every non-root exported variable.
In particular `OAT_ZERO_VLLM_GPU_RATIO=0.25` and `OAT_ZERO_AUTO_RESUME=1`
remain unchanged. Explicit repository-root exports are restored. Select the
latest structurally valid model-and-optimizer checkpoint, which must still
be step 1920 before release; this continuation must not restart from zero.

Before submission and release, require predecessor TIMEOUT/absence from the
queue, no completion receipt, and no other active or pending writer for the
same run directory. Submit one held replacement, audit its effective Slurm
record and export equality, and record its job ID durably. Use a unique
submission comment; an uncertain submission must reconcile the existing held
job or stop for accounting inspection, never create a blind duplicate.

Preserve source and aggregate before-images and hashes in
`var/artifacts/e118_countdown_s73_timeout_20260908/`. Under the shared
`e118_ledger_promotion.lock`, stage both after-images, append 31048123 to
`previous_job_ids`, promote the same source cell to the replacement ID, and
update its corresponding row in the 150-cell aggregate. Each file is replaced
atomically; recorded before/after hashes permit completion of an interrupted
pair of promotions. Release only after both ledgers agree and the held-job,
checkpoint and sole-writer checks pass. No existing live job is interrupted.

Implementation:
`ops/exp_scaling/recover_e118_countdown_s73_timeout_20260908.py`.
`prepare` records the plan and runs only `sbatch --test-only`; `apply` submits
and promotes the one replacement, and rerunning it reconciles that same
transaction. After release, verify restoration of step 1920 and fresh
optimizer progress beyond it, current metrics, and absence of new fatal or
OOM events before reporting successful recovery.
