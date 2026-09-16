# Existing Countdown timeout continuations — September 8, 2026

The user requested requeueing broken E118/E119/E120 jobs and accelerating
completion. Read-only status and accounting identify two unfinished timeouts:

| Cohort and existing cell | Failed job | Observed update | Latest valid checkpoint | Previous walltime |
| --- | --- | --- | --- | --- |
| E118 Qwen2.5-3B Countdown ReplayMaxRL, seed 73 | 31048124 | 1918 | 1728 | 12 hours |
| E119 Level-2 Qwen2.5-0.5B Countdown ReplayMaxRL, seed 46 | 31048160 (original 31014413) | 2521 | 2496 | 36 hours |

Both stderr logs explicitly report cancellation at the scheduler time limit.
Neither cell has a terminal completion receipt or another active/pending
writer for its run directory. E120-R1 has no failed unfinished cell at this
inspection. No evaluation outcomes inform checkpoint or resource selection.

Resume each existing cell from its latest valid model-and-optimizer checkpoint
with a 72-hour allocation, preserving the eight-pass, 3072-update target.
Use `allcs/cs`, nice 0, no artificial inter-job dependency, requeue eligibility,
and the unchanged PVL exclusion. The E118 replacement uses one A6000 GPU,
16 CPUs and 128 GiB host memory in the existing A6000 owner pool
`node205,node206,node207`. The E119 replacement uses one A5000 GPU, 8 CPUs and
128 GiB host-memory allocation in
`node202,node203,node204`; its predecessor ran on A5000 node203. The original
E119 SubmitLine predates the effective memory increase and still says 40 GiB;
its archived post-amendment scheduler record verifies 96 GiB. Increase that
effective allocation to 128 GiB prospectively: current same-cohort Countdown
jobs 31048154 and 31048161 exceed their 96 GiB memory.high thresholds with
observed non-cache working sets of approximately 100.48 and 101.24 GiB,
respectively, and tens of millions of memory.high events. A sampled process
was blocked in mem_cgroup_handle_over_high. These operational observations
show that 96 GiB is insufficient for these otherwise matching runs; they
support 128 GiB headroom for the failed seed-46 continuation without claiming
a direct cgroup measurement from the terminated job. These are scheduling
changes to existing cells, without modifying the scientific treatment.

Keep every non-root exported variable unchanged, including data, model,
seed, optimizer, objective, sampling, evaluation/checkpoint cadence,
offloading, `SAVE_PATH`, `RUN_STAMP`, `OAT_ZERO_AUTO_RESUME=1`, and
`OAT_ZERO_VLLM_GPU_RATIO=0.25`. Use the original frozen launchers and source
snapshots (`089bcea44b44cc70` for E118 and `50d36295558a8958` for E119),
record their hashes, and restore explicit repository-root exports. Require
the latest valid checkpoints still be steps 1728 and 2496 before release.

Implementation is
`ops/exp_scaling/recover_campaign_evening_timeouts_20260908.py`.
`prepare` only reads jobs, validates both checkpoints and invokes
`sbatch --test-only`; it writes the reviewable transaction and ledger backups
in `var/artifacts/campaign_evening_timeouts_20260908/`. `apply` submits each
replacement held, records its ID durably, and audits its exact exports,
launcher, resources, node pool, time limit, dependency and sole-writer status.
An uncertain submission must reconcile its unique scheduler comment or stop;
a retry must never create a blind duplicate.

Under `e118_ledger_promotion.lock`, preserve the E118 source, E119 continuation
ledger and E118 aggregate before-images and hashes, then stage all after-images
and hashes. Replace the same two cell IDs, retain predecessor histories,
preserve the E119 original-cell mapping and replace the corresponding E118
aggregate row without changing its 150-cell cardinality. Per-file atomic
promotion and recorded before/after hashes permit reconciliation after an
interrupted multi-file promotion. Release only after all ledgers agree and
held-job, checkpoint and sole-writer checks pass. Existing live jobs are
untouched. After release, verify actual checkpoint restoration and fresh
optimizer progress before describing the replacements as successfully
resumed; a queued continuation is reported as queued.
