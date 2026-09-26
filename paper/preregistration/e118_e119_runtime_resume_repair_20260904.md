# E118/E119 runtime recovery amendment — 2026-09-04

This operational amendment follows the user-authorized repair of E118's missing
Ninja executable and E119's checkpoint-resume exception. The audit observed
learner exceptions while Slurm allocations remained RUNNING. It does not
register a new scientific treatment or reinterpret completed endpoints.

Ninja already existed in the pinned grail-training environment. A link exposes
that executable through the existing paper310 training PATH. E118 Qwen-3B uses
node-local temporary and extension-build directories to avoid shared-filesystem
compiler stalls and CPU-specific binary reuse across nodes. A CPUAdam preflight
on an existing node302 allocation successfully compiled and updated a parameter.

The missing ZeroMathRunMixin._infer_resume_step method is implemented in the
working source and the E118/E119 frozen runtime snapshots. It reads the saved
checkpoint step, checks agreement with a numeric checkpoint tag, and refuses
unidentifiable or inconsistent progress. The existing optimizer restoration,
prompt cursor restoration, replay state, and actor synchronization remain in
place. Eleven focused resume/actor synchronization tests passed. Original source
bytes and before/after hashes are preserved in the recovery audit directories;
these runtime snapshots now carry this explicit operational amendment.

All 20 E119 Pantry cells now launch with a 96-step rolling optimizer checkpoint
interval, starting at step 96 and retaining one checkpoint. The learner saves
before evaluation at the same boundary, preventing the original repeated loss
of steps 1–95. The first recovery amendment used interval 32. Its observed
roughly 6 GB optimizer write at approximately 3 MB/s made that cadence too
expensive; revision r3 changes future allocations to 96. Already-running Pantry
job 31041178 retains interval 32 for its current attempt, allowing its first
checkpoint to finish without another restart. Their two-hour inactivity
allowance remains.
E119's watchdog observes the actual per-job stdout path, recognizes log activity
during evaluation, and detects the identified fatal startup exceptions.
A subsequent launch audit found that 13 Pantry replacement commands had lost
five settings from the previously registered Pantry policy repair. The E119
runtime guard now reapplies canonical_action_task=none, canonical_graph_actions=0,
canonical_graph_action_count=3, canonical_graph_learner_sampling=0, and
canonical_graph_fixed_shape_sampling=0. All 20 effective Pantry environments
were checked, including a full launcher dry-run from an affected replacement.
The r2 policy audit and r3 cadence audit are `durability-watchdog-amendment-r2.json`
and `durability-watchdog-amendment-r3.json` in the E119 recovery directory.

Scientific evaluation cadence, optimizer settings, intended treatment, data,
seed, training horizon, and run directories are unchanged.

The same Slurm job IDs are requeued for active affected cells; queued cells
inherit the amended runtime when allocated. Completed cells are not restarted.
Historical maximum metrics alone are insufficient evidence of successful
recovery: verification requires current-attempt checkpoint restoration and new
optimizer metrics, with the original highwater reported separately.

Evidence:

- `var/artifacts/e118_ninja_resume_runtime_recovery_20260904.json`
- `var/artifacts/e119_resume_runtime_recovery_20260904.json`
- `var/artifacts/e118_runtime_recovery_20260904/`
- `var/artifacts/e119_runtime_recovery_20260904/resume-source-patches.json`
- `var/artifacts/e119_runtime_recovery_20260904/durability-watchdog-amendment.json`
- `var/artifacts/e119_runtime_recovery_20260904/restart-audit.json`
- `var/artifacts/e118_runtime_recovery_20260904/current_attempt_verification_20260905.json`
- `var/artifacts/e119_runtime_recovery_20260904/current-allocation-health.json`
- `var/artifacts/e119_runtime_recovery_20260904/durability-watchdog-amendment-r3.json`
- `var/artifacts/e119_runtime_recovery_20260904/durability-watchdog-r3-validation.json`
- `var/artifacts/e119_runtime_recovery_20260904/cohort-progress-snapshot.json`

Verified recovery on 2026-09-05: all three restarted E118 Python learners
surpassed their pre-repair maxima. All 33 allocated E119 learners produced
current-attempt training metrics, with no current fatal errors in the final
scan. Pantry job 31041178 completed a structurally valid step-32 model and
optimizer checkpoint; saved counters and the latest pointer agree, and the
same allocation subsequently reached training step 45. Its detailed proof is
`var/artifacts/e119_runtime_recovery_20260904/live-checkpoint-verification.json`.
The original restarted E118 Graph MaxRL job remains queued for Priority.
