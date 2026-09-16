# E100 sparse-RLEP execution repair

Date: 2026-08-14. This execution amendment was written after the frozen E100
pool outcomes were observed. It does not change collection samples, replay
eligibility, training hyperparameters, endpoints, or the E100 estimand.

## Observed gate outcomes

Four collection jobs exited 137 during process teardown after writing their
complete immutable draw sidecar and `EVAL_ONLY_COMPLETE.json`: Graph/s58,
Countdown/s57, and MathIR/s55,s58. The original frozen E100 auditor was run on
those exact artifacts. Each covers 384 prompts under the registered 16 x 4,
temperature-0.7, top-p-0.95 sampler and has at least one replay-eligible
prompt. No collection row is regenerated. Fresh CPU audit jobs recheck these
artifacts and become the scientific jobs' scheduler dependencies.

Python/s57 timed out on node206 before writing any pool artifact. Its empty
attempt directory is retained. One infrastructure replacement may run the
original frozen collection command, seed, checkpoint, prompt view, sampler,
and output root on healthy A6000 nodes 205 or 207. Its scientific cell remains
behind a fresh pool audit and the already-completed global smoke audit.
Because Slurm purged that completed smoke-audit job from its active dependency
lookup, a fresh CPU job reruns the same frozen smoke auditor on the immutable
completed smoke artifacts. Repaired science jobs depend on this re-audit; the
smoke training itself is not rerun.

Python/s56 and Python/s59 completed collection but their frozen audits found
zero replay-eligible prompts. This violates E100's preregistered per-pool hard
gate. They are blocked scientific cells, not scheduler-pending work and not
failed training runs. Their unstarted training jobs are cancelled, retained
in `blocked_runs`, and excluded from the operational campaign denominator.
They remain visible as blocked in the paper matrix. We do not regenerate their
fixed pools or run a control-equivalent zero-replay treatment under the E100
label.

## Scheduler-only acceleration

Thirteen cells whose original pool audits and global smoke audit already
passed are released at normal priority. The remaining live pool/audit jobs are
also promoted from nice 100 to nice 0. This changes only queue ordering. Falcon
training keeps its measured-safe 64 GB host-memory request and original GPU
class. E97/E98 Qwen jobs retain the A100 GPU class while using 36 GB, supported
by their completed cells' approximately 29 GB peak RSS; E99 Falcon jobs retain
64 GB and their original GPU placement while moving from nice 100 to nice 0.

All original collection, audit, and training job IDs and their terminal states
remain in the primary or repair ledger. Replacement jobs are submitted held,
audited against the frozen command surface, recorded atomically, and only then
released.
