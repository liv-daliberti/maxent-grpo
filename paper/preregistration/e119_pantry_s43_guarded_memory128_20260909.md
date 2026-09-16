# E119 Pantry seed 43: checkpoint-preserving 128 GiB recovery

The user requested faster completion and recovery of stalled E119 workloads.
This operational repair may target only Dr.GRPO job 31037827 at checkpoint768
and, only after equivalent fresh pressure evidence, ReplayMaxRL job31048179 at
checkpoint960. Each cell has an independent immutable plan and transaction.

Preparation requires a complete same-cell model/optimizer checkpoint, matching
saved global/prompt counters, no newer incomplete checkpoint, no logged updates
beyond that saved state, and at least fifteen minutes without metric progress.
Two fresh cgroup observations must show noncache memory above96GiB, increasing
memory.high events, and no OOM. Target128GiB preserves a useful margin for
observed~99GiB demand; no scientific parameter or hardware route is changed.

Apply preserves the same job ID, exact original submission, account, partition,
node pool, GPU request, CPU count, walltime, source/data exports, save directory,
checkpoint bytes, and authoritative scientific identity. It writes durable
intents before same-ID requeuehold, held memory update to128GiB, and release.
An uncertain requeue is never repeated. The exact owned pending hold and one
restart increment are required before update/release. Unacknowledged actions
remain recorded for reconciliation. Scheduler placement is not guaranteed;
Slurm may use the existing owner partition's ordinary preemption policy.
No other job is cancelled or preempted by this helper.

Checkpoint shard identities and pickle metadata are audited without loading
model tensors. Logs and metrics are archived before stopping. The operation
uses a private lock, preserves all shared ledger files, and checks the current
same-cell mapping before every mutation. Existing guard plans and helpers are
unchanged. The shared ledger lock is not held during asynchronous cleanup.
