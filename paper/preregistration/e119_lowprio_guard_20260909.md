# E119 ReplayDr.GRPO44 hourly continuation watcher — 2026-09-09

The user requested recovery of broken E119 cells and faster completion. This
watcher covers only the reviewed 116 GiB, one-hour, mltheory/all continuation of
ReplayDr.GRPO Pantry seed 44, predecessor 31037832, from validated checkpoint 192.
It adds no scientific cell and changes no training, evaluation, source, seed or
horizon. The frozen Pantry wrapper retains 96-step rolling recovery checkpoints
and the unchanged evaluation schedule. The original 36-hour cs/96 GiB allocation
remains held as a reviewed fallback for an unsuitable short scheduling window.

Preparation freezes the exact staged SubmitLine, resource fields, protocol,
capacity controller/plan and helper hashes. The authoritative E119 identity and
mapping must still point to the continuation. An exact user-held predecessor is
the only duplicate-writer exemption. A readiness receipt comes from its CPU-only
Slurm supervisor; only the capacity controller performs initial release.

The 24-hour deadline is persisted at first application and survives supervisor
restarts. At most 24 same-ID retries are allowed, exclusively for an inactive,
accounted TIMEOUT with a model/optimizer ZIP checkpoint whose saved global_steps,
global_step and prompt_batches_consumed_total agree with the tag. The first retry
must exceed checkpoint 192; subsequent retries must exceed the last released
resume step. Holds and releases have durable intents and exact Restarts proof;
uncertain requeue commands are never repeated. After the five-minute release
grace, a proven owned requeue hold is normalized to a user hold with durable
Restarts and action receipts, then handed to the capacity deadline fallback.
Unrelated user or administrator holds are never claimed. Deadline checks run again
immediately before requeuehold and release. A persisted pre-deadline transition
may finish once in the first five minutes of cleanup, leaving a full one-hour
allocation inside the finite 65-minute cleanup period. The watcher never preempts
a running allocation. Running final attempts remain observed through cleanup;
expiry while still active preserves that allocation and the original hold.

Every five minutes the watcher records current progress and optional cgroup
memory observations within the existing allocation. OOM log evidence, positive
cgroup OOM counters, or increasing memory.high events with noncache memory at the
high limit block subsequent retry/fallback. A true memory failure leaves the new
job inactive and its 96 GiB predecessor held pending conservative 128 GiB repair.
The watcher never shrinks memory or automatically stops a running allocation.

The reviewed capacity fallback is allowed for inactive accounted NODE_FAIL or
BOOT_FAIL, an ordinary pending route at the deadline, or an inactive hourly
TIMEOUT with no advancing checkpoint, an exhausted retry cap, or the elapsed
deadline. These are scheduling-route failures, and fallback requires no memory
failure evidence. Other failures, invariant drift or uncertain actions stop for
review with the predecessor held. A valid terminal 3072-step receipt invokes the
capacity controller's exact predecessor-retirement helper after cleanup.

All mutations use the existing E118/E119 ledger-promotion flock and a separate
single-watcher lock. Existing campaign guards, sources, plans and observers remain
immutable. This watcher is prepared and independently reviewed before submission.
