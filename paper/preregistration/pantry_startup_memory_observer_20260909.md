# Read-only Pantry startup and memory observer — September9,2026

Observe only E120 job31033711 (Pantry seed71,116GiB after the reviewed
memory-only amendment) and E119 job31048186 (Level-2 Pantry ReplayDr.GRPO
seed46,96GiB). Preserve their full exported environments, source launchers,
run identities and effective scheduler resource fields. Report any drift;
never repair or change either scientific job. The existing six-cell
observer31158641 remains untouched.

Wait for allocation for at most48hours, then observe for at most6hours
from first observed allocation. The absolute total bound is54hours,
persisted across observer restarts. Stop each cell's startup audit after
fresh post-resume optimizer progress plus a newer structurally valid
current-job model/optimizer checkpoint; valid terminal completion also
satisfies the audit. Otherwise report pending/fatal/stale/checkpoint/memory
status and the bounded deadline. Initial durable floors are0 for31033711
and96 for31048186; actual selected resume checkpoints override those
startup-progress thresholds. Historical maximum steps are not proof of
new progress after a same-ID restart.

Record exact scheduler state, restore messages, latest optimizer record,
current log activity and optional cgroup current/high/peak/anon/shmem/kernel
and OOM/high counters. Probe at startup and periodically using small
read-only overlap steps with an8second client timeout; probe failure is
reported without modifying training. Do not infer productive progress from
allocation, model restore, or a stale historical metrics maximum.
The CPU-only supervisor uses owner node915,256MiB,noGPU, with54h10m limit.
