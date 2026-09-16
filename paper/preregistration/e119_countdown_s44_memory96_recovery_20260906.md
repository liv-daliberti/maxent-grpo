# E119 Countdown seed 44 memory recovery — September 6, 2026 UTC

The user explicitly requested more memory and restart for the two Countdown
seed-44 jobs diagnosed with active host-memory throttling on node105:
31045875 (ReplayDr.GRPO) and 31074759 (Dr.GRPO). Their memory requests increase
from 64 to 96 GiB using the same Slurm job IDs and a held requeue transaction.

The validated restart checkpoints are step 1536 for 31045875 and step 2688 for
31074759. The former repeats 95 logged, unsaved updates; the latter's saved
checkpoint precedes its corresponding metric write and is one step above the
last logged step, 2687. Model/optimizer ZIP structures, three saved progress
counters, and the frozen launcher's automatic checkpoint selection were checked.
The controller revalidates after each old writer stops and archives its logs.

Only the requested memory and operational restart state change. Existing
scientific exports, run directories, models, seeds, objectives, optimizer and
evaluation settings, CPU/GPU requests, node105 placement, account, partition,
QOS and nice values are preserved. Main and continuation ledgers are hashed
before execution and checked throughout the transaction.

Node105 has 503.02 GiB schedulable memory. The four other running allocations
reserve 352 GiB, so only one of these two 96-GiB restarts fits immediately.
Both retain their ordinary queue route. Release the Dr.GRPO continuation first
because its validated checkpoint is closer to the fixed 3072-step completion
budget; release the other after this allocation starts. No priority values are
changed. Scheduler availability determines the second start.

Plan, archived logs, checkpoint certificates, mutation receipts and startup
verification are under `var/artifacts/e119_countdown_s44_memory96_20260906/`.
