# E122 first Countdown cell: recovered node208 placement — September 10, 2026

The user requested starting E122 Qwen-0.5B Level-3 while E118/E119/E120 finish.
The original E122 resource protocol excluded node208 because it was drained for
overheating on September 9. September 10 scheduler evidence shows node208 healthy
and available with the same A6000 GPU family already approved for E122.

This additive operational amendment affects only existing, released, zero-step
Countdown Dr.GRPO seed43 job31158645. Expand its requested pool from
node205/node206/node207/node302 to include node208. Preserve its job ID, all
scientific and runtime exports, model/data/source, seed,3072-step horizon,
8CPUs,128GiB host RAM,one GPU,36-hour walltime,allcs account,lowprio partition,
medium QoS,nice value,exclusions,requeue setting,checkpoints and evaluation.
Leave the other three released E122 jobs' routes intact. One E122 allocation
uses128GiB; the parallel finishing-workload plan may allocate a116GiB E119 job
on node208 without this amendment consuming all available memory.

Prepare and validate a concrete plan without scheduler mutation. Authenticate
all frozen E122 plan/ledger/admission/runtime bytes and the original successful
release journal, plus the exact current released pending identity, sole writer,
unchanged environment and healthy node208 capacity. Use sbatch --test-only as
scheduler feasibility evidence; its predicted start is not a guarantee.

Apply exactly one `scontrol update JobId=31158645 ReqNodeList=...` after a final
pending-state check and durable intent. No hold/release/requeue/submission occurs.
No release-controller lock is acquired: that controller's nonblocking lock would
make a concurrent watcher exit, and a temporary hold on an already released job
would look like external activity. Its existing release reservation remains valid.
The frozen controller audits original archived held evidence for released cells
and current held requests only before initial release, so no source adapter,
controller handoff, journal rewrite or frozen ledger modification is needed.

If allocation races the final pending check, preserve it; never cancel or requeue.
Audit readback for the exact allowed node pool and unchanged full SubmitLine and
resources. An ambiguous scheduler response is reconciled only from its recorded
intent and exact readback, never blindly repeated. Store this route amendment
and receipts separately. Confirm the unchanged controller reports four reserved
unfinished cells and no issues, then observe actual allocation and progress.
