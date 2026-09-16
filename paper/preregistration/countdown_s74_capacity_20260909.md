# Countdown seed74 pending capacity replacement — September 9, 2026

The user requested requeueing broken E118/E119/E120 workloads and accelerating
completion. Existing E118 Qwen2.5-3B Countdown MaxRL seed74 job31151411 suffered
an actor process fatal Python `none_dealloc` error at09:26 EDT and stopped at
optimizer step1981. Its unchanged launcher automatically requeued the Slurm
ID. The saved model/optimizer checkpoint at1920 validates. This prospective
operation applies only while that predecessor remains pending. A running
predecessor is preserved and the operation stops.

The replacement changes only its account/partition/node eligibility:
allcs/cs/node202-204 becomes mltheory/lowprio/node202-203. Preserve exactly all
exported runtime and scientific values, including vLLM ratio0.40, the frozen
launcher and source snapshot,16CPUs,116GiB,one A5000,72-hour allocation limit,
queue nice value and PVL exclusion. Preserve the same run directory, seed,
optimizer settings, eight-pass/3072-step horizon and checkpoint cadence.
No outcome is inspected and no new science cell is added. Lowprio borrowing
can be preempted; existing requeue and automatic checkpoint resume remain on.

The five-cell capacity transaction must finish before this plan captures
ledger hashes. Acquire the E118 ledger promotion lock, freeze the exact
scheduler record/SubmitLine, source fingerprint, checkpoint and both E118
ledger hashes. Require no terminal receipt or duplicate writer, no incoming
dependent, no existing hold, and an intact checkpoint1920. Test the exact
replacement submission without launching it. Record the plan for review.

On application, hold only the pending predecessor and revalidate it, submit
one held replacement, then verify the full exported environment and allocation
fields. Stage and atomically promote only the E118 source and aggregate
ledgers, preserving every other cell and both E120 ledgers. Retire the held
predecessor only after the replacement is concrete and mapped. Confirm no
other queued writer, then release the audited replacement. Record durable
intent and exact IDs before submissions, holds, cancellation and release;
reconcile uncertain actions instead of repeating them. Never interrupt a
running scientific allocation or change a training recipe.
