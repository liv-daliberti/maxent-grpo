# E113-R4-R2-S6: common Qwen/Python filter-exhaustion recovery

Frozen: 2026-08-29, after jobs 30869121, 30869123, 30869124, and
30869125 failed and before any S6 replacement was submitted. The diagnosis
used scheduler accounting, exception signatures, realized step numbers,
terminal accepted-group counts, checkpoint manifests, and receipt presence.
Already-emitted training and validation fields were not used to choose the
repair, compare arms, or evaluate efficacy.

## Trigger and common cause

The four newly failed authoritative cells are Qwen-0.5B Python-factors DAPO
seeds 43, 45, 46, and 47. Each exited 1:0 on an A6000 after completing
optimizer step 11 with the exact official-verl exception:

`num_gen_batches=10 >= max_num_gen_batches=10`

Their terminal retained-prompt counts after ten generation batches were,
respectively, 50, 44, 55, and 64 of the required 128. No job failed from GPU
memory, node loss, data, verifier, optimizer, or checkpoint corruption. All
four therefore share the filter-liveness cause already established for
seed 44, but the later-step shortfalls are substantially larger.

Each failed cell has a complete `global_step_10` checkpoint containing model,
optimizer, extra state, Hugging Face assets, and data state, with the latest
checkpoint pointer equal to 10. Step 11 was not checkpointed and must be
discarded on resume.

## Bounded common repair

Use one cap of 40 generation batches for all five Qwen/Python seeds 43--47.
The worst observed yield was 44 retained groups after ten batches, implying
29.1 batches at the same average yield to realize 128 groups. Forty is a
bounded 37.5% margin above that empirical requirement. If 40 is insufficient,
the affected replacement must fail closed.

The released seed-44 replacement, job 30971120, is still pending with zero
runtime and cap 20. New evidence makes that intermediate cap inconsistent
with the observed common-domain requirement. Hold it before the transaction,
verify it never started and has no checkpoint or receipt, and replace it in
the same five-cell block. This avoids waiting for a predictable second
failure and gives all five cells identical recovery source and cap policy.

Set `E113R4_MAX_NUM_GEN_BATCHES=40` and
`E113R4_MAX_EPOCHS=960`; retain 10 as the runner default outside this block.
Seeds 43, 45, 46, and 47 resume automatically from their exact step-10
checkpoints. Seed 44 restarts from the initial model because its pending job
has zero runtime and its earlier failed prefix had no checkpoint.

The repair does not accept constant-reward groups, shrink the 128-prompt
optimizer batch, change 16 responses per prompt, skip optimizer updates,
alter seeds, or change the conditional DAPO batch definition. Model, domain,
data and hashes, prompt/response lengths, reward, loss, optimizer, evaluation
request, target 24 updates, run directories, A6000 hardware class, and
official verl commit remain fixed.

## Source and scheduler transaction

Create one content-addressed runtime snapshot shared by all five replacements.
It may differ from the R4-R2 base only in the already-tested
environment-controlled filter cap and snapshot identity metadata.

Before submission, place job 30971120 on user hold and audit that it remains
zero-runtime. Submit all five replacements held with partition `all`,
account `allcs`, QOS `long`, one A6000, 16 CPUs, 128 GiB, and a 12-hour
limit. While held, switch each to account `mltheory`, explicitly reapply QOS
`long` after the account transition, and audit the complete scheduler record
and exported environment.

Only after all five held audits pass may the authoritative ledger replace the
five prior rows and durably record their full provenance. Cancel the
superseded zero-runtime seed-44 job, verify that cancellation, and release the
five registered replacements. If any pre-ledger audit fails, cancel the new
held submissions and release the unchanged seed-44 job. After a ledger write,
fail closed rather than attempt an implicit rollback.

No other E113-R4 job may be canceled, held, requeued, reprioritized, or changed
by this transaction.
