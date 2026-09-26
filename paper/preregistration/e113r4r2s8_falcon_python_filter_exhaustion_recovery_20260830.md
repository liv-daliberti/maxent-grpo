# E113-R4-R2-S8: Falcon/Python filter-exhaustion recovery

Frozen: 2026-08-30, after the authoritative Falcon-1B Python-factors DAPO
jobs reached their natural terminal states and before any S8 replacement was
submitted. The diagnosis used scheduler accounting, exception signatures,
accepted-group counts, accepted optimizer-step numbers, checkpoint manifests,
and completion-receipt presence. Training and validation outcome values were
not used to choose the repair, compare arms, or evaluate efficacy.

## Trigger and common cause

The five affected cells are Falcon-1B Python-factors DAPO seeds 55--59. Their
S7 jobs are 30977247--30977251. Each exited 1:0 on an A6000 before accepting
optimizer step 1 with the exact official-verl exception:

`num_gen_batches=10 >= max_num_gen_batches=10`

After ten generation batches, seeds 55--58 retained respectively 24, 28, 30,
and 24 of the required 128 non-constant prompt groups. Seed 59's buffered
standard output was empty at exit, so its terminal retained-group count is
unknown and is not imputed. No affected job failed from GPU memory, node loss,
data, verifier, optimizer, or checkpoint corruption. None wrote a checkpoint
or `TRAINING_COMPLETE.json`; all five therefore restart from their unchanged
initial models and seeds.

## Bounded common repair

Use one cap of 80 generation batches for all five Falcon/Python seeds. The
worst observed yield was 24 retained groups after ten batches, implying 53.3
batches at the same average yield to realize 128 groups. Eighty is a bounded
50% margin above that empirical requirement. If 80 is insufficient, the
affected replacement must fail closed.

Set `E113R4_MAX_NUM_GEN_BATCHES=80` and `E113R4_MAX_EPOCHS=1920`; retain 10 as
the runner default outside this block. The epoch horizon remains the target 24
accepted optimizer updates times the bounded generation-attempt cap.

The repair does not accept constant-reward groups, shrink the 128-prompt
optimizer batch, change 16 responses per prompt, skip optimizer updates, alter
seeds, or change the conditional DAPO batch definition. Model, domain, data
and hashes, prompt/response lengths, reward, loss, optimizer, evaluation
request, target 24 updates, run directories, A6000 hardware class, and the
official verl commit remain fixed.

## Source and scheduler transaction

Create one content-addressed runtime snapshot shared by all five replacements.
It may differ from the S7 base only by installing the already-tested
environment-controlled filter-cap runner and snapshot identity metadata.

Submit all five replacements held with partition `all`, account `allcs`, QOS
`long`, one A6000, 16 CPUs, 128 GiB, and a 12-hour limit. While held, switch
each to account `mltheory`, explicitly reapply QOS `long`, and audit the full
scheduler record and exported environment.

Only after all five held audits pass may the authoritative ledger replace the
five failed rows and durably record their full provenance. Release the five
registered replacements only after both the ledger and separate recovery
record are durable. If any pre-ledger audit fails, cancel all new held
submissions. After a ledger write, fail closed rather than attempt an implicit
rollback.

No completed, running, or pending E113-R4 job outside these five failed cells
may be canceled, held, requeued, reprioritized, or changed by this transaction.
