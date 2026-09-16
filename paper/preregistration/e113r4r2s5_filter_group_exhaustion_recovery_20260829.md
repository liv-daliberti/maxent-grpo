# E113-R4-R2-S5: bounded filter-group exhaustion recovery

Frozen: 2026-08-29, after job 30869122 failed and before submitting its
replacement. Inspection was limited to scheduler state, runtime failure text,
accepted optimizer-step count, filter-group counts, and checkpoint/receipt
presence. No validation accuracy, paper endpoint, arm contrast, or efficacy
statistic was used.

## Trigger and diagnosis

The authoritative E113-R4 campaign has one failed scientific cell:
Qwen-0.5B, Python factors, seed 44. Job 30869122 completed two of 24 optimizer
updates and then exited 1:0 on node205. It had no checkpoint and no
TRAINING_COMPLETE receipt.

The official verl DAPO filter retained 122 of the required 128 non-constant
prompt groups after generation batch 10. The frozen
algorithm.filter_groups.max_num_gen_batches=10 liveness cap then raised the
documented "Generated too many" ValueError. This is not a GPU, memory, data,
verifier, or optimizer failure. The accepted-batch definition worked exactly
as configured; its bounded attempt budget was too small for this stochastic
cell.

## Authorized exact-cell repair

Keep 10 as the runner default for the released cells. Expose that existing
value through E113R4_MAX_NUM_GEN_BATCHES and set it to 20 only for the exact
replacement of job 30869122. Raise E113R4_MAX_EPOCHS from 240 to 480 so the
data-loader liveness horizon remains 24 optimizer updates times the bounded
20 generation attempts.

The replacement must preserve model, family, domain, seed, train/eval parquet
and hashes, prompt/response lengths, 384-prompt generation batches, 128
accepted prompt groups, 16 responses per prompt, accuracy filtering, reward,
loss, optimizer, evaluation request, target 24 updates, run directory, A6000
hardware class, and official verl commit. It restarts from the initial model;
the two uncheckpointed prefix updates are discarded rather than combined with
the replacement.

The larger attempt cap does not accept constant-reward groups, shrink the
training batch, skip an update, change the seed, or alter the conditional DAPO
batch definition. It changes only the bounded number of generation batches
available to realize that definition. If 20 batches are insufficient, the
replacement must fail closed and receive a new prospective diagnosis.

## Source, launch, and monitoring transaction

Derive one content-addressed runtime snapshot from the R4-R2 snapshot. It may
differ only in the runner's validated environment-controlled attempt cap and
snapshot identity metadata. Submit one replacement held on partition all with
one A6000, 16 CPUs, 128 GiB, a 12-hour limit, and the current mltheory account.
Audit the complete held scheduler record and exported environment before
release.

Atomically replace only job 30869122 in the authoritative ledger while
retaining its complete failed row in a recovery record. The replacement keeps
the same scientific run directory and becomes the effective job seen by
campaign_stats and the paper progress builder. Release only after the ledger
and separate recovery receipt are durable.

No other completed, running, or pending E113-R4 job may be canceled, requeued,
held, reprioritized, or changed by this repair.


## Held-audit amendment before the released retry

The first held submission, job 30971093, was rejected by the launcher audit and
canceled with zero runtime before any ledger write because Slurm assigned
QOS=short during submission, rather than the cell's established QOS=long.
No code executed and no scientific result was observed. The retry must request
--qos=long explicitly at held submission, then repeat the complete account,
partition, resource, environment, ledger, and release audit. This is a
scheduler-only correction and does not change the repair or scientific cell.


### Second held-audit amendment before the released retry

Job 30971112 repeated the held-only transaction with --qos=long at submission,
but the account handoff again reset it to QOS=short. The launcher canceled it
with zero runtime before any ledger write; no code or scientific result was
observed. The next retry must explicitly set QOS=long after the held account
handoff and then re-read the complete scheduler record. Release remains
forbidden unless that post-handoff audit passes.
