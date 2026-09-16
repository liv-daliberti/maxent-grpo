# E113-R4 / E117-R2 signal-53 batch-wave recovery

Date frozen: 2026-08-30, after the affected Slurm allocations became terminal
and before inspecting any task reward, validation accuracy, arm contrast, or
efficacy endpoint.

## Trigger and diagnosis boundary

A single cross-node failure wave affected the released E113-R4 and E117-R2
campaigns between 07:23 and 07:32 EDT. Eighteen E113-R4 science jobs, ten
E117-R2 science jobs, the E117-R2 audit job, and one unrelated job ended with
batch-step signal `53` after zero to four seconds. Nine additional E113-R4
jobs and one E117-R2 job that were already running ended with process exit
`1:0` during the same interval, without an application traceback. The event
spanned nodes 016, 202, 203, 205, 206, 207, and 208. A later unrelated job
started and remained healthy on node207.

The login host's `/tmp` filesystem was observed full after the event. That is
not sufficient evidence that compute-node temporary storage caused the
cross-node terminations, so this protocol records the root cause narrowly as
a transient site-wide batch-launch interruption. It does not claim a
space-related compute failure.

Slurm accounting still retains the terminal evidence, but the controller has
already purged the failed job records. Exact-ID `scontrol requeue` is therefore
unavailable and clean replacement job IDs are required.

## Exact E113-R4 recovery

Replace exactly the 27 failed authoritative E113-R4 cells:

- `30869134`, `30869139`, `30869140`, and `30869142` through `30869160`;
- `30972013` through `30972017`.

Keep the other 23 completed cells unchanged. Each replacement must retain
the authoritative row's model, family, domain, seed, data, verifier, official
verl DAPO objective, filter cap, epoch horizon, runtime snapshot, optimizer,
and output directory. Use the existing `all / mltheory / long` A6000 route,
128 GiB memory, and 12-hour limit.

Resume only from a complete checkpoint selected by the frozen verl
`trainer.resume_mode=auto` contract. The validated resume steps are:

- step 20: Qwen/Pantry seed 46 and Falcon/Graph seeds 58-59;
- step 15: Falcon/Countdown seeds 56-58;
- step 10: Qwen/Python seeds 43, 45, 46, and 47;
- initial model: every other failed cell.

No uncheckpointed optimizer update is credited. Existing completed receipts
remain authoritative and are not relaunched.

## Exact E117-R2 recovery

Keep completed reference job `30970803` unchanged. Replace exactly failed
science jobs `30970804` through `30970814` with the same common-source
snapshot, scientific environment, run directory, node pin, arm identity, and
`all / mltheory / none / 01:00:00` route. Their failed allocations produced no
metric or checkpoint file, so every replacement begins the unchanged 64-step
cell from its initial model state.

Replace failed audit job `30970815` with a new non-requeueable audit job whose
live `afterany` dependency contains the 11 replacement science jobs. The
durable audit receipt must separately retain `30970803` as the already
completed member of the authoritative 12-cell graph. Stage 1 remains blocked
until that official fail-closed audit passes.

## Transaction and invariants

1. Validate all 23 E113 completion receipts, the E117 completed reference, all
   38 failed accounting records, the exact checkpoint boundary, and absence
   of E117 metric/checkpoint material.
2. Submit all 38 science replacements held, normalize their scheduler routes,
   and verify their complete exported environments before changing a ledger.
3. Submit and verify the replacement E117 audit dependency while science jobs
   remain held.
4. Atomically record both ledger mappings and the incident receipt, then
   release all 38 science jobs together.
5. On any pre-ledger failure, cancel every newly submitted job and leave both
   authoritative ledgers unchanged.

This is an execution-continuity repair. It changes no treatment, source,
seed, data order, estimator, coefficient, proposal mechanism, policy loss,
reward, evaluation setting, target step count, or accepted-batch definition.
Only receipt presence and checkpoint integrity are inspected; efficacy
outcomes are not used.

## Explicit exclusion

Collateral job `30971429` and its `/n/fs/similarity/social_sim` dependency
chain are outside this transaction and must not be mutated here.
