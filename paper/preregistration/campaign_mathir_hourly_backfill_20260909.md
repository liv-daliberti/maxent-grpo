# Bounded MathIR hourly backfill — September 9, 2026

The user requested more concurrent E118/E119/E120-R1 work to finish the existing
campaigns promptly. This two-cell operational pilot uses the ordinary permitted
one-hour all-partition route for existing Qwen2.5-3B MathIR MaxRL and ReplayMaxRL
seed72 cells, currently pending jobs31158503/31158504. No scientific cell, seed,
training horizon, data, objective, optimizer, replay setting, batch, sequence
length, decoding parameter, initial evaluation or evaluation cadence changes.

Request account mltheory, partition all, one typed A5000, 16 CPUs, 116 GiB RAM,
exactly one hour, candidate nodes105/202/204, and the full existing PVL exclusion.
Keep Nice and automatic requeue settings. Only the explicit runtime export
OAT_ZERO_RESUME_STEPS changes192→48. The frozen source independently controls
resume checkpoint cadence; keep legacy SAVE_STEPS192, ratio0.40, all other
submitted exports, frozen launchers/source, run paths and automatic resume.
Existing initial evaluations remain mandatory. Recovery checkpoints retain the
previous valid state until the next distributed checkpoint write completes.

Timing-only feasibility evidence is stored in feasibility.json/feasibility.md
under the audit directory. It supports a bounded pilot rather than a guarantee:
MathIR71 A5000 startup plus48 updates and a save takes approximately22–24minutes;
a prior MathIR72 Replay checkpoint restore was unusually slow and the analogous
48-update window would total approximately56minutes. Scheduler test-only calls
accept both ordinary one-hour requests, but do not prove actual starts.

Prepare a frozen plan; stage only after independent root review. Hold each still-
pending original job, submit its hourly replacement held, audit the complete
export/resource difference and validated initial checkpoint, then promote E118
source and aggregate ledgers under their shared lock. Preserve the original
five-cell transaction and protocol unchanged. No E119/E120 ledger is written.

Before release, require a running bounded timeout guard with exact new IDs,
initial checkpoint counters, script hash, explicit positive requeue cap and
deadline. The guard permits same-ID requeue only after accounted TIMEOUT and a
strictly newer counter-validated model/optimizer checkpoint. It must stop hourly
requeues on no durable progress, cap or deadline. It then invokes the reviewed
fallback helper, rather than leaving an unfinished cell abandoned.

Keep original72-hour jobs31158503/31158504 as dormant, owned-held fallbacks.
They cannot execute while held and are not the active canonical mappings during
the pilot. The only duplicate-writer exemption is the exact corresponding old
job ID whose complete original command and JobHeldUser state are verified.
Fallback after a running allocation requires the hourly job to be absent from
the queue and accounted as TIMEOUT. A still-pending hourly request at the guard
deadline may instead be held and retired, with exact owned-hold/cancellation
receipts; a racing newly started allocation is preserved. Fallback revalidates the exact held predecessor and absence of any other writer,
atomically restores both canonical mappings, and only then releases the old job.
The original long route restores RESUME_STEPS192 and mltheory/lowprio on
nodes105/202/203/204, retaining72hours and all other original resources. It can
resume a valid checkpoint produced on a48-step boundary. No new fallback job is
submitted. Scientific completion must retire the dormant fallback without ever
releasing it for retraining.

Audit directory: var/artifacts/campaign_mathir_hourly_backfill_20260909/.
Controller: ops/exp_scaling/backfill_mathir_hourly_20260909.py.
