# E119 memory-pressure recovery, September 5, 2026

The user authorized E119 repairs, approved the prepared backfill, requested
healthy workloads before the paper review, and specifically flagged Pantry
ReplayDr.GRPO seed44 job31037832. That job was independently diagnosed and
requeued under the same ID with64GiB/A6000 placement; it had no valid checkpoint.

A read-only cgroup census then confirmed23 additional allocations whose
anonymous plus shared memory exceeded their40GiB memory.high limit, with
negligible reclaimable cache and rapidly increasing throttle events. Some
weight synchronizations took hundreds of seconds while learning took seconds.
Two allocations naturally requeued before preparation, leaving38 ordinary
pending40GiB jobs and21 running pressure candidates in the reviewed plan.

Operational repair: raise ordinary pending E119 requests to64GiB. Requeue only
explicitly selected, currently pressure-confirmed running E119 allocations,
under their existing job IDs, using held-state resource and full scientific
export audits. Pending intentional holds are preserved. Pantry allocations use
already permitted A6000 routes or an already safe A100 route. Other cohorts,
objectives, seeds, datasets, optimizer settings, CPU/GPU counts, evaluation,
targets, and run directories are unchanged. Increased per-job RAM may reduce
simultaneous allocations; the purpose is to eliminate severe memory throttling.

Before stopping a learner, archive its logs and metric history and inspect the
latest valid model/optimizer checkpoint and saved counters. Revalidate after
the writer stops. The actual runtime uses the same partial-ZIP rejection and
highest-valid-step selection, so incomplete checkpoints are preserved as
evidence and skipped. Recovery may repeat updates after the last durable save.
Three candidate cells have no valid checkpoint (31037836,31048185,31048188);
if still without one at action time, the repair must restart their unchanged
scientific cell from initialization. Job31048195 is initially deferred near
its next durable checkpoint; it will be reviewed separately before recovery.

All changes are recorded per job under
var/artifacts/campaign_health_capacity_20260905/e119_memory_pressure_recovery/.
The prepared script passed9 focused safety checks and independent review.
No new scientific experiment or evaluation-outcome selection is introduced.

The follow-up census identified three more running cells crossing the same
limit (31048155,31048208,31048192), all with valid checkpoints and unchanged
scientific identities. The node205 diagnosis used an explicitly validated fast
owned peer because launching a probe inside the throttled learner could stall.
The generic recovery source remained hash-pinned and unchanged.

The deferred31048195 checkpoint failed to finish: its optimizer file stayed at
24,800 bytes for about nine minutes, compared with about5.93GB in its last valid
save. Its model metadata reached1152 while logged training stopped at1151.
The independently reviewed decision is to retain the partial1152 save and
recover from validated960, repeating192 actual updates. This explicitly
supersedes the initial near-checkpoint deferral; no fresh restart is needed.
See deferred_31048195_rollback_review.json for the saved-counter evidence.

Repeated follow-up samples found four additional pressure cases among learners
that had previously looked healthy:31045873,31048201,31048153,31045875. The
step1152 save for31048201 was allowed to finish; all saved counters validated
before its64GiB migration. The partial384 save for31048153 was only one-third
written with roughly45minutes estimated remaining, so the reviewed repair
uses valid192 and preserves the partial save, repeating192 actual updates.

To prevent continuing the same40GiB failure cycle, the final two otherwise
healthy E119 learners,31048196 and31075763, receive explicitly preventive
64GiB migrations only at validated checkpoints1920 and1344, respectively,
with at most16 recorded unsaved updates. They are not classified as crashed
or currently throttled. This completes a64GiB baseline for incomplete E119
cells. The lower possible allocation concurrency is an explicit resource
tradeoff; all scientific exports and CPU/GPU counts remain unchanged.

Final receipt reconciliation:68 generic transactions completed and were
released (41 pending resizes,27 requeues), with no outstanding repair holds.
Together with the separate primary repair and earlier64GiB Countdown repair,
all70 unfinished E119 cells request64GiB. Actual initialization restarts were
31037832,31037836 and31048188. Although a fresh restart had been allowed for
31048185, its96 checkpoint became valid and was selected after the writer
stopped, so that cell resumed rather than resetting.
