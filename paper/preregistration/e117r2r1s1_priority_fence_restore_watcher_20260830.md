# E117-R2-R1-S1 priority-fence restoration watcher

Frozen: 2026-08-30 after installation of the audited E117 owner backfill
priority fence and before watcher submission.

Status: scheduler-safety automation only. This watcher does not inspect an
incomplete scientific endpoint and cannot alter E117 science or another user's
work.

Submit one CPU-only, no-requeue Slurm job dependent on the `after` event for
all E117 replacement jobs `30977267`--`30977277`. The watcher must validate the
installed fence receipt and byte-exact restoration application before
submission. Submit held, record and validate its dependency and resource
identity, then release it.

When all eleven E117 jobs have accounting start evidence, run the registered
fence application's fail-closed `--restore` mode. Restore only still-pending
owner competitors: E113 to Nice 0 and E115 to its pre-existing Nice 100.
Already running or terminal competitors are recorded and left unchanged.

If held validation fails, cancel only the newly submitted watcher. The watcher
does not cancel, preempt, hold, requeue, or modify a scientific job.
