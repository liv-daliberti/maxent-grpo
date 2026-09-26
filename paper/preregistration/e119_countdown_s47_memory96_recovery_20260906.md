# E119 Countdown ReplayDr.GRPO seed47 memory recovery — September6,2026UTC

The user explicitly requested fixing memory pressure in job31048162 after the
status report identified active64-GiB throttling and the unsaved updates a
checkpoint restart would repeat. The same Slurm jobID is requeued into an owned
hold, its host-memory request increased64→96GiB, then audited and released.

At preparation, the latest valid model/optimizer checkpoint was576 and latest
logged step680, requiring104logged updates to repeat. The controller accepts
at most one existing192-step checkpoint interval and always reselects the
latest valid checkpoint. This task-specific bound replaces the earlier96-update
bound used for different recoveries. No fresh initialization is permitted.
Independent validation confirmed all three saved step counters, prompt traversal
and populated replay-bank metadata, and the actual frozen auto-resume selector.
Logs and metrics are archived before stopping and after writer cleanup.

Preserve the full original scientific/runtime exports and run directory,
accountallcs,partitioncs,QOSmedium,8CPUs,oneGPU,36-hour walltime,nice0,and existing
node203/node204/node205/node207 placement. Only MinMemoryNode and operational
restart state change. Main and continuation ledgers remain byte-identical.
Node207 has nominal96-GiB capacity at preparation; scheduler admission is not
promised. No other job or resource request is modified.

Evidence and transaction state:
`var/artifacts/e119_countdown_s47_memory96_20260906/`.
