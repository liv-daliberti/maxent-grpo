# E111 Graph completed stale-requeue cancellation

Date recorded: 2026-08-19, before canceling the stale scheduler instance.
No evaluation endpoint was used.

Original Qwen-3B Graph job `30674758` has a valid
`oat_zero_training_complete_v1` receipt with terminal step 65, beyond the
frozen 64-step gate target. Checkpoint cleanup completed without errors, but
the watchdog left the same job ID pending for another restart.

Cancel only pending job `30674758`. Preserve every run artifact. The E111
auditor already requires both a validated receipt with terminal step at least
64 and parsed training telemetry with last step at least 64 before overriding
scheduler state. This is scheduler cleanup only; treatment, optimizer state,
and outcomes are unchanged. PointMaze remains excluded.
