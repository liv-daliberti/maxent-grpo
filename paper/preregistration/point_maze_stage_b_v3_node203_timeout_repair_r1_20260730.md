# PointMaze Stage-B v3 node203 timeout repair r1

Frozen on 2026-07-30 after jobs 30203027--30203030 failed and before any
replacement cell was submitted.  The four affected cells are both frozen
arms for seeds 46 and 47.  Each stopped at the evaluation following update
60/96 with the same `interactive PointMaze worker request timed out` transport
error on node203.  Each has exactly 60 training rows, 30 completed evaluation
coordinates (rounds 0 through 58), 60 state-replay rows, and no terminal
receipt.  Their partial artifacts are quarantined and have no scientific
status.

The repair restarts all four affected cells from the original immutable model
checkpoint.  It does not resume, carry forward an optimizer or replay state,
replace a seed or row, add a pass, or use a partial metric to select an
attempt.  Arms, seeds, model, data, protocol, source snapshot, execution
snapshot, learning rate, sampling, compute traversal, evaluation schedule,
and terminal auditor remain byte-identical to the initial attempt.

The sole execution repair is scheduler placement on node202.  The six
cohort cells already placed there crossed the same update-60 evaluation
boundary without a worker timeout.  This placement decision uses only the
shared transport failure and scheduler state, not reward, route, mode, or
evaluation values.  The four replacement jobs remain non-requeueable and the
independent audit is rebound to the six original jobs plus these four fresh
jobs.  Original identities, submissions, logs, scheduler records, and partial
artifacts remain preserved under job-specific names.
