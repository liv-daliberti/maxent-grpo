# E111 Qwen-3B Python/MathIR timeout requeue

Date frozen: 2026-08-18, before requeueing either job.  No endpoint evaluation
result is inspected; the previously recorded post-freeze Pantry training-reward
exposure is not used for this decision.

Jobs `30674760` (Python) and `30674761` (MathIR) ended their latest allocations
in scheduler state `TIMEOUT` at 18:14:40 and 18:09:40 EDT, respectively.  They
were not automatically returned to the queue despite `Requeue=1`.

Issue exactly one `scontrol requeue` for each original job ID.  Do not reset
state, submit replacements, change run directories, or alter environments.
On the next allocation, the frozen runtime-ops selector must choose the highest
checkpoint whose model and optimizer ZIP directories are complete.

This is scheduler/storage recovery only.  Model, optimizer updates, RNG
restoration, seed, data order, MaxEnt estimator/coefficient, ReplayDr
objective/weight, proposal policy, verifier, evaluation, and target steps are
unchanged.  PointMaze remains excluded.

The audit record must preserve TIMEOUT accounting, exact commands, successful
same-ID post-action scheduler records, no replacement jobs, no environment or
treatment change, and no endpoint outcome use.
