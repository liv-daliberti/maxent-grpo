# E119 Pantry ReplayDr.GRPO seed44 recovery — September6 UTC / September5 EDT

The user explicitly requested repair of job31037832 after its latest completed
update193 recorded3718.878 seconds for weight synchronization (about62minutes).
The preceding rollout took roughly23minutes. Two bounded live-node probes could
not be admitted. Thus the current cause is not proven; the96-GiB restart is a
conservative operational intervention motivated by severe degradation and the
related campaign memory-pressure observations, not a claim of diagnosed cause.

Use the same Slurm job ID. Requeue into a transaction-owned hold, validate the
stopped writer’s checkpoint192, change only MinMemoryNode from64 to96GiB, audit,
and release. Keep effective partitioncs/accountallcs, A6000 routesnode205/207,
CPU/GPU counts, QOS/nice, full immutable SubmitLine, scientific exports, models,
seeds, objectives, datasets, optimization, evaluation, and terminal budget.
Both scientific and continuation ledgers remain byte-identical.

Independent validation found checkpoint192 structurally valid, with three saved
progress counters at192 and91 valid replay exemplars across66 prompts. The
model metadata hash is checked before and after stopping the learner so the
validated replay state is unchanged. At preparation, only one logged update
will repeat; the executable allows at most16 and forbids initialization restart.
No partial/newer checkpoint exists at preparation; a changed save requires
refreshed review. Preserve all stdout/stderr/metrics before stopping and after
cleanup in separate archive directories. Allow600seconds for normal Slurm
cleanup, preserving the owned hold if cleanup requires further inspection.

Current memory commitments onnode205/207 do not admit96GiB immediately even
after this64-GiB job stops. Ordinary queuing is expected; do not lower the request
to fit, change GPU class, or take over another user’s allocation. No broader
placement or experiment change is included. The previous restart count6 becomes7,
within the existing watchdog’s12-restart allowance.

Evidence, plan, reviews, archived diagnostics, and receipt are under
`var/artifacts/e119_pantry_s44_memory96_20260906/`.
