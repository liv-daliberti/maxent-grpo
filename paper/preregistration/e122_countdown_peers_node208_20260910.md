# E122 Countdown peers: node208 routing — September 10, 2026

The user requested accelerating E122 today while E118/E119/E120 finish.
The earlier first-cell amendment successfully started Dr.GRPO seed43 on node208.
The remaining released first-seed jobs31158679 (ReplayDr.GRPO),31158680 (MaxRL),
and31158681 (ReplayMaxRL) remain pending on the original node205/206/207/302 pool.
This additive operational amendment expands those exact three existing released
jobs to include the recovered node208, preserving their original pool too.

Preserve each job ID, immutable plan and ledger, all scientific/runtime exports,
source/data/model, seed,3072-step horizon,8CPUs,128GiB host RAM,one GPU,36-hour
walltime,allcs/lowprio/medium,nice0,exclusions,requeue,checkpoints and evaluation.
Preserve the existing release controller and its cap of four unfinished slots.
No hold, release, requeue, submission or controller-lock operation is authorized
by this routing script. The existing active cell is unaffected.

Authenticate the frozen campaign and successful release ownership before
preparation, record scheduler feasibility with sbatch --test-only, and bind the
script/protocol/dependencies/frozen files by hash. Immediately before each
same-ID pending route update, require unchanged scheduler identity, resources,
SubmitLine, exports and sole writer, plus healthy A6000 node208 in lowprio.
Available scheduler memory is recorded; it does not authorize oversubscription.
The scheduler admits only jobs whose unchanged128GiB request fits, so the
expanded pool remains useful as existing allocations finish even when all three
peers cannot fit simultaneously.

Persist an intent before the single scontrol update and its acknowledgement
before readback. Preserve allocations that start during preparation. Audit the
exact approved node pool and every preserved field. An uncertain command is
reconciled without blind retry. Store each peer's plan/transaction separately.
The original released-job reservations remain valid; current held requests are
untouched and the frozen controller continues to authenticate them on release.
