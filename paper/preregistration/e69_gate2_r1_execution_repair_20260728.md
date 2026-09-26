# E69 Gate 2 R1 execution-only repair

Date frozen: 2026-07-28, after all four replacement Graph cells had
terminated, while the replacement Python route-successor cell had accumulated
17 scheduler restarts without reaching its first durable checkpoint, while
both MATH-dev cells remained nonterminal, and before Gate 3 or any MATH-500
evaluation.

## Why an R1 is required

Gate 2's scientific configuration is unchanged, but the accepted physical
attempts are not clean enough for the frozen integrity gate. Graph jobs
`30160592`, `30160594`, `30160595`, and `30160596` terminated after unequal
scheduler histories (respectively 7, 4, 2, and 0 recorded restarts). Python
route-successor job `30160205` accumulated 17 restarts and repeatedly reset
before its first step-384 checkpoint.

These are execution-integrity failures. They are not scientific regressions
and are not whitelisted. The existing Graph and Python traces remain archived
as excluded infrastructure provenance and are never spliced into R1.

Terminal Graph outcomes and partial Python telemetry were available before
this document was frozen. They did not determine the replacement cells,
settings, seed, stopping rule, or gate. The repair rule is symmetric and
execution-only: replace the complete four-arm Graph cohort and the single
corrupted Python route-successor cell.

## Frozen R1 cohort

Submit exactly five fresh seed-43 jobs:

1. Graph coloring compute-matched Dr.GRPO;
2. Graph verified first-global replay;
3. Graph verified entropy-gated singleton escape;
4. Graph verified route successor; and
5. Python factors verified route successor.

All five jobs use the already frozen E69 source snapshot
`b4075dbaababd0c972cc1ad4df8da34d4271274f21d7859ec9e477dc1c511976`
and execution snapshot
`c8aa9471d5af05264d2cc04338612f9f2e16a10af34c5ebaea67ac6aa6ddb41a`.
Model, data, seed 43, arm definitions, 16 samples, learning rate `2e-7`,
three proposal-control groups, one replay group, replay/proposal settings,
six-pass stopping rule, checkpoint cadence, evaluation surface, CPU, memory,
and three-day limit remain unchanged.

The only placement change is to the non-low-priority `all` partition on the
homogeneous A5000 pool `node105,node202,node203,node204`, requesting one
`gpu:a5000` per job under account `mltheory`. Cross-arm accelerator class is
matched.

Jobs are submitted held. Before release, the launcher audits all five Slurm
records for the exact frozen source, execution, protocol identity, seed,
six-pass settings, fixed control/replay budgets, A5000 request, node pool,
partition, and account. It then writes an atomic one-to-one repair identity,
cancels only corrupted Python job `30160205` and the obsolete Gate-2
transition job, registers a new fail-closed transition dependent on both
MATH-dev cells and all five R1 cells, and releases the R1 jobs together.

The four completed Graph source jobs are not deleted or canceled; the R1
identity excludes them wholesale.

## Temporal route evidence

The already prospective, clean Countdown and MathIR checkpoint observations
remain valid parent evidence. Their temporal reproduction counts are carried
forward with exact artifact hashes. Graph and Python observations start empty
and must be rebuilt solely from the fresh R1 successor jobs at exact passes
1--6. No prior Graph or Python snapshot can enter the R1 temporal artifact.

The mechanism gate remains unchanged: at least three ModeBench domains must
show a nonzero conservative post-replay neutral-reproduction lower bound.
Python is the pivotal intended third domain; Graph can also satisfy the
unchanged gate, but no domain is whitelisted.

## Advancement rule

The two existing MATH-dev jobs continue untouched. Gate 3 remains sealed until
all 18 effective Gate-2 physical cells are terminal, every integrity check
passes, every exact route checkpoint is present, the frozen outcome gate
passes, and the resulting audit status is `pass`.

In particular, Python must supply both a clean terminal R1 result and nonzero
temporal route reproduction before advancement. A failed or zero-reproduction
Python R1 does not authorize Gate 3. MATH-500 remains sealed.
