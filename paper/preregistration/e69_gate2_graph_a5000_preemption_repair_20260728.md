# E69 Gate 2 Graph preemption repair

Date frozen: 2026-07-28, after Graph control job `30160181` was preempted
before its first durable checkpoint, before any Graph Gate 2 terminal result,
and before any Gate 3 submission.

## Infrastructure failure

The matched Graph cohort initially ran on `node101` (`gpu:a40`). Slurm
preempted control job `30160181` at optimizer step 106 and automatically
requeued it with zero current runtime and
`ReqNodeNotAvail, May be reserved for other job`. The first durable checkpoint
is step 192, so that cell has no valid resume point. The three sibling arms
remain nonterminal.

Restarting only the control from zero beside continued sibling trajectories
would create asymmetric attempt history. Resuming the control is impossible
without splicing an uncheckpointed partial attempt. The entire four-arm Graph
cohort is therefore replaced together.

## Frozen replacement

Submit four fresh seed-43 Graph jobs, one per frozen Gate 2 arm, held on
`node202` with exactly one `gpu:a5000` each. Node202 had enough same-model
A5000 capacity for the complete matched cohort. After all four held jobs pass
an environment audit, cancel jobs `30160181--30160184` and release the four
replacements.

The replacement changes only accelerator placement. Source and execution
snapshots, model, dataset, seed, arms, rollout and fixed-control token budgets,
replay/proposal settings, optimizer, six-pass stopping rule, evaluation
surface, CPU, memory, partition, account, and time limit remain unchanged.
Every Graph arm uses the same accelerator model.

All partial attempts `30160181--30160184` are excluded wholesale. Their metrics,
evaluations, and checkpoints remain archived as infrastructure provenance and
are never combined with replacement traces. The repair identity records each
excluded attempt's latest optimizer step and the exact one-to-one mapping.

Some nonterminal evaluations were mechanically available when the
infrastructure failure occurred. No metric value, arm ranking, or outcome
informed this repair; it is triggered solely by the preemption, missing
control checkpoint, and scheduler/node inventory. No terminal Gate 2 outcome
or MATH-500 outcome was available.

## Gate 3 prospective placement

No Gate 3 job has been submitted. Future Graph jobs use the same A5000 pool
`node105,node202,node203,node204` already registered for Countdown and Python,
rather than the reservable single A40 node. Cross-arm accelerator class and
all scientific compute controls remain matched.
