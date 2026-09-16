# E112-R1 second user-requested private interim unblinding

Frozen on 2026-08-23 before reading any E112-R1 task-evaluation endpoint that
was not already included in the 2026-08-20 fourteen-cell private freeze. This
amendment responds to the explicit request to assess what the currently
available corrected verified-support MaxEnt results say.

## Repeated-look deviation

The original E112 analysis specification requires all 75 treatment cells and
all 75 exact comparators before emitting efficacy results. The first private
look already broke continuous confirmatory outcome blindness. This second look
is an additional sequential unblinding and must be disclosed in any eventual
E112-R1 report. It does not restore confirmatory status and must not be
described as a preregistered interim analysis.

No result from either private look may be used to cancel, reprioritize,
relaunch, retune, select, or otherwise change an E112-R1 job or its comparator.
The scientific treatment, comparator mapping, seeds, domains, horizons,
evaluation procedure, scheduler placement, checkpointing, and terminal
analysis remain unchanged.

## Outcome-blind membership freeze

Membership is every E112-R1 ledger cell possessing a valid
`TRAINING_COMPLETE.json` marker when the new freeze artifact is written. The
selection step reads only the immutable ledger and completion markers, not an
evaluation log. The artifact records the exact model scale, domain, seed, job
ID, run directory, terminal step, and SHA-256 of every admitted marker.

Each treatment cell is paired to its preregistered ReplayDr.GRPO comparator.
The private output reports exact paired seed effects and exact `n`. It reports
no pooled effect, hypothesis test, stopping decision, or model-scale trend.
Incomplete five-seed blocks receive no mean or uncertainty interval. The
official complete-data decision rules remain unevaluated until 75/75.

## Output boundary

The output must remain under `var/artifacts/private_interim/`, visibly labeled
`PRIVATE EXPLORATORY INTERIM — NOT FOR PAPER OR SELECTION`. Its JSON must bind
the new freeze, treatment ledger, comparator ledgers, completion markers, and
evaluation logs by SHA-256. It is not copied into `paper/figures/` and cannot
supply a paper efficacy point. The manuscript may update only the operational
terminal/running/pending count and disclose that a second private look occurred.

PointMaze is excluded. The campaign continues under its existing registered
execution plan regardless of the signs observed here.
