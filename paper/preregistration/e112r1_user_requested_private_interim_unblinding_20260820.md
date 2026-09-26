# E112-R1 user-requested private interim unblinding

Frozen on 2026-08-20 before reading any E112-R1 task-evaluation endpoint.
This amendment responds to the explicit request for a private view of the
currently available E112-R1 subset.

## Deviation from the original analysis freeze

The original E112 protocol prohibited inspection of any E112 or E109 task
outcome before the full paired cohort was ready.  This private interim look
breaks uninterrupted analyst outcome blindness.  The eventual paper and
confirmatory result record must disclose that fact; they must not describe the
E112-R1 efficacy analysis as continuously outcome-blind through cohort
completion.

The scientific treatment, comparator mapping, seeds, domains, horizons,
evaluation procedure, scheduler placement, checkpointing, and terminal
analysis remain unchanged.  No running or pending job may be canceled,
reprioritized, relaunched, retuned, or otherwise selected using this interim
look.

## Frozen selection rule

The private subset is every E112-R1 ledger cell possessing a valid
`TRAINING_COMPLETE.json` marker at the instant the freeze artifact is written.
Membership is selected only from completion markers and the immutable released
ledger, before any evaluation file is opened.  The freeze artifact records the
exact model, domain, seed, job ID, run directory, terminal step, and SHA-256 of
every admitted completion marker.

Each admitted treatment cell is paired to its preregistered ReplayDr.GRPO
comparator.  The plot reports exact paired seed points and exact `n` only.  No
mean, interval, hypothesis test, stopping decision, or cross-cell pooling is
permitted for an incomplete five-seed block.

## Output boundary

The interim output is exploratory and private.  It must be written under
`var/artifacts/private_interim/`, never under `paper/figures/`, and visibly
labeled `PRIVATE EXPLORATORY INTERIM — NOT FOR PAPER OR SELECTION`.  Its data
artifact must bind the freeze, treatment ledger, comparator ledgers,
completion markers, and exact evaluation logs by SHA-256.

PointMaze is excluded.  The interim results may be used only to understand how
the running cohort currently looks; they may not affect execution or the
registered final analysis.
