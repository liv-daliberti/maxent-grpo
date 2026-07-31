# ConstructiveCode v2 replay r1: frozen-snapshot relocation repair

**Status: FROZEN BEFORE R1 CANDIDATE EXECUTION — 2026-07-29**

## Antecedent

Materialization job 30184636 completed all four CodeContests+ shards and all
four overlay shards, producing the four-task v2 slate. Dependency job 30185057
then submitted executable gate job 30185883. That job failed after two seconds,
before any candidate execution or replay output, with
`ConstructiveCode v2 v1 exclusion root drift`.

The cause is isolated: the materialized manifest names the immutable exclusion
ledger by its repository-logical path
`var/data/constructive_code_review_slate_v1`, while the launcher correctly
copies that ledger under a content-addressed execution snapshot and passes the
copied absolute path. The loader compared those two physical paths.

## Only permitted repair

R1 may change only the root-identity preflight:

1. require the manifest's logical ledger path to equal exactly
   `var/data/constructive_code_review_slate_v1` and to be relative;
2. read exclusions from the launcher-supplied frozen copy;
3. retain the existing per-task task-record hash, replay-ledger hash, and
   submission-code hash checks; and
4. retain the launcher's whole-tree SHA-256 of that copied v1 ledger.

All selected tasks, source revisions, candidate hashes, known labels, checkers,
input suites, Python 3.10 runtime image, testlib, sandbox, worker count, replay
counts, equivalence thresholds, and pass/fail criteria remain unchanged. No
failed-r0 output exists to select on, and no model is sampled.

## R1 decision

R1 executes the original 1,600 submission-suite replays. Pass only under the
original v2 protocol and audits. Any semantic or checker failure is final for
this slate. Another pre-execution infrastructure failure may be diagnosed but
cannot relax an executable criterion or substitute a task.
