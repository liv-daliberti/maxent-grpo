# ConstructiveCode v4 executable slate and dual-suite gate

**Status: FROZEN AFTER V3 JOB 30187935 STOPPED IN MATERIALIZATION AND BEFORE V4 MATERIALIZATION OR MODEL SAMPLING — 2026-07-30**

## Immutable antecedent

V3 preserved replay disjointness from the earlier v1 source-review slate and
required 100 held-out explicit-Python-3 programs per label. It stopped before
creating a slate because task 359_B had only 31 held-out correct Python-3
programs. No candidate program, official checker, or language model was
executed. The v3 failure remains immutable and is not relabeled.

Replay disjointness is not required for this executable checker-admission gate:
the programs are fixed known-correct/known-incorrect probes, not training or
evaluation examples for the later Qwen model. V4 therefore selects from the
full pinned CodeContests+ submission source and reports any overlap with the
sealed v1 ledger rather than excluding it.

## Frozen slate

V4 retains the exact v3 12 task IDs, four witness families, adapters, and
problem-ID-disjoint 4/4/4 train/development/evaluation assignment. For every
task and known label, it selects exactly 64 unique nonempty submissions whose
explicit source label is `py3`, `python3`, or `pypy3`, ordered by raw-code
SHA-256. Selection never depends on execution outcome. Every selected program
is replayed against both pinned official suites: CodeContests-O corner cases
and CodeContests+ 5x. The complete gate is 12 × 2 labels × 64 programs × 2
suites = 3,072 execution records.

The pinned source revisions, candidate index, checker sources, testlib header,
Python 3.10.20 Apptainer image, sandbox limits, canonicalizers, and released
checker equivalence audit remain unchanged. Source access is permitted only
for materialization; execution is networkless. No evaluation split is exposed
to a language model during this gate.

## Pass and authorization boundary

For each task, at least one of the two preregistered official suites must meet
the unchanged v3 correctness, incorrect-rejection, checker-equivalence,
canonicalization, isolation, timeout, output-bound, and throughput criteria.
The passing suite is frozen before model sampling. A pass authorizes only split
construction and one development-only Qwen2.5-Coder-0.5B viability probe. A
failure stops ConstructiveCode; no task, program, suite, threshold, split, or
canonicalizer is replaced after observing execution.
