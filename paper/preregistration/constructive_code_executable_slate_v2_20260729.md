# ConstructiveCode executable slate v2

Status: frozen before v2 source materialization or replay  
Frozen: 2026-07-29

## Separation from v1

ConstructiveCode v1 remains failed and ineligible. Its source and replay
outcomes are exploratory development evidence for this separately versioned
attempt. V2 fixes the runtime-language contract prospectively and uses fresh
submission hashes.

## Frozen tasks

One task is selected from each family that passed every criterion in the
antecedent 5-correct/5-incorrect diagnostic:

| Family | Problem | Adapter |
|---|---|---|
| ordered sequence | Codeforces 327B, Hungry Sequence | `fixed_integer_sequence_v1` |
| unordered set | Codeforces 1294C, Product of Three Numbers | `multi_case_status_integer_set_v1` |
| assignment | Codeforces 1283C, Friends and Gifts | `implicit_assignment_v1` |
| unordered partition | Codeforces 1102B, Array K-Coloring | `status_label_partition_v1` |

No task substitution is allowed after v2 materialization.

## Frozen source and replay selection

- CodeContests+ revision:
  `96c850540fade31d384a25766461e0da6b08f5fc`;
- CodeContests-O revision:
  `1a765191567b429f633bbd1c6e67b5890dfaf267`;
- candidate index:
  `var/artifacts/constructive_code_candidate_source_index.json`;
- accepted language labels: exactly `py3`, `python3`, or `pypy3` after lowercase
  normalization;
- reject every exact code SHA-256 already present for that task in
  `var/data/constructive_code_review_slate_v1`;
- sort remaining unique hashes lexicographically within known label; and
- freeze the first 100 correct and first 100 incorrect programs per task.

Materialization fails rather than lowering counts, admitting an ambiguous
language label, or reusing a v1 hash.

## Frozen executable gate

Each of 800 programs is run in the pinned Python 3.10
Landlock/seccomp/networkless worker against:

1. the complete frozen CodeContests-O corner-case suite; and
2. the complete frozen CodeContests+ 5x input suite.

The released checker and canonical adapter inspect every accepted output. For
each task and each suite require:

- all 100 known-correct and all 100 known-incorrect replay decisions;
- at least 0.90 true-positive and true-negative rates;
- wrapper/released-checker decision equality for every replay;
- at least two accepted correct behavior keys;
- zero identity, hash, timeout, isolation, or output-bound violations; and
- median candidate execution no more than 0.50 seconds and p95 no more than
  1.0 second.

The joint task passes only if both suites pass. Failure stops v2. A passing
result authorizes split construction and Qwen2.5-Coder-0.5B development-only
viability sampling, not training.

