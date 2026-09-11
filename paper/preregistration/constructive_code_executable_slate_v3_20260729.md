# ConstructiveCode executable slate v3

Status: frozen before v3 source materialization, replay, or model sampling  
Frozen: 2026-07-29

## Antecedent evidence and purpose

V1 failed its Python-runtime contract. V2 repaired that contract and completed
all 1,600 frozen replays, but its joint-dual-suite rule failed for 1283C on the
CodeContests-O suite (TNR 0.83) and 1294C on the Plus-5x suite (TNR 0.89).
Released-checker and wrapper decisions agreed on every v2 replay; there were no
timeouts, isolation violations, output-bound violations, or latency failures.
No language model has been sampled on any ConstructiveCode task.

V3 is a prospective benchmark-construction attempt, not a repair of either
stopped cohort. It has two purposes: apply the extension plan's declared
"CodeContests-O overlay, Plus-5x fallback" rule at task level, and create
problem-ID-disjoint train, development, and evaluation splits.

## Frozen tasks and splits

Exactly one task from every witness family is assigned to each split. These
assignments are frozen before v3 materialization and cannot be changed after a
replay or model outcome.

| Split | Family | Problem | Adapter |
|---|---|---|---|
| train | ordered sequence | 327B, Hungry Sequence | `fixed_integer_sequence_v1` |
| train | unordered set | 659C, Tanya and Toys | `counted_integer_set_v1` |
| train | assignment | 1208C, Magic Grid | `matrix_assignment_v1` |
| train | unordered partition | 1102B, Array K-Coloring | `status_label_partition_v1` |
| development | ordered sequence | 359B, Permutation | `fixed_integer_sequence_v1` |
| development | unordered set | 988A, Diverse Team | `status_integer_set_v1` |
| development | assignment | 1283C, Friends and Gifts | `implicit_assignment_v1` |
| development | unordered partition | 1399D, Binary String To Subsequences | `multi_case_label_partition_v1` |
| evaluation | ordered sequence | 361B, Levko and Permutation | `sentinel_integer_sequence_v1` |
| evaluation | unordered set | 1294C, Product of Three Numbers | `multi_case_status_integer_set_v1` |
| evaluation | assignment | 1408A, Circle Coloring | `multi_case_implicit_assignment_v1` |
| evaluation | unordered partition | 149C, Division into Teams | `two_group_partition_v1` |

The problem IDs are disjoint across splits. Evaluation tasks remain unavailable
to all model-viability decisions.

## Frozen source and replay selection

- CodeContests+ revision:
  `96c850540fade31d384a25766461e0da6b08f5fc`;
- CodeContests-O revision:
  `1a765191567b429f633bbd1c6e67b5890dfaf267`;
- candidate index:
  `var/artifacts/constructive_code_candidate_source_index.json`;
- accepted language labels: exactly `py3`, `python3`, or `pypy3` after lowercase
  normalization;
- exclude every exact code SHA-256 already present for the same task in the
  frozen v1 replay ledger;
- sort remaining unique hashes lexicographically within known label; and
- require the first 100 correct and first 100 incorrect programs per task.

Materialization fails rather than reducing a count, changing a task, reusing a
v1 hash, or admitting an ambiguous language label.

## Frozen executable gate

Every selected program is executed in the pinned Python 3.10
Landlock/seccomp/networkless worker against both complete frozen suites:

1. CodeContests-O corner cases; and
2. CodeContests+ 5x inputs.

For both suites of every task require:

- exactly 100 known-correct and 100 known-incorrect replay records using the
  same submission hashes across suites;
- wrapper/released-checker decision equality on every replay;
- zero identity, hash, timeout, isolation, and output-bound violations;
- median candidate execution at most 0.50 seconds and p95 at most 1.0 second.

A suite is equivalence-admissible only if it also preserves at least 0.90 TPR,
at least 0.90 TNR, and at least two accepted-correct behavior keys. A task is
admitted only when at least one of its two suites is equivalence-admissible.
Its primary suite is selected deterministically: CodeContests-O if admissible,
otherwise Plus-5x. A task that passes neither suite stops v3. A hard execution
or checker-wrapper equality failure on either suite also stops v3; the fallback
rule can waive only a suite's empirical label-rate or multi-key failure.

V3 passes only if all 12 frozen tasks pass. No task substitution is allowed.
A pass freezes the primary suite per task and authorizes split construction
plus development-only Qwen2.5-Coder-0.5B viability sampling. It does not
authorize evaluation sampling, main-cohort training, or a paper claim.
