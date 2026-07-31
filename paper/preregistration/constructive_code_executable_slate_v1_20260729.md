# ConstructiveCode executable slate v1

Status: v1 admission failed; row ineligible; no policy sample  
Frozen: 2026-07-29  
Compatibility amendment: 2026-07-29, before admission replay or policy sampling
Outcome recorded: 2026-07-29, after full frozen replay

## Outcome

The full run replayed all 794 frozen programs across 13,898 candidate-test
executions. Throughput passed (0.068 s median, 0.203 s p95), but admission
failed. The slate contained 242 `py2` programs despite its Python 3.10 runtime,
and checker-equivalence criteria failed for 482A, 1153B, and 149C; only 988A
passed. This outcome does not authorize filtering or task substitution. A
Python-3-only attempt requires a separately frozen v2 before any execution.

## Selected tasks

One task is selected prospectively from each implemented witness family:

| Family | Problem | Adapter | Overlay tests | Correct / incorrect replays |
|---|---|---|---:|---:|
| ordered sequence | Codeforces 482A, Diverse Permutation | `fixed_integer_sequence_v1` | 55 | 100 / 100 |
| unordered set | Codeforces 988A, Diverse Team | `status_integer_set_v1` | 43 | 100 / 100 |
| assignment | Codeforces 1153B, Serval and Toy Bricks | `matrix_assignment_v1` | 11 | 100 / 94 |
| unordered partition | Codeforces 149C, Division into Teams | `two_group_partition_v1` | 56 | 100 / 100 |

The 94 incorrect assignment replays are all unique labelled Python programs
available in the bounded source payload; the gate requires all 94 rather than
silently duplicating or replacing them.

## Semantic identity

- 482A preserves the ordered permutation.
- 988A collapses selected-index order and represents an accepted infeasible
  `NO` as a fixed sentinel set.
- 1153B inserts row-major cell-position keys, then collapses only pair order.
- 149C collapses member order and team order while preserving the two groups.

Every adapter is derived from the input and task identity, never from
model-authored metadata. The exact raw program output remains SHA-256 bound to
the released checker decision before the adapter view is canonicalized.

## Execution and equivalence gate

- testlib repository: `MikeMirzayanov/testlib`;
- commit: `1e4e8a24c79c6bad3becbdb5a332ffc352b7d5dd`;
- `testlib.h` SHA-256:
  `bb323e3c89285214966076e0d23d5a295c5f6126da7ff198c1276ddb95ecb1a0`;
- checker compiler: GCC 11.5.0, C++17, `-O2 -pipe`;
- candidate runtime: the already audited Python 3.10.20 SquashFS image and
  Landlock/seccomp launcher;
- per execution: 3x the source Python CPU limit rounded up to a whole second,
  wall time equal to that CPU limit plus two seconds, the source memory limit,
  16 MiB per captured output stream, 32 descriptors/files, and 256 KiB source;
- trusted checker wall timeout: 2 seconds; and
- empty answer file, because all selected checker sources are independently
  audited not to read `ans`.

The initial 5+5 diagnostic is not an admission sample. It exposed that a valid
`1208_C` output can exceed 4 MiB and that known-correct Python programs can
consume the source's base-language time allowance. The amendment applies the
same 16 MiB output ceiling and 3x Python CPU multiplier to every task, without
changing checker logic, adapter identity, replay ordering, thresholds, or the
64 KiB isolation-smoke attack. It was frozen before the first 20+20 admission
replay and before any model sample.

For each task, execute every selected program on every frozen CodeContests-O
input unless an earlier candidate failure makes the whole suite rejected.
Require:

- wrapper/released-checker decision equality for every replay;
- zero false accepts relative to the released checker;
- at least 0.90 true-positive and true-negative rates;
- at least two accepted behavior keys per task;
- exact input, output, checker-source, checker-binary, adapter-source,
  runtime-image, and launcher hashes; and
- median candidate launch no more than 0.50 seconds and p95 no more than 1.0
  second over the full replay.

This first executable gate covers the aligned CodeContests-O overlay. The
selected tasks remain ineligible until the CodeContests+ 5x suite is
materialized and passes the same decision/key audit. A passing executable gate
still does not authorize model sampling; split freeze and the 0.5B coder
viability preregistration follow.
