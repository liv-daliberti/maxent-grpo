# ConstructiveCode v6 admitted-slate curation and executable gate

Date frozen: 2026-07-30, before any ConstructiveCode language-model sample.

ConstructiveCode v5 remains a failed experiment. Its immutable 2,304-record
dual-suite gate admitted 10 of 12 tasks but rejected 1208C and 1408A; the
CodeContests-O suite for retained task 1399D also exceeded the frozen p95
latency limit. V6 is a new benchmark version, not a relabeling or repair of
v5. It performs benchmark curation from the complete pre-model v5 executable
evidence. The v5 audit, identity, replay manifest, checker-equivalence audit,
raw replay ledger, and all v5 failure records remain bound into the v6 gate.

## Frozen task and suite slate

V6 retains exactly the ten v5 tasks with at least one fully admissible official
checker suite. It freezes CodeContests-O when that suite passed every source,
checker-equivalence, execution, isolation, output, timeout, median-latency, and
p95-latency check; otherwise it freezes Plus-5x. This rule was applied to the
complete v5 outcomes before model sampling and yields:

| Task | Witness family | Frozen suite | Split |
|---|---|---|---|
| 327B | ordered_sequence | CodeContests-O | train |
| 659C | unordered_set | CodeContests-O | train |
| 1283C | assignment | Plus-5x | train |
| 1102B | unordered_partition | CodeContests-O | train |
| 359B | ordered_sequence | CodeContests-O | development |
| 988A | unordered_set | CodeContests-O | development |
| 1399D | unordered_partition | Plus-5x | development |
| 361B | ordered_sequence | CodeContests-O | evaluation |
| 1294C | unordered_set | CodeContests-O | evaluation |
| 149C | unordered_partition | CodeContests-O | evaluation |

The problem-ID-disjoint split is therefore 4 train / 3 development / 3
evaluation. Training contains one task from every frozen witness family. The
single executable-admitted assignment task is assigned to training; v6 makes
no claim that every split contains every family.

## Independent gate replay audit

The v6 audit does not trust or copy v5 task-status labels. It reloads the raw
v5 replay manifest, raw JSONL execution ledger, and checker-equivalence audit,
then reruns the original executable-gate checks independently for each of the
ten frozen task-suite pairs. The exact gate contains 960 records: 10 tasks ×
48 known-correct programs × one suite plus 10 tasks × 48 known-incorrect
programs × one suite. The same program hashes, released checker sources,
official tests, Python 3.10 runtime, sandbox, thresholds, and limits are reused;
no replay is rerun or replaced. Reuse is appropriate because these immutable
records were collected before any language-model sample and their source and
execution identities are hash-bound.

Admission requires 960/960 uniquely identified records, exact 48/48 labels per
task, checker-wrapper agreement, checker-equivalence pass, valid execution
records, no timeout/isolation/output violation, median candidate invocation at
most 0.5 seconds, p95 at most 1.0 second, and a pass for every frozen task-suite
pair. It also requires that v5 remain an exact failed 2,304-record gate with
the excluded task set exactly {1208C, 1408A} and that no v5 viability receipt
or any v6 model-derived artifact exists when the v6 gate is materialized.

A pass authorizes only the separately frozen development-only
Qwen2.5-Coder-0.5B viability probe. Evaluation task files remain unloaded and
no language-model sampling occurs in this gate.
