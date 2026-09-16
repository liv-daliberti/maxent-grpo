# E112-R1 final-analysis A2 greedy-trace contract closure

Frozen: 2026-08-25T11:02:51-04:00 while E112-R1 is 49/75 terminal,
E109 is 13/15 terminal, and the complete official E112 result does not exist.
No endpoint metric was inspected to make this amendment. Only non-metric
greedy-trace metadata and file presence were read.

This extends the fail-closed metadata principle in final-analysis A1 to the
registered greedy pass@1 evaluation. It changes no endpoint calculation,
contrast, aggregation, interval, or decision rule.

## Trigger

The final builder computes greedy pass@1 from the registered
`<step>_multi_answer.json` score files. Each evaluation also emits a
`deterministic_greedy_trace_neutral` JSONL row that records how the score file
was generated, but the builder previously ignored that trace. Consequently,
score-file presence alone did not prove the registered neutral greedy
benchmark, seed, sample count, schema, and temperature.

## Exact greedy-trace contract

For every treatment and comparator trajectory, every registered checkpoint
must contain at least one trace row with exactly:

- `evaluation_kind = deterministic_greedy_trace_neutral`;
- `benchmark = multi_answer`;
- `draw_index = null`;
- `sample_count = 1`;
- `schema_version = 1`;
- `seed = 0`; and
- `temperature = 0.0`.

Missing registered trace steps or any metadata drift is a hard error before
the existing greedy score-file reader runs. Exact duplicate retry traces are
allowed; a conflicting trace fails. Off-grid traces remain outside the frozen
17-checkpoint analysis.

The shared endpoint reader performs this optional trace validation. E112
always supplies the contract; older callers retain their prior behavior. The
official E112 result emits the greedy contract and binds this amendment by
path and SHA-256 digest.

## Boundary

This validates evaluation identity, not efficacy. It does not change the
greedy score calculation or address E112's bundled historical-comparator
estimand, prior private looks, four-draw evaluation Monte Carlo limitation, or
derived adjusted-breadth decision rule.
