# ConstructiveCode v4 Qwen2.5-Coder-0.5B viability

**Status: FROZEN BEFORE THE V4 EXECUTABLE-GATE OUTCOME OR ANY MODEL SAMPLE — 2026-07-30**

## Conditional authorization

This development-only probe launches only if
`var/artifacts/constructive_code_v4_gate_audit.json` reports `status=pass`,
all 12 frozen tasks admitted, 3,072/3,072 expected suite replays terminal, and
no hard execution, checker-equivalence, identity, timeout, isolation, output,
or latency violation. A failed v4 gate stops this protocol without sampling a
model. Passing authorizes only a separately frozen paired ConstructiveCode
online-training smoke; this probe is not a paper seed.

## Frozen model, rows, and execution

- model: local Qwen2.5-Coder-0.5B-Instruct snapshot
  `ea3f2471cf1b1f0db85067f1ef93848e38e88c25`;
- development tasks, in order: 359B, 988A, 1283C, and 1399D;
- evaluation tasks 361B, 1294C, 1408A, and 149C are never loaded;
- each task uses the primary complete checker suite selected by the v4 gate:
  CodeContests-O when admissible, otherwise Plus-5x; and
- candidates run in the exact hash-pinned Python 3.10 networkless
  Landlock/seccomp worker used by the v4 executable gate.

The v4 source slate differs from v3 only in its pre-outcome source-selection
rule: 64 explicitly Python-3 correct and 64 explicitly Python-3 incorrect
submissions are selected per task from the full pinned source; any overlap
with the older v1 audit ledger is reported rather than excluded.

## Frozen prompt and sampling

The system message is: “Write a complete Python 3 program that solves the
problem. Return only the program source, without Markdown fences or
explanation.” The user receives only the unchanged public statement and
public input/output specification. No reference program, candidate witness,
canonical key, checker source, test, reward, suite outcome, or gate statistic
enters context.

- 64 independent samples per task; first 16 form the frozen prefix;
- seed `77101`, request seed
  `77101 + 10000 * row_index + sample_index` for zero-based indices;
- temperature 1, top-p 1, no top-k truncation;
- at most 1,024 generated tokens; no best-of, continuation, or repair; and
- strip only an exact surrounding bare/Python Markdown fence, then execute
  exactly once without feedback, retry, or resampling.

## Pass criterion and failure policy

The probe passes only if all 256 requests and terminal worker records exist,
there are no hard execution or identity violations, at least one task has an
accepted candidate in its first 16 samples, and at least one task has two
checker-accepted canonicalizer-distinct behavior keys across all 64 samples.
Ordinary syntax, runtime, and wrong-answer outcomes are verified negatives;
timeouts, isolation/output violations, checker-wrapper disagreement, missing
suite identity, nonfinite latency, or missing records fail the probe.

The receipt records candidate/request hashes, checker and behavior outcomes,
task-level prefix/full counts, model/tokenizer, source/gate/execution
identities, runtime latencies, and proof that no evaluation row was loaded.
No threshold, prompt, model, task, seed, token budget, or sample-count change
is allowed after the outcome.
