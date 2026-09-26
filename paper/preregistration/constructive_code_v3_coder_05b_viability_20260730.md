# ConstructiveCode v3 Qwen2.5-Coder-0.5B viability

**Status: FROZEN BEFORE THE V3 EXECUTABLE-GATE OUTCOME OR ANY MODEL SAMPLE — 2026-07-30**

## Conditional authorization

This development-only probe may launch only if
`var/artifacts/constructive_code_v3_gate_audit.json` reports `status=pass`, all
12 frozen tasks admitted, 4,800/4,800 expected suite replays terminal, and no
hard execution, checker-equivalence, identity, timeout, isolation, output, or
latency violation. A failed v3 gate stops this protocol without sampling a
model.

The probe can authorize only a separately frozen paired ConstructiveCode
online-training smoke. It is not a final seed and cannot be reported as a
paper result.

## Frozen model and rows

- model: the complete local Qwen2.5-Coder-0.5B-Instruct snapshot
  `ea3f2471cf1b1f0db85067f1ef93848e38e88c25`;
- split: exactly the four v3 development tasks frozen before materialization:
  359B, 988A, 1283C, and 1399D, in that order;
- evaluation tasks 361B, 1294C, 1408A, and 149C are never loaded by the
  evaluator;
- each task uses only the primary complete checker suite selected by the
  passing v3 gate: CodeContests-O when admissible, otherwise Plus-5x;
- every candidate runs in the exact hash-pinned Python 3.10
  Landlock/seccomp/networkless worker used by v3.

## Frozen prompt and sampling

The system message is:

> Write a complete Python 3 program that solves the problem. Return only the
> program source, without Markdown fences or explanation.

The user message is the unchanged public problem statement followed by the
unchanged public input/output specification from the v3 row. No reference
program, witness, canonical key, accepted behavior count, checker source,
test input, expected output, suite-selection reason, reward, development
outcome, or evaluation artifact enters model context.

- 64 independent samples per task; the first 16 are the frozen prefix;
- seed `77101`, with request seed
  `77101 + 10000 * row_index + sample_index` for zero-based row and sample
  indices;
- temperature 1, top-p 1, no top-k truncation;
- maximum 1,024 generated tokens, no best-of or repair continuation;
- strip only an exact surrounding Markdown code fence if the model violates
  the no-fence instruction; otherwise execute the emitted text byte-for-byte;
- one execution per emitted candidate, no syntax repair, retry, resampling,
  feedback, or post-outcome prompt change.

## Frozen viability criterion

This is the integer realization of the extension plan's predeclared rule for
four development prompts:

1. at least one of four tasks has an accepted candidate in its first 16
   samples; and
2. at least one of four tasks has at least two checker-accepted,
   canonicalizer-distinct behavior keys across all 64 samples.

All 256 requests and all emitted candidates must be present. Every execution
must have a terminal worker record. Infrastructure errors, timeouts, isolation
violations, checker-wrapper disagreement, missing primary-suite identity,
nonfinite latency, or output-bound violations fail the probe rather than count
as ordinary incorrect programs. Ordinary syntax, runtime, or wrong-answer
failures count as verified negatives.

The receipt reports all candidate hashes, request seeds, checker decisions,
canonical keys, task-level prefix/full counts, execution latencies, model and
tokenizer hashes, gate/source/execution identities, and proof that no
evaluation row was loaded. Passing does not permit changing the Coder model,
task list, prompt, token budget, sample count, or threshold for later jobs.
