# E112-R1 final-analysis A3 paired prompt-surface closure

Frozen: 2026-08-25T11:08:56-04:00 while E112-R1 is 49/75 terminal,
E109 is 13/15 terminal, and the complete official E112 result does not exist.
No response, reward, metric, or endpoint value was inspected to make this
amendment.

This adds an exact paired-evaluation identity check. It changes no treatment,
comparator, checkpoint, draw, endpoint, effect, interval, or decision rule.

## Trigger

The ledgers bind treatment and comparator run identities and intended dataset
paths, but paths alone do not prove that the emitted evaluations used the same
prompt and request surface. The sampled JSONL rows retain enough response-free
information to prove the pairing directly.

## Response-free identity projection

For every registered sampled row, compute a canonical SHA-256 digest of the
ordered prompt projection containing only:

- `answer_keys`
- `answer_mode_count`
- `option_ids`
- `prompt`
- `prompt_index`
- `reference`

Separately compute the request identity from:

- `option_ids`
- `prompt_index`
- `request_seeds_by_option`

The projection explicitly excludes `metrics`, `responses`, and `rewards`.
Canonical JSON uses sorted object keys, compact separators, UTF-8, and the
original prompt order.

## Fail-closed invariants

The final builder now requires:

1. one exact prompt count and prompt digest across all 17 checkpoints and all
   four draws within a trajectory;
2. one exact request digest per draw across all 17 checkpoints;
3. complete step-by-draw coverage with exact sampled-row metadata from A1;
4. exact equality of the complete prompt/request identity object between each
   E112 treatment trajectory and its registered ReplayDr comparator.

Conflicting retry rows, missing identity fields, grid drift, or paired mismatch
abort materialization before any official result is written. The final result
retains only the prompt count and SHA-256 digests, not prompt text or labels.

A metadata-only pilot on the completed Qwen-0.5B MathIR seed-47 pair produced
the same response-free SHA-256 digest on treatment and comparator. This pilot
does not substitute for the builder's required all-75-pair validation.

## Boundary

This proves paired evaluation identity, not causal treatment identity. E112
remains a bundled contrast against historical comparators and remains
non-confirmatory after private interim inspection. The registered four-draw
Monte Carlo limitation and derived adjusted-breadth decision also remain.
