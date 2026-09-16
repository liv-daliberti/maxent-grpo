# E112-R1 final-analysis A7 response-free identity erratum

Recorded: 2026-08-28, after the author-requested private 14-, 33-, and
50-terminal-cell looks and after the first two-scale public builder attempt
failed closed. This correction cannot restore confirmatory outcome blindness.

## Defect

A3 described `answer_keys` as response-free and included it in the prompt
identity digest. The evaluator actually constructs `answer_keys` from each
generated completion after model inference. It is therefore an outcome field:
it should differ across evaluation draws and may differ between treatment and
comparator whenever their sampled answers differ. Hashing it makes paired
identity depend on the efficacy outcome being measured.

A3 also omitted the registered row-level sampled seed from the request digest.
Some Graph rows predate nested per-option seed recording and therefore store an
empty `request_seeds_by_option` list. Their distinct registered row seeds still
identify the four executed requests and are already checked against the frozen
draw contract.

The failure was diagnosed at Qwen2.5-0.5B Graph seed 43. Across its complete
17-checkpoint by four-draw grid, the original digest changed. Removing only
`answer_keys` produced one invariant prompt digest, one invariant request
digest per draw, four distinct request surfaces, and exact treatment/comparator
identity across all 68 rows.

## Corrected executable projection

The ordered prompt projection now contains only:

- `answer_mode_count`, which is derived from the frozen reference answer;
- `option_ids`;
- `prompt`;
- `prompt_index`; and
- `reference`.

The separate request digest now hashes the registered row-level sampled seed
together with the ordered `option_ids`, `prompt_index`, and
`request_seeds_by_option` projection. The sampled-row contract, prompt ordering,
prompt count, checkpoint grid, draw grid, treatment/comparator binding,
endpoints, intervals, and decision rules are unchanged. A regression test varies
`answer_keys` over steps and draws while requiring the response-free identity
to remain invariant; another covers empty nested seed metadata.

## Reporting boundary

The corrected audit must still pass all frozen pairs before a two-scale result
is written. The 50-cell result, if emitted, is an author-requested exploratory
bundled historical-comparator contrast. This post-outcome integrity correction
does not authorize a confirmatory claim, a component-isolated claim, any
Qwen2.5-3B estimate, or any campaign mutation from the observed endpoints.
