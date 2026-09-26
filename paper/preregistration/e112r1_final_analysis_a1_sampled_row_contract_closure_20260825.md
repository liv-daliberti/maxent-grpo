# E112-R1 final-analysis A1 sampled-row contract closure

Frozen: 2026-08-25T10:57:34-04:00 while E112-R1 is 49/75 terminal,
E109 is 13/15 terminal, and the complete official E112 result does not exist.
No additional endpoint metric was inspected to make this amendment. The audit
read only scheduler state, file presence, and non-metric sampled-row metadata.

This is a fail-closed validation and provenance amendment. It changes no cell,
comparator, checkpoint, draw, endpoint, contrast, aggregation, interval, or
decision rule frozen in the E112 paired-analysis specification.

## Trigger

The frozen launch ledgers record the intended neutral sampled-K evaluation
configuration, and the emitted JSONL rows record their realized benchmark,
sample count, schema version, draw seed, and temperature. The shared endpoint
reader previously required the right evaluation kind, registered step,
draw index, and finite metric payload, but did not compare those additional
row fields with the registered contract. A row generated with a drifted seed,
K, temperature, benchmark, or schema could therefore enter the official table
despite a correct launch export.

The E112 builder also imported its endpoint reader from the E105 builder while
hashing only the E112 wrapper. A later change to the shared reader would not
have been visible in final-result provenance.

## Exact sampled-row contract

For every treatment and comparator row at every registered checkpoint and
draw, the E112 builder now requires:

- `evaluation_kind = fixed_seed_sampled_k_neutral`;
- `benchmark = multi_answer`;
- `sample_count = 8`;
- `schema_version = 1`;
- `temperature = 1.0`; and
- `seed = registered domain seed base + draw_index`.

The registered seed bases are:

- Graph Coloring: 610100
- Countdown: 610200
- Python Factors: 610300
- MathIR: 610400
- Pantry Plan: 76299

The same values are present in all E112 treatment exports and the exact E78,
E79, E80-R1, and E109 comparator exports. Contract drift is a hard error; the
builder writes no result.

## Shared implementation binding

The optional contract check lives in the single shared E105 endpoint reader,
so the extraction algebra remains unified. E112 always supplies the exact
contract above. Existing callers that do not supply a contract retain their
previous behavior.

The official E112 result provenance now records the path and SHA-256 digest of
the shared endpoint builder as well as this amendment. The result also emits
the complete sampled-row contract in its machine-readable metric contract.

## Interpretation boundary

This closure increases confidence that registered evaluations were actually
generated as declared. It does not repair E112's bundled historical-comparator
estimand, restore confirmatory blinding, add evaluation Monte Carlo draws, or
alter the registered derived adjusted-breadth decision. Those limitations
remain disclosed, and component attribution remains assigned to E117 C/P/F.
