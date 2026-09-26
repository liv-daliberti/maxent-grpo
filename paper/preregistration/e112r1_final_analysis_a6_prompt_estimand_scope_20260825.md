# E112-R1 final-analysis A6 prompt-estimand scope

Frozen: 2026-08-25T13:37:23-04:00 while E112-R1 was 49/75 terminal,
E109 was 13/15 terminal, and the complete official E112 result did not exist.
No additional task-evaluation endpoint was inspected to make this amendment.
The private interim history remains disclosed and irreversible.

Status: interpretation and machine-readable disclosure only. This amendment
changes no cell, prompt, draw, endpoint, contrast, aggregation, interval,
figure coordinate, threshold, or registered decision.

## Scope correction

Within each registered model--domain family, the endpoint target is the exact
finite evaluation prompt bank hashed by the paired prompt-surface audit. The
five paired training seeds measure fitted-policy variation and the four common
generation draws measure evaluation Monte Carlo variation conditional on that
bank. Neither axis estimates variation over newly generated prompts.

The official result must therefore state:

- `prompt_target = registered_finite_evaluation_bank_within_domain`;
- `prompt_population_inference = false`;
- `prompt_population_se = null`;
- the four draws change response-generation randomness, not prompt identity;
- family, scale, and 15-family summaries remain conditional on the five frozen
  prompt banks and registered model/domain grid.

The exact prompt count and prompt/request digests remain stored per run and
must match within every treatment/comparator pair. A prompt mismatch is still a
hard build failure. This disclosure does not treat prompts as independent
replicates, add a pooled interval, or generalize to the domain generators.

## Interpretation boundary

The historical-comparator bundle, private-interim limitation, original
adjusted-breadth decision, A5 paired-draw diagnostic, and baseline-centered
sensitivity remain unchanged. Replication to the independent E117 reserve is a
different same-plumbing experiment; it cannot retroactively make E112 a prompt-
population or isolated-component estimate.
