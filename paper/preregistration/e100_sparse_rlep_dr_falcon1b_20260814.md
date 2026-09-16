# E100 preregistration: sparse RLEP-Dr on Falcon3-1B

Date frozen: 2026-08-14, before submission of any E100 collection, smoke, or
scientific cell.

## Question and estimand

Does sparse, prompt-matched RLEP-Dr preserve verified-answer breadth better
than the completed Falcon3-1B E79 Dr.GRPO control? The estimand is the paired
terminal-pass difference, E100 minus E79 control, within domain and seed.

## Frozen cohort

- Model: Falcon3-1B-Instruct at E79 revision
  `28ba2251970a01dd1edc7ba7dad2eb71216ccfdf`.
- Domains: Graph Coloring, Countdown, Python Factors, MathIR, and PantryPlan.
- Seeds: 55--59 (25 scientific cells), paired to completed E79 controls.
- Training: 384 prompts x 8 passes, group size 16, and E79's optimizer,
  decoding, prompt surfaces, verifier, placement, and 192-step checkpoints.

For every domain/seed, the paired terminal E79 control is the immutable prior
policy. It generates four draws of 16 candidates per training prompt at
temperature 0.7 and top-p 0.95. Responses are never borrowed across prompts,
seeds, domains, or models; empirical response frequency is preserved.

On a prompt with at least two frozen verified trajectories, RLEP-Dr adds two
frequency-preserving replay rows to the ordinary 16-row Dr.GRPO update. On an
ineligible prompt it performs the unchanged 16-row E79 Dr.GRPO update. No
prompt is dropped or reweighted, and canonical/mode-balanced replay is off.

## Hard gates

Each of the 25 immutable pools must have one completion marker and one draw
sidecar, exactly 384 prompt references per draw under the registered 16 x 4,
T=.7, top-p=.95 sampler, and at least one replay-eligible prompt. The audit
records eligible and ineligible counts; there is no outcome-dependent minimum
eligible fraction.

A non-scientific 32-update Graph/s55 learner smoke is fixed in advance. Its CPU
audit must observe a terminal receipt, both eligibility branches, exactly two
replay rows for eligible updates and zero for fallback updates, and finite
RLEP loss/advantage telemetry. Every scientific job depends on that smoke
audit and its own pool audit. If the fixed smoke slice does not exercise both
branches, the gate fails closed; another cell is not selected after seeing the
pools.

## Outcomes and reporting

Report terminal pass@8, distinct@8, all five paired seed effects, pool
eligibility, realized replay-update fraction, replay-row counts, and replay
gradient diagnostics. Half-pass checkpoints are trajectories, not independent
samples. A missing receipt, hash or configuration drift, wrong replay dose,
canonical replay, non-finite value, or missing endpoint is a failure.

All jobs are submitted held and audited before an atomic ledger is written and
released. Infrastructure retries may use only this frozen configuration.
