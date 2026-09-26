# E72 baseline and decoding controls — current objective contract

## Scientific claim

The supported common mechanism is verified replay: rehearse successful,
validator-accepted outputs so correct modes do not disappear between collection
batches. Semantic MaxEnt over canonical verified outcomes and known-mode balance
remain explicit refinements whose usefulness can vary by domain.

The paper does not claim that the comprehensive x-Mode Dr.GRPO stack is uniformly
necessary. It also does not claim an isolated incremental effect for semantic
MaxEnt until a clean same-plumbing comparison is run under the current code.

## Active comparison arms

| Arm | Objective | Role |
|---|---|---|
| matched Dr.GRPO | task advantage only; passive verified-support telemetry | collapse control |
| ordinary verified rehearsal | task advantage + verified-mass replay | common-mechanism baseline |
| replay + balance | rehearsal + known-mode balance | domain-dependent balance comparison |
| x-Mode Dr.GRPO | rehearsal + semantic MaxEnt + known-mode balance | comprehensive current instantiation |
| semantic MaxEnt, no replay | task advantage + semantic MaxEnt | diagnostic only; not current paper evidence |

All semantic scoring is validator-gated. The open-set unseen bucket belongs to
the predictive Shannon-entropy estimator. Passive support tracking remains
available for matched controls and has zero objective influence.

## Paper evidence boundary

The registered ordinary-rehearsal and replay-plus-balance table predates the
fixed-coefficient revision. Within its frozen objective contract, rehearsal
improves the mean over matched Dr.GRPO in all five domains; balance is
domain-dependent. These endpoints are historical mechanism evidence, not
estimates of the current implementation, and the fixed-coefficient rerun will
replace them. Original preregistrations and results remain immutable
provenance rather than being rewritten.

## Decoding controls

The decoding frontier remains valid as a control on frozen endpoint behavior:
rerun terminal checkpoints over temperature, sample budget, and nucleus
truncation. It answers whether sampling alone reopens collapsed support; it does
not establish the value of any training component.

## Integrity checks

- Fail closed on validator/key disagreement.
- Score a rollout group against one immutable pre-group support snapshot.
- Commit support only after group scores are fixed.
- Keep proposal-only replay support separate from on-policy semantic counts.
- Checkpoint bank, scheduler, optimizer, and data/request cursors
  atomically.
- Do not silently resume bank state from a superseded schema.
- Report per-domain paired intervals without pooling domains.
