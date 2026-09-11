# E49E curated singleton objective dual-audit gate

**Status: FROZEN BEFORE ANY CURATED-CONTRACT AUDIT REQUEST — 2026-07-24**

The generated repair runs exposed a redundant-label false-negative: an
executor derived `-16.5`, the independent exact validator correctly matched
it to `-33/2`, and all actions were executable, but the same executor's
free-form self-label claimed the surfaces differed.  Another executor derived
the correct digit sum while objecting to an irrelevant trailing decimal zero.

For the precommitted coverage-only singleton contracts, acceptance therefore
uses the objective parts of the same two completed execution records:

1. both records must be identity-bound, schema-complete, cover every declared
   action, finish normally, and have independently recomputed derived answers
   that exactly match the frozen reference;
2. at least one of the two must mark every action valid, use only declared
   actions, and affirm that the route is self-contained.

The redundant language-model fields `matches_reference_answer` and `status`
cannot veto a route when the external exact validator and the structural
execution fields satisfy this gate.  Conversely, a wrong derived answer from
either executor, or the absence of one fully valid/self-contained execution,
still rejects the contract.

This rule applies only to the 11 precommitted singleton coverage contracts.
It does not alter raw E49E certification, pairwise novelty decisions,
multi-route support, controls, entropy observations, or any generated V2b
record.
