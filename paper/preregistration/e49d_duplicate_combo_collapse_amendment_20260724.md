# E49D duplicate-combo collapse amendment — 2026-07-24

**Status: FROZEN BEFORE ANY E49D TRAINING LAUNCH**

For the right-triangle side-length problem, the proposal generator returned
two nominal strategy IDs with exactly the same ordered action combo. The
existing parser correctly rejected the duplicate, and the rescue proposal
then failed its combo grammar. No menu for the row was materialized.

E49D now deterministically collapses byte-identical proposed action combos,
retains the first proposal in proposal order, and renumbers the remaining
strategy IDs before mathematical auditing. The collapsed route receives no
special trust: both answer-bound v4 auditors must still certify it before it
can become even a singleton menu.

This is the canonicalization operation implied by maximal certified support:
identical action sequences are one candidate, never two novel outcomes. It
does not merge nonidentical combos, repair mathematics, introduce an answer,
or change any certification, runtime, reward, controller, data, schedule, or
outcome rule. Earlier v4 records remain valid.
