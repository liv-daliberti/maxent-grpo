# E47G-CAL: fail-closed representative MATH strategy calibration

**Status: FROZEN BEFORE LAUNCH — 2026-07-24**

E47F exposed one implementation error on problem 15 before completing
calibration or producing manual labels: when a stored representative was
omitted by the judge, the round was recognized as structurally invalid, but a
subsequent diagnostic traversal still indexed that omitted assignment and
raised `KeyError`. No invalid key was admitted. E47F is preserved as a failed
partial and cannot advance training.

E47G keeps E47F's frozen data, prompts, bounded schedule, Qwen72 identity,
two temperature-zero permutation passes, transitive union-component rule,
manual audit construction, and every numerical gate unchanged. It makes one
fail-closed implementation correction: after candidate ambiguity accounting,
an ambiguous stored representative or a merge of two stored representatives
immediately returns an all-rejected round without traversing the unusable
partition graph or modifying either bank.

The exact behavior is covered by a regression test in addition to the
validator-gating, omission, permutation-disagreement, transitive-chain,
checkpoint, and passive-control tests. The persistent state schema is bumped
to `math_strategy_canonicalizer_two_pass_union_v5`, so a pre-fix checkpoint
cannot be silently loaded.

The E47F proof remains the E47G proof: admitted components are connected
components of the union of both pass-equivalence graphs. Distinct admitted
components are therefore separated in both passes; disagreement can merge or
reject support but cannot manufacture a novel strategy boundary.

E49 may launch only from a complete
`var/artifacts/e47g_bounded_math_strategy_calibration_v1` artifact whose
unchanged frozen gates all pass.
