# E117 Stage-1-A5 cross-model broad-scope amendment

Frozen: 2026-08-25T11:31:13-04:00 while E117-R1 remains at 0/12
terminal, zero realized optimizer updates, and before any Stage-1 seed, split,
job, ledger, sampled response, or outcome exists. This amends only the
analysis-only Stage-1 development contract. It does not authorize a launch or
change E112-R1's already-frozen analysis.

## Trigger

The registered 3-of-4 broad-development rule weights sentinels, not model
families: Countdown, Graph, and Python are Qwen-0.5B, while MathIR is Falcon-1B.
Consequently, all three Qwen sentinels could pass and Falcon could fail while
the executable contract still called the component broad. That pattern is
multi-domain evidence on one model family, not broad model support.

This defect was identified from the frozen sentinel layout and a synthetic
fixture. No E117 or Stage-1 outcome was inspected.

## Corrected scope gate

A component is a `broad_development_candidate` only when both conditions hold:

1. it is actionable in at least three of the four registered sentinels; and
2. its actionable sentinels represent both registered model families,
   Qwen-0.5B and Falcon-1B.

Thus a qualifying 3-of-4 set must contain Falcon MathIR and at least two Qwen
sentinels. Four-of-four also qualifies. Three Qwen-only sentinels are retained
and reported but receive `scope=do_not_advance`, not a broad label. The frozen
Countdown-only domain-specific rule remains unchanged; all other patterns
continue not to advance.

The result records the registered and actionable model families,
`cross_model_support`, and the final broad-candidate Boolean. The executable
schema advances to `e117_stage1_paired_vector_statistics_v6`.

## Boundary

Cross-model support is necessary for a broad development label, not evidence
of population-wide generalization. Stage 1 still has only three training seeds
and four sentinels. A confirmatory claim still requires a separately frozen
split with at least five fresh paired training seeds and no tuning on Stage-1
outcomes. No endpoint, component contrast, uncertainty calculation, effect
threshold, or pass-safety rule changes here.
