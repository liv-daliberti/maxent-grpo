# E117 Stage-1-A1 primitive-vector advancement amendment

Frozen: 2026-08-25 while E117-R1 remains at 0/12 terminal and zero realized
optimizer updates, and before any Stage-1 seed, split, job, ledger, or outcome
exists. This changes only the development-screen decision rule in the
analysis-only E117 Stage-1 contract. It does not authorize a launch and does
not change E112-R1's already-frozen final analysis.

## Trigger

The original Stage-1 contract correctly declared `(pass@8, raw distinct
correct modes@8)` to be the primitive endpoint vector and
`raw distinct@8 - pass@8` to be derived. Its advancement gate nevertheless
used only the derived subtraction. That creates a sign paradox: a component
can improve both primitive coordinates and still fail because its pass gain is
larger than its raw-distinct gain.

For example, effects `(pass@8=+0.50, raw distinct@8=+0.30)` are a strict
Pareto improvement in the registered primitive vector but yield adjusted
breadth `-0.20`. The subtraction usefully decomposes total correct-mode gain
into first-success and beyond-first-mode parts; it is not an independent
endpoint and must not veto simultaneous primitive gains.

This defect was identified from the frozen algebra and a synthetic regression
fixture. No Stage-1 outcome exists or was inspected.

## Corrected development gate

All pairing, C/P/F contrasts, sentinels, fresh seeds, common evaluation draws,
fixed checkpoints, uncertainty axes, scope rules, and confirmation boundary
remain unchanged.

For each component and sentinel:

1. Terminal and normalized-AUC `raw distinct correct modes@8` effects must
   each be strictly greater than +0.05, greater than two corresponding
   evaluation Monte Carlo SEs, and positive in at least two of three paired
   training seeds.
2. The existing correctness safeguards remain unchanged: terminal pass@8
   versus C must be at least -0.03 on the three-seed mean and at least -0.10
   in every paired seed. P-C uses P versus C for safety; F-P continues to use
   F versus C.
3. Report terminal and normalized-AUC effects for pass@8, raw distinct@8, and
   the exact derived adjusted-breadth decomposition, with both uncertainty
   axes and every paired seed/draw effect. The adjusted-breadth effect has no
   independent advancement veto.
4. A positive adjusted-breadth effect supports the narrower interpretation
   that correct-mode gain extends beyond first successes. A nonpositive value
   limits that interpretation but does not erase a raw-distinct improvement
   that passes the registered correctness safeguards.

The executable result schema advances to
`e117_stage1_paired_vector_statistics_v2` and records both the primitive
advancement basis and `derived_adjusted_breadth_veto=false`.

## Scope and confirmation

The same component must remain actionable in at least three of four sentinels
to become a broad development candidate. Countdown alone remains explicitly
domain-specific, with Graph retained as the negative boundary. Three training
seeds remain a development screen; any confirmatory claim still requires a
separately frozen fresh split, at least five fresh training seeds, common
evaluation draws, and no tuning on Stage-1 outcomes.
