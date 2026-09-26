# E117 Stage-1-A7 total-effect estimand amendment

Frozen: 2026-08-25T11:43:53-04:00 while E117-R1 remains at 0/12
terminal, zero realized optimizer updates, and before any Stage-1 seed, split,
job, ledger, sampled response, or outcome exists. This amends only the
analysis-only Stage-1 development contract. It does not authorize a launch or
change E112-R1's already-frozen analysis.

## Trigger

E117-A4 correctly recognizes proposal consumption, admission, retention, and
support as post-treatment mediators after the common step-1 boundary. Enabling
proposal replay or semantic PPO can change the policy, which can then change
future anchors and realized proposal support. The executable Stage-1 labels
did not state this causal boundary and could invite a controlled-direct-effect
interpretation that the C/P/F design does not identify.

This issue was identified from the frozen treatment graph. No E117 or Stage-1
outcome was inspected.

## Effective estimands

- `P-C` is the total effect of enabling validator-positive proposal admission
  and uniform replay relative to proposal-shaped compute with pre-mutation
  discard. It includes downstream policy-mediated changes in realized support.
- `F-P` is the total effect of enabling v7 semantic PPO on top of the
  proposal/replay system. It includes any downstream semantic-induced changes
  in proposal consumption, admission, retention, and replay.

Neither contrast holds realized post-treatment support fixed. In particular,
`F-P` is not a controlled direct semantic effect and must not be described as
one. If `F-P` advances, the next decomposition requires the already-motivated
semantic-without-proposal arm before attributing the gain to a standalone
semantic pathway or interaction.

The executable result now emits these estimands and boundaries. Its schema
advances to `e117_stage1_paired_vector_statistics_v8`.

## Boundary

This is an interpretation correction, not a new endpoint or gate. All C/P/F
contrasts, uncertainty calculations, thresholds, safety rules, sentinel scope,
and advancement decisions remain unchanged. Stage 1 remains development-only;
confirmation still requires a separately frozen split and at least five fresh
paired training seeds.
