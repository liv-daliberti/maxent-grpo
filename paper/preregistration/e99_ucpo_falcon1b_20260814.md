# E99 preregistration: UCPO on Falcon3-1B

Date frozen: 2026-08-14, before submission of any E99 smoke or scientific
cell.

## Question and estimand

Does Uniform-Correct Policy Optimization (UCPO, tau=0.2) preserve verified
answer breadth better than the completed Falcon3-1B E79 Dr.GRPO control while
retaining comparable accuracy? The estimand is the paired terminal-pass
difference, E99 minus E79 control, within domain and seed. Half-pass
checkpoints are trajectories, not independent samples.

## Frozen cohort

- Model: Falcon3-1B-Instruct at E79 revision
  `28ba2251970a01dd1edc7ba7dad2eb71216ccfdf`.
- Domains: Graph Coloring, Countdown, Python Factors, MathIR, and PantryPlan.
- Seeds: 55, 56, 57, 58, 59 (25 scientific cells).
- Training: 384 prompts, 8 passes, 16 rollouts per prompt, learning rate 2e-7,
  beta=0, one PPO epoch, and checkpoints every 192 updates.
- Comparator: the already-completed E79 `control` cell with the same domain
  and seed; it is not rerun.
- Placement, data, Falcon prompt surface, verifier, decoding, optimizer, and
  passive compute-matching traversal are inherited cell-by-cell from E79.
- Intervention: UCPO advantage redistribution with fixed tau=0.2 is the only
  live-gradient change relative to E79 control.

No coefficient tuning, checkpoint selection, seed exclusion, domain-specific
rule, or post-hoc replacement is allowed. A separate 32-query Graph/s55
learner smoke is operational only; every scientific cell depends on it.

## Outcomes and failure rules

Report terminal pass@8, distinct@8, all five paired seed effects, and UCPO
mechanism telemetry (eligible-group rate, inverse-weight range, and maximum
advantage-mass error). A non-finite value, mass error outside numerical
tolerance, missing terminal receipt, or configuration drift is a failure.
Infrastructure retries may use only this frozen configuration.

Jobs are submitted held. Their Falcon revision, placement, objective exports,
smoke dependency, and horizon are audited before an atomic ledger is written;
only then are they released.
