# PantryPlan support-retention final v1

**Status: FROZEN AFTER THE REPAIR-V1 GATE AND SPLIT CORRECTION, BEFORE FINAL SUBMISSION — 2026-07-30**

The repair-v1 calibration passed: at three passes the treatment exceeded the
compute-matched control in terminal distinct correct, did not reduce pass@8,
retained at least 60% of its own initial distinct support, and retained at
least 90% of the control's terminal sampled mean. Applied novelty and verified
replay gradients were nonzero and the control derivatives remained zero.

The final is a fresh five-seed, two-arm cohort on seeds
76411–76415. It trains from the unchanged Qwen2.5-0.5B-Instruct warm start for
12 passes (384 optimizer updates), 16 rollouts per prompt, learning rate 2e-7,
and otherwise the exact passing repair-v1 dose:

- novelty beta 0.50;
- verified replay alpha 0.20;
- replay-mass alpha 0.20;
- one global replay group per update;
- replay/novelty warmup 64 updates; and
- four K=8 evaluation draws every eight updates.

The control executes the exact compute-matched replay traversal with zero
exploration and replay derivatives. Paired arms share training seeds and
evaluation-draw schedules.

Final evaluation uses only the immutable final view whose `eval/` is the
previously untouched 64-row source `dev/`. The 128 rows used during
calibration are excluded. The final audit requires all ten cells, exact update
and evaluation schedules, source/ops/model/data identities, matched traversal
within seed, nonzero verified reward, treatment exploration/replay
application, exact 0.20 replay coefficients, zero control derivatives, and
finite terminal endpoints. Outcomes are reported as a secondary post-outcome
repair and never replace the original frozen PantryPlan campaign.
