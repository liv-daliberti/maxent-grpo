# AntMaze, PointMaze, and PantryPlan repair diagnosis

**Status: FROZEN AFTER THE ORIGINAL FIVE-SEED OUTCOMES AND BEFORE ANY REPAIR
MODEL SAMPLE — 2026-07-30**

This document records a secondary, explicitly post-outcome repair campaign.
It does not replace, relabel, or overwrite any cell in the frozen 80-run
campaign. Repaired cohorts use new artifact names and are reported separately.

## AntMaze

All ten original AntMaze cells were saturated at initialization. For every
seed in both arms, pass-0 greedy success, mean@8, and pass@8 were 1.0, while
distinct@8 was 2.0. Pass-12 values were identical except for treatment seed 44,
whose mean@8 was 0.9921875. The environment therefore measured neither task
learning nor a meaningful success/diversity tradeoff.

Repair requirement: certify a prospectively harder held-out map family with
longer routes and multiple forced decisions. No original evaluation map may be
substituted or reclassified. A frozen-model viability gate must show nonzero
verified completions without pass-0 saturation before a five-seed pair can
launch.

## PointMaze

The original train slate was saturated rather than usefully learnable. Across
five seeds, Dr.GRPO averaged 15.9833 verified episodes per 16-rollout group and
the treatment averaged 15.9771. Only 8 of 480 Dr.GRPO updates and 11 of 480
treatment updates had nonzero task advantage. The treatment consequently
optimized almost entirely its verified exploration terms; the control was
almost always stationary.

The original endpoint pattern was also internally inconsistent with a healthy
training signal: greedy success rose from 0 to 0.25 in every seed, while
mean@8, pass@8, and distinct@8 usually stagnated or declined. The sampled
evaluation request seeds also included the checkpoint update, so adjacent
checkpoints did not use common random numbers.

Repair requirements:

1. Use train-only geometries not seen by the PointMaze warm start.
2. Require a development gate with a nontrivial mixture of verified and failed
   rollouts and nonzero task advantage on at least 25% of updates in both arms.
3. Remove checkpoint update from evaluation sampling seeds, while retaining
   independent fixed draw/sample/row seeds.
4. Keep binary official verification, within-prompt Dr.GRPO centering,
   compute-matched replay traversal, and the public action mask unchanged.

## PantryPlan

The original five-seed means at pass 0 were greedy 0.3594, mean@8 0.3447,
pass@8 0.9004, and distinct@8 2.4355 in both arms. At pass 12, Dr.GRPO reached
greedy 0.5469 and mean@8 0.5425 but fell to pass@8 0.5652 and distinct@8
0.6184. Verified MaxEnt reached greedy 0.5547 and mean@8 0.5308 while retaining
pass@8 0.6891 and distinct@8 1.0953. Thus the treatment materially delayed
collapse but did not preserve enough of the initial verified support.

The Pantry special-case path produced no semantic-Shannon advantage. Its
effective support mechanisms were online novelty and split mass/balance
replay. Treatment replay was applied in every update at alpha 0.10; novelty was
nonzero in 903 of 1,920 updates.

Repair requirement: calibrate stronger verified replay on a newly generated
development split. The first registered dose is replay alpha 0.20, with
novelty beta 0.50, learning rate 2e-7, and all task, verifier, sampling, and
compute-matching settings otherwise unchanged. A five-seed comparison may use
the new evaluation split only after the dose and acceptance criteria are
frozen.

