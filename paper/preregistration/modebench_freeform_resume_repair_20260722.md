# ModeBench free-form resume-contamination repair

**Status: FROZEN REMEDIATION ADDENDUM (2026-07-22)**

## Scope and invariant

This addendum repairs the Qwen2.5 3B free-form ModeBench trajectories after a
resume-ordering defect was found: restored learner weights were not synchronized
to rollout actors before the resumed boundary evaluation and first rollout.
Every replacement uses the corrected source snapshot, the original frozen
dataset/model/hyperparameters, and seeds 43, 44, and 45. No curve smoothing or
post-hoc hyperparameter change is permitted.

A trajectory may resume only from the last optimizer checkpoint written by the
clean predecessor before the earliest contaminated attempt. A model-only export
does not qualify because it drops optimizer, scheduler, RNG, and controller
state. If that exact optimizer checkpoint has been pruned, the replacement must
replay from the pinned pretrained initialization.

All replacements use fresh run prefixes so automatic recovery and analysis
cannot splice onto an abandoned contaminated branch. An external predecessor
checkpoint is a one-time bootstrap only; watchdog requeues prefer checkpoints
written by the new branch and fall back to the clean bootstrap only if the new
branch has not yet written one.

## Audited disposition

| Cohort | Task | Seed | Earliest contaminated boundary | Retained clean optimizer state | Repair start |
|---|---|---:|---:|---|---|
| E25-v2 MaxEnt dual | Countdown | 43 | 576 | no | initialization |
| E25-v2 MaxEnt dual | Countdown | 44 | 576 | no | initialization |
| E25-v2 MaxEnt dual | Countdown | 45 | none; interrupted clean at 288 | no | initialization |
| E25-v2 MaxEnt dual | Graph coloring | 43 | 576 | no | initialization |
| E25-v2 MaxEnt dual | Graph coloring | 44 | 480 | no | initialization |
| E25-v2 MaxEnt dual | Graph coloring | 45 | 576 | no | initialization |
| E28 Dr.GRPO | Countdown | 43 | 96 | yes | clean step 96 |
| E28 Dr.GRPO | Countdown | 44 | 96 | yes | clean step 96 |
| E28 Dr.GRPO | Countdown | 45 | 864 | yes | clean step 864 |
| E28 Dr.GRPO | Graph coloring | 43 | 144 | no | initialization |
| E28 Dr.GRPO | Graph coloring | 44 | 48 | no | initialization |
| E28 Dr.GRPO | Graph coloring | 45 | 48 | no | initialization |

The 0.5B E22-v2 controls/conservative treatment and E27 aggressive treatment
were audited across all seeds and tasks. Each is a single uninterrupted attempt
from step zero through its terminal evaluation. They never crossed the faulty
resume path, so they are certified clean and are not rerun. Rerunning them would
be a new replication, not a contamination repair.

## Replacement identities

- `gce25_freeform_conditional_dual_3b_repair_v2`
- `cde25_freeform_conditional_dual_3b_repair_v2`
- `gce28_freeform_drgrpo_3b_repair_v2`
- `cde28_freeform_drgrpo_3b_repair_v2`

The held `repair_v1` staging jobs were rejected before release because their
critical recovery fields were inherited through Slurm `--export=ALL` rather
than recorded explicitly in `SubmitLine`. They performed no training. Version
2 pins those fields individually so the held-job audit is independently
reproducible.

Jobs are submitted held, audited for source, method, seed, task, resource, and
repair-start identity, and only then released. The repaired 3B figure inputs
replace the abandoned prefixes only after clean metrics exist; until then,
contaminated post-boundary points are not scientific evidence.
