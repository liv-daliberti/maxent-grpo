# E19 matched canonical Dr.GRPO 0.5B control

**Status: FROZEN POST-HOC MATCHED-CONTROL EXTENSION, BEFORE E19 SUBMISSION OR
E19 OUTCOMES (2026-07-19).**

E19 adds the pure canonical Dr.GRPO controls that were absent from E16's 0.5B
grid. E16 outcomes were visible before this extension was requested, so every
E19-versus-E16 comparison is explicitly post-hoc and exploratory. E19 does
not alter, replace, restart, or retroactively amend any E16 run.

One control is shared by all three E16 MaxEnt treatments within each
task/seed. It is unnecessary and statistically incorrect to duplicate an
identical control separately for fixed, proportional, and dual MaxEnt.

## Treatment contrast

The contrast changes exactly the direct entropy term:

- E19 uses the ordinary Dr.GRPO reward update on the same canonical policy,
  with no entropy bonus.
- E16 uses the identical Dr.GRPO reward update plus fixed, proportional, or
  Haarnoja-dual direct complete-action MaxEnt.

E19 therefore uses `critic_type=drgrpo`, `xdr_tau=inf`, uniform candidate
aggregation, no reward-standard-deviation normalization, Dr.GRPO's shared
`1/T_max` update normalization, `maxent_alpha=0`,
`policy_entropy_coef=0`, `beta=0`, and every entropy or aggregation controller
disabled. Historical free-text Dr.GRPO changes the policy support and is not
an eligible control.

## Frozen grid

- Environments: canonical graph coloring and canonical Countdown.
- Method: canonical Dr.GRPO (`alpha=0`) only.
- Paired seeds: 43, 44, and 45 (six jobs total).
- Model: Qwen2.5-0.5B-Instruct revision
  `7ae557604adf67be50417f59c2c2f167def9a775`.
- Group size 16; one PPO epoch; learning rate `2e-7`; `beta=0`.
- Five complete prompt-pool passes.
- Graph: 192 train / 96 evaluation prompts, 960 optimizer updates,
  evaluation and checkpoint every 48 prompts.
- Countdown: 384 train / 128 evaluation prompts, 1,920 optimizer updates,
  evaluation and checkpoint every 96 prompts.
- Learner-side canonical rollout batch 1; train batch 16; per-device train
  batch 4; rollout temperature 1; `T_max=192`.
- Sampled pass@8, mean@8, coverage@8, distinct@8, and greedy pass@1 use the
  same repaired canonical evaluator as E16.
- Automatic resume and watchdog requeue are disabled.

The immutable Python source is E16's eligible V3 snapshot with SHA-256
`0e9929f09bac51ddcd85a6d5a506bf0613279a6fd983e0147da2f39a7aa320ec`.
The frozen E16 shell/tooling execution surface has SHA-256
`b7f01de3e9a9247afe6d95f984ce77c0e0615e4aa067a7fa77c701da15bc9597`.
The launcher must import both source and operations only from
`var/artifacts/source_snapshots/e16_canonical_full_0e9929f09bac51ddcd85a6d5a506bf0613279a6fd983e0147da2f39a7aa320ec/`.

Fresh prefixes are `gce19_canonical_drgrpo_05b_v1` and
`cde19_canonical_drgrpo_05b_v1`. No pre-existing run may be relabeled into
this cohort.

## Submission and reporting contract

Configuration-only validation must pass for both tasks before any job is
submitted. Full submission creates all six jobs held, verifies the manifests
and resolved scheduler exports, and releases the cohort atomically. A partial
cohort remains held. Re-running the full launcher against an existing manifest
is prohibited.

All three seeds must be shown individually and as a seed mean. Reporting must
include paired E16-minus-E19 differences for every sampled endpoint and may
not describe the post-hoc contrast as part of E16's original preregistration.
Exact held-out full-support audits should additionally report action entropy,
valid probability, conditional valid-mode entropy, and effective support; no
training-prompt entropy sensor may substitute for those audits.

Until all six E19 jobs complete their registered horizons, the manuscript may
show available trajectories only as explicitly interim. It must not place an
E19 treatment effect in the abstract or conclusion.
