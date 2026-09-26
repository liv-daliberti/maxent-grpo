# E18 matched canonical Dr.GRPO 3B control

**Status: FROZEN POST-HOC MATCHED-CONTROL EXTENSION, BEFORE E18 SUBMISSION OR
E18 OUTCOMES (2026-07-19).**

E18 adds the pure Dr.GRPO control that was absent from E17's canonical-action
3B grid. E17 outcomes were already visible before this extension was requested,
so comparisons involving E18 are explicitly post-hoc/exploratory. E18 does not
alter, replace, or restart any E17 run.

The treatment contrast is deliberately one term only:

- E18: original Dr.GRPO reward update, with no entropy bonus.
- E17: the identical Dr.GRPO reward update plus the registered direct
  completion-sequence MaxEnt term (fixed, proportional, or Haarnoja-dual
  coefficient control).

Concretely, E18 uses `critic_type=drgrpo`, `xdr_tau=inf`, no reward-standard-
deviation normalization, Dr.GRPO's shared `1/T_max` update normalization,
uniform candidate aggregation, `maxent_alpha=0`, `policy_entropy_coef=0`, and
`beta=0`.

## Frozen grid

- Environments: canonical graph coloring and canonical Countdown.
- Method: matched canonical Dr.GRPO only.
- Seeds: 43, 44, and 45 (six jobs total).
- Model: Qwen2.5-3B-Instruct revision
  `aa8e72537993ba99e69dfaafa59ed015b17504d1`.
- Group size 16; one PPO epoch; learning rate `2e-7`.
- Five complete prompt-pool epochs.
- Graph: 192 train / 96 evaluation prompts, 960 optimizer updates, evaluation
  and checkpoint every 48 prompts.
- Countdown: 384 train / 128 evaluation prompts, 1,920 optimizer updates,
  evaluation and checkpoint every 96 prompts.
- Greedy pass@1 and sampled pass@8/mean@8/coverage@8 use the same repaired
  canonical evaluator as E17.
- Placement: one 48 GB A6000 per job, learner-side canonical rollout batch 1,
  ZeRO stage 2, optimizer and activation offload, actor synchronization at the
  quarter-epoch evaluation boundary.

The immutable Python source is E17's E16-derived repaired snapshot with SHA-256
`0e9929f09bac51ddcd85a6d5a506bf0613279a6fd983e0147da2f39a7aa320ec`.
The frozen six-file execution surface is E17's snapshot with combined receipt
`a8414bcd1932787ee1cea7e8c0b23868282f45903eeb9d495b23e4649f397f0c`.
Fresh prefixes are `gce18_canonical_drgrpo_3b_v1` and
`cde18_canonical_drgrpo_3b_v1`.
