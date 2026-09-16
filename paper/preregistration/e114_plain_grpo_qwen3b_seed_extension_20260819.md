# E114 preregistration: five-seed Qwen2.5-3B plain-GRPO completion

Date frozen: 2026-08-19, before submission of any E114 scientific cell.

## Question and estimand

E95 contains only Qwen2.5-3B seed 70, although the aligned scale block uses
seeds 70--74. E114 completes the ordinary-GRPO comparator without replacing or
rerunning E95. The estimand remains the paired terminal-pass difference between
ordinary GRPO and the E80-R1 Dr.GRPO control within domain and seed.

## Frozen cells

- Model: Qwen2.5-3B-Instruct at E80-R1 revision
  `aa8e72537993ba99e69dfaafa59ed015b17504d1`.
- Domains: Graph Coloring, Countdown, Python Factors, MathIR, and PantryPlan.
- New seeds: 71--74, giving 20 new scientific cells. Together with E95 seed 70,
  this forms five seeds in every domain.
- Training: the exact E80-R1 cell-specific data, native Qwen prompt surface,
  optimizer, decoding, 384 prompts x 8 passes, and 192-step checkpoints.
- Only scientific change from the paired E80-R1 control:
  `critic_type: drgrpo -> grpo`. Passive compute-matching replay remains
  compute-only and contributes no objective.

## Execution and reporting

Every job is submitted held, checked against its seed, variant, model/runtime
snapshot, and frozen horizon, then recorded atomically before release. Existing
E95 seed-70 artifacts are immutable and are referenced during analysis rather
than copied into E114. Report all five paired seeds, including instability or
failure; infrastructure retries may preserve only this frozen configuration.

