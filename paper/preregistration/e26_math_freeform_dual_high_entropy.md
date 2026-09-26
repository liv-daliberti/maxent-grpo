# E26 high-entropy base-preserving free-form dual on MATH

**Status: FROZEN FOR LAUNCH (2026-07-21).**

E26 is an explicitly exploratory, outcome-informed replacement for only the
retired E21 MATH Haarnoja-dual treatment. It does not restart E21's cancelled
Dr.GRPO, fixed-MaxEnt, or proportional-control jobs. The completed/cancelled
E21 Dr.GRPO curves remain historical comparison data; no claim of a fresh
matched concurrent control is made.

## Motivation and calibration

E21 used unrestricted free-form `conditional_token_mean` entropy, but its
dual target was calibrated per seed to 80% of the first 64 updates and allowed
alpha to weaken below the intended base dose. The treatment therefore is not
the base-preserving E22/E25 controller requested here. Moreover, the E21 dual
spent 68.8%, 76.4%, and 80.4% of post-warmup updates at its `0.00015` alpha
ceiling for seeds 43, 44, and 45, while late conditional-token entropy fell to
approximately 0.16--0.21 nats.

The first E26 draft froze the pooled warmup mean itself (`0.3653633594512939`
nats) as the target, with alpha capped at `0.00030` and log-alpha Adam LR
`0.005`. Jobs 30033993--30033995 completed only their step-zero evaluation and
were cancelled before any optimizer metric was written; this draft contributes
no training outcome.

The replacement freezes a single target at `0.4567` nats, or 125% of the
pooled E21 warmup mean (rounded to four decimals). This is approximately 56%
above the pooled legacy target. Alpha starts at and may not fall below
`0.000075`; its upper bound is `0.00060`, four times the saturated E21 ceiling.
The Haarnoja log-alpha Adam learning rate is doubled to `0.010`. Because the
target is configured explicitly, the controller updates from its first entropy
observation; the retained 64-step warmup field is not an adaptation gate in
fixed-target mode. The exact observed inputs are frozen in
`paper/results/e26_math_freeform_dual_high_entropy_calibration.json`.

## Treatment and runtime

Only `maxent_dual` is launched, at seeds 43, 44, and 45. The entropy objective
is E22/E25's base-preserving unrestricted free-form conditional-token mean:
non-EOS vocabulary entropy is averaged within each sampled response and then
equally across responses; EOS has no direct entropy derivative; the sampled
state distribution is detached; and expected-length control is disabled.

Everything else retains E21's MATH contract: Qwen2.5-0.5B-Instruct at pinned
revision `7ae557604adf67be50417f59c2c2f167def9a775`, the 8,515 admitted training
prompts, MATH-500 evaluation, group size 16, one PPO epoch, learning rate
`2e-7`, one prompt epoch, free-form generation up to 1,024 tokens, and
evaluation/checkpoint boundaries every 2,129 prompts. Each run requests one
A5000 and 64 GiB host memory. The frozen source and execution hashes are
`217547637154ed74c2356eafec6dc885914793f8bb530c2b6918ca9a2c245448` and
`05d43e4ae78d75d9e98c5b3d07d6ebd88cc545cc9039774ad0b6bf7cfebc05ce`.

The launcher stages all three jobs held, audits the exact treatment, target,
alpha bounds, seed, resource placement, and run identity, and releases only a
complete three-seed cohort. E26 is reported as a separate row and must not be
relabeled as the completed E21 dual.
