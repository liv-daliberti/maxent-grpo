# E22-v2 base-preserving free-form conditional-token MaxEnt on ModeBench

**Status: FROZEN FOR LAUNCH (2026-07-20).**

E22-v2 is a prospective engineering correction to E22-v1. The v1
Haarnoja-style controller was allowed to reduce the entropy coefficient from
its intended base dose of `0.000075` to `0.00005`. All three graph-coloring
seeds finished at that lower bound. This follow-up preserves v1 unchanged and
uses fresh prefixes, fresh jobs, and a matched free-form Dr.GRPO control.

## Calibration and intervention

The policy and entropy objective remain unrestricted free-form generation with
`conditional_token_mean`: non-EOS vocabulary entropy is averaged within each
sampled response, then across responses. The sampled state distribution is
detached, EOS receives no direct entropy derivative, and response length does
not multiply the entropy payment.

The dual may strengthen but may not weaken the intended MaxEnt dose:

- initial and minimum alpha: `0.000075`;
- maximum alpha: `0.00015`;
- log-alpha Adam learning rate: `0.005`;
- fixed graph-coloring entropy target: `1.2981748407085736` nats;
- fixed Countdown entropy target: `1.0776875459899504` nats.

The targets are the three-seed pooled means of the per-seed conditional-token
entropy means over updates 1--64 of the frozen E22-v1 cohort. The exact
calculation and per-seed inputs are recorded in
`paper/results/e22_freeform_dual_v1_calibration.json`. Replacing v1's target of
80% of each seed's warmup mean prevents the controller from being instructed
to weaken the treatment immediately after warmup. Because v1 outcomes were
observed before this calibration, E22-v2 is an outcome-informed engineering
follow-up, not a confirmatory replication.

## Matched arms

Each domain has two arms at seeds 43, 44, and 45:

- free-form Dr.GRPO (`alpha=0`, no entropy controller);
- base-preserving free-form conditional-token MaxEnt (Haarnoja dual).

The control shares the model, data, prompt format, unrestricted text action
space, optimizer, group size, sampling settings, horizon, evaluator, and
checkpoint cadence. It is the only baseline used for the E22-v2 treatment
comparison; canonical-action Dr.GRPO is not a matched control for this policy.

## Tasks and runtime

| Task | Training pool | Evaluation pool | Horizon | Evaluation cadence |
|---|---:|---:|---:|---:|
| Graph coloring | 192 | 96 | five passes | every 48 prompts |
| Countdown easy3 | 384 | 128 | five passes | every 96 prompts |

Both tasks use Qwen2.5-0.5B-Instruct revision
`7ae557604adf67be50417f59c2c2f167def9a775`, group size 16, one PPO epoch,
learning rate `2e-7`, `beta=0`, rollout temperature 1, `top_p=1`, the ordinary
`qwen_boxed` prompt, and unrestricted responses up to 192 tokens. Expected
length control remains disabled. The source and execution snapshots are the
runtime-gated E21 V10 surfaces:

- source hash `217547637154ed74c2356eafec6dc885914793f8bb530c2b6918ca9a2c245448`;
- execution hash `05d43e4ae78d75d9e98c5b3d07d6ebd88cc545cc9039774ad0b6bf7cfebc05ce`.

E22-v2 must be plotted in a separate free-form panel with its matched
free-form Dr.GRPO control. It must not be pooled with canonical-action MaxEnt
or presented against canonical-action Dr.GRPO as though that were its control.
