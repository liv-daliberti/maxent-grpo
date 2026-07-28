# E27 aggressive 0.5B free-form conditional-token MaxEnt on ModeBench

**Status: FROZEN FOR LAUNCH (2026-07-21).**

E27 replaces only the completed E22-v2 Haarnoja-dual treatments for Countdown
easy3 and graph coloring. The completed E22-v2 free-form Dr.GRPO seeds remain
historical matched references; no new control, fixed-MaxEnt, or proportional
arm is launched. E22-v2 artifacts remain immutable and are not pooled into the
new treatment trajectories.

## Frozen intervention

- Model: Qwen2.5-0.5B-Instruct revision
  `7ae557604adf67be50417f59c2c2f167def9a775`.
- Seeds: 43, 44, and 45 for each task; six treatment jobs total.
- Objective: unrestricted free-form `conditional_token_mean`, with non-EOS
  vocabulary entropy averaged within responses and then equally across
  responses. The sampled state distribution is detached and expected-length
  control is disabled.
- Controller: base-preserving Haarnoja log-alpha Adam with initial/minimum
  alpha `0.000075`, maximum alpha `0.00060`, and learning rate `0.010`.
- Fixed targets: graph coloring `1.622718550885717` nats and Countdown
  `1.347109432487438` nats, exactly 125% of the respective pooled E22 warmup
  references. Since the targets are explicit, adaptation begins with the first
  entropy observation; the retained 64-step warmup field is not a gate.

## Runtime and reporting

Graph coloring uses the frozen 192/96 train/evaluation pools and evaluates
every 48 prompts. Countdown uses the frozen 384/128 easy3 pools and evaluates
every 96 prompts. Both run five passes with group size 16, one PPO epoch,
learning rate `2e-7`, temperature one, `top_p=1`, unrestricted `qwen_boxed`
responses up to 192 tokens, and one A5000/64 GiB per seed on node105.

Jobs are submitted held and released only after auditing the treatment-only
arm, task target, alpha bounds, controller LR, seeds, placement, and budgets.
The active monitor and free-form figures use the E27 prefixes; E22-v2 dual
outcomes remain historical and must not fill missing E27 checkpoints.
