# E22 free-form conditional-token MaxEnt on ModeBench

**Status: FROZEN FOR LAUNCH (2026-07-20).**

E22 extends the E21 free-form token-policy intervention to the two 0.5B
ModeBench domains: graph coloring and Countdown. It is deliberately separate
from E16's canonical-action experiments. The policy uses ordinary unrestricted
Qwen text generation, a learned EOS token, and the normal verifier-facing
answer format. No canonical action restriction, answer index, or
gold-derived support is used.

## Intervention

The treatment is E21's `conditional_token_mean` objective. At each sampled
response state, entropy is computed over the non-EOS vocabulary conditional
on continuing. Sampled states are detached, active positions are averaged
within each response, and response means are averaged across the batch. Thus
the entropy term has no direct EOS-logit derivative and does not accumulate a
larger regularization payment merely because a response is longer.

Only the Haarnoja-style dual arm is added:

- initial alpha: `0.000075`;
- target: 80% of mean conditional-token entropy during the first 64 updates;
- log-alpha Adam learning rate: `0.005`;
- alpha bounds: `[0.00005, 0.00015]`.

The Dr.GRPO reward update uses group size 16, one PPO epoch, learning rate
`2e-7`, `beta=0`, rollout temperature 1, and `top_p=1`. The entropy mean is
already normalized once per response and receives no additional horizon
normalization. Expected-length control is disabled.

## Tasks and horizon

| Task | Training pool | Evaluation pool | Horizon | Evaluation cadence |
|---|---:|---:|---:|---:|
| Graph coloring | 192 | 96 | five passes | every 48 prompts |
| Countdown easy3 | 384 | 128 | five passes | every 96 prompts |

Both tasks use the released exact ModeBench artifacts, the ordinary
`qwen_boxed` prompt, unrestricted responses up to 192 tokens, greedy pass@1,
and sampled pass@8/mode coverage at temperature 1. Seeds are 43, 44, and 45.
The model is Qwen2.5-0.5B-Instruct revision
`7ae557604adf67be50417f59c2c2f167def9a775`.

The source and execution snapshots are the already runtime-gated E21 V10
surfaces:

- source hash `217547637154ed74c2356eafec6dc885914793f8bb530c2b6918ca9a2c245448`;
- execution hash `05d43e4ae78d75d9e98c5b3d07d6ebd88cc545cc9039774ad0b6bf7cfebc05ce`.

E22 is a descriptive policy-space extension. It must be labeled
**free-form conditional-token MaxEnt (Haarnoja dual)** and must not be pooled
with canonical-action MaxEnt as though the action spaces were identical.
