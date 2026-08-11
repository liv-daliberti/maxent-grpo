# E80: Qwen2.5-3B verified-replay scale replication

**Frozen before submission on 2026-08-05.**

## Question

Does verified replay improve retention of validator-positive execution modes in
Qwen2.5-3B-Instruct relative to a compute-matched Dr.GRPO control?

E80 is a fresh five-domain, five-seed comparison of verified replay only. The
optimizer is chosen before any E80 outcome and is shared exactly across arms.
Semantic MaxEnt, adaptive coefficients, balance losses, support-escape
actuators, counterfactual proposals, and checkpoint selection are excluded.

## Why this 3B recipe

The earlier E74 Qwen2.5-3B runs used a constant learning rate of 2e-7 with no
warmup. They showed that 3B can learn, but not that this was a safe eight-pass
recipe. On Graph Coloring, the seed-43 control rose from sampled pass@8 0.607
at initialization to 0.936 at pass 2.25, then fell to 0.668 by pass 4. On
PantryPlan, the control fell from 0.828 at initialization to 0.703 at pass 0.5
and reached the 0.555 single-output floor by pass 1. These are descriptive
development results, and the old treatment also contained components that E80
does not contain.

E76 supplies the conservative anchor. Its completed Qwen2.5-3B Graph control
at constant 5e-8 and beta_KL=0 remained stable through six passes and ended at
sampled pass@8 0.723 rather than collapsing. E76 did not finish enough 3B
cells to estimate a replay effect or select a universal recipe. Therefore E80
does not claim a completed hyperparameter sweep: it uses the stable local scale
as the target effective step size and adds standard schedule guardrails.

Official public recipes are convention checks, not learning-rate oracles.
Hugging Face Open-R1's Qwen GRPO demonstration uses BF16, gradient
checkpointing, 16 generations, one training epoch, 10% warmup, and cosine
decay to one tenth of the peak rate:
https://github.com/huggingface/open-r1/blob/main/recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/config_demo.yaml.
Its absolute 1e-6 learning rate is not transplanted because its batch, loss
normalization, data, and model differ. Current TRL documentation also records
beta_KL=0 as common GRPO practice and explains why response-level reward
normalization and length normalization require care:
https://github.com/huggingface/trl/blob/main/docs/source/grpo_trainer.md.
The Dr.GRPO analysis identifies the response-length bias that motivates the
repository's existing Dr.GRPO control:
https://arxiv.org/abs/2503.20783.

The resulting common recipe is:

- AdamW with peak learning rate 1e-7;
- linear warmup for 10% of the fixed 3,072-update horizon, followed by cosine
  decay to 1e-8 (the installed OAT `cosine_with_min_lr` schedule);
- Adam betas (0.9, 0.999), epsilon 1e-8, no weight decay;
- beta_KL=0, PPO clip range 0.2, gradient norm cap 1;
- one PPO epoch per rollout batch, group size 16, BF16;
- rollout temperature 1 and top-p 1.

The peak is half the unstable E74 rate. Because warmup occupies 0.8 pass and
the rate subsequently decays, its horizon-average step size is close to the
locally stable 5e-8 anchor. The exact coefficient is a fixed reproducibility
choice, not evidence that it is universally optimal.

## Design

- Model: `Qwen/Qwen2.5-3B-Instruct`, pinned to cached revision
  `aa8e72537993ba99e69dfaafa59ed015b17504d1`.
- Interface: the model's native Qwen `<|im_start|>` / `<|im_end|>` chat surface
  and each domain's already frozen Qwen prompt template.
- Domains: Graph Coloring, Countdown, Python Factors, MathIR, and PantryPlan.
- Data: each domain's released 384-prompt training pool and fixed 128-prompt
  evaluation split.
- Seeds: 65, 66, 67, 68, and 69, paired within domain and arm. These do not
  overlap the E74 development seed 43 or E76 tuning seed 53.
- Training: exactly eight passes over 384 prompts, for 3,072 prompt updates.
- Evaluation and resumable checkpoints: every 192 prompt updates, corresponding
  to passes 0, 0.5, 1.0, ..., 8.0. Pass 8 is the terminal endpoint.
- Memory-only accommodations: one A100, ZeRO-2, optimizer and activation
  offload, vLLM GPU ratio 0.25, and evaluation batch size 32. These are shared
  by both arms and do not change the objective.

The response budgets, validators, canonicalizers, evaluation draw seeds, and
sampling surface are inherited from the E78 Qwen comparison. The complete
cohort has 5 domains x 2 arms x 5 seeds = 50 runs.

## Arms

### Compute-matched Dr.GRPO (`control`)

The control executes the same verified-bank insertion, replay scheduling,
teacher-forced traversal, and backward-compute envelope as the treatment, but
the applied verified-replay derivative is exactly zero.

### Verified replay (`replay`)

For each scheduled prompt-local bank B_x, retain at most one policy-generated,
validator-positive exemplar per observed verified mode and minimize

    L_replay(x) = mean_{b in B_x} -s_theta(b | x).

Here `s_theta` is the length-normalized teacher-forced log score. One bank is
scheduled per prompt update in deterministic global round-robin order,
including singleton banks. Capacity is 16 observed modes per prompt. The fixed
loss weight is 0.10; it is not an adaptive controller and was not tuned on E80.

## Exact exclusions

Both arms hard-disable semantic Shannon shaping, semantic separate advantages,
quality-gated or signed semantic pressure, conditioned-bank balance KL,
canonical-bank entropy shaping, token entropy bonuses, adaptive coefficients,
counterfactual proposals, singleton escape, novelty reward, and reference KL.
No exhaustive support, evaluation outcome, desired mode count, or desired
entropy is available to training, scheduling, stopping, or checkpoint choice.

## Outcomes and estimands

At every registered half-pass checkpoint report, separately by domain:

- greedy pass@1;
- sampled mean correctness@8;
- sampled pass@8;
- mean distinct correct modes@8; and
- excess multiplicity, `distinct@8 - pass@8`.

The primary comparison is the paired seed difference `replay - control` at
pass 8 for `distinct@8` and `pass@8`. The secondary trajectory summary is
trapezoidal AUC over the complete pass-0 through pass-8 half-pass grid. Show all
five paired seed differences and their mean/range; do not pool domains into one
effect and do not select a best checkpoint. E80 is a model-scale replication,
so effect sizes may be shown beside E78 but are not pooled across model sizes.

Mechanism telemetry reports banked-mode survival, verified bank size, replay
opportunities, applied replay gradient, and replayed verified score. The
control must have exact-zero applied replay gradient whenever an opportunity
exists; treatment must have a finite nonzero applied gradient on at least one
eligible update.

## Integrity and failure policy

All 50 cells are submitted held from one hash-bound runtime snapshot and are
released only after their scheduler environments pass an exact audit. A
malformed environment, source mismatch, duplicate run directory, non-finite
loss, proposal leakage, traceback, or missing pass-8 endpoint fails closed.
Infrastructure interruption may resume only from the same run's hash-bound
checkpoint and exact bank, optimizer, data-cursor, and request-stream state.
No failed scientific run is silently replaced or excluded.

