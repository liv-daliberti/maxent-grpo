# E79: aligned Falcon3-1B verified-replay comparison

**Frozen before submission on 2026-08-04.**

## Question

Does verified replay improve retention of validator-positive execution modes in
Falcon3-1B-Instruct relative to a compute-matched Dr.GRPO control when both arms
use the same, independently chosen base optimization recipe?

The base recipe is selected without consulting any replay-arm outcome. E79 is a
fresh five-domain, five-seed comparison of verified replay only; semantic
MaxEnt, adaptive coefficients, explicit balance losses, support-escape
actuators, and counterfactual proposals are excluded.

## Base-recipe evidence

The completed Falcon **control-only** cells of E76 Stage A screened constant
learning rates 5e-8, 1e-7, and 2e-7, with beta_KL in {0, 0.01}, on held-out
Graph Coloring and PantryPlan validation pools. At pass 6, the zero-KL 2e-7
cell had positive change in `pass@8 + distinct@8` on both domains (+0.1875 and
+0.7734). The lower zero-KL rates had worst-domain changes of -0.4766 (5e-8)
and +0.0313 (1e-7). These are descriptive one-seed tuning data, not confirmatory
evidence. Their authoritative job identities are in
`var/artifacts/e76_tuned_scale_stage_a_jobs.json`; the corrected Pantry prompt
surface is recorded in
`var/artifacts/e76_tuned_scale_stage_a_falcon_pantry_repair.json`.

Current public GRPO implementations provide a useful convention check rather
than a transferable learning-rate oracle. Hugging Face TRL v0.24 defaults to
AdamW, learning rate 1e-6, beta_KL=0, one update per generation batch,
clip epsilon 0.2, gradient norm 1, no weight decay, and Adam betas (0.9, 0.999):
https://huggingface.co/docs/trl/v0.24.0/grpo_trainer. The current verl reference
configuration likewise uses learning rate 1e-6, constant scheduling, one PPO
epoch, clip ratio 0.2, gradient clipping at 1, Adam betas (0.9, 0.999), and no
active policy KL loss:
https://github.com/volcengine/verl/blob/main/verl/trainer/config/_generated_ppo_trainer.yaml.
Exact learning rates depend on loss normalization, effective batch size, and
model/task geometry, so E79 retains the locally supported 2e-7 rate rather than
copying 1e-6 across frameworks.

The frozen common recipe is therefore:

- AdamW, learning rate 2e-7, constant schedule, no warmup;
- Adam betas (0.9, 0.999), epsilon 1e-8, and weight decay 0;
- beta_KL=0, PPO clip range 0.2, gradient norm cap 1;
- one PPO epoch per rollout batch, group size 16, BF16;
- rollout temperature 1 and top-p 1.

Zero KL is chosen despite the larger mean Pantry gain of one beta_KL=0.01
screen cell because zero KL is the common modern GRPO convention, avoids a
second reference-model path, and the 2e-7 zero-KL cell already improved both
screened domains. No replay result participates in this choice.

## Design

- Model: `tiiuae/Falcon3-1B-Instruct` pinned to cached revision
  `28ba2251970a01dd1edc7ba7dad2eb71216ccfdf` (about 1.67B parameters despite
  the family label). The tokenizer's published Falcon chat surface is used.
- Domains: Graph Coloring, Countdown, Python Factors, MathIR, and PantryPlan.
- Data: each domain's released 384-prompt training pool and fixed 128-prompt
  evaluation split.
- Seeds: 55, 56, 57, 58, and 59, paired within domain, physical node, and arm.
  These do not overlap the E76 tuning seed 53 or the earlier E73 seeds 43--47.
- Training: exactly eight passes over 384 prompts. Group size is 16 and all
  base optimizer settings above are shared exactly by both arms.
- Evaluation and resumable checkpoints: every 192 prompt updates, corresponding
  to passes 0, 0.5, 1.0, ..., 8.0. Pass 8 is the terminal endpoint; checkpoint
  selection is forbidden.

The Falcon response budgets are the non-binding E73-amended contracts: 192
tokens for Graph and Countdown, 512 for Python, 128 for MathIR, and 8 canonical
bits for PantryPlan. Model windows are respectively 512, 512, 768, 384, and
704 tokens. The Pantry run uses `falcon_pantry_support_mask`; the other domains
use `falcon_boxed`. These interface settings are identical across arms.

The complete cohort has 5 domains x 2 arms x 5 seeds = 50 runs.

## Arms

### Compute-matched Dr.GRPO (`control`)

The control executes the same verified-bank insertion, scheduler, replay batch
materialization, teacher-forced traversal, and backward-compute envelope as the
treatment, but the applied verified-replay derivative is exactly zero.

### Verified replay (`replay`)

For each scheduled prompt-local bank B_x, retain at most one policy-generated,
validator-positive exemplar per observed verified mode and minimize

    L_replay(x) = mean_{b in B_x} -s_theta(b | x).

Here `s_theta` is the length-normalized teacher-forced log score. One bank is
scheduled per prompt update in deterministic global round-robin order,
including singleton banks. Capacity is 16 observed modes per prompt. The fixed
loss weight is 0.10; it is a reproducibility setting, not an adaptive
controller or a tuned Falcon coefficient.

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
effect and do not select a best checkpoint.

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
