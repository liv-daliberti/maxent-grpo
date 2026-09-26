# E25 prospective 3B free-form conditional-token MaxEnt scaling

**Status: FROZEN FOR LAUNCH (2026-07-21).**

E25 scales E22-v2's base-preserving free-form conditional-token Haarnoja-dual
treatment from Qwen2.5-0.5B-Instruct to Qwen2.5-3B-Instruct for graph coloring
and Countdown easy3. It is a treatment-only exploratory scale extension: no
new matched Dr.GRPO arm is authorized here, and canonical-action controls must
not be presented as matched free-form baselines.

## Frozen treatment

- Seeds: 43, 44, and 45 in each task, six jobs total.
- Objective: unrestricted free-form `conditional_token_mean`; non-EOS token
  entropy is averaged within each sampled response and then across responses.
- Alpha: initial/minimum `0.000075`, maximum `0.00060`.
- Controller: base-preserving Haarnoja log-alpha Adam, learning rate `0.010`.
  Because the targets are configured explicitly, adaptation starts with the
  first entropy observation; the retained 64-step warmup field is not a gate.
- Fixed targets are 125% of the task-specific E22-v2 pooled warmup means:
  graph coloring `1.622718550885717` nats and Countdown
  `1.347109432487438` nats.
- Group size 16, one PPO epoch, learning rate `2e-7`, `beta=0`, sampling
  temperature one, `top_p=1`, ordinary `qwen_boxed` prompting, and maximum
  response length 192. Expected-length control remains disabled.

Graph coloring uses the frozen 192/96 pools in `exact_answer_mode_probe` and
evaluates/checkpoints every 48 prompts. Countdown uses the 384/128 easy3 pools
and evaluates/checkpoints every 96 prompts. Each task runs for five passes.

The superseded v1 jobs 30033851--30033856 used the pooled-mean targets,
maximum alpha `0.00015`, and controller LR `0.005`. They were stopped on user
request before relaunch: graph seeds reached approximately update 40,
Countdown seeds 43 and 44 reached approximately updates 39 and 66, and
Countdown seed 45 had not started. Their artifacts remain exploratory and are
not combined with the clean v2 trajectories.

## Model and runtime

The model is the pinned Qwen2.5-3B-Instruct snapshot
`aa8e72537993ba99e69dfaafa59ed015b17504d1`; its digit-tokenizer files must be
byte-identical to E22-v2's 0.5B tokenizer. E25 uses E22-v2's immutable E21 V10
source and execution snapshots (`217547637154ed74c2356eafec6dc885914793f8bb530c2b6918ca9a2c245448`
and `05d43e4ae78d75d9e98c5b3d07d6ebd88cc545cc9039774ad0b6bf7cfebc05ce`).

Each job requests one node302 A100-80GB GPU and 96 GiB host memory, with
ZeRO-2, CPU optimizer offload, activation offload, vLLM sleep at ratio `0.25`,
rollout batch one, global train batch 16, and backward microbatch one. All six
jobs are submitted held and released only after auditing every method, seed,
task target, objective, model, budget, and resource invariant.
