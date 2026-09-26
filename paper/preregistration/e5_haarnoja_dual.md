# E5 prospective analysis plan: Haarnoja-style entropy-dual xDr

**Status: EXPLORATORY, SPECIFIED BEFORE ANY E5 OUTCOME (2026-07-17).** This
experiment was designed after the fixed-temperature and proportional-feedback
trajectories had been inspected. It is therefore prospective with respect to
the E5 runs below, but mechanism-informed and post-hoc relative to E1 and E4.
Initialization-only metrics are outcomes: the launch must follow this frozen
specification.

## Question

E4 changes xDr's candidate-aggregation temperature with a one-sided,
memoryless proportional rule. Does the automatic entropy-tuning construction
of Haarnoja et al. provide a better controller when its learned positive dual
strength is mapped to xDr's intervention strength?

This is a transfer of the **controller**, not an implementation of Soft
Actor-Critic. The training algorithm remains on-policy group-relative RL; no
critic, replay buffer, off-policy update, or token-level entropy reward is
added. Token entropy is only the controller's sensor, and xDr candidate
aggregation remains the actuator.

## Fourth arm and dual update

The existing grid contains Dr.GRPO, fixed xDr at tau = 0.05, and E4
proportional-feedback xDr. E5 adds one arm, `xdr_sac_dual`, at every existing
environment-by-scale cell. Existing controls are reused and are not rerun.

xDr's empirical intervention becomes stronger as its candidate temperature
falls, so define the dimensionless positive dual strength

```text
alpha_xdr = base_tau / tau,       base_tau = 0.05.
```

For the first 64 globally averaged post-update token-entropy observations,
run at tau = 0.05 and set

```text
target_entropy = 0.8 * mean(first 64 entropy observations).
```

Thereafter minimize Haarnoja et al.'s signed entropy-dual objective through a
log parameter,

```text
J(alpha_xdr) = alpha_xdr * (observed_entropy - target_entropy)
alpha_xdr = exp(log_alpha_xdr),
```

using scalar Adam with learning rate 0.003, beta1 = 0.9, beta2 = 0.999, and
epsilon = 1e-8. Map the updated dual to the next candidate temperature as

```text
tau_next = clip(0.05 / alpha_xdr, 0.005, 0.5).
```

The update is signed and cumulative. Below-target entropy increases
`alpha_xdr` and strengthens xDr by lowering tau; above-target entropy decreases
`alpha_xdr` and can relax xDr above its initial tau. The symmetric factor-ten
bounds keep this exploratory controller finite. The entropy observation is
globally reduced so every learner rank applies the same next-step temperature;
the dual, optimizer moments, target, and warmup state are checkpointed.

This mapping is an empirical control hypothesis, not a mathematical entropy
guarantee. In SAC, the learned multiplier directly weights policy entropy. In
E5, it controls inverse candidate-aggregation temperature because that is the
direction in which the landed xDr results resist policy sharpening.

## Matched 18-run extension

Run seeds 43, 44, and 45 in both Countdown and graph coloring at each of
Qwen2.5-0.5B-Instruct, Qwen2.5-3B-Instruct, and
Qwen2.5-7B-Instruct: 2 environments x 3 scales x 3 seeds = 18 new runs.

Each E5 cell inherits its corresponding maintained compute-divergence recipe:

- Countdown uses the 384-prompt easy3 pool at all scales; graph coloring uses
  192 prompts at 0.5B and 1,024 prompts at 3B/7B.
- Group size is G=16 at 0.5B and G=32 at 3B/7B.
- Every run has a hard ceiling of five complete prompt-pool passes.
- Learning rate, optimizer, rollout sampling, completion budget, evaluation
  cadence in prompt units, and inline exact-mode evaluation match the other
  arms in the same cell.
- The 3B jobs use two A5000 GPUs with optimizer/activation offload, and the 7B
  jobs use two A100-80GB GPUs with the same two-prompt layout and offload.
  These choices change placement, not the statistical recipe.
- Model checkpoints are not retained; inline metrics and evaluations retain
  the complete trajectory and avoid unnecessary shared-storage growth.

## Outcomes and interpretation fixed before launch

The primary descriptive comparison is E5 minus fixed xDr on mean
multi-answer coverage@8 at the last common evaluation at or before five
passes, paired by training seed within each environment and scale. Report all
seed trajectories and the paired contrast; do not select an earlier checkpoint
because it is favorable.

Secondary outcomes are pass@1, pass@8, mean@8, distinct correct modes@8,
token-entropy trajectory, mixed/all-correct/all-zero group fractions, learned
alpha and tau trajectories, time outside the initial tau, and fraction of
updates at either bound. Mean@8 and pass@1 are accuracy guardrails.

- Better late coverage with a stable entropy and mixed-group trajectory would
  support cumulative dual control over the proportional E4 rule.
- Better late coverage while entropy continues to decline is resistance or
  delay, not immunity.
- No improvement over fixed or proportional xDr means this controller transfer
  did not improve the actuator.
- Saturation at tau = 0.005 with continued entropy loss is actuator failure;
  saturation at tau = 0.5 with a coverage loss indicates harmful relaxation.
- Any coverage gain accompanied by materially worse pass@1 or mean@8 is a
  tradeoff, not free stabilization.

No E5 result changes the confirmatory status of the landed 3B comparison.

## Operational recovery amendment, 2026-07-19

All three graph-coloring 7B jobs failed at the identical learner-step-1024
vLLM 0.8.4 CuMem sleep-allocator call (`none_dealloc`), after landing step 896.
This shared runtime failure was unrelated to the dual trajectory. Same-stamp
recovery disables vLLM sleep, lowers its KV reservation to 0.10, and uses two
A6000s with microbatch 4 plus optimizer/activation offload. The frozen E5
controller, data, G=32, global batch, five-pass ceiling, seeds, and evaluation
cadence are unchanged, and all three seeds receive the same recovery.

## Maintained entry points

```bash
bash ops/exp_scaling/launch_haarnoja_dual_extension.sh all
python ops/exp_scaling/refresh_haarnoja_dual_curves.py
python ops/exp_scaling/plot_divergence.py
```

The first command submits only the fourth arm. The refresh command emits six
tidy scaling-curve artifacts, including empty arrays while jobs are merely
queued, so the four-method grid has an explicit prospective row from launch.
