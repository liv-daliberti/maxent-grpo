# E6 prospective analysis plan: Candidate MaxEnt projection grid

**Status: EXPLORATORY, SPECIFIED BEFORE ANY E6 TRAINING OUTCOME
(2026-07-17).** The objective and its three temperature modes were integrated
after the landed xDr, E4 proportional-feedback, and E5 Haarnoja-dual
trajectories had been inspected. E6 is prospective only with respect to the
runs listed here. Any short technical smoke is excluded from outcome analysis.

**Dated outcome amendment (2026-07-17, after launch): TERMINATED ON A
PRE-SPECIFIED ACCURACY GUARDRAIL.** The 0.5B jobs collapsed to two-token
responses, zero rollout reward, and all-zero groups within the first pass in
both environments. The fixed graph-coloring seed inspected at step 250 had
best rollout reward 0.8125 at step 21 and current reward 0; all three methods
showed the same qualitative failure. Jobs 30006209--30006262 were cancelled;
the 3B and 7B jobs never trained. These traces are retained as evidence
against the per-candidate `1/T_i` recipe and are not silently replaced. A
separately frozen E7 plan corrects the projection scale and requires a
collapse-length smoke before any larger launch.

## Question and objective

Does optimizing the appendix's candidate-distribution projection behave
differently from detached xDr surrogate reweighting when both methods use the
same realized candidate weights? Do proportional or Haarnoja-style control of
the projection target temperature improve its five-pass trajectory?

For prompt x and candidate i, use the same rollout-policy utility as xDr,

```text
u_xi = A_xi * T_xi / T_max,
q_x = softmax(u_x / tau_score),             tau_score = 1.0,
w*_x = argmax_w <w, log q_x> + tau_target H(w),
```

with no reference tilt (beta = 0) and no probability floor. The closed-form
target is `w*_xi proportional to q_xi ** (1 / tau_target)`. The training loss
replaces the signed clipped-PPO surrogate with the detached, length-normalized
sequence projection

```text
L_projection = -mean_x sum_i stopgrad(w*_xi)
                         * log pi_theta(y_xi | x) / T_xi.
```

All-equal reward groups therefore have a uniform target and a sampled
projection gradient; they are not zero-gradient Dr.GRPO groups. This is a
defining difference and will be reported rather than hidden as an
implementation detail.

At `tau_score = 1` and `tau_target = 0.05`, the target weights are exactly
`softmax(u / 0.05)`, identical to fixed xDr's candidate weights at
`tau_agg = 0.05`. The fixed-arm contrast therefore isolates the projection
loss from the signed-surrogate aggregation loss. Neither temperature is the
rollout sampling temperature, which remains 1.0.

## Three new methods

1. `xdr_maxent`: fixed Candidate MaxEnt projection at `tau_target = 0.05`.
2. `xdr_maxent_tau_control`: the same projection with E4's one-sided
   proportional controller applied to `tau_target` after a 64-observation
   warmup. Target ratio = 0.8, EMA decay = 0.9, gain = 20, and minimum
   `tau_target = 0.005`; the controller cannot raise the temperature above
   0.05.
3. `xdr_maxent_sac_dual`: the same projection with E5's signed cumulative
   Haarnoja-style dual controller applied to `tau_target`. It optimizes
   `log(alpha)`, where `alpha = 0.05 / tau_target`, using scalar Adam with
   learning rate 0.003, betas (0.9, 0.999), and bounds
   `tau_target in [0.005, 0.5]` after the same 64-observation warmup and 0.8
   target ratio.

Token entropy is a controller sensor only. No token-entropy reward, semantic
labels, exact answer keys, reference-policy tilt, replay buffer, critic, or
off-policy SAC update enters any E6 loss.

## Matched 54-run extension

Run all three methods with seeds 43, 44, and 45 in Countdown and graph
coloring at Qwen2.5-0.5B-Instruct, Qwen2.5-3B-Instruct, and
Qwen2.5-7B-Instruct: 3 methods x 3 seeds x 2 environments x 3 scales = 54 new
training runs. Existing Dr.GRPO, signed-surrogate xDr, and E4/E5 controller
arms are reused.

Each environment/scale cell inherits the maintained compute-divergence recipe:

- Countdown uses the 384-prompt easy3 pool; graph coloring uses 192 prompts at
  0.5B and 1,024 prompts at 3B/7B.
- Group size is 16 at 0.5B and 32 at 3B/7B.
- Every run has a hard ceiling of five complete prompt-pool passes.
- Seeds, data order, optimizer, learning rate, rollout distribution,
  completion budget, verifier, evaluation cadence in prompt units, and exact
  mode-coverage evaluation match the existing cell.
- Projection targets are computed while groups are intact, detached, and
  converted to `G * w*` row multipliers so shuffled memory-bounded minibatches
  implement the exact group-mean objective.
- No model checkpoints are retained. Inline training and evaluation metrics
  preserve each trajectory while the campaign cleanup protects shared space.

## Outcomes and fixed interpretation

The first primary descriptive contrast is fixed Candidate MaxEnt projection
minus fixed signed-surrogate xDr at the last common evaluation at or before
five passes, paired by training seed within each environment and scale, on
multi-answer coverage@8. Because their target weights match at the fixed
temperature, this estimates the consequence of changing the optimization
objective in this recipe.

The second family compares each controller branch with fixed Candidate MaxEnt
on the same endpoint and pairing. Report all seed trajectories; do not select
an earlier favorable checkpoint.

Secondary outcomes are pass@1, pass@8, mean@8, distinct@8, token entropy,
mixed/all-correct/all-zero group fractions, target entropy, effective target
rollouts, projection loss, learned temperature, and (for the dual) alpha.
Pass@1 and mean@8 are accuracy guardrails.

- A fixed-projection gain at flat guardrails supports the full MaxEnt
  projection over surrogate reweighting under matched candidate weights.
- A loss or collapse is evidence against the projection recipe, including the
  possibility that reinforcing sampled all-zero groups is harmful.
- A controller gain over fixed projection supports temperature control only
  for this objective; it is not evidence of universal entropy immunity.
- Continued entropy decline despite a temperature bound is actuator failure.
- Any coverage improvement with materially worse pass@1 or mean@8 is a
  tradeoff, not free mode preservation.

E6 remains exploratory and cannot change the confirmatory status of the
landed 3B signed-surrogate xDr comparison. The optional reference-policy tilt
in the appendix is not evaluated by E6.

## Maintained entry points

```bash
bash ops/exp_scaling/launch_maxent_projection_extension.sh smoke
bash ops/exp_scaling/launch_maxent_projection_extension.sh all
python ops/exp_scaling/refresh_maxent_projection_curves.py
python ops/exp_scaling/plot_divergence.py
```

The smoke uses a separate stamp and is excluded. The full launcher submits
only the three E6 methods; it does not duplicate any earlier control.
