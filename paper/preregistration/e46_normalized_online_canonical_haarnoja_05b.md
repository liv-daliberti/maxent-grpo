# E46: normalized online-canonical Haarnoja control at 0.5B

**Status: FROZEN BEFORE LAUNCH — 2026-07-23**

## Scope and provenance

E46 is a prospective adaptive extension of E44-OGS, selected after inspecting
E44-OGS learning curves and seeing late-pass quality/breadth collapse after
verified support growth had largely saturated. It does not alter, restart, or
pool itself with the frozen E44-OGS cohort. All E46 conclusions are
exploratory.

E46 adds one new treatment arm on the same graph-coloring and Countdown easy3
tasks, with the same model, data, seeds, ordering, optimizer, rollout budget,
validator-bound canonicalizer, and evaluation protocol as E44-OGS. The frozen
E44-OGS Dr.GRPO and fixed-alpha arms are the declared comparators. This is an
exact matched-seed extension but not a contemporaneously launched three-arm
cohort; that limitation must accompany any comparison.

## Question

Does controlling canonical entropy relative to the current verified support
prevent the late collapse seen with a fixed canonical-entropy coefficient?

Raw entropy is not comparable as support grows because its maximum is
`log |B_x^+|`. E46 therefore controls

`rho_x = H(q_x) / log |B_x^+|`.

## Frozen sensor

For each prompt `x`, the bank contains only validator-positive canonical
outcomes actually produced by the policy. After every complete `G=16` candidate
group has been scored and atomically committed, let `C_x(a)` be the cumulative
validated occurrence count and `K_x = |B_x^+|`. With the same pseudocount
`lambda=1` used by E44-OGS,

`q_x(a) = (C_x(a) + lambda) / (sum_b C_x(b) + lambda K_x)`.

The controller sensor is the exact, unclipped Shannon entropy

`H(q_x) = -sum_a q_x(a) log q_x(a)`

and `rho_x = H(q_x) / log K_x`. Numerical roundoff is clamped to `[0,1]`.
Only groups with `K_x >= 2` are controller-eligible. A singleton bank has no
defined normalized entropy and is skipped; it is never recorded as zero.
When an optimizer round has no eligible group, the EMA and Adam state are left
unchanged.

This is a post-update empirical bank-distribution sensor. It is not an exact
enumeration of the neural policy's probability over arbitrary free-form
strings, and cumulative counts make it a deliberately slow sensor. The
per-row training advantage remains E44-OGS's bounded leave-one-out estimator;
the control sensor does not replace it.

## Frozen controller and objective

E46 retains E44-OGS's novelty bonus `beta=0.50`, pseudocount `1`, surprisal
clip `5`, fail-closed admission, group snapshot, and placement of the detached
canonical advantage after ordinary Dr.GRPO task centering.

The entropy coefficient begins at `alpha_0=0.10`. Let `rho_t` be the mean
eligible normalized entropy for optimizer round `t`, and

`m_t = 0.9 m_{t-1} + 0.1 rho_t`,

with the first observation initializing `m_t=rho_t`. The normalized target is
`rho*=0.80`. E46 performs Haarnoja-style dual descent in log-alpha:

`e_t = m_t - rho*`,

`L_alpha = alpha_t e_t`,

`g_t = d L_alpha / d log(alpha_t) = alpha_t e_t`.

Adam uses learning rate `0.003`, `beta1=0.9`, `beta2=0.999`, and
`epsilon=1e-8`, followed by projection to
`alpha in [0.10, 0.50]`. Thus entropy below target raises alpha; entropy above
target can lower it only to the fixed E44-OGS coefficient. The coefficient
used to score a round is frozen for that round. Its sensor is observed only
after the policy optimizer step, and the updated coefficient applies first to
the next round.

The bank and dual-controller states are checkpointed independently. Adaptive
resume fails closed if either state is absent or any target, bound, optimizer,
EMA, sensor-unit, or controller-rule field differs.

## Cohort

- Model: local immutable
  `Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775`.
- Tasks:
  - graph coloring: 192 training prompts, 96 neutral evaluation prompts;
  - Countdown easy3: 384 training prompts, 128 neutral evaluation prompts.
- New arm: `online_canonical_haarnoja`.
- Comparator arms: frozen E44-OGS `grpo` and
  `online_canonical_maxent`.
- Seeds: `43,44,45`.
- Group size: `16`.
- Budget: ten complete prompt-pool passes.
- Learning rate: `2e-7`.
- One PPO epoch, `beta=0`, maximum norm `1`.
- Rollout temperature `1`, top-p `1`, maximum response length `192`.
- One node302 A100 GPU per E46 job. Pending time caused by node saturation does
  not change the protocol.

Run prefixes:

- `gce46_normalized_canonical_haarnoja_05b_v1`;
- `cde46_normalized_canonical_haarnoja_05b_v1`.

## Evaluation, telemetry, and interpretation

Evaluation is identical to E44-OGS: initialization, every quarter pass, and
ten passes; greedy pass@1 plus four fixed temperature-1, K=8 draws with seeds
`440100--440103`; raw outputs, rewards, and canonical keys retained.

In addition to all E44-OGS telemetry, E46 must log post-update `H(q_x)`,
`log K_x`, normalized entropy, controller-eligible fraction, alpha used for the
current round, observed ratio, ratio EMA, target, dual error/gradient/loss,
alpha before update, alpha for the next round, skipped-observation indicator,
and optimizer/observation counts.

The adaptive mechanism clears its exploratory gate only if, in both tasks:

1. terminal pass@8 and coverage@8 exceed fixed-alpha E44-OGS in the
   all-three-seed mean;
2. at least two of three paired seeds improve both endpoints over fixed alpha;
3. mean@8 and greedy pass@1 decline by no more than `0.03` relative to fixed
   alpha;
4. alpha rises above `0.10` after the normalized-entropy EMA crosses below
   `0.80`, with zero validator/key parity failures; and
5. late-pass breadth has a smaller peak-to-terminal decline than fixed alpha.

Dr.GRPO remains a contextual baseline. Beating fixed alpha but not Dr.GRPO
supports only a collapse-control claim, not an overall performance claim.
Graph-only success is domain-specific. E46 makes no unrestricted-MATH claim.
