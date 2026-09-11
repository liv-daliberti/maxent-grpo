# E15 prospective canonical-action dose calibration

**Status: FROZEN BEFORE E15 CONFIGURATION, SUBMISSION, TRAINING, OR OUTCOMES.**

E15 is a narrow, single-seed engineering follow-up to E14. E14 established
that fixed canonical MaxEnt at `alpha=0.05` was behaviorally safe and increased
exact valid-mode effective support by 19.28%, but missed the frozen 25% gate.
E15 asks only whether either of two slightly larger fixed coefficients,
`alpha=0.075` or `alpha=0.10`, crosses that same gate without exhausting the
remaining correctness headroom. It is not a replication, a scale experiment,
an adaptive-controller experiment, or evidence for a paper claim.

This document freezes the E15 design before either E15 arm is configured or
submitted. The earlier E14 outcomes below are antecedent evidence, not E15
outcomes.

## Frozen antecedent evidence and authorization

The launcher fails closed unless it can replay all C0, M01, and M05 evidence
behind these exact artifacts:

- E14 C0 approval:
  `var/artifacts/e14_c0_20260718_1349_fixed_shape_e14_c0_approval.json`,
  SHA-256
  `4ab8b0173034931f287b489d042f9aad735fc7986e789de7e31c9363d1972d97`;
- E14 final comparison:
  `var/artifacts/e14_canonical_smoke_decision_20260718_1425.json`, SHA-256
  `50456aee52b0ab9ad3fe32deaf8da33caabe7d3a99c90eafb44cf493faa794de`;
- canonical logical Python-source hash:
  `8f0ee26fe1a4482efcf55a96e3f3de0a689aa2f94813f39df66887f9d6bf6329`.

The E14 comparison must replay to `status=no_viable_dose`, contain exactly
M01 and M05, and retain its single-seed and scale/domain firewalls. In
particular, M05 must replay as runtime-valid and behaviorally safe, with an
exact entropy gain above `log(1.25)`, exact mean valid-probability retention
above 80%, and valid-support ratio strictly between 1 and 1.25. The user's
explicit direction after that outcome authorizes this separate E15 protocol;
the E14 record by itself does not authorize expansion.

## Frozen scope and identities

Both arms use exactly:

- environment: graph coloring;
- model: `Qwen/Qwen2.5-0.5B-Instruct`, frozen revision
  `7ae557604adf67be50417f59c2c2f167def9a775`;
- seed: `9005`;
- the E14 train/eval datasets with 192/96 prompts and exactly three hidden
  nodes per prompt; their combined content hash is
  `8e8bd8d4986920784cb067f99f4dc1b401f553b0d40a378dd4ada5dab29b48d6`;
- the canonical response space `{1,2,3}^3`, exactly 27 complete actions;
- action token IDs `(16,17,18)` and exactly three generated action tokens;
- the HF learner's restricted inverse-CDF, fixed-shape causal-placeholder
  sampler; vLLM remains evaluation-only;
- group size 16, one PPO epoch, learning rate `2e-7`, batch size 16,
  per-device batch size 4, `beta=0`, and shared Dr.GRPO outer normalization
  `T_max=192`;
- exactly 128 optimizer updates. OAT's strict post-update query API therefore
  receives `max_queries=2032`, yielding 2,048 consumed trajectories;
- evaluation and saved-model boundaries at updates 32, 64, 96, and 128;
- no automatic resume or watchdog requeue.

The Python training implementation is unchanged from E14. The launcher must
compute the logical source-tree hash above and make a fresh immutable source
snapshot at submission. A path-dependent hash over `sha256sum` output is not
an acceptable identity because it changes when an identical tree is copied.

## Frozen arms and objective

There are exactly two E15 arms:

| Arm | Fixed coefficient | Launcher phase |
|---|---:|---|
| M075 | 0.075 | `m075` |
| M10 | 0.10 | `m10` |

For a canonical three-action trajectory (a=(a_1,a_2,a_3)), each arm uses
the E14 direct on-policy objective

\[
  J(\theta)=J_{\mathrm{Dr.GRPO}}(\theta)
  +\alpha H\!\left(\pi_\theta(a\mid x)\right).
\]

The entropy term is estimated with E14's unclipped exclusive-prefix ratios
against the complete behavior probabilities recorded by the same learner
policy. It has no extra division by action length; only the shared outer
Dr.GRPO normalization by 192 remains. The coefficient is fixed at its arm's
assigned value on every update. Proportional feedback, Haarnoja dual updates,
length targets, token-entropy bonuses, xDr weighting, coefficient schedules,
and every other adaptive controller are prohibited.

The two arms are fully specified in advance and may be submitted together.
Neither arm's outcome may change the other's coefficient, budget, stopping
rule, or gate.

## Runtime and endpoint validity

An arm is runtime-valid only if the post-run E15 validator establishes all of
the following from immutable artifacts:

1. the exact source, dataset, runtime, antecedent approvals, arm, seed,
   coefficient, and query budget match this document;
2. Slurm terminates `COMPLETED` with exit code `0:0`, with a single
   restart-invalid metrics stream containing exactly updates 1--128 and the
   terminal update-128 alias;
3. all canonical rollouts are valid three-token actions, behavior
   probabilities normalize, behavior/current overlap diagnostics are finite,
   and no controller telemetry is present;
4. the assigned coefficient and direct entropy loss are active on every
   update with the same E14 units and shared normalization;
5. `step_00128` exists and an exact enumeration of all 27 actions on all 96
   eval prompts passes probability normalization, entropy identity, and
   teacher-forcing/prefix-tree cross-checks at E14's unchanged tolerances.

A runtime-invalid arm has no scientific classification and must be rerun from
scratch under a new stamp only after the implementation failure is understood.
Its partial outcome cannot be used to alter this protocol.

## Frozen scientific gates

All endpoint quantities are exact means over the 96 frozen evaluation
prompts at `step_00128`, except reward, which is the mean rollout reward over
updates 97--128. The frozen E14 C0 reference is:

- exact action entropy: `1.345020968071439` nats;
- exact mean valid probability: `0.31407401670345525`;
- exact mean valid-mode effective support: `2.3263799784010906`;
- final-32 mean rollout reward: `0.330078125`.

A runtime-valid E15 arm is **behaviorally safe** iff all three conditions hold:

1. exact mean valid probability is strictly greater than `0.05`;
2. exact mean valid-probability retention relative to C0 is at least `0.80`;
3. final-32 mean rollout reward is strictly positive.

It is **diversity-effective** iff both conditions hold:

1. exact action-entropy gain relative to C0 is at least
   `log(1.25) = 0.22314355131420976` nats;
2. exact mean valid-mode effective support divided by C0's value is at least
   `1.25`.

An arm is **viable** iff it is both behaviorally safe and
diversity-effective. These are engineering thresholds, not hypothesis-test
significance levels, and the single-seed results will be reported as such.

If exactly one arm is viable, it is selected. If both are viable, select the
arm with larger exact mean valid-mode effective support; if their values are
within 5% multiplicatively, select the smaller coefficient M075. If neither
is viable, E15 selects no dose. The thresholds and tie rule will not be
weakened after seeing outcomes.

## Scope firewall

E15 authorizes only M075 and M10 at seed 9005 in the frozen 0.5B graph-coloring
setup. It does not authorize another seed, Countdown, a 3B/7B model, an
adaptive controller, insertion into the analytical 18-cell grid, or a main
paper claim. A viable E15 arm authorizes only drafting and reviewing a new,
separately frozen three-seed 0.5B replication protocol. It does not submit
that replication automatically.

## Retrospective outcome — appended after E15 completion

**This section was written after both E15 arms and their exact endpoint audits
completed. All preceding text remains the frozen prospective protocol.**

Both treatments were runtime-valid, behaviorally safe, and
diversity-effective under the unchanged gates. M075 (`alpha=0.075`; training
job `30012428`, audit job `30012432`) achieved exact action entropy
`2.251622008861948`, mean valid probability `0.28169776775686256`, mean
valid-mode effective support `3.310869216608205`, and final-32 rollout reward
`0.291015625`. Its valid-probability retention was `0.8969152262692207` and
its valid-support ratio was `1.4231850546116498` relative to C0.

M10 (`alpha=0.10`; training job `30012427`, audit job `30012431`) achieved
exact action entropy `2.7974740052946436`, mean valid probability
`0.261672194741692`, mean valid-mode effective support `4.437708099723962`,
and final-32 rollout reward `0.271484375`. Its valid-probability retention was
`0.833154545824017` and its valid-support ratio was `1.9075594446845168`
relative to C0.

Because both arms were viable and M10's exact mean valid-mode effective
support was 34.03% larger than M075's—outside the frozen 5% tie band—the
preregistered rule selected **M10**. This remains a single-seed,
Qwen2.5-0.5B, graph-coloring engineering calibration. It does not authorize
Countdown, larger models, adaptive control, the analytical grid, or a main
paper claim; it authorizes only drafting and review of a separately frozen
three-seed 0.5B replication protocol.

The immutable comparison is
`var/artifacts/e15_canonical_dose_decision_20260718_2229.json` (SHA-256
`680e739411a5ed9f14754a86c285945e8b20d11a4362c48235de26d9ba9d7de1`).
The complete retrospective record, including treatment and audit hashes, is
in [`../results/e15_canonical_dose_calibration.json`](../results/e15_canonical_dose_calibration.json).
