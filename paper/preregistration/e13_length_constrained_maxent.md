# E13 prospective engineering gate: length-constrained sequence MaxEnt

**Status: EXPLORATORY ENGINEERING GATE, FROZEN BEFORE E13 OUTCOMES
(2026-07-17).** E12 found no fixed coefficient that was both behaviorally
safe and entropy-effective. At `alpha=0.002`, final-16 raw sequence entropy
was `48.464` nats, but mean response length was `60.762`, a batch reached
`121.938` tokens, and as many as 10/16 responses reached the generation
horizon. All smaller E12 coefficients were behaviorally safe but retained at
most `2.113` raw nats. E13 therefore holds the only entropy-effective E12
coefficient fixed and adds the missing expected-length constraint. It does
not tune the entropy coefficient again.

## Constrained objective and estimator

For the operational response length

```text
L(Y) = number of active generated response tokens,
```

horizon-truncated responses contribute `T_max`. E13 studies

```text
maximize_theta  E[R] + alpha H(pi_theta)
subject to      E[L] <= L_target,
```

with `alpha=0.002` and `L_target=16`. Its nonnegative Lagrange multiplier has
the actor objective

```text
J(theta, lambda) = E[R] + alpha H(pi_theta)
                   - lambda (E[L] - L_target).
```

The constant `lambda * L_target` has no actor gradient. For E10--E12's
exclusive prefix ratio

```text
W_(t-1) = pi_new(y_<t | x) / pi_old(y_<t | x),
```

the exact finite-horizon change-of-measure identity is

```text
E_pi_new[L] = E_pi_old [sum_t W_(t-1)].
```

The detached, unclipped right-hand side is the length value estimate used by
the dual controller. The actor's conservative PPO cost surrogate uses the
opposite clipping branch from the positive entropy bonus:

```text
S_L(Y) = sum_t max(W_(t-1), clip(W_(t-1), 1-epsilon, 1+epsilon)).
```

Thus a decrease in estimated cost cannot be credited beyond the clipping
boundary, while an increase remains fully penalized. With
`c=(G-1)/G`, the maintained minimized actor loss is

```text
L_actor = L_Dr.GRPO - c * alpha * S_H / T_max
                      + c * lambda * S_L / T_max.
```

Reward, raw entropy, and raw response length therefore retain exactly one
shared outer Dr.GRPO `1/T_max` normalization. Applying `c` to both auxiliary
terms preserves their stated objective units relative to Dr.GRPO's
self-including group-mean reward estimator.

## Projected length dual

The controller observes the detached unclipped new-policy length estimate
once per optimizer update. It is deliberately an additive projected dual,
not a log-parameterized entropy-coefficient controller: complementary
slackness requires the length price to be able to equal exactly zero.

```text
length_ema_0 = L_target
length_ema_k = 0.9 * length_ema_(k-1) + 0.1 * expected_length_k
relative_violation_k = (length_ema_k - L_target) / L_target
lambda_(k+1) = clip(lambda_k + eta * relative_violation_k, 0, 0.02).
```

`lambda_k` is used for actor update `k`; the observation from that update
sets `lambda_(k+1)`. There is no warmup. The target is fixed independently of
the run, the initial multiplier is zero, and controller checkpoints must
record raw-response-token units and all controller state.

## Two-arm engineering smoke

Both arms use graph coloring, Qwen2.5-0.5B-Instruct, seed 9005, group size 16,
four-row backward microbatches, one PPO epoch, PPO clip 0.2, `T_max=192`, 128
optimizer updates, and evaluations every 32 prompt updates. They differ only
in the dual step size:

| Arm | Dual step `eta` |
|---|---:|
| slow | `0.00005` |
| fast | `0.00020` |

This is a single-seed engineering gate, not comparative evidence. E12's
completed `alpha=0.002` arm is the frozen unconstrained anchor and is not
pooled into E13. A fresh third training arm is unnecessary only because
preflight tests must prove that `lambda=0` leaves E12's actor loss unchanged.

Before submission, exact enumeration under distinct old and new two-step
variable-length policies must verify both the value and gradient of the
unclipped length identity. Tests must also cover exclusive-prefix indexing,
response masking, the conservative `max` clipping branch, zero-price loss
equivalence, projected-dual direction and bounds, and checkpoint continuity.

## Runtime integrity and outcome gates

Each arm must contain the frozen step-128 endpoint and contiguous telemetry
for steps 97--128. Every observed update must have finite policy loss,
gradient, reward, entropy, prefix-ratio, response, length-surrogate, and dual
telemetry. The assigned entropy coefficient, target, multiplier maximum, EMA
decay, and dual step size must appear on every row. Raw sequence entropy
divided by 192 must equal its separately named per-`T_max` diagnostic. The
sampled-prefix length must equal the actor's operational response length.
`lambda_used` must equal the previous update's `lambda_next`, and all logged
EMA, relative-violation, and projected-dual transitions must reproduce the
frozen equations above. The ordinary MaxEnt proportional and Haarnoja-alpha
controllers must be absent.

An arm is **constraint-safe** only when:

- it has a positive rollout reward in its final 32 updates;
- final-16 mean actor response length is at most 18 tokens;
- final-16 mean unclipped expected new-policy length is at most 18 tokens;
- no final-16 batch has mean response length above 64 tokens;
- no final-16 batch has more than 2/16 horizon-truncated responses;
- final average evaluation accuracy (reported as pass@1) exceeds 0.05;
- `lambda_next` is not pinned at `0.02` for all final eight updates.

The two-token allowance around the mathematical target is a frozen
finite-sample gate, not a change from the constraint `E[L] <= 16`. An arm is
**entropy-effective** only when final-16 mean raw sequence entropy is at least
`4.9840172700576` nats, the same half-target threshold used in E12. Peak
entropy, peak length, pass@1, pass@8, and coverage@8 are reported
descriptively and cannot rescue a failed gate.

An arm is viable only if it is both constraint-safe and entropy-effective.
Among viable arms, select the one with the largest final-16 entropy. If all
arms within 5% of that maximum are considered tied, choose the smallest dual
step size among the tied arms. No viable arm means no analytical-grid release:

- length violation with a multiplier pinned at its maximum triggers an
  estimator/price-range audit;
- length satisfaction with entropy below threshold in both arms is evidence
  that this entropy target is not feasible under the frozen 16-token budget
  in this setup, and coefficient chasing stops;
- missing, inconsistent, or nonfinite telemetry is an implementation failure,
  not a scientific outcome.

A viable E13 arm authorizes a separately frozen three-seed replication. It
does not automatically submit that replication or any 54-cell analytical
grid.

## Frozen outcome

Jobs 30007892 and 30007893 used byte-identical source snapshots, completed
the exact step-128 endpoint on node023, and exited zero. The prospective
checker reproduced every controller transition and selected the slow arm:

| dual step | final-16 raw entropy | actor / expected length | max final-16 length | max no-EOS | pass@1 | pass@8 | coverage@8 | decision |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.00005 | 9.675 | 12.97 / 12.98 | 27.50 | 2/16 | 0.182 | 0.573 | 0.144 | viable; preferred |
| 0.00020 | 9.594 | 11.14 / 11.13 | 35.69 | 2/16 | 0.177 | 0.510 | 0.138 | viable |

Both arms are within 5% of the maximum final-16 entropy, so the frozen
tie-break selects `eta=0.00005`. Its final-16 mean multiplier is `0.001198`;
the fast arm's is `0.000030`. The preferred arm did have a transient
mid-training excursion (peak batch mean length 86.25 and peak 7/16 no-EOS)
before its price caught up, while the fast arm peaked at length 40.25 and
3/16 no-EOS. E13 therefore demonstrates endpoint recovery in this one-seed
engineering gate, not per-update constraint satisfaction or replicated
effectiveness. It authorizes a separately frozen three-seed replication; no
replication or analytical-grid job was automatically submitted.
