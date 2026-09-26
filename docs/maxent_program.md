# On-policy MaxEnt-GRPO research program

This project has one broader research program and one landed experimental
method. Keeping them separate is essential to the paper's provenance.

## Direct objective

The maintained MaxEnt method optimizes the policy distribution itself:

```text
J(theta) = E_x E_{Y ~ pi_theta(.|x)}[R(x,Y)]
           + alpha E_x[H(pi_theta(.|x))].
```

Dr.GRPO applies one shared `1/T_max` update normalization outside both reward
and entropy. It does not change their objective-level tradeoff. The active E11
path does not divide entropy by `T_max` a second time.

For a sample from behavior policy `pi_old`, define the new-policy categorical
entropy `h_t = H(pi_new(.|x,Y_<t))` and exclusive prefix ratio
`W_(t-1) = pi_new(Y_<t|x) / pi_old(Y_<t|x)`. The exact finite-horizon identity
is

```text
H(pi_new) = E_{Y ~ pi_old} sum_t W_(t-1) h_t.
```

Differentiating `h_t` sees the entire vocabulary distribution; differentiating
the prefix ratio accounts for how earlier actions change later-prefix
visitation. `src/oat_drgrpo/on_policy_maxent.py` implements and exactly tests
this prefix-ratio estimator under distinct old and new autoregressive policies. The
trainer uses one PPO epoch per fresh rollout group and clips positive prefix-
occupancy increases with PPO's ratio interval. Entropy is never mixed into the
group-relative reward advantage.

## What is landed

xDr uses the same Gibbs geometry but retains the signed Dr.GRPO update:

```text
w_xdr_i proportional to exp(U_i / tau)
L_xdr = sum_i stopgrad(w_xdr_i) L_DrGRPO_i.
```

The fixed-arm 0.5B and 3B results evaluate this xDr surrogate. E4 and E5 add
controllers over xDr's aggregation temperature; they are not MaxEnt-policy
experiments. E6's per-candidate projection failed its collapse guardrail. E7
fixed the length scale but still fit a finite candidate target sampled from
`pi_old`, whose population target is proportional to
`pi_old(y) exp(u(y)/tau)`. That is a KL-regularized improvement step, not
`E[R] + alpha H(pi)/T_max`, so the E7 grid was cancelled after the audit. E8 then
used a sampled, centered-surprisal advantage. Its controller raised alpha as
intended, but the advantage dispersion went to zero as rollout groups
concentrated. E9 added the full categorical causal gradient at the rollout-
policy tangent; E10 supplied the missing finite-step old/new prefix ratio but
still optimized normalized `H/T_max`. E11 retains that estimator and changes
the maintained objective to standard sequence entropy `H`.

## Current status

| Component | Code status | Experimental status |
|---|---|---|
| Dr.GRPO baseline | training path | landed |
| xDr detached aggregation | training path | landed at 0.5B and 3B |
| xDr entropy-feedback tau | training path and tests | partial exploratory trajectories at 0.5B, 3B, and 7B |
| Per-candidate `1/T_i` projection | retired E6 path | failed 0.5B guardrail; larger jobs cancelled |
| Fixed-`T_max` candidate projection | retired E7 reference code | smoke passed, then grid cancelled after objective audit |
| Sampled-advantage on-policy MaxEnt | retired E8 estimator | smoke plus partial 0.5B failure diagnostic |
| Standard prefix-ratio MaxEnt | E11 training path and exact tests | E12 found no safe entropy-effective fixed dose; analytical grid held |
| Expected-length-constrained MaxEnt | E13 training path and exact tests | single-seed engineering gate passed; replication pending |
| Adaptive MaxEnt coefficient | proportional and Haarnoja-dual training paths | blocked before an adaptive standard-objective smoke |

The E4/E5 controllers observe token entropy and act on signed-surrogate xDr's
aggregation temperature. E11's controllers instead observe the raw importance-
weighted sum of new-policy categorical entropies and act on its coefficient
`alpha`. A separately named per-`T_max` diagnostic preserves historical
comparability. The fixed, proportional, and dual branches differ only in how
that coefficient is selected. E6--E11 are separately recorded
under `paper/preregistration/`.

The historical E10 smoke verified the exact ratio-corrected gradient path, but proportional
and dual control retained only 17.7% and 11.7% of their frozen target. A fixed
`alpha=0.50` stress arm ended at 168.75 tokens, with 14/16 responses lacking
EOS and zero evaluation accuracy. The maintained gate fails closed, and no
analytical direct-MaxEnt grid job has been submitted. This is an estimator
repair with a negative control result, not a landed MaxEnt effectiveness claim.

E11's live literal-coefficient gate verifies the new units: raw entropy 62.43
nats corresponds to 0.3251 after division by the 192-token budget. It also
fails behaviorally at step 32, with mean length 87 and 7/16 responses lacking
EOS. No adaptive or analytical E11 job was released.

E12 then calibrated fixed standard-objective coefficients
`{0.0005, 0.0010, 0.0015, 0.0020}` for 128 updates. The first three arms were
behaviorally safe but retained only 1.72--2.11 raw entropy nats in their final
16 updates, below the frozen 4.984-nat threshold. The `0.0020` arm retained
48.46 nats only with mean length 60.76, a 121.94-token batch, and as many as
10/16 no-EOS rollouts. The preregistered rule therefore selects no fixed
coefficient. The next MaxEnt experiment must impose an explicit expected-
length constraint; the analytical grid remains held.

E13 implements that constraint directly: maximize `E[R] + alpha H` subject
to expected generated response length at most 16 tokens. An exclusive-prefix
importance estimator supplies expected new-policy length, the actor uses the
conservative PPO cost branch, and a separate projected multiplier prices
violations. At fixed `alpha=0.002`, both dual-rate arms passed the frozen
single-seed gate. Their final-16 entropy/actor-length pairs were 9.675/12.97
and 9.594/11.14; the preregistered tie-break selected dual rate `0.00005`.
The preferred arm transiently reached mean length 86.25 and 7/16 no-EOS
before recovery, so this is endpoint engineering evidence, not a claim of
per-step safety or replicated effectiveness. A three-seed replication and
the analytical grid remain unsubmitted.

The earlier semantic/remix trainer is not part of this program's maintained
implementation. Future MaxEnt experiments should start from E11's standard-
objective prefix-ratio gradient rather than inherit that system, E8's sampled-
group carrier, or E10's normalized coefficient units.
