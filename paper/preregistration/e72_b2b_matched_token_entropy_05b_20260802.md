# E72 B2b: does breadth follow from entropy as such, or from where it is spent?

**Status: FROZEN BEFORE SUBMISSION — 2026-08-02**

## Question

What happens if Dr.GRPO is held at *xGRPO's own measured token entropy* by an
explicit entropy controller, with no rarity credit, no discovery credit, and no
replay?

The review names matched token entropy as an untested family, and it is the
family with the strongest prior: if the treatment merely keeps the policy from
becoming deterministic, then any method that keeps entropy up should buy the
same breadth, and the paper's mechanism claim collapses to "stay stochastic."
This arm is designed to be able to falsify that claim, which is why the target
is taken from the treatment's telemetry rather than tuned.

## Why the existing arms do not answer it

The manuscript already reports `grpo_entropy`, a fixed token-entropy bonus. That
arm answers a weaker question: it shows a *particular* entropy coefficient does
not buy breadth, and a reviewer may reasonably reply that the coefficient was
too small. B2b removes that reply by construction. The coefficient is not fixed
at all --- a Haarnoja-style dual controller adapts it every update to hold the
policy at a *measured* target, and the target is the entropy the treatment
itself runs at.

| arm | entropy pressure | outcome-directed credit |
| --- | --- | --- |
| matched Dr.GRPO (published control) | none | no |
| `grpo_entropy` (published) | fixed coefficient | no |
| **B2b (this arm)** | **dual-controlled to the treatment's measured entropy** | **no** |
| xGRPO | none explicit | yes |

The measured gap this arm is asked to close is large. Over the final 200 logged
updates the treatment sits at .616 nats per token on graph coloring against the
control's .0017, .402 against .0018 on PantryPlan, .184 against .0083 on
Countdown, .082 against .030 on MathIR, and .057 against .0018 on Python
factors. B2b is required to run at the first number in each pair.

## Arm

Runtime variant `matched_token_entropy_ablation`: Dr.GRPO plus a dual-controlled
MaxEnt term, base coefficient .05, dual bounds [.005, .5], controller learning
rate .003, EMA decay .7. Rarity coefficient, novelty credit, replay, and the
separate-advantage, success-conditioned, and open-set adaptation switches are
all off.

**The target, and one conversion.** Targets come from
`var/artifacts/e72_token_entropy_targets.json`, built by
`ops/exp_scaling/extract_e72_entropy_targets.py` from the treatment's own runs,
and are injected per domain by the launcher; the variant refuses to start
without one, so no run can silently fall back to a ratio-derived target. The
controller regulates *sequence* entropy while the manuscript's telemetry records
the masked-mean *per-token* entropy, so the target is the treatment's measured
per-token entropy multiplied by its measured mean response length, both from the
same runs over the same window: 2.503 (graph coloring), 1.803 (Countdown), 1.569
(Python factors), 0.329 (MathIR), 2.411 (PantryPlan). The per-token objective is
not an alternative here --- under it the learner stops emitting the
sequence-entropy observation the dual controller reads.

This conversion is the arm's one soft spot and is registered as such: where
response length is fixed by a canonical action space it is exact, and elsewhere
it carries the treatment's own length variation. The realized per-token entropy
is therefore measured on B2b's own runs and reported next to the treatment's, so
a reader can see how closely the match actually held rather than taking the
label on trust.

**One departure from compute matching, forced.** B2b carries no replay
bookkeeping, including the zero-derivative traversal the published
compute-matched control performs, because the argument validator treats
canonical bookkeeping and token-policy MaxEnt as separate treatments. The
objective is unaffected --- that traversal's derivative is identically zero in
the control --- but the arm is Dr.GRPO plus entropy rather than the
compute-matched control plus entropy, and it is reported as such.

## Cohort

Five domains x seeds 43--47 = 25 runs, from the pinned Qwen2.5-0.5B-Instruct
initialization on the common design: 12 passes, 4,608 optimizer updates,
$G = 16$, $\beta_{KL} = 0$, rollout temperature 1, the fixed 128-prompt
evaluation split, greedy decoding plus four deterministic temperature-one
$K = 8$ replicates. Per-domain data, template, response budget, and evaluation
draw seeds are inherited from the published runs. Each run is pinned to the GPU
model that trained its paired reference seed.

Comparators are the published seed-matched arms: matched Dr.GRPO and xGRPO. No
coefficient is tuned.

## Analysis, fixed in advance

Primary quantity: `distinct@8` at terminal pass 12, per domain, five paired
seeds, no pooling. Reported with paired per-seed rows and a seed-level paired
bootstrap interval (10,000 resamples). Equivalence margin $\delta = 0.15$, as in
the B3a, B1b, and confirmation protocols.

Secondary, reported whatever the primary shows: realized mean `train/entropy`
over the final 200 updates, per domain, against the treatment's, as the check on
whether the match held.

## Registered interpretations

- **P1.** B2b within $\delta$ of xGRPO in at least three domains -> matched
  token entropy is sufficient for breadth. The mechanism claim is then wrong as
  stated, Section 4.2 is re-scoped to "entropy, however obtained," and the
  reviewer's hypothesis is confirmed against us.
- **P2.** B2b materially above matched Dr.GRPO but below xGRPO in most domains
  -> entropy buys part of the breadth and direction buys the rest; report the
  decomposition and weaken any claim that entropy contributes nothing.
- **P3.** B2b at or near matched Dr.GRPO -> entropy at the treatment's own level
  does not buy breadth, and the distinction is *where* the entropy is spent
  rather than how much there is. This is the outcome that most strengthens the
  manuscript and therefore the one to state most carefully, with the realized
  entropy check shown alongside so the claim rests on a demonstrated match.
- **P4.** B2b above xGRPO anywhere -> reported as-is.

A P3 reading is only admissible for domains where the realized entropy check
shows B2b actually reached the treatment's level. Where the controller failed to
hold the target, the domain is reported as an inconclusive match rather than as
evidence that entropy does not help; that distinction is registered now so it
cannot be blurred later.

## Failure policy

CUDA OOM, non-finite loss or coefficient, traceback, malformed checkpoint,
identity mismatch, missing seed, or failure to reach the terminal budget is a
run failure; a failed run resumes only from its own source-bound checkpoint. A
domain missing any of its five terminal seeds is unreported.

Arm-specific integrity, checked on every logged update rather than assumed: the
dual controller's coefficient stays strictly inside its bounds rather than
pinned at either end for the whole run, the configured absolute target is the
one in the artifact, and rarity, novelty, and replay advantages are exactly
zero. A run violating any of these is discarded, not reinterpreted.

---

## Amendment 1 — controller units corrected, first cohort discarded (2026-08-02)

**Status: FROZEN BEFORE ANY AMENDED RUN REACHED ITS FIRST EVALUATION**

The cohort launched under the protocol above was cancelled at roughly 3,000
optimizer updates and is discarded in full. It did not measure matched entropy
and could not have, for a units error described here rather than corrected
silently.

**What went wrong.** The protocol above converted the treatment's measured
per-token entropy into a *sequence* target by multiplying by response length,
because the dual controller reads `maxent_sequence_entropy` under the default
`sequence` objective. Sequence entropy is a sum over generated tokens, so the
converted target was already exceeded at step one: Countdown's target of $1.803$
sat below the $8.40$ its eleven-token step-one responses produced. The
controller could therefore only push down. It pinned its coefficient at the
$.005$ floor by step $849$, and the residual positive bonus on a *sum* rewarded
length until responses reached the 192-token cap at $8.2$ nats per token. The
arm was a runaway entropy maximizer, not a matched-entropy control.

The registered soft spot in the protocol above anticipated an approximation
error in that conversion. This was worse and different: the target was
unreachable in principle, not merely imprecise.

**The correction.** The conversion is removed rather than repaired. Under
`maxent_objective=conditional_token_mean` the controller observes
`maxent_conditional_token_entropy`, a per-token mean, so the treatment's
measurement is the target directly --- $.616$ nats per token on Graph coloring,
$.184$ Countdown, $.057$ Python factors, $.082$ MathIR --- with no conversion at
any step. The entropy term also becomes a mean rather than a sum, so the length
incentive that drove the runaway is absent by construction.

PantryPlan is a separate and cleaner case. Its canonical action task overrides
the objective, pinning the controller to exact canonical sequence entropy over
the 64-mode support, and the argument validator requires the sequence objective
there. Its target is therefore the treatment's own measured
`canonical_exact_sequence_entropy`, $2.409$ nats against a $\log 64 = 4.159$
ceiling --- an exact quantity in the controller's own units, with no conversion
and no length variation.

**What is unchanged.** The cohort, comparators, primary quantity, equivalence
margin, registered interpretations P1--P4, and failure policy all stand exactly
as above. The secondary realized-entropy check becomes more important, not less,
and is now the first thing read on the amended cohort.

**What this costs the arm's standing.** The amended cohort is a first
measurement of matched token entropy, not a replication of one, and nothing was
learned from the discarded runs about the hypothesis --- only about the
instrument. The discarded runs are retained under
`var/data/superseded/b2b_sequence_target_20260802/` rather than deleted.

**Integrity checks, tightened.** In addition to those above: the realized
per-token entropy must come within a factor of two of the target by pass four
in at least three domains, and the dual coefficient must not sit at either
bound for more than half the run. A cohort failing either is reported as an
instrument failure rather than as evidence about entropy.

---

## Amendment 2 — controller calibrated on a screen; second cohort discarded (2026-08-02)

**Status: FROZEN BEFORE ANY CALIBRATED RUN REACHED ITS FIRST EVALUATION**

The cohort launched under Amendment 1 was also cancelled and is discarded. The
units correction there was necessary and insufficient: the arm reached 8.2 nats
per token on Graph coloring against a $.616$ target within half a pass.

**The binding constraint, identified from telemetry.** `train/reward` is
$0.000$ from step one on these domains --- Dr.GRPO earns nothing for the first
several hundred updates. The dual coefficient's floor is a *strictly positive*
entropy bonus, so during that window it is the only force acting on the policy.
It drives output toward uniform; once there no verified outcome is ever sampled,
every advantage is identically zero, and the entropy term remains the only
force. The state absorbs, and lowering the coefficient afterwards cannot escape
it. The controller was behaving correctly throughout --- it spent none of the
run at a bound --- but it cannot express what the situation requires, which is
*zero* pressure while the policy sits above target. Every domain starts above
target, because the base model is more stochastic than the trained treatment.

**Calibration, measured not chosen.** `ops/exp_scaling/screen_e72_b2b_controller.py`
ran Countdown for two passes under four controller settings, including the
discarded setting as a negative control. Only the coefficient floor and its
adaptation rate varied; the entropy target and every arm coefficient were held.

| floor | adaptation | entropy (target $.184$) | `pass@8` at pass 2 |
| --- | --- | --- | --- |
| $5\times10^{-3}$ (discarded) | $.003$ | $11.88$ | $.000$ |
| $10^{-4}$ | $.01$ | $1.25$ | $.545$ |
| $10^{-6}$ | $.003$ | $0.91$ | $.502$ |
| $10^{-6}$ | $.01$ | $0.79$ | $.383$ |

The discarded setting destroys the policy; every lower floor preserves learning,
against the published control's $.594$ terminal `pass@8`. The cohort adopts the
$10^{-4}$ floor at adaptation rate $.01$: it preserved learning best, and it
sits four log units from a useful coefficient rather than nine, so the
controller can climb off the floor within a few hundred updates once the policy
falls below target.

**What remains untested, and is registered as such.** At two passes entropy is
still above target in every setting, so the screen does not show the controller
*holding* the target from below --- the regime the arm's claim depends on. The
full cohort is therefore also the first test of that behaviour, and the pass-four
integrity check in Amendment 1 is the gate: if realized entropy is not within a
factor of two of target in at least three domains, this is reported as an
instrument failure and no P1--P4 reading is taken. That outcome is a legitimate
result about the difficulty of holding a policy at a prescribed entropy under
sparse verifier reward, and will be reported rather than retried indefinitely.

**Standing.** Two cohorts discarded, neither reaching an analysis. Both are
retained under `var/data/superseded/`. The amended cohort is a first
measurement, not a replication.

---

## Amendment 3 — the controller now observes the measured estimator (2026-08-02)

**Status: FROZEN BEFORE ANY RUN UNDER THIS AMENDMENT REACHED ITS FIRST EVALUATION**

The cohort under Amendment 2 is also discarded. Its coefficient floor was right
and its policy trained normally, but it still did not hold the target, for the
same reason as the first two in a third disguise.

**One mistake, three disguises.** The target is a measurement of
`train/entropy`, the masked-mean token entropy this manuscript reports. The dual
controller was never regulating that quantity. Under the sequence objective it
observed a sum over generated tokens, roughly $250\times$ the target; under the
per-token objective it observed `maxent_conditional_token_entropy`, a different
estimator over content tokens with EOS excluded, running $10$--$30\times$
higher. Each time the controller concluded the policy was far above target when
it was in fact below, parked its coefficient at a bound, and never applied the
pressure the arm exists to apply. Measured at pass six of the Amendment 2
cohort: realized `train/entropy` was $.117$ against a $.616$ target on Graph
coloring and $.0053$ against $.184$ on Countdown --- the second below even its
own control --- while the controller's own observable read $1.75$ and $1.02$.

**The correction is to the instrument, not to the target.** A new argument,
`maxent_observe_masked_mean_entropy`, points the controller at `entropy`, the
key under which the learner already logs the manuscript's own measurement, with
a matching entropy-units tag so a mismatched pairing raises rather than
silently regulating the wrong quantity. Every domain's target is now that
measurement verbatim and in one unit: $.616$ Graph coloring, $.184$ Countdown,
$.057$ Python factors, $.082$ MathIR, $.402$ PantryPlan. PantryPlan keeps the
sequence *objective*, which its canonical action space requires, while
observing the same estimator as every other domain --- the objective and the
observation are now independent, which is what made the previous conflation
possible.

**Unchanged.** Cohort, comparators, primary quantity, equivalence margin,
registered interpretations P1--P4, the controller calibration from Amendment 2
(floor $10^{-4}$, adaptation rate $.01$), and the failure policy all stand.

**Standing, stated plainly.** Three cohorts discarded, none reaching an
analysis, all three retained under `var/data/superseded/`. Every one was caught
by telemetry within a few passes and none produced a number that entered this
paper, but the arm has consumed roughly three cohorts of compute to establish
only that holding a policy at a prescribed entropy requires the controller and
the target to name the same quantity. If the amended cohort also fails its
pass-four gate, B2b is reported as an instrument failure and the matched-entropy
question is left open rather than pursued further before submission.
