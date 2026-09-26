# Strengthening the theory in the current ModeBench / Re:Max paper

Research assessment, 2026-09-21. Read against `paper/main.pdf`, the main text and Appendix P in `paper/main.tex`, and the cited primary papers. This is a proposed theory package, not an edited manuscript or a claim of new neural-training guarantees. The supporting notes contain proofs; the verification script checks selected identities by exact enumeration and numerical differentiation.

## Recommendation

Yes: strengthen the substance, and change which results carry the main-text argument. Do not try to make the standard replicator specialization look like a general theorem about LLM optimization. The useful contribution can instead be:

> Outcome-only advantages do not add a mode-selective component to the per-prompt mean gradient. Sampling nevertheless creates diversity loss even from an exactly balanced policy. Verified mode replay can reverse this pressure, with a quantitative condition on its dose and a finite recovery bound. Its effect under neural optimization depends on measurable gradient interference.

The first claim can be stated for general differentiable policies. The second requires an actual sampled-update analysis. The third should predict changes in PCMD during finite training, rather than relying on an asymptotic probability floor that may be astronomically small. The fourth connects the mechanism to the implementation without pretending to prove arbitrary AdamW convergence.

The recommended minimum package is **a strengthened P.6, a stochastic symmetry-breaking proposition, and a finite-time replay theorem**, accompanied by one focused mechanism experiment. Exact discrete-gradient and finite-evaluation corollaries are inexpensive useful additions. A general stochastic neural-collapse theorem is not the right target: counterexamples show that some versions of that claim are false.

## What is already present

The appendix contains more than the quoted objection suggests. In addition to P.4 and P.6, it has the outcome-binary class extension, replay's fixed-bank barrier, convergence to bank weights in the categorical model, a complete-response retention lemma allowing shared parameters conditional on joint-loss control, a visibility bound, and reference-KL comparisons. Those should not be proposed as new work.

The weak points are the restrictive update model, a mostly asymptotic account of retention, and the missing link from available replay gradients to useful updates in the actual model. Section 3.1 opens with language implying that binary rewards and fresh sampling make collapse follow; its later geometry qualification is doing essential work, not describing a technical detail. The main text should lead with the result that survives general parameterization and explicitly identify the conditions for the collapse and recovery results.

Sources already close to the current collapse argument include [Sinha et al.](https://arxiv.org/html/2601.21669v1), which derives probability-weighted softmax-logit amplification, and [UCPO](https://arxiv.org/html/2605.00365v1), which analyzes binary-objective indifference, sampling imbalance, and conditional uniformity. [MaxRL, Theorem 5](https://arxiv.org/html/2602.02710v3) already derives the practical estimator's truncated mean gradient. The distinct contribution should therefore be the precise class characterization, finite-sampling effects, and the quantitative role of verified memory. Novelty of the complete proposed package still needs a broader literature check; the algebra being new to this manuscript is not by itself a novelty claim.

## 1. Make P.6 a theorem about neural mean gradients and their noise

Fix a prompt, a differentiable response distribution with the usual score-function regularity, binary verifier reward, and iid on-policy samples. Let

\[
P(\theta)=\Pr_\theta(R=1),\qquad
\widehat g_a=G^{-1}\sum_i a(R_i,R)\nabla_\theta\log\pi_\theta(Y_i).
\]

The advantage is detached and depends only on the reward outcomes, as in the implemented estimators. Then

\[
\mathbb E[\widehat g_a\mid\theta,x]=c_a(P)\nabla_\theta P.
\]

**Independent logits are unnecessary for this identity.** Condition on the reward vector and use
\(\mathbb E[\nabla\log\pi(Y)\mid R=1]=\nabla\log P\) and the corresponding failure identity. This works directly with autoregressive sequence scores; there is no need to identify one response string with one mode or to assume aggregation commutes with training.

The useful stronger result also characterizes what the identity leaves out. Set \(v=\nabla P\), \(\Sigma_r=\operatorname{Cov}(\nabla\log\pi(Y)\mid R=r)\), and

\[
b_R=G^{-1}\left[Ra(1,R)/P-(G-R)a(0,R)/(1-P)\right].
\]

Then the law of total covariance gives exactly

\[
\operatorname{Cov}(\widehat g_a)
=\operatorname{Var}(b_R)vv^\top
+G^{-2}\mathbb E\left[Ra(1,R)^2\Sigma_1
+(G-R)a(0,R)^2\Sigma_0\right].
\]

Thus two objectives can have the same normalized mean direction and different noise. For an observable \(h\) with Hessian norm at most \(L\), a normalized SGD step \(\theta^+=\theta+\eta\widehat g_a/c_a\), where \(c_a>0\), satisfies

\[
\left|\mathbb E[h(\theta^+)-h(\theta)]
-\eta\langle\nabla h,\nabla P\rangle\right|
\le\frac{L\eta^2}{2c_a^2}\mathbb E\|\widehat g_a\|^2.
\]

This provides a finite-step statement and separates drift from fluctuations. It does not establish whole-training equivalence.

Two boundaries must be explicit:

- **Shared prompts:** the aggregate update is \(\sum_x\nu_x c_a(P_x)\nabla P_x\). Changing the coefficients can change its direction, through difficulty weighting and interference. A concrete two-prompt counterexample in the supporting note has MaxRL increase a prompt's diversity while Dr.GRPO leaves its instantaneous diversity unchanged, despite purely binary rewards.
- **Response-dependent weights and optimization:** inverse-length factors introduce score/length covariance terms; PPO reuse and clipping, Adam moments, and weight decay can also change the update. These are mechanisms to quantify, not details that the per-prompt identity eliminates.

For inverse-length weighting \(h(Y)=1/L(Y)\), the exact residual beyond a scalar correctness gradient is a weighted sum of \(\operatorname{Cov}(h,\nabla\log\pi\mid R=r)\). This turns the existing length exclusion into a quantitative, testable extension.

**Paper value:** P.6 becomes a theorem covering the neural policy's local expected gradient, with an explicit account of why different practical algorithms may still behave differently. Promote that statement to the main text; retain the replicator result as a specialization.

## 2. Prove finite-step diversity loss, including sampling noise

Let \(D(q)=\mathrm{PCMD}(q)=1-\sum_c q_c^2\), and write
\(V(q)=\sum_cq_c^3-(\sum_cq_c^2)^2=\operatorname{Var}_{C\sim q}(q_C)\).

Under the existing categorical mean flow,

\[
\dot D=-2c(P)P(1-P)V(q).
\]

This is an exact change in the metric the paper reports. It proves monotonic PCMD decline at every nonuniform interior correct distribution, not merely an eventual winner. At matched correctness,

\[
\frac{dD}{d\operatorname{logit}P}
=-\frac{2V(q)}{\sum_cq_c^2+\sum_ar_a^2},
\qquad r_a=p_a/(1-P),\ a\in\mathcal N.
\]

The estimator coefficient cancels. The prediction of a common PCMD-versus-correctness curve is restricted to the deterministic categorical flows initialized identically.

**Discrete deterministic strengthening.** For exact mean-gradient updates at a finite positive step size,

\[
q_c^+=\frac{q_c\exp(hq_c)}{\sum_dq_d\exp(hq_d)},
\qquad h=\eta c(P)P(1-P)>0.
\]

This distribution majorizes \(q\), so PCMD decreases unless \(q\) is uniform. The winner-take-all conclusion extends to positive finite steps with infinite accumulated learning rate, under the coefficient regularity in the supporting proof. No infinitesimal limit is needed. This remains exact-gradient tabular optimization, not stochastic AdamW.

**More valuable stochastic strengthening.** Start with perfectly uniform correct modes, \(q_c=1/m\), where the deterministic conditional flow is stationary. For one sampled centered group update, every \(\eta>0\) gives

\[
\mathbb E[D(q^+)]<1-1/m,
\qquad \mathbb E[q^+]=\operatorname{Uniform}(m).
\]

Proof: PCMD is maximized uniquely at uniformity. A group with exactly one correct response has positive probability and makes one correct logit larger than the others. Every group yields PCMD at most its initial maximum, and this event yields strictly less. This is a statement about actual finite sampling that mean-flow analysis misses entirely.

For the paper's centered estimator with group multiplier \(w(R)\), the small-step expansion is

\[
\mathbb E[D(q^+)]
=1-1/m-\eta^2\frac{m-1}{m^3}
\mathbb E\left[\frac{w(R)^2R(G-R)^2}{G^4}\right]+O(\eta^3).
\]

For Dr.GRPO the expectation is
\((G-1)P(1-P)[1+(G-2)(1-P)]/G^3\), approximately \(P(1-P)^2/G\) at large \(G\). This predicts a quadratic learning-rate effect and a group-size effect on initial symmetry breaking. Those are per-update predictions; match drift and account for rollout cost when comparing objectives.

This one-step theorem does not assert that expected PCMD decreases at every state, that the initial leader always wins under noise, or that neural stochastic optimization eventually collapses. A more ambitious but tractable special case is fixed-step \(G=2\) categorical training, which becomes an exponentially reinforced urn at mixed-group times; its optional proof is in the inertness note.

**Operational finite-time collapse.** Replace impossible finite-time softmax support loss with invisibility at a specified evaluation budget. If the initial unique correct leader has mass \(a\), define
\(A=\sum_{c\ne *}q_c(0)/(a-q_c(0))\). At effective time \(\tau\),

\[
\Pr(\text{any other correct mode appears in }K\text{ draws})
\le KAe^{-a\tau}.
\]

The proof and a correctness-indexed version are in the finite-time note. This is an operational sufficient bound, often conservative, and it preserves the distinction between low probability and zero support.

## 3. Replace the qualitative replay floor with a finite-time balancing theorem

Take a fixed bank \(B\) of \(k\ge2\) categorical correct modes, with equal replay weights. Let \(M=\sum_{b\in B}p_b\), \(r_b=p_b/M\), and \(D_B=1-\sum_br_b^2\). Full correct-support coverage is not needed. Then

\[
\frac{d}{dt}\log\frac{p_i}{p_j}
=-[\rho-c(P)(1-P)](p_i-p_j),\qquad i,j\in B,
\]

and therefore

\[
\boxed{\dot D_B=2M[\rho-c(P)(1-P)]
\operatorname{Var}_{b\sim r}(r_b).}
\]

For a nonuniform bank, replay increases bank-conditional diversity precisely when

\[
\boxed{\rho>c(P)(1-P).}
\]

If \(B=\mathcal C\), \(M=P\) and \(D_B\) is the paper's PCMD. For a partial bank this theorem concerns the distribution conditional on the bank, not overall PCMD or unbanked modes.

This explains an important distinction the existing asymptotic result hides: **any positive dose may suffice eventually in the ideal model, while a small dose can still permit diversity to decline throughout a practical training interval.** At equal correctness MaxRL also has stronger concentration pressure in this model, even though its larger correctness coefficient may assist discovery. Faster fresh learning and replay dose should therefore be analyzed together.

There is an explicit recovery bound. Let

\[
H(t)=\log\frac{\max_b r_b(t)}{\min_b r_b(t)},\qquad
S(T,t)=\int_T^t M(u)[\rho-c(P(u))(1-P(u))]du.
\]

If the integrand is nonnegative on the interval,

\[
e^{H(t)}-1\le(e^{H(T)}-1)e^{-S(T,t)/k},\qquad
\min_b r_b(t)\ge\frac{1}{1+(k-1)e^{H(t)}}.
\]

This converts accumulated replay strength into a finite time to recover a desired minority-mode mass. With lower bounds on the bracket and on \(M\), it becomes an explicit bound in training time. A finite-step exact-gradient analogue with a step-size condition is proved in the replay note.

The implemented loss averages **mean-token** exemplar scores. Unequal exemplar lengths produce unequal categorical weights, proportional to inverse length; one exemplar is also only part of a key event. Consequently the exact uniform-bank threshold is a clean mechanism theorem, not an exact equation for the implemented LM. Keep that distinction adjacent to the statement, and test equal-length or sequence-sum variants rather than assuming it away.

**Paper value:** this is the most useful new result for explaining why the proposed method works, when its dose is sufficient, and what to measure. It is more informative than another exponentially tiny probability floor.

## 4. Connect available replay signal to the actual optimizer

This is the key empirical bridge. A nonzero logit gradient does not establish that a rare exemplar's likelihood improves after a shared-parameter update.

Let \(\ell_b(\theta)=\log\pi_\theta(e_b\mid x_b)/L_b\), \(h_b=\nabla\ell_b\), and \(r=k^{-1}\sum_jh_j\). For a step
\(\Delta=\eta H(g_{\rm fresh}+\lambda r)+u\), Taylor's theorem gives

\[
\Delta\ell_b\ge
\eta h_b^THg_{\rm fresh}
+\frac{\eta\lambda}{k}\sum_j h_b^THh_j
+h_b^Tu-\frac{B_b}{2}\|\Delta\|^2,
\]

when the Hessian norm is bounded by \(B_b\) along the step. The replay contribution depends on a row sum of the score Gram matrix, not just a positive self-inner-product. Shared gradients can conflict; the supporting note gives a one-parameter example where uniform replay initially lowers one rare banked exemplar's probability.

For AdamW, the preconditioner changes when the gradient changes. Do not silently treat it as a fixed matrix in an intervention. Use cloned model **and optimizer** states for replay-on/off comparisons, or analyze the actual realized update \(\Delta\) directly. Measure complete-exemplar likelihood and key probabilities separately.

A practical experiment is to select a few existing checkpoints, obtain matched fresh-only and replay-on updates across doses, and measure the likelihood changes of banked exemplars absent from the fresh batch. Record the first-order gradient predictions and their discrepancy from the actual step. This tests exactly the implication the current appendix cannot establish for the implementation.

A general stochastic energy-descent theorem is possible under smoothness, unbiasedness, variance, and step-size assumptions, but these assumptions would need to be checked against PPO, alternating replay, Adam state, and changing banks. Adding a standard SGD lemma without addressing those mismatches would not solve the reviewer's objection.

## Optional method-specific extension: the variance cost of rediscovery

The existing bounded-score argument leaves an obvious question: why not cancel rarity with inverse-probability weighting? A short proposition can quantify the price for estimators whose direct mode contribution is zero unless that mode appears.

Let \(E_b\) be the event that a fresh group contains mode \(b\), with
\(\alpha=1-(1-p_b)^G\). If a scalar direct-contribution estimator \(T_b\) is zero off \(E_b\) and has expectation \(w_b>0\), Cauchy--Schwarz gives

\[
\operatorname{Var}(T_b)\ge w_b^2(\alpha^{-1}-1).
\]

For rare modes this scales at least as \(w_b^2/(Gp_b)\). The standard estimator \(T_b=w_bN_b/(Gp_b)\) has variance \(w_b^2(1-p_b)/(Gp_b)\). Stored exemplars remove the requirement to rediscover that response before evaluating its training score; the implemented full-bank pass is particularly direct.

This is **not** a lower bound for all neural estimators or all algorithms: analytic expectations, model-based information, control variates with contributions outside the event, and shared-parameter transfer may fall outside the premise. Present it as a sampling-cost explanation for fresh inverse weighting, not a universal impossibility theorem. It complements the substantive distinction between knowing a mode once and repeatedly finding it again.

## Correctness and scope repairs before adding stronger claims

1. **Replay's expanded advantage class is too broad as stated.** P.13 allows any bounded reward-measurable advantage but uses increasing \(\Psi\) and the upper bound \(\Psi(1)\). Require nonnegative correctness coefficient, or replace \(\Psi(1)\) with \(\sup_{s\in[0,1]}\Psi(s)\) for retention alone. Convergence to correctness one needs an additional condition such as \(\rho+Pc(P)>0\). With \(c=-1,\rho=.2,k=2\), a stationary policy has correctness .2 and banked probabilities .1; the current floor formula incorrectly exceeds 29.

2. **The KL recovery rate needs repair.** In the two-correct-mode reduction the exact dynamics give a leading recovery time \(1/[2\beta s\log(1/s)]\), rather than the displayed coefficient one. The general proof's remainder is of the same order as its claimed leading term. Also \(\Theta(1/[s\log(1/s)])\) is not \(\Theta(1/s)\). Restrict the theorem to a proved reduction or derive the correct multi-mode comparison bounds.

3. **A bounded KL penalty can exclude a boundary on sufficiently low sublevels.** In fact \(\inf_{p:p_b=0}\mathrm{KL}(p\|\mu)=-\log(1-\mu_b)\). The correct contrast is that every finite cross-entropy sublevel excludes loss of a protected coordinate, whereas an arbitrary finite reverse-KL sublevel need not. Failure of one loose energy argument is not an impossibility theorem for KL retention or convergence.

4. **Per-prompt mean-direction equivalence is not equivalence of shared, stochastic training.** Tighten statements such as “no outcome-binary objective can change which mode wins” and “group size is not a remedy” to their actual assumptions. The multi-prompt counterexample and stochastic covariance formula show precisely why.

5. **Absent samples do not imply zero neural response.** The fresh-sample statement concerns a direct, sampled contribution. Shared parameters and normalization can change an unsampled mode; memory makes a verified target available without rediscovery, but does not universally guarantee its probability increases.

6. **Do not infer held-out guarantees from training-bank retention.** The complete-exemplar bound protects those prompt--response pairs under the same decoding law. It gives neither uniform whole-key probabilities nor a theorem about unseen prompts. Keep held-out improvements as experimental evidence.

7. **The Bernstein sign condition is sufficient, not necessary.** Some negative control points do not imply a negative polynomial coefficient anywhere. A coefficient can also have an interior zero and stop a trajectory rather than reverse it. The prose after P.7 should not identify every excluded advantage with an objective that sacrifices correctness.

These are local mathematical repairs; they do not invalidate the empirical comparison. They matter because broadening the narrative without repairing them would expose more weaknesses to a theory reviewer.

## Focused validation plan

| Check | Purpose | Status |
|---|---|---|
| Exact enumeration of small groups | Check drift, covariance, and the stochastic PCMD expansion | Completed for selected finite cases |
| Tabular dose and step-size sweep | Test the replay threshold and finite-time contraction, with equal and unequal lengths | Proposed |
| Small shared-parameter autoregressive model with enumerable outputs | Separate independent logits, parameter sharing, stochastic SGD, and Adam | Proposed |
| One-step interventions at existing LM checkpoints | Check whether replay improves absent bank exemplars and whether interference predicts exceptions | Proposed |
| Limited dose/length-loss ablation in a domain with several discoverable modes | Establish relevance of effective dose and normalization to actual key diversity | Proposed |

Use the existing training setup and optimizer states for the neural intervention. Match initial policies and fresh samples. For stochastic objective comparisons, report both matched learning rate and matched mean-update scale; otherwise a larger coefficient confounds stronger drift with stronger noise. Distinguish bank-conditional diversity, full success-conditional PCMD, exemplar likelihood, and bank occupancy.

The controlled experiment should be allowed to falsify the mechanism's applicability. A poor fit is evidence to narrow the explanatory claim, not a reason to relabel the fitted quantity as a neural guarantee.

## Proposed main-text organization

Give Section 3.1 approximately one clear theorem statement and two compact consequences, with detailed proofs in Appendix P:

1. **Per-prompt mean-gradient limitation:** state the neural version of P.6 and immediately qualify its stochastic and multi-prompt scope.
2. **How diversity is lost:** show the PCMD derivative in the categorical specialization, then state that finite sampling can break exact symmetry even when the conditional mean flow cannot.
3. **How memory changes the update:** display the replay-versus-fresh pressure threshold, identify the fixed-bank/equal-weight assumptions, and explain the finite-time implication.

Suggested prose, after the new propositions are integrated and the repairs made:

> For a fixed prompt, any detached advantage depending only on binary verifier outcomes has an expected on-policy gradient parallel to the gradient of correctness. This statement holds for differentiable neural policies: changing the advantage changes its scale, but introduces no separate mode-balancing term. It does not imply identical stochastic or shared-prompt training trajectories, because gradient covariance and prompt weighting can differ. Under independent categorical logits, this mean update decreases success-conditional mode diversity whenever the correct distribution is nonuniform. Finite sampling can also break an exactly uniform allocation: one sampled centered update strictly decreases its expected PCMD. Verified replay adds an independently available gradient for previously discovered modes. In the equal-weight fixed-bank model, its balancing effect exceeds the fresh concentration pressure when its effective dose satisfies \(\rho>c(P)(1-P)\), yielding an explicit finite recovery bound. Our neural update audit tests whether this restoring effect survives shared parameters and the implemented optimizer.

The last sentence is proposed language for a **future completed experiment**, not something currently established. Retain the standard replicator convergence theorem in the appendix as supporting context. Do not advertise a general theorem that binary-reward LLM training must collapse or that any positive replay coefficient prevents practical mode loss.

## Proof notes and numerical checks

- [Neural inertness, covariance, discrete collapse, and counterexamples](../audits/theory_strengthening_20260921_inertness.md).
- [Finite-time PCMD, sampled symmetry breaking, and visibility](../audits/theory_strengthening_20260921_finite_time.md).
- [Replay threshold, recovery, discrete updates, and optimizer response](../audits/theory_strengthening_20260921_replay.md).
- [Verification script](../audits/verify_theory_strengthening_20260921.py).

Run `python paper/audits/verify_theory_strengthening_20260921.py`. On this workspace it passed: 200 full-/partial-bank drift cases agreed to maximum absolute error 2.3e-11; the score covariance decomposition agreed to 2.3e-16; exact group enumeration approached the predicted quadratic PCMD-loss coefficients for Dr.GRPO and MaxRL. The script also reproduces the signed-advantage counterexample and the factor two in binary KL recovery dynamics. These are mathematical sanity checks, not substitutes for proofs or evidence about the trained neural models.

`paper/main.tex` and `paper/main.pdf` were not edited for this assessment.
