# Independent second review: replay proof chain

Reviewed 2026-09-05. Scope: `paper/main.tex`, the replay section beginning at `app:theory-replay` (approximately lines 1232–1505 when assigned), and the relevant current implementation in `src/oat_drgrpo/canonical_replay.py`, `learner/grpo.py`, `online_canonical_bank.py`, and `actor.py`. This review does not edit either manuscript or any scheduler state. References below use labels because the parent agent is editing line positions concurrently.

## Verdict and priorities

**No mathematical blocker in the current fixed-bank barrier, weighted/full-bank limit, or deterministic Euclidean GD lemma.** The incomplete-bank limit is correct and consequential: under these assumptions replay ultimately puts all probability on its frozen bank, including eliminating unbanked correct modes. The proofs now establish convergence, rather than merely identifying the minimizer or obtaining a subsequence with small gradient.

Recommended additions, in order:

1. Add the shared-parameter, multiple-prompt, length-normalized **exemplar energy lemma** below. It is a stronger and more accurate bridge to LM scoring than categorically excluding cross-prompt interference. It assumes descent of a joint potential and complete response probabilities; it does not certify the actual Adam/PPO/alternating training loop.
2. If space permits, add the **finite evaluation visibility** calculation. The simple exponential energy barrier is usually extremely loose; binary-KL contraction gives a sharper probability certificate from the actual replay loss. This connects the mathematical statement to what an evaluation can observe without claiming empirical detectability from positivity alone.
3. State the incomplete-bank limiting `distinct@K` formula explicitly. This is an immediate, useful memory-capacity consequence, not another substantial theorem.
4. Optional appendix material: admission-energy accounting and the underweight-time bound. Neither is necessary for the central paper. Do not add a general claim for arbitrary changing banks or merely recurrent finite updates.

Two wording refinements are worthwhile even without additions:

- Interpret prompt frequencies as **effective objective weights**, including averaging, relative step size, and any shared length normalizers. The numerical dose `.10(15/256)` is an illustrative categorical coefficient, not an identified LM dynamical dose.
- In the scope remark, replace an absolute statement that cross-prompt interference is outside all retention theory with the narrower statement that the actual shared-parameter stochastic update has not been proved to decrease a suitable joint potential. Shared parameters alone do not destroy the energy argument.

## 1. Barrier theorem: every step checked

For finite independent logits and fixed normalized positive bank weights, write

\[
 R_w=-\sum_{b\in B}w_b\log p_b,
 \qquad F=\rho R_w-\Psi(P),\qquad c=\Psi'\ge0.
\]

The uniform result is `w_b=1/k`. Since `p_b in (0,1)` initially, `R_w` is finite and nonnegative. The dynamics are exactly `dot z=-grad F`, so

\[
 F(t)+\int_T^t\|\nabla F(s)\|^2ds=F(T).
\]

`Psi(P) <= Psi(1)` implies `R_w(t) <= C_T`, with the manuscript's exact `C_T`. Every term in `R_w` is nonnegative, so `-w_b log p_b <= C_T` and `p_b >= exp(-C_T/w_b)`. There is no cancellation of negative terms. The bound is uniform over all later times, not just finite-time softmax positivity. The bank need not cover the correct set.

The potential is not coercive in logits: adding a common constant does nothing, and eliminating incorrect/unbanked coordinates requires divergent relative logits. **Coercivity is not needed or claimed.** Compactness is used in probability space; the proof does not pretend the logit trajectory has a convergent subsequence in Euclidean space.

A sharper lower bound on the energy is available:

\[
 R_w\ge H(w),\qquad F\ge F_*:=\rho H(w)-\Psi(1).
\]

Here `w` is extended by zero outside the bank. This follows from cross-entropy nonnegativity of `KL(w||p)`, and equality is attained only at the boundary distribution `p=bar w`. It is useful for rate/visibility statements but is not necessary for the existing proof, which correctly uses the weaker `F >= -Psi(1)`.

Global existence causes no hidden problem for independent logits: softmax derivatives, the replay gradient `p-bar w`, and bounded `c(P)` make the velocity bounded and locally Lipschitz. For a globally smooth parameterization, finite initial energy and a global lower bound also preclude finite-time escape along exact gradient flow: on a finite interval, path length is at most `sqrt((t-T)(F(T)-F_*))`; a finite limiting parameter point permits continuation. State a globally defined smooth model/flow if formal completeness is desired.

## 2. Full-bank and weighted convergence

For full coverage,

\[
 R_w=H(w)-\log P+\mathrm{KL}(w\|q).
\]

The derivative in `P` is `-rho/P-c(P)<0`; the unique boundary minimizing distribution is indeed `P=1,q=w`. This minimizer argument **alone** would not prove convergence, but the manuscript adds the necessary gradient/stationarity argument.

Bounded Hessian and velocity imply uniform continuity of `grad F(z(t))`. Energy dissipation implies square integrability of the gradient, so its norm tends to zero. One can prove this directly: a gradient component bounded away from zero at infinitely many separated times would, by uniform continuity, give disjoint intervals of uniformly positive integral, contradicting square integrability.

At a probability limit, the correct-coordinate equations are

\[
 0=\rho(p_c-w_c)-c(P)p_c(1-P).
\]

Summing them under full coverage gives

\[
 0=\rho(P-1)-c(P)P(1-P)=-(1-P)(\rho+c(P)P),
\]

hence `P=1` and then `p_c=w_c`. Incorrect-coordinate equations are `(rho+c(P)P)p_a=0`. The coefficient is strictly positive because `rho>0`; positivity of `c` is not additionally needed. Every subsequential limit is the same and the probability simplex is compact, proving convergence of the entire probability trajectory.

The uniform proof's intermediate argument that all correct coordinates are equal is also valid: its shared multiplier must be positive because it equals `rho/(m p_c)`. Summing coordinates is simpler and also immediately generalizes to nonuniform weights.

## 3. Smoothness, GD steps, and incomplete-bank limit

The stated global smoothness bound is valid. For `U=u_a`, `Y=1[a in C]`, under `a~p`, differentiation gives

\[
 u^T\nabla^2P\,u=E[(Y-P)(U-EU)^2].
\]

The cancellation is exact: differentiating `E[YU]-P E[U]` yields `E[YU^2]-2E[YU]E[U]-P E[U^2]+2P(E[U])^2`, which is precisely the displayed centered expression. Since `|Y-P|<=1`, its absolute value is at most `Var(U)`. Coordinate range is at most `sqrt(2)||u||`; the bounded-variable variance bound is range squared divided by four. Thus `||Hess P||op <= 1/2`. For replay, `Hess R_w=diag(p)-pp^T`, independent of the target, and the same variance argument gives `1/2`.

Also

\[
 \|\nabla P\|^2=(1-P)^2\sum_Cp_c^2+P^2\sum_Np_a^2
 \le2P^2(1-P)^2\le1/8.
\]

The chain rule gives

\[
 \|\nabla^2F\|_{op}\le \rho/2+M_1/2+M_2/8=L.
\]

This is conservative but correct. Under constant `0<eta<2/L`, the descent coefficient `a=eta(1-L eta/2)` is strictly positive. Consequently `sum_n ||grad F(z_n)||^2 <= (F(z_0)-F_*)/a`; the gradient tends to zero. The coordinate gradient is continuous as a function of `p` even when some probabilities tend to zero. Incorrect-coordinate stationarity forces all incorrect probabilities to zero, hence `P=1`; then every correct coordinate satisfies `p_a=bar w_a`. This includes **unbanked correct coordinates**, which have zero target weight. The manuscript's incomplete-bank limit is correct.

For general finite-group GRPO, the removable endpoint singularities can be made explicit. The identity

\[
 \binom Gm m(G-m)=G(G-1)\binom{G-2}{m-1}
\]

gives

\[
 c_G(P)=\frac{G-1}{G}\sum_{j=0}^{G-2}
 \binom{G-2}{j}w(j+1)P^j(1-P)^{G-2-j}.
\]

Thus `c_G` is a polynomial with positive Bernstein coefficients, proving smoothness on the closed interval and endpoint positivity directly. This closes any apparent concern about the denominator `P(1-P)` in the earlier expectation formula; it does not require differentiating a singular quotient at an endpoint.

For MaxRL,

\[
 c(P)=\sum_{j=0}^{G-2}(1-P)^j,
 \quad M_1=G-1,
 \quad M_2=\sum_{j=1}^{G-2}j=(G-1)(G-2)/2.
\]

Hence `L=rho/2+(G-1)(G+6)/16` is algebraically correct, including `G=2`. For Dr.GRPO, `M_1=(G-1)/G,M_2=0`. The proof requires Euclidean independent-logit GD; it neither supplies a neural learning-rate prescription nor justifies finite alternating replay, clipping, stochastic noise, or arbitrary preconditioning. The existing final paragraph states those limits correctly.

A short immediate convergence-rate fact, if wanted, is

\[
 \min_{0\le n<N}\|\nabla F(z_n)\|^2
 \le \frac{F(z_0)-F_*}{N\eta(1-L\eta/2)}.
\]

This is a stationarity rate, **not** an `O(1/N)` probability-convergence rate. No global strong convexity or exponential convergence has been established.

## 4. Exemplar length and prompt-frequency mapping

The singleton categorical reduction is correct:

\[
 S=-\frac1k\sum_b\frac1{L_b}\log p_b=A R_w,
 \quad A=\frac1k\sum_bL_b^{-1},
 \quad w_b=\frac{L_b^{-1}}{\sum_jL_j^{-1}}.
\]

With fixed lengths, using coefficient `rho_LM` on `S` is equivalent to coefficient `rho_LM A` on `R_w`. Consequently the independent-category limit favors shorter exemplars. Equal weighting of **mean-token scores** does not create equal sequence probabilities or equal execution-key probabilities. The current wording makes this distinction correctly.

For different prompt frequencies, if the physical averaged vector field is

\[
 \dot z_x=\nu_x c(P_x)\nabla P_x-\rho_{raw}\mu_x\nabla R_x,
\]

rescaling time by `nu_x>0` gives `rho_x=rho_raw mu_x/nu_x`. Equal effective frequencies cancel. Merely saying round-robin occurs does not establish this ratio, especially with a changing eligible set, priority scheduling, minibatch averaging, or different score normalizations.

Implementation checks:

- `canonical_replay_uniform_verified_likelihood_loss` uses an average over prompt groups and a normalized weighted average over keys within each group. Its uniform score derivative is exactly `-1/(number_of_groups * group_size)`.
- `_score_canonical_replay_rows` divides summed response-token log probabilities by each row's token count.
- The learner multiplies the replay objective by `alpha * (G-1)/G * 1/G` for the per-rollout objective; `.10*(15/256)=.005859375` is correct for `G=16`.
- `scheduled_global_replay_groups` cycles through a fixed budget of eligible prompt banks, but membership and priority selections may change; constant frequency is an averaged-model premise, not an exact property of every finite prefix.
- Fresh Dr.GRPO also contains a fixed response-length normalizer in the learner, whereas replay uses row-specific lengths. Any translation of raw optimizer loss scales to the categorical `rho` must include the common fresh normalization and per-example inverse-length factor. The numerical example should remain explicitly illustrative; it does not measure the actual LM replay/fresh ratio.

The Dr.GRPO potential contribution in the example is `k c/rho =16*(15/16)/(.1*15/256)=2560`, multiplied by `1-P(T)`. The worst-case `exp(-2560)` statement is correct and intentionally vacuous. Positive initial replay loss further weakens it.

## 5. Recommended extension: shared-parameter exemplar energy

This is the most useful additional result. It removes the singleton-category assumption for a qualitative retention statement, and handles shared prompts exactly when a common energy decreases.

Let a finite frozen collection of verified pairs be `(x_j,e_j)`, with lengths `L_j>0` and fixed coefficients `kappa_j>0`. Let `pi_theta(e_j|x_j)` be the probability of the **complete response event**, including its termination rule, and let its execution key be `b_j`. Define

\[
 S(\theta)=\sum_j\kappa_j\frac{-\log\pi_\theta(e_j\mid x_j)}{L_j},
 \qquad F(\theta)=S(\theta)-J(\theta),
\]

where `J(theta)<=J_max`. Assume initial exemplar probabilities are positive and `F(theta(t))<=F(theta(T))` for all `t>=T`. Exact gradient flow of this globally defined smooth potential is one sufficient condition; the statement itself needs only this monotonicity. Put `D=F(theta(T))+J_max`. Then

\[
 p_\theta(b_j\mid x_j)\ \ge\ \pi_\theta(e_j\mid x_j)
 \ \ge\ \exp\left(-\frac{L_jD}{\kappa_j}\right).
\]

**Proof.** `S(theta(t))=F(theta(t))+J(theta(t))<=D`. Each summand in `S` is nonnegative. Therefore `kappa_j[-log pi(e_j|x_j)]/L_j<=D`, which gives the second inequality. The first follows because the complete event producing `e_j` is contained in the event producing key `b_j`. Notice `D>=S(theta(T))>=0`, so the floor cannot exceed one. No independence of parameters across prompts, full bank coverage, parameter identifiability, or convergence of parameters is required.

For a categorical exact mean model with multiple prompts, take

\[
 J(\theta)=\sum_x\nu_x\Psi_x(P_x(\theta)),
 \qquad J_{max}=\sum_x\nu_x\Psi_x(1),
\]

with fixed nonnegative effective prompt weights and positive replay coefficients for each protected exemplar. Then the sum of fresh mean gradients is exactly `grad_theta J`; shared-parameter interference is already included. The floor may become much weaker because it uses the total multi-prompt energy budget, but remains positive.

Suggested TeX:

```tex
\begin{lemma}[Retention from a joint exemplar potential]
Fix verified complete-response exemplars $(x_j,e_j)$, lengths $L_j>0$,
and weights $\kappa_j>0$. Suppose
\[
 F(\theta)=\sum_j\kappa_j
 \frac{-\log\pi_\theta(e_j\mid x_j)}{L_j}-J(\theta),
 \qquad J(\theta)\le J_{\max},
\]
and $F(\theta(t))\le F(\theta(T))$ for every $t\ge T$.
If $e_j$ realizes key $b_j$, then
\[
 p_\theta(b_j\mid x_j)\ge\pi_\theta(e_j\mid x_j)
 \ge\exp\!\left[-\frac{L_j(F(\theta(T))+J_{\max})}{\kappa_j}\right].
\]
\end{lemma}
\begin{proof}
The weighted exemplar loss is at most $F(\theta(T))+J_{\max}$.
Each of its summands is nonnegative, which bounds the surprisal of
$e_j$. A verified complete exemplar is contained in its key event.
\end{proof}
```

Essential scope text:

> The lemma permits shared parameters and multiple prompts when the joint potential decreases. It neither implies uniform key probabilities nor establishes this descent property for the implemented stochastic, clipped, alternating optimizer. The likelihood must describe the complete response event under the relevant sampling law; a prefix score without its termination event does not by itself bound the probability of a completed verified response.

**Code-level event caution:** `materialize_canonical_replay_batch` preserves stored response tokens verbatim and appends no EOS. The actor stores `sample.token_ids` verbatim; it distinguishes stop termination from length truncation. Standard listed Qwen/Falcon templates disable custom stop strings. Admission requires both validator positivity and positive task reward, but these observations do not independently certify that every generic/template path's stored token sequence contains a stochastic termination event. The proposed lemma should state the complete-event condition, not infer it solely from teacher-forced scoring. Deterministic-horizon action policies are another valid complete-event convention. Temperature, top-p/masking and stopping policies must also match the probability law being bounded.

Counterexample to dropping the event condition: a model can give a valid-answer prefix probability one while its probability of stopping immediately after that prefix tends to zero and every continuation is invalid. Prefix likelihood remains high, but completed verified-answer probability vanishes.

## 6. Recommended extension: finite evaluation visibility

This result is about a frozen checkpoint and iid draws from the same policy. It does not need a training-convergence theorem.

For categorical target `w` extended by zero to every nonbank category, measured replay loss `R_w=r` gives

\[
 \mathrm{KL}(\bar w\|p)=r-H(w)=:d\ge0.
\]

Coarsen the category space into `{b}` and its complement. The log-sum inequality gives

\[
 \operatorname{kl}(w_b\|p_b)\le d,
\]

where `kl` is Bernoulli KL. For `0<w_b<1`, let `delta_b` be the unique root in `(0,w_b]` of `kl(w_b||delta_b)=d` (with `delta_b=w_b` when `d=0`). The binary divergence decreases strictly from infinity to zero on this interval, so `p_b>=delta_b`. For `w_b=1`, use `delta_b=e^{-r}`. This is sharper than discarding all other loss terms to get `e^{-r/w_b}`.

The contraction can also be derived without citing a theorem: fix `p_b=t`; minimizing the cross-entropy of the other bank entries subject to their total mass being at most `1-t` assigns them probabilities `(1-t)w_j/(1-w_b)` and no outside mass. The resulting minimum is `H(w)+kl(w_b||t)`.

For any valid per-key floors `delta_b`, `K` iid complete-response draws yield

\[
 E[D_{B,K}]=\sum_b[1-(1-p_b)^K]
 \ge\sum_b[1-(1-\delta_b)^K],
\]

and, by the union bound,

\[
 \Pr(\text{at least one banked key is unseen})
 \le\sum_b(1-p_b)^K
 \le\sum_b(1-\delta_b)^K
 \le k e^{-K\delta_{min}}.
\]

Thus `K >= log(k/alpha)/delta_min` is sufficient to observe every banked mode with probability at least `1-alpha`. The exact sum is often tighter. This bound uses neither a false independence assumption between the missing-key events nor replacement of expectation by a realized distinct count.

Example, for illustration only: uniform `k=16` and actual categorical replay loss `r=log 16+0.1` give binary-KL lower root `0.0051810431094`, whereas the crude floor is `1.0944832171e-20`. The sufficient 95% all-key visibility budget from the KL floor is 1114 draws. This demonstrates both the value of the sharper certificate and why an eight-draw evaluation need not see all protected keys. It is not an empirical measurement from the runs.

For LM exemplars, the simple per-exemplar energy floor above can be used directly. Alternatively, if the scored tokens specify disjoint complete exemplar events and `S_x=-(1/k)sum_b log pi(e_b|x)/L_b`, then `S_x=A R_w` with the inverse-length weights already defined in the manuscript. Apply the KL calculation to the distribution over these complete exemplar events plus an 'other response' event. Execution-key probabilities dominate their exemplar probabilities. Do not compute this certificate from the bank-softmax mean-token entropy diagnostic: that diagnostic is a different distribution.

## 7. Immediate capacity consequence of the existing limit

The existing incomplete-bank limit implies

\[
 \texttt{pass@}K\to1,\qquad
 E[\texttt{distinct@}K]\to\sum_{b\in B}[1-(1-w_b)^K].
\]

For a uniform bank this is `k[1-(1-1/k)^K]`. With fixed `k=16,K=8`, it equals approximately `6.45248842`, even though all sixteen banked modes retain probability `1/16`. Thus full retention and full visibility in a small batch are distinct. The limiting expression increases with uniform bank size for `K>=2`, while the uniform per-key replay score pressure is `rho/k`. This gives a clear memory/pressure tradeoff in the model. It is an asymptotic categorical expression, **not an upper bound on finite-time LM distinct counts**: unbanked modes can still be sampled during training and the LM does not necessarily reach this categorical limit.

## 8. Optional admission accounting; do not claim arbitrary recurrence

Finite bank growth can be analyzed without pretending the potential is unchanged at an admission. For uniform banks and fixed `rho`, adding one new key of surprisal `ell_new` to a size-`k` bank changes replay loss by

\[
 R_{new}-R_{old}=\frac{\ell_{new}-R_{old}}{k+1}.
\]

The potential therefore jumps by `rho*(ell_new-R_old)/(k+1)`. Between admissions it decreases under the corresponding exact gradient flow. If positive jumps sum to `A<infinity` and every permanently protected coordinate has target weight at least `w_min>0`, then

\[
 F(t)\le F(T)+A,
 \qquad p_b(t)\ge
 \exp\left(-\frac{F(T)+A+\Psi(1)}{\rho w_{min}}\right).
\]

For capacity `K`, no eviction and finitely many admissions with positive probabilities, take `w_min=1/K`; the finite cumulative admission cost yields a global post-admission retention bound. This result does not establish discovery, does not protect an evicted key, and does not bound the cost of admitting an arbitrarily improbable exemplar. In the actual finite-capacity/no-replacement setting, the existing theorem already applies after the last admission; this accounting merely quantifies the transition.

Arbitrary recurrent replay or pointwise positive but vanishing weights is not sufficient justification for the fixed-potential proof. Energy may rise at changes or during separate fresh updates. A simple continuous example with two always-correct categories demonstrates that positive weights at every finite time need not imply uniform retention: choose `w_2(t)=p_2(t)/2`, `w_1=1-w_2`, with replay ascent `dot z=rho(w-p)`. Then `dot p_2=-rho p_2^2(1-p_2)`, so `p_2(t)->0`, although `w_2(t)>0` at every finite time. The missing assumption is a persistent positive lower target weight or other controlled-energy condition.

## 9. Optional quantitative refresh budget

Root is separately adding the more useful instantaneous gradient-availability proposition. The following is a valid optional consequence, not required additional theorem material.

For a banked coordinate,

\[
 [\nabla F]_b=\rho(p_b-w_b)-c(P)p_b(1-P)
 \le\rho(p_b-w_b).
\]

Whenever `p_b<=a w_b` with fixed `0<a<1`, the gradient norm is at least `rho(1-a)w_b`. Under continuous flow, the total amount of time spent in this underweight region is bounded by

\[
 \frac{F(T)-F_*}{\rho^2(1-a)^2w_b^2}.
\]

For deterministic GD the number of such iterates is at most

\[
 \frac{F(z_0)-F_*}
 {\eta(1-L\eta/2)\rho^2(1-a)^2w_b^2}.
\]

These follow by summing the gradient dissipation over that region. They do not say every replay update increases `p_b`: a positive coordinate-logit update is not the same as positive probability derivative, and parameter sharing may change the direction. They do not bound real training iterations of an adaptive stochastic optimizer.

## Suggested integration scope

Keep the established replay theorem, weighted paragraph, and finite-step/incomplete-bank lemma. Add the joint-exemplar energy lemma plus its full-response/scoring-law caveat. Include finite visibility and the capacity expression as one compact paragraph or proposition if space permits. Gradient availability can remain the other principal new result. Admission accounting, occupation time, and rate details can remain in this audit unless a reviewer asks; promoting all of them would obscure the main chain.
