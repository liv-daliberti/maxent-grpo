# Entropy scope and sampled-score review — 2026-09-05

**Verdict:** the current paper does not prove that MaxEnt alone inevitably
collapses. Its categorical collapse theorem excludes entropy, and its semantic
appendix explicitly disclaims an unbiased entropy gradient or general retention
claim. The universal statement would be false. Exact categorical entropy can
prevent extinction; the implemented historical predictor is a different,
bounded, sample-only update. Its absence of replay's boundary force establishes
a mechanism distinction, not inevitable collapse.

No shared manuscript, bibliography, runtime, scheduler, or ledger was changed
by this review. Optional integration text is in `entropy_theory.tex`; root owns
manuscript integration. Numerical/source checks are reproducible with:

```sh
python3 paper/audits/theory_second_review_20260905/verify_entropy_review.py
```

The resulting `verify_entropy_review.json` passes 48 exact-gradient
finite-difference cases (maximum error 7.49e-11), 36 exact finite-group
correctness-leakage cases, minimum-logit barrier checks, source-exact predictor
checks, and bounded-score enumeration. These checks supplement the proofs;
they are not proofs of neural training behavior.

## 1. What the manuscript and code currently establish

- `paper/main.tex:901` explicitly sets reference KL to zero in the categorical
  group reduction; `:1009` defines the entropy-free vector field
  `c_G(P) grad P`. The collapse result applies to that field.
- `paper/main.tex:1724` identifies the registered no-token-entropy protocol.
  `:1773` distinguishes logged token entropy from canonical-key entropy;
  `:1777` lists token entropy as a missing direct control.
- `paper/main.tex:1899` gives a correct exact conditional-entropy identity.
  `:1911` explicitly says the predictor estimator is not claimed unbiased;
  `:1942` disclaims a general entropy/support-retention conclusion.
- `src/oat_drgrpo/semantic_shannon.py:179` builds the open-set predictor from
  persistent successful counts and leave-one-out peers, with one pseudocount
  per explicit key and one structural unseen bucket. Lines 229–236 clip the
  score and center under the predictor. Lines 1400–1468 score successful
  current rows; lines 1500–1505 persist counts after the group. There is no
  history-count eviction in this fixed branch.
- `src/oat_drgrpo/learner/grpo.py:2496` dispatches the successful-row scorer;
  `:2523` detaches its tensor; `:3943` adds it after task-advantage centering.
  `src/oat_drgrpo/semantic_shannon.py:1605` adds the two advantages without
  another centering step. The active implementation applies no extra outer
  clamp in this branch despite a stale docstring mentioning one; the score's
  own bound is `|A_sem| <= eta`.
- The fixed E83 protocol explicitly states `eta=.10`, `C=5`, unit
  pseudocount, no second centering, no outer clamp, and no applied replay
  derivative (`paper/preregistration/e83_semantic_maxent_without_replay_05b_20260807.md:84`).
  Its historical statement that this therefore isolates entropy ascent is too
  strong; the current manuscript appropriately supersedes that interpretation.
- The repository also contains separate exact-sequence-entropy infrastructure:
  `src/oat_drgrpo/on_policy_maxent.py:1` and `:195` use exclusive prefix
  importance ratios so their derivatives include prefix visitation effects.
  Learner `grpo.py:679` applies the common Dr.GRPO scale, including `(G-1)/G`.
  This is a different branch from fixed semantic surprisal. Conversely,
  `grpo.py:857` implements an ordinary sampled-prefix token-entropy bonus.
  Do not describe every repository MaxEnt branch as the same estimator.

One wording correction is necessary: replace “nor preserves a key after it
disappears from on-policy history” (`main.tex:1938`) with “nor supplies a replay
gradient for a remembered key after it stops appearing in fresh rollouts.”
Counts remain remembered; what is absent is a selected exemplar score row.

## 2. Proposed theorem: exact categorical entropy prevents extinction

Let `p=softmax(z)` have `d=m+n` finite independent categories, correct mass
`P`, and objective

\[
J(z)=\Psi(P(z))+\beta H(p(z)),\quad \beta>0,
\qquad 0\le\Psi'(P)\le M<\infty.
\]

Assume sufficient smoothness for gradient flow (the paper's Dr.GRPO and
implemented MaxRL potentials are smooth). Define

\[
\bar z=d^{-1}\sum_i z_i(0),\qquad
\ell=\min\{\min_i z_i(0),\bar z-M/\beta\}.
\]

Then every coordinate remains uniformly positive:

\[
 p_i(t)\ge d^{-1}\exp\{d(\ell-\bar z)\}>0 \quad(t\ge0).
\]

**Proof.** The exact gradient is

\[
\dot z_i=p_i[\Psi'(P)(1_{i\in C}-P)
                 +\beta(-\log p_i-H(p))].
\]

Its components sum to zero, preserving `zbar`. Since
`log sum exp z >= zbar+log d` and `H(p)<=log d`,
`-log p_i-H(p)>=zbar-z_i`. At any face `z_i=ell`, the vector field is
at least `p_i[-M+beta(zbar-ell)]>=0`. The region `z_i>=ell` is forward
invariant. Conservation of the sum gives `z_i<=d*zbar-(d-1)*ell`, and the
stated probability bound follows.

If `Psi` is also concave and twice continuously differentiable, the flow
converges to the unique interior maximizer. The logit trajectory is compact
on its fixed-mean hyperplane. Along gradient ascent, `dJ/dt=||grad J||²`;
compact smooth gradient dynamics have only stationary limit points. Interior
softmax stationarity is simplex stationarity. `Psi(P)+beta H(p)` is strictly
concave, and its maximum is interior because an `epsilon log(1/epsilon)`
entropy increase dominates any bounded linear task loss at a missing
coordinate. Thus all limit points induce the same probability vector.

For Dr.GRPO, `Psi(P)=aP`, `a=(G-1)/G`, so

\[
p_i^*=\frac{\exp(a1_{i\in C}/\beta)}{m\exp(a/\beta)+n}.
\]

For implemented MaxRL, `Psi'` is positive and nonincreasing, so the same
uniqueness/convergence result applies. Both optima distribute mass uniformly
within the correct class. For generic GRPO standardization, bounded `Psi'`
still suffices for the nonextinction part, but concavity must be checked
before asserting the unique-maximizer part.

This proof received an independent PASS from `/root/e118_health`. It shows
why “entropy has no retention bound” would also be too broad. Exact entropy
can have a retention guarantee despite its logit gradient approaching zero
at the probability boundary. Its unbounded surprisal changes the dynamics.

## 3. Exact verified-key entropy and exact string entropy differ

For `q_c=p_c/P`, exact `H(q)` is uniquely maximized by uniform correct-key
mass at any fixed positive `P`. Maximizing `Psi(P)+beta H(q)` over the
probability simplex, with strictly increasing `Psi`, gives `P=1` and uniform
`q`. This is an objective statement; the preceding all-category gradient-flow
theorem is not automatically a theorem about conditional entropy.

Pure entropy flow on the correct-key simplex also converges to uniform from
finite independent logits: its minimum logit has nonnegative velocity and its
maximum has nonpositive velocity, giving a compact fixed-mean trajectory;
the only interior stationary point is uniform. We do not need an unproved
global conditional-entropy-plus-task flow assertion to refute universal
MaxEnt collapse.

For raw completion strings `Y` and deterministic execution key `C=f(Y)`,

\[
 H(Y\mid\mathrm{correct})=H(C\mid\mathrm{correct})
                       +H(Y\mid C,\mathrm{correct}).
\]

Entropy can increase inside a single key's surface realizations. If one
correct key has `N` raw completions and a second has one, exact raw entropy
is maximized by uniform mass on `N+1` strings, yielding key masses
`(N/(N+1),1/(N+1))`. For `N=1,000,000`, expected `distinct@8` is about
`1.000008`, despite maximal raw entropy and perfect correctness. A policy
supported only on the first key loses merely `log(1+1/N)` raw-entropy units.
This is arbitrarily severe sampled key concentration, not exact extinction
at a fixed finite `N` optimum.

The chain rule identifies full expected sequence entropy with the sum of
expected token entropies over all prefixes, including visitation dependence.
An average entropy measured on sampled tokens, a length-normalized local
entropy loss, clipped prefix ratios, and entropy of executed keys are not
interchangeable. The theorem above concerns exact independent output logits;
it does not transfer unchanged to a shared autoregressive neural optimizer.

## 4. Proposed lemma: bounded sample-only scores versus replay

Condition on history and draw a group `A_i iid p`. Let detached scores
`a_i` depend arbitrarily on the whole group but satisfy `|a_i|<=B`. At the
on-policy point, set

\[
\widehat g=G^{-1}\sum_i a_i(e_{A_i}-p).
\]

Then

\[
 |E\widehat g_b|\le E|\widehat g_b|
             \le2Bp_b(1-p_b).
\]

**Proof.** Apply the triangle inequality, bound each score by `B`, and use
`E|1{A_i=b}-p_b|=2p_b(1-p_b)`. Group-dependent or leave-one-out scores do
not invalidate this argument. On a group with no sampled `b`, the coordinate
is exactly `-p_b G^{-1}sum_i a_i`; if applied scores sum to zero, it is zero.

For fixed semantic surprisal, `B=eta=.10`. Pseudocounts keep predicted
scores finite but do not enumerate or score unseen exemplars; clipping bounds
the score even for a newly observed extremely rare key. Remembered counts
and proposal-only support can alter a sampled row's score, but supply no
additional `e_b` term themselves.

For fixed weighted replay, its separate update is `rho(w-p)`. A banked
coordinate has limit `rho w_b>0` as `p_b->0`, and is at least `rho w_b/2`
whenever `p_b<=w_b/2`. This supplies a probability-independent restorative
force and underlies the manuscript's replay log-barrier proof.

This distinction does **not** establish eventual collapse of every bounded
surrogate. In particular, “no sampled row” does not mean “no probability
change”: softmax normalization can change an absent category, and neural
parameters couple categories. Exact entropy has an unbounded surprisal and
does not satisfy a single global finite `B`. Its gradient also vanishes at
the boundary, yet Section 2 proves retention.

## 5. Precise estimator mismatch and a code-matched counterexample

Freeze historical/leave-one-out predictor scores `s_c` and a baseline `b`
independently of the scored action. With `q=p(.|C)`,

\[
 E_p[1_C(A)(s_A-b)\nabla\log p_A]
 =P E_q[s_c\nabla\log q_c]
   +(E_qs-b)\nabla P.
\]

**Proof.** Substitute `grad log p_c=grad log q_c+grad log P` on correct
actions and use `E_q grad log q_c=0`.

The second term is correctness pressure caused by mismatch between the
predictor-centered score and its current conditional expectation. Even at an
unclipped exact predictor, `s=-log q`, `b=H(q)`, a per-unconditional-row
average gives `P grad H(q)`, not `grad H(q)` itself. Success normalization
and score clipping must be accounted for before claiming exact unbiasedness.

For the fixed `C=5` predictor with one observed correct key, let `N>=1`
be its historical-plus-peer count. Seen and unseen masses are
`u=(N+1)/(N+2)` and `v=1/(N+2)`. The seen advantage is exactly

\[
s_N=-\frac{\eta v}{5}
       [\min\{\log(N+2),5\}+\log u]<0.
\]

Consider one correct category and one incorrect category, with correct mass
`0<P<1` and prior successful count `h>=1`. True conditional verified-key
entropy is identically zero. Conditional on `M=k` successes in a group,
every successful row gets `s_{h+k-1}<0`; every unsuccessful row gets zero.
Therefore

\[
E\widehat g_{\rm sem}
=\left[\sum_{k=1}^G {G\choose k}P^k(1-P)^{G-k}
                 \frac{k}{G}s_{h+k-1}\right](e_C-p).
\]

The bracket is strictly negative. Its induced correct-mass derivative is
the bracket times `2P(1-P)^2`, hence strictly negative, whereas the exact
conditional-entropy gradient is zero. This is a complete finite-group
counterexample to unbiased conditional-entropy interpretation, not a proof
that multiple modes inevitably disappear. The numerical audit verifies it
for `G=2,3,16`, four correctness masses, and three history sizes, using the
actual current predictor function extracted from its AST.

`paper/preregistration/e104_theory_clarification_20260817.md` already
identifies this effect as correctness leakage from the open-set predictor.
Retain that careful diagnosis: the problem is not that constant baselines
are intrinsically invalid. Here the baseline is applied only to successes,
so the correct comparison is the success-conditioned decomposition above.

## Suggested short paper wording

“Our collapse theorem concerns the verifier-only update and does not apply
to exact entropy regularization. In a finite categorical model, a positive
exact Shannon-entropy term prevents extinction. The fixed Semantic MaxEnt
comparator instead applies bounded, detached historical-surprisal scores to
sampled successes. Such scores can promote alternatives but supply no
probability-independent gradient for a missing key; verified replay supplies
that gradient through a stored exemplar. This distinction does not imply
that every entropy method collapses.”

The two proof fragments use labels `lem:entropy-scope` and
`lem:sampled-score-bound`; root may merge the latter into its gradient-
availability discussion to avoid duplicate statements.
