# Second proof and theoretical scope review — 2026-09-05

The existing categorical mean-update, collapse, shared-correctness, replay-retention, and deterministic finite-step arguments withstand a second independent review under their stated assumptions. Both paper versions now have a clearer proof chain and additional results linking collapse, entropy, replay signals, bank coverage, and finite evaluation. The new statements are proved in the appendix; they are not presented as established guarantees for the implemented neural optimizer or as claims of novelty for elementary inequalities.

Two headline interpretations need correction:

- **“More right eventually collapses” needs an update-geometry condition.** Collapse follows for independent logits and for a fixed geometry `I + λ vvᵀ` with finite λ and a unique initially largest correct mode. A purely shared correctness direction preserves every conditional correct-mode ratio while correctness approaches one. Increasing correctness alone is therefore insufficient to imply collapse.
- **“MaxEnt alone inevitably collapses” is false as a universal statement.** Exact Shannon entropy over a finite categorical output space, with a fixed positive coefficient and the stated bounded correctness potential, keeps all category probabilities uniformly positive. The historical, clipped, sampled semantic score in the experiments is a different estimator. Its bounded-score property does not prove inevitable collapse either. The manuscript now states the valid distinction instead of importing the base theorem into an entropy-regularized update.

## What the revised proof chain establishes

| Question | Precise result and assumptions | Location |
|---|---|---|
| Why do the three fresh objectives share a collapse mechanism? | Their exact on-policy group expectations are positive scalar multiples of the correctness gradient. The denominator-free binomial expression makes endpoint regularity explicit. MaxRL retains the finite-group `G−1` potential. | GRPO and MaxRL mean-update lemmas; [full manuscript](/n/fs/similarity/maxent-grpo/paper/main.tex:922) |
| Can correctness improve while conditional breadth shrinks? | Under independent categorical logits, a unique initial maximum wins. Tied maxima preserve a smaller uniform support; a fully uniform initialization is exceptional. Collapse is asymptotic, not a finite-time zero probability. | Collapse theorem and tie corollary; [full manuscript](/n/fs/similarity/maxent-grpo/paper/main.tex:1050) |
| How much does a shared correctness direction delay collapse? | At matched correctness, the shared-direction theorem preserves more sampled modes. The added corollary gives `log q_c / log(1−P) → 1/(λ+1+1/n)` for every minority mode and fixed finite λ. This is a logarithmic asymptotic, not an exact finite-accuracy power law or wall-clock rate. | New collapse-rate corollary; [full manuscript](/n/fs/similarity/maxent-grpo/paper/main.tex:1235) |
| Does entropy itself force collapse? | No. Exact finite categorical entropy gives an explicit positive floor. With a concave correctness potential the policy converges to a unique interior optimum; linear Dr.GRPO gives a Gibbs distribution, uniform conditional on correctness and with positive incorrect mass. | New entropy lemma; [full manuscript](/n/fs/similarity/maxent-grpo/paper/main.tex:1313) |
| What differs for a bounded sampled semantic score? | For detached, possibly group-dependent advantages bounded by B, the expected magnitude of a rare category's logit component is at most `2B p_b(1−p_b)`. Replay instead tends to the positive component `ρ w_b`. This distinguishes boundary behavior but does not establish inevitability of collapse. | New bounded-score lemma; [full manuscript](/n/fs/similarity/maxent-grpo/paper/main.tex:1371) |
| What exactly does “replay refreshes the gradient” mean? | A fresh group has a mixed reward with probability `1−P^G−(1−P)^G ≤ G min(P,1−P)`. All-equal groups have zero fresh task gradient. Fixed-bank replay has expected logit vector `ρ(w−p)`; near a banked single-mode collapse its norm remains `ρ sqrt(1−1/k)` for a uniform k-key bank. | New gradient-availability lemma; [full manuscript](/n/fs/similarity/maxent-grpo/paper/main.tex:1412) |
| Which modes are retained? | A fixed positive replay weight gives a uniform probability floor for every banked mode. Full coverage yields the stated uniform or weighted limit. A frozen incomplete bank ultimately excludes unbanked correct modes too in this categorical model. | Retention theorem and finite-step/incomplete-bank limit; [full manuscript](/n/fs/similarity/maxent-grpo/paper/main.tex:1459) |
| Can the retention argument accommodate shared parameters and real response scores? | Yes, if a fixed joint potential combining weighted complete-response negative log likelihoods and a bounded-above fresh objective does not increase. Each stored response then has a positive probability floor, and its execution key has at least that probability. | New complete-exemplar energy lemma; [full manuscript](/n/fs/similarity/maxent-grpo/paper/main.tex:1736) |
| Does positive probability mean a key appears in eight samples? | No. For bank floors δ_b, expected banked distinct count is at least `Σ[1−(1−δ_b)^K]`; the probability of missing any banked key is at most `k exp(−K δ_min)`. Loose energy floors may require impractically large K. | Finite evaluation visibility paragraph immediately after the energy lemma |

The fresh-group probability is an opportunity for a task signal, not a lower bound on its magnitude. The replay vector is a fixed-bank expectation; a single sampled exemplar differs. A nonzero categorical logit vector can be annihilated by a degenerate neural Jacobian. Replay also need not increase the next group's mixed-reward probability: that probability is largest at correctness one half.

## Historical semantic score: a concrete counterexample

The semantic appendix now decomposes a successful-row score into a conditional-mode term and a correctness term:

`E[1_correct (s−b) ∇log p] = P E_q[s ∇log q] + (E_q[s]−b) ∇P`.

A historical predictor's mean need not equal the current conditional mean. Even exact current conditional surprisal with its matching baseline produces `P ∇H(q)` when averaged over all unconditional rows. It is not the unscaled conditional-entropy gradient.

The added code-matched counterexample uses one correct key, for which conditional key entropy is identically zero. The historical predictor's reserved unseen bucket nevertheless gives that seen key a strictly negative semantic advantage. This is a precise estimator mismatch; it is not a proof of long-run collapse. The audit also corrects the history description: fixed historical counts remain stored when a key stops appearing in fresh rollouts, but remembered counts do not themselves provide a teacher-forced exemplar score row. The old preregistration is preserved as historical evidence, and the current manuscript supplies the clarified interpretation.

## Changes that make the proofs easier to follow

- Added a roadmap and separated the optimization clock, training group size, evaluation sample size, mode counts, and bank size.
- Defined the all-equal group endpoint convention and exhibited the positive polynomial extension of the expected-gradient multiplier.
- Made the tie cases and finite-time versus limiting support explicit.
- Filled in the convergence step from finite gradient-energy integral plus uniform continuity to vanishing gradients, rather than relying on identifying a minimizer alone.
- Explained the categorical Hessian/variance bound used for deterministic step-size guarantees.
- Stated exactly where complete response events, termination, length normalization, bank weights, and the sampling law enter the language-model bridge.
- Removed an overly broad statement that no method could protect a never-sampled mode. The coverage restriction is now explicitly a limitation of the replay term's guarantee.
- Tightened the full-paper conclusion and limitations while preserving their results, substantive scope, and incomplete-evidence caveats, restoring the nine-page main-text boundary.

## Additional work a reviewer is most likely to request

1. **Connect the energy premise to the actual optimizer.** The current proof does not establish descent for clipped PPO, Adam/preconditioning, alternating fresh/replay steps, or stochastic minibatches. A useful next theoretical result would bound the increase of a suitable joint potential under those updates, with explicit noise and step-size conditions. A small-step deterministic categorical result is already present; it is not an Adam theorem.
2. **Separate admission from retention.** Measure per-key discovery time, bank admission, eviction, replay exposure, and later key probability or survival. The fixed-bank result does not prove the policy discovers missing keys, and bank capacity limits protected support. A changing-bank theorem would need explicit admission/eviction and energy accounting assumptions.
3. **Make the entropy comparison exact where tractable.** On an enumerated categorical or action-menu problem, compare exact entropy, the historical clipped estimator, and replay at specified doses. Measure the predictor baseline's correctness component. A failed historical estimator cannot justify a universal claim about exact entropy regularization.
4. **Audit complete-response likelihood and evaluation sampling.** Confirm whether replay scores include EOS/termination or use a fixed deterministic horizon, and align temperature/truncation with the sampling law being bounded. Current replay materialization preserves supplied token sequences; the generic code path does not by itself certify that every stored row includes the required terminal event. Prefix probability alone is not a key-probability lower bound.
5. **Validate mode retention beyond a coarse endpoint.** Report fixed-bank survival and within-correctness breadth at matched accuracy, with sufficient evaluation draws or interval estimates. A positive floor may be too small to observe in eight samples. The collapse-rate exponent is also a testable prediction only in a controlled geometry; current cross-model runs do not identify that geometry.

These are bounded next steps, not reasons to add every possible theorem. The most valuable additional theory beyond this revision is an optimizer-aware energy bound or an admission/retention guarantee. The elementary stationarity-rate, binary-KL refinement, and admission accounting suggestions remain in the detailed replay review instead of expanding the manuscript further.

## Validation and artifacts

- Independent audits cover the base/shared flow, replay chain, and entropy/implementation correspondence. The final integration review found no remaining mathematical or scope inconsistency after the coverage sentence was corrected.
- Collapse-rate verification passed 20 cases over four category configurations and five shared-direction strengths. Stable log-probability integration avoided underflow; maximum late-interval slope error was approximately `1.87e−14`. This is numerical corroboration, not proof.
- Exact entropy gradients passed 48 finite-difference cases with maximum discrepancy approximately `7.49e−11`; 48 minimum-logit boundary checks also passed.
- The historical-score check evaluated the actual source implementation for five singleton counts and 36 finite-group expectation cases. Nine bounded-advantage expectation checks and three raw/key entropy examples also passed.
- Final build and packaging checks, PDF/source hashes, source-manifest consistency, and unchanged bibliography/asset checks are recorded in [validation.json](theory_second_review_20260905/validation.json). Both main-text page limits are preserved; the full paper's prose check covers 197 blocks.

Updated [full paper](../main.pdf), [workshop paper](../mathai2026/main.pdf), and [workshop source bundle](../mathai2026/mathai2026-source.zip). The [exact revision patch](theory_second_changes_20260905.patch) is relative to the state after the completed reference audit; it does not mix this work into the earlier reference-only patch. Figures, experimental results, both bibliographies, and the workshop main text are unchanged by this theory revision.

Detailed receipts: [base and shared-flow review](theory_second_review_20260905/base_flow_review.md), [replay review](theory_second_review_20260905/replay_review.md), [entropy review](theory_second_review_20260905/entropy_review.md), [final integration review](theory_second_review_20260905/final_integration_review.md), [rate verification](theory_second_review_20260905/verify_collapse_error_rate.json), and [entropy/source verification](theory_second_review_20260905/verify_entropy_review.json).
