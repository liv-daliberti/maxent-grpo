# Appendix P.1–P.2 review, 21 September 2026

Scope: the introduction to Appendix P, P.1 (categorical abstraction and mean updates), and P.2 (winner-take-all collapse), covering the corresponding text on incoming PDF pages 91–93. The replacement stops immediately before `\subsection{Conditional trajectories of outcome-binary mean flows}`. No figure or table captions occur in this block. No main-body text, measurements, assets, later subsections, or bibliography entries were edited.

`replacement.tex` is the replacement block for root integration. `original.tex` is the exact incoming block; `manifest.json` records boundaries and both hashes. `changes.patch` is its localized review diff. This agent has not written to live `paper/main.tex`, avoiding interference with the other agents' disjoint blocks.

## Presentation changes

- Replaced the opening account of what the authors separate, apply, derive, and claim with a direct description of estimator means, sampling covariances, frequency amplification, replay guarantees, and neural assumptions.
- Removed the publication/novelty narrative (“published convergence guarantee,” “sharpened,” “We invoke,” “not a new general theorem”) while preserving the underlying scientific attribution. The MaxRL result remains a short, explicitly attributed specialization of Tajwar et al., Theorem 5; the replicator equation retains Hofbauer–Sigmund and Harper citations.
- Kept the genuine limits: independently optimized logits do not follow from outcome probabilities; variable response-length weighting can invalidate the equivalence; PPO clipping and adaptive optimization can alter the expected update; limiting concentration does not imply finite-time support loss or certify stochastic neural trajectories.
- Made GRPO/Dr.GRPO attribution explicit next to each definition: Liu et al. (2025) for Dr.GRPO, Shao et al. (2024) for GRPO.
- Used “solution modes” in the categorical setup and collapse conclusion.
- Replaced the theorem's publication-positioning title with its mathematical conclusion, “Winner-take-all collapse in the categorical mean flow.”

## Mathematical precision fixes

1. **The common length factor is fixed.** A2 and the GRPO qualifier now specify a fixed shared normalization factor. A common but sample-dependent group factor can correlate with the identities of successful modes and rotate the mean gradient; equality only within a group is insufficient. Exact counterexample: three equiprobable categories, two correct, group size two, and a common multiplier of two for mixed groups containing correct category 0 versus one for those containing category 1 gives mean `(1/9, 1/18, -1/6)`. The correctness gradient's two correct coordinates are equal, so these vectors cannot be proportional. The correction states the premise required by the existing formulas and leaves the formulas unchanged.
2. **The asymptotic remark requires all assumptions.** It previously cited only (A1)–(A3), despite requiring no competing regularizer and a finite nondegenerate start/group size from (A4)–(A5). It now cites (A1)–(A5).
3. **Optimizer exclusions are precise.** AdamW's preconditioning and momentum violate Euclidean mean flow. Finite repeated PPO updates are outside the stated infinitesimal on-policy limit; the new prose avoids treating “PPO” as a single algebraic geometry at ratio one.
4. **No distinct-coefficient claim at degenerate coincidences.** The MaxRL transition states the common mean-gradient direction and its convention-dependent coefficient, without suggesting that coefficients of every included estimator must differ at every group size.

## Mathematical review and verification

All **13 display equations** and **12 labels** are byte-identical; LaTeX environment boundaries are identical. `validate.py` passes **509 checks**, using exact rational arithmetic, including **18 exhaustive group configurations totaling 2,676 category sequences**. These are deterministic checks of mathematical identities, not training, model calls, Monte Carlo experiments, or new measured results.

The checks verify:

- The group-baseline coefficient identity for Dr.GRPO, centered MaxRL, and a separate positive count-dependent weight across 2- and 3-correct-category laws and group sizes 2–4.
- The denominator-free binomial representation and coefficient bounds across group sizes 2–12 and five correctness values.
- The MaxRL coefficient, finite geometric sum, endpoint values, and potential derivative, including the exact `G−1` truncation.
- The softmax chain rule, monotonic correctness identity, gradient lower bound, conditional replicator equation, and pairwise log-ratio equation.
- The sample-dependent common-factor counterexample supporting the A2 clarification.

The remaining asymptotic proof was reviewed analytically. Bounded smooth logit velocity ensures global finite-time existence; the correctness-gradient lower bound excludes an interior correctness limit; finite accumulated effective time would force finite limiting logits and contradict correctness tending to one; the ratio differential inequality eliminates every initially smaller correct mode. Exact initial ties remain tied, and finite logits preserve positive probability at every finite time. These arguments are retained.

The local `src/oat_drgrpo/maxrl.py` definition confirms `G R_i / R − 1` on groups with a success and an all-zero all-failure update. No optimizer or training code was changed.

## Primary-source checks

- [Tajwar et al., Maximum Likelihood Reinforcement Learning, v3](https://arxiv.org/html/2602.02710v3), Section 4.3 and Appendix D, Theorem 5: the practical centered estimator skips the baseline on all-failure groups, giving the `G−1` finite series. Its unconditional-baseline alternative retains order `G`.
- [Liu et al., Understanding R1-Zero-Like Training](https://arxiv.org/html/2503.20783), Sections 3.1–3.2: response-length and reward-standard-deviation normalization in GRPO; Dr.GRPO removes those terms and uses a fixed normalization factor.
- [Shao et al., DeepSeekMath](https://arxiv.org/abs/2402.03300): GRPO attribution checked.
- [Harper, Information Geometry and Evolutionary Game Theory](https://arxiv.org/html/0911.1383v1), Section 1.1: the frequency-dependent replicator equation and mean-fitness definition. The retained equation specializes fitness to each mode's own probability.
- [Sinha et al.](https://arxiv.org/html/2601.21669v1) and [Lochab et al.](https://arxiv.org/html/2605.00365v1): the related-work placement concerns collapse or diversity indifference among equally rewarded outcomes; no theorem from either is used as a substitute for the displayed proof.

Full-paper compilation and final page/cross-reference checks are assigned to root after integrating all disjoint replacements.
