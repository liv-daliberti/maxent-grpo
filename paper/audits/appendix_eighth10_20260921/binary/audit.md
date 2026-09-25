# P.3: outcome-binary conditional trajectories

Scope: incoming PDF pages 93–95, from `Conditional trajectories of outcome-binary mean flows` up to (but excluding) `Neural score identities and the limits of conditional inertness`. This includes deleting the two stale source comments immediately before the latter heading. The replacement is in `replacement.tex`; no live source was edited and no PDF build was run by this review.

## Presentation changes

- Replaced the introductory account of which theorem was stated first, what question it invites, and a “one line” proof characterization with the mathematical conclusion and its conditioning argument.
- Distinguished conditional score means from realized scores. Conditional means are unchanged by conditioning on the other rows' rewards because the group draws are independent; realized scores still depend on the sampled category.
- Stated the fixed reward-dependent advantage premise explicitly before deriving a polynomial in correctness. Detachment alone is not a claim that arbitrary policy-dependent advantage values define one fixed polynomial for all policies.
- Defined the Bernstein coefficients as reward changes of a single row while the remaining rewards stay fixed. Replaced “checkable by inspection” and “what the objective puts between being right and being wrong” with exact sign and convex-combination statements.
- Replaced “the collapse theorem was never about the three estimators it was stated for,” “hold verbatim,” and similar narration with the class assumptions and explicit limiting probabilities.
- Replaced “What a stronger fresh objective does buy” with “MaxRL amplification of the correctness gradient.” The comparison is stated at the same policy, so the coefficient ratio is not represented as a fixed end-to-end training speedup or compute comparison.
- Removed both stale integration comments (“Insert after…” and “This fragment does not modify…”).
- Preserved all six labels, all theorem/proof/remark environments, and scientific limits concerning positivity, initial imbalance, sampled neural updates, covariance, and individual-mode retention. This section has no figures, figure captions, or local citation commands to edit.

## Mathematical findings

1. **The centered MaxRL Bernstein gaps were already correct.** The applied advantage is `G r/R - 1` when `R>0`, and zero when `R=0`. Thus `d_0=G-1`, and `d_j=G/(j+1)` for `1<=j<=G-1`. In particular, the last comparison is between two different groups: `a(1,G)=0` (all successes) and `a(0,G-1)=-1` (one failure). Therefore `d_(G-1)=1`, despite the zero update on all-success groups. Added this endpoint calculation explicitly; did not change the correct coefficient.
2. **The monotonicity sentence required the `G=2` edge case.** The MaxRL/Dr.GRPO coefficient ratio is nonincreasing, strictly decreasing only for `G>2`. For `G=2` it is identically 2. The proof now uses the exact finite sum; its `1-P` term exists precisely for `G>2`.
3. **The starvation discussion omitted the high-correctness boundary.** `h_G(P)=1-P^G-(1-P)^G` vanishes as `P` approaches either 0 or 1. The group-size remark now names both boundaries, and distinguishes this total-correctness property from a probability floor for an individual solution mode.
4. **The collapse extension retains sufficient hypotheses.** A fixed finite reward table makes `c_a` smooth and bounded on `[0,1]`. Nonnegative Bernstein gaps with one positive give strict positivity on the interior. These are exactly the coefficient properties used to establish global finite-time existence, `P -> 1`, and infinite effective time in the earlier collapse proof. The replacement does not assume positivity at the endpoints.
5. **Sign and stationary-point statements remain precise.** For an interior policy, `dP/dt=c_a(P)||grad P||^2`; a negative coefficient decreases correctness and a zero coefficient makes that correctness level stationary. Mixed-sign gaps remain inconclusive without checking the polynomial.

## Validation

`validate.py` performs 5,864 exact deterministic checks using rational arithmetic; `validation.json` records the counts and SHA-256 hashes.

- All six labels and all mathematical environment counts match the incoming block; no citations are lost; targeted process phrases and source comments are absent.
- Group sizes 2–32: directly evaluate all Dr.GRPO and centered MaxRL Bernstein gaps, including both all-equal endpoints.
- Group sizes 2–32 and 19 interior correctness values: compare the original binomial expectation with the Bernstein expression for Dr.GRPO, MaxRL, and a signed synthetic finite reward map; compare both named estimators with their exact closed forms.
- Check the coefficient ratio's monotonicity and `G=2` equality case, endpoint values, and the mixed-group bound and symmetry.
- Fully enumerate categorical groups with four categories and group sizes 2–5. For each of three advantages, verify every coordinate of the expected score update against `c_a(P) grad P` and verify the conditional log-ratio dynamics.

These checks verify algebra and discrete estimator identities; they do not simulate training, run models, change experimental measurements, or replace the proofs. Root integration remains responsible for compilation, page layout, cross-reference checks, and final release artifacts.
