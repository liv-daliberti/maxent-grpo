# Independent review: early theory, incoming PDF pages 91–94

Reviewed the theory introduction, P.1–P.3, and the short P.3 continuation needed to close its statements. The saved `before.tex` and live source already contain the cleaner categorical and binary-estimator replacements; the incoming page extracts contain earlier wording. This review therefore checks the replacement itself rather than treating the old PDF as the current source. No manuscript source was edited.

## Findings

No remaining mandatory mathematical or presentation correction was found in this scope.

- The introduction now states results and assumptions directly. The earlier discussion of the order of derivations, what was invoked, the novelty of the standard replicator theorem, and what question an earlier theorem invites is absent. The assumption and collapse statements consistently say solution modes.
- The categorical model correctly needs independent mode logits, a fixed common normalization factor, Euclidean infinitesimal mean updates, no competing regularizer, and a nondegenerate finite-logit start. The repaired remark correctly cites all (A1)–(A5), rather than only (A1)–(A3). It also distinguishes AdamW preconditioning/momentum from finite repeated PPO updates.
- The group-centered expectation and its denominator-free Bernstein representation are correct. Conditioning on the full reward vector is legitimate because the original responses are independent; conditioning on each response's own reward fixes its conditional score mean. Zero-update conventions at all-equal groups avoid undefined reciprocal reward variance.
- The centered MaxRL coefficient is exactly `[1-(1-P)^(G-1)]/P`. An independent count derivation gives conditional coefficient `(G-r)/(G P(1-P))` for `r>0`, and zero at `r=0`. The numerator follows from `E[(G-R)1(R>0)] = G(1-P)-G(1-P)^G`. Thus the two endpoint limits, the finite geometric sum, and the integrated harmonic potential are correct.
- The order-`G` alternative in the following paragraph is also correct: retaining advantage `-1` on an all-failure group adds `(1-P)^(G-1)` to the current coefficient. This distinction is a scientific estimator definition, not process/history wording, and should stay.
- The collapse proof establishes both `P -> 1` and unbounded effective time. The lower gradient bound follows separately from Cauchy–Schwarz over correct and incorrect categories. Bounded total variation would give finite limiting logits and contradict `P -> 1`, so the time reparameterization cannot stop after finite effective time. The log-ratio estimate then eliminates all coordinates strictly below the initial maximum. Tied maxima remain uniform among themselves. These conclusions do not imply finite-time support loss.
- P.3 now makes the fixed finite advantage table explicit before asserting polynomial regularity. The Bernstein gap positivity condition is sufficient, with strict positivity only needed in the interior. The extension of the collapse proof does not require a positive coefficient at `P=0` or `P=1`.
- The MaxRL endpoint gap `d_(G-1)=1` is correct even though all-success groups have zero updates: it compares `a(1,G)=0` against `a(0,G-1)=-1` in a different group. No formula should be changed here.
- The P.3 continuation correctly handles the `G=2` coefficient-ratio edge case and both low- and high-correctness scarcity of mixed groups. Neither statement gives an individual-mode retention guarantee.

## Optional local improvement

The MaxRL lemma's proof currently cites the external estimator theorem and then evaluates its geometric sum. It is correct, and the lemma already defines the local estimator completely. If a more self-contained proof is desired, replace that proof alone with:

```tex
\begin{proof}
\enforcehalffinalproseline
For $r>0$, conditioning on $R=r$ and using the score identities gives
\[
 \E[\widehat g_G^{\mathrm{MaxRL}}\mid R=r]
 =\frac{G-r}{GP(1-P)}\nabla_zP.
\]
The update is zero for $r=0$, and
$\E[(G-R)\1[R>0]]=G(1-P)-G(1-P)^G$.
Averaging therefore gives Equation~\eqref{eq:maxrl-expected-gradient}.
The finite sum $c_G^{\mathrm{MaxRL}}(P)=\sum_{j=0}^{G-2}(1-P)^j$
gives the endpoint values.
\end{proof}
```

This is optional, not a detected error, and may add vertical space. The attribution immediately before the lemma should remain if this alternative is used. No figure caption edits are needed: the assigned theory scope contains no figures or captions.

## Verification

A source scan found 32 cross-reference uses targeting 17 distinct local labels, all present in `main.tex`; no broken local reference target was found. Targeted process/history wording, `reasoning mode(s)`, and `execution mode(s)` are absent in this scope. Existing exact categorical and binary checks were inspected; this review did not repeat those deterministic checks or run experiments. The core proof steps and both MaxRL conventions were independently checked algebraically above.

Reviewed source block SHA-256: `020c0e78ea2c974669ae7e6f9bb96cbfde3862f5aa6a30a7180c50bba9170dd5`.

## Additional independent algebra check

At root's request, `validate_algebra.py` adds an independent finite-state validation, recorded in `algebra_validation.json`. All 1,480 checks pass. It enumerates 1,344 ordered groups across 60 combinations of two policies, two score geometries, three small group sizes, and five reward-only advantages. Exact rational arithmetic and symbolic radicals cover both mean and covariance formulas, including standardized GRPO, leave-one-out, and signed advantages. It also verifies the response-weight residual identity, the vector Cauchy–Schwarz and aggregate bounds, Bernstein coefficients, and MaxRL endpoint polynomials and the `G=2` ratio. Twenty-five deterministic finite-step examples cover nonuniform distributions, uniform distributions, and tied maxima at step factors from 0.001 to 1,000. These examples corroborate the formulas and do not replace the proofs.

The added coverage checks standardized-estimator covariance with exact radicals and a shared-parameter feature geometry, independently of the earlier subsection validators. No model, training, Monte Carlo, or experimental run occurred.

`algebra_snapshot.tex` preserves only the scoped theory source, and `algebra_equation_signatures.json` records its 48 displayed equations. The script's final comparison found all 48 byte-identical in live `paper/main.tex`. Run `python paper/audits/appendix_eighth10_20260921/validate_algebra.py` from the repository to reproduce the arithmetic and compare the live formulas again; the script requires SymPy.
