# Appendix P.4 review, 2026-09-21

Scope: the subsection `Neural score identities and the limits of conditional inertness` up to, but excluding, `Diversity at finite steps and sampling budgets` (incoming PDF pages 95–97). The exact replacement block is `replacement.tex`. No live manuscript or experiment files were edited by this reviewer.

## Editorial changes

- Removed the subsection-replacement instruction and the introduction's account of an extension being inserted. The introduction now states the scientific distinction between total correctness and allocation among solution modes, and states the exact scope of proportional per-prompt mean gradients.
- Rephrased the finite-update comparison in terms of two estimators with the same mean step. Preserved the fact that coefficient normalization is not used in the training updates, because this is necessary to interpret the corollary rather than manuscript-process language.
- Replaced a call for additional optimizer analysis with the explicit scope limitation: fresh on-policy score updates, excluding Adam moments and subsequent PPO epochs. Preserved the finite-step, sampling covariance, multi-prompt, and parameter-sharing limitations.
- Used `solution mode` for the observable and surviving-mode prose.
- Made the prompt-sampling law fixed and stated the integrability requirement for its aggregate mean.
- Qualified the zero residual for common response length: that inference applies to inverse-length weighting, not to an arbitrary detached response-dependent multiplier. Preserved the warning that equal mean lengths need not eliminate within-outcome covariance.
- Presented SetPO's verified-key specialization directly rather than as an account of connecting pieces of the paper. All attribution, version-specific locators, and scientific assumptions remain.

The two immediately preceding source comments (`Insert after ...` and `This fragment does not modify Section 3.1`) lie outside the replacement boundary and should also be removed by the integrator. They are not rendered.

## Mathematics and validation

All displayed equations, labels, citations, and citation version locators are byte-identical. No measured result, numerical table, figure, coordinate, or interval is changed. This subsection contains no figure captions.

`validate.py` passes **102 checks**, including exact rational enumeration of 18 advantage/group-size cases (2,016 weighted response groups in total):

- conditional scores, the neural mean, and total covariance;
- the finite-step remainder, including equality for a quadratic observable;
- the response-dependent mean residual and its Cauchy–Schwarz bound;
- the two-prompt Dr.GRPO/MaxRL example, the `9/1024` conditional-diversity derivative, and both prompts' positive correctness derivatives;
- SetPO's mixture influence, zero-mean influence, and the shared-parameter verified-key gradient.

The proof descriptions were also checked directly: conditioning on the full reward vector preserves score independence; finite second moments of the score and weighting variable suffice for the mean/residual identities; the time-change statement is restricted to positive coefficients, a common autonomous preconditioner, and uniquely existing flows; and the metric influence alone does not supply a sign for optimizer-induced change.

Attribution checked against locally supplied theory audits containing the cited source statements: `paper/audits/theory_strengthening_20260921_inertness.md` and `paper/audits/theory_source_reuse_20260921_rlvr.md`, plus `paper/audits/theory_source_impl_20260921/setpo_bridge.tex`. No new outside-source claim or citation was introduced. No experiment, training, model call, or bootstrap was run.

Final numerical/prose preservation checks and block hash are in `validation.json`; human-readable differences are in `changes.patch`. Root handles the integrated PDF and Overleaf build.
