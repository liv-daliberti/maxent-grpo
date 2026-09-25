# Appendix P implementation, 2026-09-21

The user authorized appendix extensions and requested that Section 3.1 remain unchanged. Its source matches `section31.before.tex` byte-for-byte. Other manuscript editing occurred concurrently; every integration preserved the current source outside Appendix P.

The rebuilt `paper/main.pdf` contains the revised appendix on pages 95–117. `appendix.diff` isolates the theory changes. Final wording and layout fixes are authoritative in `paper/main.tex`; the standalone fragments are drafting records.

Added 11 formal results with proofs: neural per-prompt mean/covariance; finite-step neural remainder; response-weight residual; PCMD drift; finite expected-gradient collapse; sampled symmetry breaking; finite-budget invisibility; replay threshold; recovery exposure; discrete replay contraction; local neural exemplar response. A shared-prompt counterexample states the limit of mean-direction equivalence.

Sinha et al. (arXiv:2601.21669v1) supplies the softmax log-ratio and ideal inverse-weighted fields. UCPO (arXiv:2605.00365v1) supplies binary-objective indifference, count-level symmetry breaking, and the conditional uniform target. These foundations are explicitly credited.

Corrections cover signed-advantage retention, positive-coefficient convergence, Bernstein positivity, KL sublevels, two-mode recovery constants and logarithmic factors, stationary versus transient interpretation, and Jensen occupancy summaries. The numerical KL study's description was corrected without changing its values.

Validation: exact finite-group covariance and stochastic-PCMD checks passed; 20,000 replay contraction/order checks passed; independent mathematical review passed. LaTeX/BibTeX builds have no undefined citations/references, duplicate labels, or overfull hboxes. Representative theorem pages were visually inspected. Appendix P has zero paragraph-fill violations; 25 remain elsewhere, and the initial PDF already failed that global gate. The document-wide contract reports the same pre-existing missing main role label `sec:results-maxrl` as before this task. Main-text work was left unchanged.

The proof-preservation reference was deliberately updated to the reviewed 58-block chain in `paper/audits/proof_chain_20260921/main.tex`; the September 19 reference remains intact. Its targeted preservation check passes.

No neural training or new empirical mechanism experiment was run. Neural results remain conditional mathematical statements; categorical results retain their stated geometry and update assumptions. See `validation.json`, `theory_checks.json`, `replay_discrete_checks.json`, and `proof_check.json`.
