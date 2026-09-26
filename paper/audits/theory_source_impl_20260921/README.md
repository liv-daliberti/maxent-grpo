# Appendix P: applying established theory

Implemented on 21 September 2026 following the user's authorization. The
authoritative text is `paper/main.tex`; `paper/main.pdf` has been rebuilt.
Appendix P occupies pages 94–120 in this build. The main body, including
Section 3.1, matches the initial source snapshot exactly. Concurrent edits
to other appendices were preserved.

Five attributed corollaries were added:

- **P.13, SetPO:** the verified-key equality kernel gives PCMD and its
  gradient for differentiable policies with shared parameters.
- **P.22, van Erven–Harremoës:** binary KL data processing sharpens replay
  retention, with bank-mass and complete-response execution-key bounds.
- **P.24, Anceaume et al.:** a uniform allocation maximizes finite-budget
  coverage tails at fixed correctness under independent sampling.
- **P.29, Howard et al.:** a conditional stochastic energy budget gives
  simultaneous exemplar retention; Wang et al. supplies a scoped sufficient
  optimizer condition.
- **P.33, Mei et al.:** exact independent-logit Dr.GRPO with reference KL
  has a trajectory probability floor and geometric convergence to the
  reference-conditional target. The reduction does not cover general MaxRL
  or the implemented AdamW updates.

The MaxRL mean-estimator proof now invokes the original practical-estimator
theorem. The PCMD drift proof invokes Harper's replicator variance identity.
Conditional scores, likelihood replay, natural-gradient ratio preservation,
and the Gibbs optimum receive explicit source attribution. Bibliography URLs
are pinned to the checked versions. Existing estimator covariance, sampled
PCMD, finite-step collapse, replay-threshold, and recovery proofs are retained.

The surrounding KL discussion now distinguishes proved Dr.GRPO convergence
from MaxRL's stationary characterization. Occupancy summaries and hosted
reference diversity are not presented as bounds on held-out neural training.

Validation is recorded in `validation.json`:

- All three independent mathematical reviews pass; see `review_*.md`.
- `verify_imports.py` passes nonlinear finite-difference checks of the SetPO
  specialization, 200 sharp-retention cases, 100 exact reward-transform
  checks, and 54 exact coverage-enumeration comparisons.
- LaTeX and BibTeX complete without unresolved references/citations,
  duplicate labels, or overfull boxes. Representative retention and
  convergence pages were inspected visually.
- Appendix P has zero paragraph-fill violations. There are 26 document
  violations outside it; the initial PDF also had 26 outside Appendix P.
- The reviewed 68-block statement/proof chain is preserved in
  `paper/audits/proof_chain_source_reuse_20260921/main.tex`, with earlier
  references retained. Its dedicated preservation check passes.

The repository-wide contract already failed before this task on
`main role label sec:results-maxrl missing or duplicated`; its final output
is retained in `global_contract.final.txt`. This unrelated main-text check
was not weakened to accommodate the theory changes.

`appendix.diff` isolates this task's theory changes. Standalone `.tex`
fragments are drafting records; subsequent wording and layout corrections
are present in `appendix.final.tex` and the manuscript. The numerical
16-mode example is illustrative, not a measured training loss. No new
training or neural mechanism experiment was run.
