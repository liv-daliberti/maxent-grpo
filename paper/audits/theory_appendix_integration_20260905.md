# Theory appendix integration — September 5, 2026

Integrated the reviewed theoretical extensions into both manuscripts with full proofs, explicit assumptions, and primary-literature attribution at the relevant steps. The full paper is [paper/main.pdf](../main.pdf); the workshop version is [paper/mathai2026/main.pdf](../mathai2026/main.pdf), with an updated [standalone source bundle](../mathai2026/mathai2026-source.zip).

| Result | Full paper | Workshop | Literature connection |
|---|---|---|---|
| Exact categorical natural gradient | A.4, p. 18 | E.4, p. 17 | Fisher/Shahshahani geometry; the correctness-plus-replay flow is derived explicitly |
| Exact conditional entropy versus uniform replay | A.6, p. 21 | E.6, p. 20 | Confidence penalties/label smoothing, log-barrier policy gradient, regularized MDPs, and exact tabular entropy convergence |
| Controlled stochastic retention | A.10, p. 26 | E.10, p. 25 | Variable-metric descent, Ville's inequality, almost-supermartingales; includes the sharper bounded-noise finite-horizon corollary |
| Admission and changing-bank retention | A.11, p. 28 | E.11, p. 26 | Conditional Borel–Cantelli, coupon collection, switched-energy accounting, archive and replay precedents |
| Sharp survival certificates and measurement | A.12, p. 30 | E.12, p. 29 | KL data processing, binary-KL inversion, entropy thresholds, exact binomial intervals, confidence sequences, and missing mass |

The expansion adds ten formal statements across five subsections. Eighteen bibliography entries were added, each with a primary-source URL; existing Geist, Ecoffet, and Rolnick entries are reused where relevant. The two bibliographies are identical, with 111 entries and 48 cited keys across the manuscripts. The theory roadmap and final scope remark now connect the extensions to the preceding collapse, fixed-bank, and complete-exemplar arguments.

## Mathematical and evidence boundaries

The optimizer theorem assumes a fixed joint potential, smoothness, predictable preconditioning, conditional noise centering, and controlled accumulated bias/noise costs. Its infinite-horizon conclusion requires a finite budget. The stronger finite-horizon corollary explicitly inherits noise centering and deterministic initialization; this was clarified during independent review.

Admission means actual insertion after all verification, scheduling, capacity, and selection gates. The coverage bound is in counted opportunities. Retention through bank changes separately assumes controlled positive objective jumps and persistent positive coefficients. These conditions have not been established for the actual neural optimizer or proposal machinery.

Exact conditional entropy and full-support categorical replay share an ideal optimum. Published exact tabular entropy guarantees and the Fisher comparison prevent a universal claim that correctness or MaxEnt must collapse under every update rule. Sharp probability certificates require normalized complete-response likelihoods, including termination and matched decoding. The numerical example is explicitly hypothetical. Finite observations and changing checkpoints retain their separate statistical interpretation.

No new training measurements, results, scheduler actions, or experimental asset changes are included. The full paper's main-text change is limited to the sentence identifying the optimizer/admission extensions as conditional. Workshop main text and preamble are byte-identical to their pre-integration versions. Its supplementary limitations now describe the expanded theoretical scope accurately.

## Validation

- `make -C paper` passes the current manuscript/evidence contract, frozen domain-prompt check, LaTeX/BibTeX build, and rendered paragraph-layout gate (234 natural-prose blocks).
- `make -C paper/mathai2026 bundle` passes the four-page main-text limit, six main/seven supplementary figures, anonymity, unchanged official style and frozen-asset checks. The 42-file source bundle matches the current source files byte for byte.
- Final PDFs have 52 and 53 pages, respectively; main-text budgets remain nine and four pages. Final compiler logs contain no unresolved citations/references, duplicate labels, overfull boxes, or fatal errors.
- Core theory is identical across versions except for the two pre-existing document-specific cross-references. All 18 new bibliography keys resolve and are used.
- Representative full-paper pages 18, 26, and 30 were visually reviewed for readable equations, theorem statements, and proof layout. Prose reflow preserved formulas and premises; the switching potential was split over two aligned lines.
- Full proof reviews and source-use notes are retained below. The earlier numerical corroboration is recorded in the [standalone supplement validation](theory_literature_extensions_20260905/validation.json); those checks were not rerun as part of this manuscript-only integration.

[Final validation receipt](theory_appendix_integration_20260905/validation.json), [integration checks](theory_appendix_integration_20260905/integration_checks.json), and [exact source patch](theory_appendix_integration_20260905/integration.patch) provide the final file-level record. Authoring/review details are in the [optimizer/admission notes](theory_appendix_integration_20260905/optimizer_admission_notes.md), [entropy/geometry notes](theory_appendix_integration_20260905/entropy_geometry_notes.md), and [survival notes](theory_appendix_integration_20260905/survival_integration_notes.md).
