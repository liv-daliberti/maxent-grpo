# Worked examples for theory sections A.4–A.7

Four complete replacement subsections are staged from the current `paper/main.tex`. Each adds a short opening and one worked example immediately after its first proof. Existing formal statements, proofs, and citations are unchanged; no main manuscript, workshop source, or bibliography was edited.

| Replacement | Subsection label | What the worked example reveals |
| --- | --- | --- |
| `natural_gradient_section.tex` | `app:theory-natural-gradient` | At the running fixture, exact Fisher correctness learning changes total correct mass while keeping correct-mode proportions fixed. Uniform replay moves the rare mode's conditional share upward and supplies its 0.05 all-time floor. Instantaneous derivatives are explicitly distinguished from a finite optimizer step. |
| `entropy_section.tex` | `app:theory-entropy` | A separate two-correct/one-incorrect fixture with G=4 and beta=1/4 has exact Gibbs limit approximately (0.48786, 0.48786, 0.02429). Exact full entropy protects the incorrect outcome too; the example is explicitly unrelated to a prediction for the historical sampled semantic score. |
| `entropy_comparison_section.tex` | `app:theory-entropy-comparison` | The running fixture gives different forward/reverse KL values. Holding its conditional shape fixed while raising correct mass from 0.5 to 0.9 leaves conditional entropy unchanged and reduces replay loss by approximately 0.58779. |
| `gradient_availability_section.tex` | `app:theory-gradient-availability` | With G=16 and p=(0.600,0.389,0.001,0.010), only 14.85% of fresh groups are mixed, the rare key is absent with probability 0.98412, and deterministic uniform-bank replay still supplies a positive 0.03323 logit coordinate. Mixed-group and rare-key-absence events are explicitly distinguished. |

`entropy_verify_examples.py` recomputes each displayed numerical value and checks its rounding. The Fisher example is calculated from the full probability-space vector field and then reduced to P/q derivatives. The Gibbs example checks equality of simplex first-order derivatives. The mixed-group probability is computed independently as a binomial sum. The script also compares all theorem/lemma/corollary/proof blocks and all citation commands with the current manuscript. All checks passed; results are in `entropy_example_verification.json` and source hashes in `entropy_staging_manifest.json`.

The natural-gradient and KL examples reuse the parent's running example. The Gibbs and near-perfect-accuracy examples explicitly introduce different hypothetical fixtures. All numerical values are illustrative, not run measurements. No new source citations, global macros, theorem labels, or theorem assumptions were introduced. Final document layout and any local reflow remain with the parent integration pass.

Final integration note: root subsequently aligned example links, clarified complete-response wording, and adjusted headings/prose for the manuscript layout. Any handoff hashes above describe that review snapshot; current integrated file hashes are recorded in `validation.json`. All formal statements/proofs and citations remain unchanged.
