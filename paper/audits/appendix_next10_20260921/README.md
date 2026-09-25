# Second appendix prose and caption pass — 2026-09-21

## Scope and presentation

The target is the next ten content pages after the first pass: PDF pages 28–37. The pass covers the end of the prompt examples, Replay Algorithm, Fixed Semantic MaxEnt, terminal training results, and the training-curve/comparator text that enters this range after reflow. Figures 18–22 are in the target range; the linked Level-2 training panels (Figures 23–24) were recaptioned with their companion curves. Three table captions and Algorithm 1 were also revised.

Removed editorial justifications, campaign/completion narration, logging and checkpoint-recovery details, registration/admission-history language, and machine-readable-record references. Actual algorithm steps, support rules, uncertainty, missing observations, and limitations remain explicit. Four subsection titles and their contents entries were synchronized. Captions lead with a supported finding, followed by setup, visual encodings, populations, and uncertainty.

## Corrections identified during the review

- **Semantic MaxEnt normalization:** the predictor denominator is `N_x + S_{−i} + |K_i| + 1`, where `S_{−i}` counts successful eligible peers, rather than all `G−1` peers. Verified against `open_set_success_semantic_signal` and `SemanticShannonTracker` in `src/oat_drgrpo/semantic_shannon.py`. A mixed-validity four-response example matches the implemented advantages exactly; the corrected predictor sums to one, while the old formula sums to 7/9 in that example. All other displayed Semantic MaxEnt equations are unchanged.
- **Absolute versus paired diversity:** arm-specific summaries require 30 defining prompts per seed; paired retention effects require 20 per arm. Their seed populations can differ. Diversity-effect Student-t intervals use `n−1` degrees of freedom for `n>=2`; correctness/raw-count intervals require five seeds. The captions no longer claim all intervals require complete five-seed blocks.
- **Level-2 Graph:** the paired Dr.GRPO comparison uses three seeds and means approximately .039 to .353, giving +.314. The previous .111 versus .346 comparison mixed one control seed with five replay seeds. MaxRL's .112 to .325 comparison uses five paired seeds.
- **Cross-level averages:** correctness uses five common seeds across Graph/Countdown/Python/PantryPlan. Diversity uses arm-specific eligible seed means across Graph/MathIR/PantryPlan. Its bars are descriptive differences, not matched-seed effects.
- **MathIR:** each Level-2 prompt has one shortest derivation and four longer alternatives; near-zero diversity is not a structural absence of alternatives. The unobserved route length in the 27,450-draw analysis is four states. The 67%/52% coverage figures concern 115 solved Level-1 prompts. Length normalization and learned-judge limitations are stated without unsupported guarantees.
- **Training curves:** partial cohorts are included as indicated by rings or dotted segments. A universal decline across Graph, Countdown, and PantryPlan was unsupported; the caption now states the supported PantryPlan result. Level-2 Graph curve values use the plotted five-prompt threshold, and MathIR is described as near zero rather than exactly zero.
- **Supporting comparators:** their diversity estimand uses per-prompt differences over at least 30 common eligible prompts, unlike the retention matrix. Coverage and interval summaries are scoped to the correct methods. The weighting ablation supports additional sampled modes; its correctness effect is inconclusive, not evidence of no difference.
- **Notation:** the Level-2 interaction table defines `P_8`, `D_8`, and `B_8`, avoiding reuse of the main text's single-draw correctness symbol `P` for pass@8.
- **Figure 19 footer:** now identifies connector style by control-arm seed count and states that arm counts can differ. Every numerical record and existing display setting is unchanged; only the footer and its metadata entry changed.

## Validation

- Independently reviewed the methods and result descriptions against implementation, result files, and plotting code.
- Exact prompt generator check passes for all five domains; prompt contents are unchanged.
- All ten target pages visually inspected; references, equations, tables, and captions checked after compilation.
- No undefined references/citations, duplicate labels/destinations, overfull boxes, or wrapfigure collisions.
- All 90 numbered appendix headings appear in the contents.
- Main body remains nine pages with all 12 figures; its extracted PDF text is byte-identical to the incoming main body. References begin on page 11.
- Full PDF is 123 pages, down from 125. No experimental values or training code were changed; no new experiments were run. The recently revised compute appendix is unchanged.

Validated main.tex SHA-256: `2e9f1a172a2affc748c0770c67b02eb5bf320a2d5c46bba0ced46e1570aa652c`.

Overleaf export: standalone compilation passed. The packaged source, revised figure, figure metadata, and reference PDF exactly match the installed deliverables.
