# Streamlining review — September 11, 2026

This is an editorial cut plan for the long and short manuscripts. It does not change either manuscript or its evidence. Recommendations follow the current three-experiment story: verified-mode retention, the fresh-objective × replay factorial, and transfer to matched Level 2. All requested model scales and accuracy/mode training figures stay in both papers. Source anchors are more useful than page numbers while layouts are being rebuilt.

## Recommended first pass

| Priority | Material | Concrete action | What must remain |
|---|---|---|---|
| 1 | Bundled verified-support discovery experiment: long `main.tex:3686`, short `appendix.tex:3302` | Remove this subsection and its figure from the compiled papers; preserve the experiment in the research archive. It jointly changes proposals, semantic regularization, and replay, and its figure has no main-text dependency. | Fixed Semantic MaxEnt is a different comparator used in main Figure 4. Keep its result and enough estimator definition to reproduce it. Remove the historical four-seed interval exception that exists only for the bundled experiment. |
| 1 | Dated endpoint update: long `main.tex:4599`, short `appendix.tex:4232` | Remove the September 6-to-11 completion narrative, receipt table, and newly-completed-effects table. Put all current effects in one results section using the existing current tables. | Exact admitted seed counts, partial blocks, uncertainty, and the frozen cutoff. Keep the audit history in its existing files. |
| 1 | Weighting ablation: long `main.tex:3203`, short `appendix.tex:2821` | Consolidate the six earlier displays and current E120 table. Retain a current all-scale result table and a compact statement of the registered Qwen-0.5B analysis. Move seed dumps, the opposite-sign presentation, and separate September 6 Falcon Graph update to the artifact. | Uniform versus fresh-frequency replay is an important control for key balancing. Preserve neutral effects and the Falcon accuracy tradeoff. Keep the original primary estimand, frozen input, aggregation, and bootstrap convention distinct from later extensions. |
| 1 | “Aligned Scale Extensions”: long `main.tex:4388`, short `appendix.tex:4018` | Merge into current Level-1 results. Three scales now belong to the primary design. Remove the standalone newly-completed 3B Python table and chronology; retain relevant interpretation beside the full result display. | All three scales, per-domain effects, factorial interactions, and the distinction between objective-pair seeds and common four-arm seeds. |
| 1 | Repeated short-paper opening: `appendix.tex:86–475` | Condense the metric/design primer, remove the second narrative of Experiments 1–3, and attach unique uncertainty statements to the actual results. | Formal metrics, domain identities, protocol, exact model/method scope, and any caveat absent from the main text. |
| 2 | Theory: long `main.tex:896–2988`, short `appendix.tex:514–2606` | Keep a concise collapse/replay proof chain; move peripheral extensions into a preserved technical note. The current theory occupies roughly 23 pages before supporting empirical results. | Mean-update derivation including binary MaxRL, collapse assumptions, recurrent replay argument, response-to-mode bridge, and limits of finite-sample/LM interpretation. See dependencies below. |
| 2 | Fixed-bank mechanism study: long `main.tex:4239`, short `appendix.tex:3869` | Reduce three tables plus the score-distribution figure to the figure and one compact summary. Move detailed sequence-score and bootstrap tables to the artifact. | This is current limitation evidence, cited by both conclusions. Preserve the declining tail, length normalization, and the fact that exemplar scores do not establish exact mode-probability guarantees. |
| 2 | Reproducibility inventory: long `main.tex:4840–5007`, short `appendix.tex:4472–4589` | Keep one protocol/model-settings table, a concise exclusion policy, and links to data/models/results. Put scripts, source-file inventories, retry histories, and regeneration commands in maintained documentation. | Scientific matching, held-out data, masking/admission/loss scaling, evaluation contract, and enough immutable provenance to reproduce claims. |

## Optional cuts after the first pass

- **Training AUC sensitivity figure** (`main.tex:3105`, `appendix.tex:2723`): it has no main-text dependency and remains a Qwen-0.5B-only sensitivity analysis. It can move to the archive, especially for the short supplement. The new trajectory figures do not replace its statistical calculation; preserve that calculation as an artifact.
- **Token-entropy trajectory table** (`main.tex:3460`): keep the stronger five-seed raw-output collapse audit and the explanation of vanished fresh gradients; move this secondary diagnostic and detailed draw-spread discussion to the artifact.
- **Absolute-reference table** (`main.tex:3514`): emphasize Graph/Pantry, where a uniform reference is actually defined. Other rows largely repeat trained/frozen endpoints available elsewhere. Preserve the Pantry interpretation limitation.
- **Telemetry-only plot** (`main.tex:4209`): compress the actuation checks to a short methods validation or retain this plot only in the long supplement. It confirms implementation activity, not causal identity-level survival.

## Theory dependencies

The least disruptive large cut is the stochastic-optimizer, dynamic-admission, and sharp-certificate material in long `main.tex:2252–2988` (A.10–A.12; roughly eight pages). Empirical results do not use those certificates. Preserve a consolidated paragraph explaining the unverified optimizer/admission assumptions, the difference between token likelihood and mode probability, and what finite-K observations cannot establish.

The natural-gradient and extensive entropy comparisons are secondary to the central method. Their useful conclusion can survive in a concise scope discussion with necessary citations.

The model-geometry theorem needs a separate editorial decision: both main texts currently cite it (long `main.tex:154`, short `main.tex:116`). If its extended proof is archived, remove or restate those main-text claims at the same time. This does not remove any empirical 0.5B/1B/3B comparison. A compact 8–10-page core theory is an editorial target, not a verified layout estimate.

E121 does not validate the theoretical probability bounds. Keeping its observed declining tail is essential to an honest account of what replay guarantees in the trained model.

## Proposed reading order

Use the same organization in both supplements:

1. **Benchmark and protocol:** formal metrics, domain/key table, Level-2 construction, model/method scope, and uncertainty policy.
2. **Algorithm:** binary MaxRL advantage, replay update, admission, balancing, masking, and control matching.
3. **Current results:** one coverage table; Level-1 and Level-2 terminal results and interactions; all five accuracy/mode trajectory figures; direct comparators; weighting ablation.
4. **Mechanism and limitations:** baseline collapse precheck, compact raw-output/gradient audit, and compact fixed-bank score study.
5. **Core proofs:** the claims needed by the main text, with their assumptions.
6. **Reproducibility:** frozen data/model/result links and the concise source-exclusion contract.

The short supplement should not reprint a second main paper before this material. Preserve its necessary technical definitions, related work, and evidence, but eliminate repeated result narration. Dataset construction and algorithm details currently appear very late and should move ahead of the experimental appendix.

## Consistency fixes to include in editing

- The direct-comparator table still uses denominators of 75 for UCPO/RLEP. For the selected 0.5B/1B scope, explicitly report UCPO 50/50 and RLEP 47/50 admitted cells, with the scope stated. Keep historical 3B registrations in the archive rather than presenting them as missing requested experiments.
- Replace “Still needed,” “finish PantryPlan,” “newly completed,” and “now joins” with current coverage and exact n. Preserve partial-result visibility.
- Put detailed effects in Results, rather than inside Reproducibility. Use one authoritative current result table per family.
- Standardize accuracy as `pass@8` and verified modes as `distinct@8`. Keep `mean@8`, extra modes, and collision statistics where the particular analysis needs them, rather than letting them become competing primary stories.
- The new hosted-model evaluation is context, not an old training experiment. Keep its inference-only interpretation explicit. The long-paper macro table is currently defined in both hosted main-text and appendix inputs with the same label; retain one displayed copy. For streamlining, keep a short contextual main-text mention and the detailed hosted protocol/domain results together in the supplement. It need not become a fourth central training experiment.
- Old bank-normalization, starvation-fallback, and open-bank probes are already excluded from the active mechanism claims. Disk presence alone is not compiled-paper content.
- If figure/table sections are removed, update manuscript references, figure inventory, build contracts, short-paper snapshot, and submission checks together. Preserve archived data and audits.

## Preserve across every cut

All four primary methods at 0.5B, 1B, and 3B; Level-2 evidence at its registered model scale; both accuracy and modes; UCPO/RLEP only at 0.5B and 1B; the baseline precheck; the real fixed-semantic and frequency-weighting controls; source-integrity exclusions; partial cohorts; unfavorable or inconclusive effects; and the distinction between descriptive results and complete-block inference.
