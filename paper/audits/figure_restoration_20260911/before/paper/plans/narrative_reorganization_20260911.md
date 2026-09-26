# Mode Collapse in RLVR & ModeBench: implemented organization

Status: **implemented in both manuscripts on September 11, 2026**. The broader reorganization and figure update are complete. The title remains **Mode Collapse in RLVR & ModeBench**. The [completion audit](../audits/narrative_reorganization_20260911/README.md) records the builds, figure placement, source preservation, and visual review. The [original planning document](../audits/narrative_reorganization_20260911/before/paper/plans/narrative_reorganization_20260911.md) preserves the reasoning, proposed budgets, and optional future analyses.

## The argument now implemented

A binary reward records that a response succeeded; its execution identity records which success occurred. ModeBench supplies those identities. The saved-output analysis measures concentration during training, conditional theory explains an amplification mechanism, and matched replay comparisons test an intervention. Hosted observations show why this measurement matters beyond the local suite while retaining their different sampling and causal scope.

The main empirical claim is specific: all twelve Dr.GRPO/GRPO × scale × Graph/Pantry contrasts increase correct-key collision in the primary analysis. Eleven agree in direction across both stream orientations. All six ReplayDr.GRPO comparisons on those domains decrease collision, with both orientations agreeing. MaxRL replay and other domains retain their heterogeneous or undefined findings. This supports observed collapse and demonstrated mitigation without asserting universal concentration reduction or literal extinction of modes.

Full-test success and sampled breadth remain independent empirical outcomes. They are not used as substitutes for conditional concentration, and the main text states the lower per-sample correctness on the replay comparisons' jointly eligible prompts.

## ICLR main and figure placement

The compiled main occupies seven pages within the existing nine-page limit; references begin on page eight. Five main figures replace eight earlier main displays.

| Section | Scientific role | Main figure |
|---|---|---|
| 1. Introduction | Frame verified alternatives as the common object of measurement, theory, and intervention. | — |
| 2. ModeBench: From Correctness to Verified Alternatives | Define five domain keys, success, sampled breadth, and conditional collision. | Compact domain table and two identities. |
| 3. Concentration During Training and in Hosted Models | Establish longitudinal concentration, then the relevance of the measurement in hosted deployments. | 1: concentration changes and matched replay effects; 2: hosted correctness and breadth. |
| 4. A Conditional Mechanism and Canonical Replay | State the categorical mechanism, actual replay loss, and bank-mass/balance identity with their assumptions. | Equations in main; schematic in appendix. |
| 5. Controlled Replay Comparisons | Compare replay under Dr.GRPO and MaxRL across domains and scales; separate concentration from full-test outcomes. | 3: domain-specific replay effects on pass@8 and distinct@8. |
| 6. What Matters, and Where the Benefit Weakens | Test key weighting and replication within Level 2, with cohort and interpretation limits. | 4: uniform-minus-frequency weighting; 5: within-Level-2 replay effects. |
| 7. Related Work and Discussion | Position measurement, objective design, and verified replay. | — |
| 8. Conclusion | State demonstrated outcomes and limits of bank coverage, exemplar scores, and neural guarantees. | — |

Four new figures are generated from frozen sources, with exact estimate, interval, cohort, builder, and rendered-output hashes. The hosted overview is retained unchanged. All nineteen earlier figures remain compiled; detailed examples, schematic, comparator catalogue, absolute endpoint displays, admission, and temperature curves now sit beside their supporting appendix material. The [figure inventory](../FIGURE_MANIFEST.md) lists all twenty-three assets and both editions' placements.

## Appendix organization

Both supplements now follow the evidence rather than experiment chronology:

1. Claim and evidence map.
2. Measurement identities and interpretation.
3. Complete conditional mathematical results.
4. Benchmark construction, examples, and admission.
5. Verbatim prompts and task contracts.
6. Implemented replay algorithm and controlled design.
7. Complete controlled results, endpoints, and trajectories.
8. Longitudinal concentration, prechecks, and exemplar-score diagnostics.
9. Hosted observations, temperature, and prompt/protocol sensitivities.
10. Admitted local prompt intervention; incomplete hosted panel disclosed.
11. Reproducibility and disclosures, with workshop related work retained before them.

All eighteen formal statement/proof blocks in each manuscript are byte-identical to the validated pre-reorganization versions and identical across editions. The proof and measurement details precede the empirical supplement, making the mathematical basis easier to find. The new main equations point to that full chain.

## Mathematical and empirical boundaries

The main contains the occupancy identities, collision definition and two-draw identity, idealized concentration derivative, implemented mean-token exemplar loss, and the categorical mass/balance decomposition. The collapse statement specifies a finite isolated prompt, at least two correct and one incorrect category, finite initial logits, independent category logits, common length normalization, group size at least two, and Euclidean infinitesimal expected on-policy updates without a competing regularizer.

The retention statement remains conditional on the stated categorical dynamics, a fixed bank, and a positive replay dose. It protects banked modes; it does not establish all-mode coverage, visibility in eight draws, neural-optimizer convergence, or held-out transfer. Fixed-exemplar scores remain surrogates, and their declining tail is acknowledged.

The concentration figure exposes eligible-prompt coverage and both disjoint-stream orientations. All 135 contrasts remain in the full analysis, including sparse adverse and undefined cells. Eleven nominal streams replace a false 32-iid interpretation of the overlapping saved groups. Hosted pair-pooled collision and training equal-prompt collision keep different aggregate estimands. The retrospective census extension remains explicitly documented after inspection of the initial effects.

Level 2 is a within-level replication under another construction. The new effect figure uses the common four-arm intersection: four complete domains with five seeds and descriptive Pantry with one. Different cross-level populations, prompts, syntax, and generation budgets prevent an isolated difficulty-effect claim.

## Workshop adaptation

The workshop remains four main pages in the official anonymous style, with references starting on page five. It uses three main figures: concentration on page one, the replay factorial on page three, and key weighting on page four. Hosted and Level-2 findings appear briefly in main, with the complete displays in the supplement. The same twenty-three figures and full proof chain are present overall.

## Completed checks and remaining scope

The figure and numerical contracts retain all existing endpoint, source, cohort, and uncertainty checks. Only the editorial expectations for main figure count, ordering, and appendix placement changed. The updated checks additionally reconstruct all new figure metadata and verify rendered bytes and preservation of the mathematical chain. Both manuscripts compile and have been visually reviewed; the workshop source ZIP is independently checked against its build receipt.

No new model generation or training was needed. Optional common-stratum standardization, retrospective reserve evaluation, and a controlled paired difficulty intervention remain future scientific options, not unfinished requirements of this reorganization. Their costs and interpretation limits are preserved in the original plan and [cross-level audit](crosslevel_evidence_and_options_20260911.md).
