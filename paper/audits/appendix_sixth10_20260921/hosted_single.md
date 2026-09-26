# Appendix N: single-deployment concentration

Only `paper/results/frontier_hosted_20260911_appendix.tex` was edited. The three row fragments, copied result JSON, and experiment records remain unchanged. No live rendering generator for this hand-authored fragment was found after searching live `ops`, repository code references, and retained hosted code; no generator synchronization applies.

## Presentation changes

- Replaced exploratory/current/inherited/completed/accepted-population narration with the deployment, task population, request settings, and direct purpose of the evaluation.
- Removed experiment IDs, construction admission history, provenance/hash/receipt inventories, audit/cache/reconciliation chronology, first-response timing, post-hoc labeling, example selection history, and the reproduction-record paragraph from the rendered appendix.
- Recast all three table captions as supported findings followed by sample counts, grading, estimands, weighting, and interval definitions.
- Linked the level-construction description to `app:data-levels` and retained the distinct native Level-1 versus matched-reference populations and unpaired cross-level prompts.
- Defined mean@8, pass@8, distinct@8, and pair-weighted collision directly. Retained the exact collision equation and uniform-correct-key reference. Clarified that bootstrap intervals describe empirical prompt-population uncertainty with all eight generations kept together.
- Described the fixed normalization operations and exclusions directly, preserving strict successes and original keys. Explained that adding correct responses changes eligible pairs and pair weights, so collision can rise or fall after normalization.
- Retained that 107 unsuccessful API/transport attempts were retried and excluded from the 15,360 returned responses rather than graded as wrong answers. Distinct response IDs and stateless requests do not establish independent provider randomness, and hidden reasoning content is not exposed.
- Retained all per-level/domain caveats, symbolic execution versus numerical shortcuts, prompt guidance, Pantry support-versus-quantity interfaces, unknown Countdown total support, unseen-mode possibility, and lack of causal training/difficulty/scale identification.

## Verification

`validate.py` passes 132 checks. These compare all 33 displayed table rows and intervals to the source summary; reconstruct all 30 strict/normalized domain-level correctness, pass@8, distinct@8, and collision point estimates and count totals from existing response records; verify numerical prose; verify request settings and attempt counts; and assert unchanged table environments, equation, labels, data fragments and cross-references. No API calls, grading changes, new samples, or bootstrap reruns were performed.

The normalized cache has 15,361 physical entries for 15,360 response identities because its documented correction appends a replacement for one earlier entry. Validation uses the same last-record-per-identity convention, matching all published final counts and estimates. This storage detail is recorded here, not narrated in the paper.

All original subsection labels remain. The surrounding main.tex content was reviewed read-only at the parent agent's request. Full-document typesetting and page-boundary checks are handled by the parent.
