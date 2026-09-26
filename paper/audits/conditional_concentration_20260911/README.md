# Saved-output conditional concentration: analysis and provenance

The final analysis covers 475 registered runs, 892 available checkpoint evaluations, and 135 reported contrast blocks. It uses saved verified keys; no new training, model inference, hosted API calls, or regrading was launched.

Start with [the result report](../../results/conditional_concentration_20260911/report.md), [the simultaneous metric changes](../../results/conditional_concentration_20260911/metric_tradeoffs.md), and [the overview figure](../../results/conditional_concentration_20260911/conditional_concentration_overview.pdf). The manuscript includes exact identities, assumptions, eligibility, and every registered contrast in `conditional_concentration_20260911_appendix.tex` and its generated tables.

## Analysis phases

1. `analysis_plan.md` and its binding were frozen before the first concentration contrasts.
2. `analysis_plan_amendment_1.md` documented overlapping child-generation seeds before those contrasts: four saved K=8 groups yield eleven nominal streams under the audited runtime mapping. Selection uses earliest saved positions regardless of their outcomes. The original K=8 metrics retain all four original groups.
3. `verified_samples.jsonl.gz` preserves the original source-admitted sample cache. `conditional_concentration_20260911_initial.json` preserves the first completed result.
4. The paper's source census was refreshed while the analysis was underway. `amendment_2_current_census.md` explicitly records the subsequent extension **after inspection of the initial results**. It includes all thirteen added terminal checkpoints and removes one newly conflicted initial checkpoint. Both exact census snapshots are preserved in `source/`.
5. `verified_samples_completed_cohort.jsonl.gz` and its collection receipt support the final result. The Level-1 terminal replay cohorts contain 74 Dr.GRPO and 75 MaxRL pairs before conditional eligibility. Initial-history availability and sufficient correct outputs remain separate requirements.

The estimator identity is promptwise and conditional on fixed-law iid sampling. Historical dependency identity is incomplete, nominal stream IDs repeat across prompts, and common eligibility can select on shared randomness. Distinct-stream and disjoint-orientation empirical means are reported descriptively; neither their nominal intervals nor the source certificate proves an all-prompt population effect.

## Reproduce from the frozen completed cache

Run from the repository root with Python 3.11:

```sh
/usr/local/anaconda3/2024.02/bin/python ops/exp_scaling/analyze_paper_conditional_concentration.py analyze \
  --cache paper/audits/conditional_concentration_20260911/verified_samples_completed_cohort.jsonl.gz \
  --receipt paper/audits/conditional_concentration_20260911/collection_receipt_completed_cohort.json \
  --cohort-binding paper/audits/conditional_concentration_20260911/amendment_2_current_census_binding.json \
  --output /tmp/conditional_concentration_reproduced.json
```

The analysis validates original and extended hashes before computing effects. Output timestamps differ on repetition. Raw-source reconstruction is available through the original intake and incremental extension scripts; collection deliberately refuses to overwrite frozen caches.

```sh
/usr/local/anaconda3/2024.02/bin/python -m pytest -q \
  tests/test_paper_collision_statistics.py \
  tests/test_paper_collision_sample_sources.py \
  tests/test_paper_grpo_collision_sources.py \
  tests/test_paper_collision_cohort_extension.py
/usr/local/anaconda3/2024.02/bin/python paper/audits/conditional_concentration_20260911/validate_final_analysis.py
```

All 85 tests passed. Independent statistical review did not inspect empirical concentration effects. It covers the collision identity, shared-randomness counterexamples, the 32-to-11 stream correction, outcome-independent representative selection, separate orientation eligibility, original K=8 metrics, and fixed-across-seed populations. Source tests cover reward gating, frozen-prefix hashes, exact cohort preservation, duplicate-origin conflicts, and the complete-census extension.

`analysis_claim_validation.json` checks the manuscript's numerical claims and every block's undefined-value, seed-count, and eligibility contract. The report renderer binds input, code, and output hashes in its `build_manifest.json`.

## Manuscript validation

Both manuscript builds passed. ICLR has nine main pages and 74 total PDF pages, with its source/evidence, references, figure, and line-fill checks passing. The workshop retains four content pages, all seven main figures, twelve supplementary figures, and its official style and anonymous submission checks. The current paper Makefile uses its designated Python 3.10 interpreter for previously frozen source-contract comparisons; the new concentration analysis uses Python 3.11. The separate interpreter audit explains last-bit Student-t differences without changing scientific inputs.

Build logs: `main_build_final.log`, `workshop_build.log`. Workshop synchronization preserved previous source and asset bindings in `workshop_sync/`. No main-paper figure was silently replaced; the new concentration figures are standalone artifacts for the planned narrative reorganization.
