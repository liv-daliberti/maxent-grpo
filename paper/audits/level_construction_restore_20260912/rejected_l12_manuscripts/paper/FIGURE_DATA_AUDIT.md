# Figure coverage and scientific boundaries

The current papers retain 26 scientific figures: eight main and eighteen supplementary in each version. ICLR replaces the hosted main plot with per-model/per-level averages of empirical `pass@8` and mean `distinct@8`, and preserves the complete plot in the hosted appendix. The nineteen original figures, four newer quantitative views, two later sampling-budget figures, and the separate Methods construction panel are all present. See the [placement inventory](FIGURE_MANIFEST.md) and [restoration source map](audits/figure_restoration_20260911/figure_placement.json). The added construction plot uses the same measured admission rows as the earlier combined panel.

## What the new figures establish

| New figure | Frozen source | Measurement and boundary |
|---|---|---|
| `concentration_story` | `results/conditional_concentration_20260911.json` | 24 Graph/Pantry comparisons: two baseline objectives before/after and two matched replay additions across three scales. Equal-prompt collision on explicit common eligibility; separate disjoint-stream orientations. Graph/Pantry have substantial initial breadth; all five domains and 135 contrasts remain reported. |
| `replay_factorial_effects` | `figures/experiment1_retention_comparator_matrix.json`; `figures/e118_all_scale_factorial_progress.json` | All three models and five Level-1 domains, Dr.GRPO and MaxRL replay effects. Pass@8 probability points and distinct@8 expected keys retain distinct axes and exact source intervals. |
| `replay_key_weighting` | `results/e120_primary_breadth.json` | Frozen Qwen2.5-0.5B uniform-minus-frequency weighting, five seeds. Original paired-bootstrap intervals, success and extra modes; correctness is not held fixed. Expanded scales remain supporting evidence. |
| `replay_level2_effects` | `results/level2_factorial_contrasts_20260911.json` | Within-Level-2 contrasts on the all-four-arm intersection. Four complete domains have n=5; Pantry has n=1, no five-seed interval. This is not a cross-level difficulty effect. |

All estimates and intervals are copied from their frozen result records; the rendering scripts perform no new resampling or outcome selection. JSON sidecars bind exact plotted values, cohorts, source bytes, builder bytes, and PDF/PNG outputs. The scientific build reconstructs these records and checks artifact hashes. Figures are retained as vector PDFs; the workshop uses a compact construction display.

## Cohorts and uncertainty

The primary Level-1 replay cohorts contain 74 Dr.GRPO pairs and 75 MaxRL pairs. Falcon Countdown's Dr.GRPO comparison uses four registered admissible seeds; every MaxRL domain, including all five 3B domains, uses five. Partial cohorts retain exact n and receive no five-seed interval.

Level 2 has 90/100 admitted arm endpoints: four complete domain factorials and partial Pantry. Pantry has D/RD/M/RM arm counts 3/4/1/2; its two-arm Dr replay intersection has two seeds, but only seed43 is common to all four methods. The new Level-2 figure deliberately uses that declared four-arm population for both contrasts. Broader arm availability does not silently enlarge this figure's cohort.

The original weighting analysis stays bound to its frozen September4 input. The later weighting census has 44/45 treatment endpoints, with 3B Graph n=4 and Pantry n=5; those later results do not replace the primary 0.5B analysis or its bootstrap.

## Concentration and sampling

The completed saved-output analysis covers 475 registered runs and 892 available checkpoint evaluations. Under the audited vLLM child-seed mapping, four eight-output groups yield eleven nominal streams. The analysis selects the earliest saved position per stream without consulting correctness, and checks separate lower-five/upper-six orientations. Historical runtime identity is incomplete and seeds recur across prompts; distinct identifiers do not establish iid sampling.

Per-prompt collision is undefined with fewer than two correct representatives. Paired eligible-prompt means and their nominal seed intervals describe the stated observable population. Changing eligibility and common random streams limit a population interpretation. The original K=8 success and breadth metrics still use their intact groups. The report exposes simultaneous per-sample correctness decreases, stream-sensitive Graph comparisons, sparse adverse Level-2 Pantry results, and the difference between mitigation and restoration to the initial distribution.

The initial protocol and stream amendment precede the first effects. A second, explicitly documented census extension followed inspection of those effects after the paper's source census refreshed. It includes all thirteen newly available terminal checkpoints and withdraws one newly conflicted initial checkpoint. Both source snapshots, caches, and result phases are preserved; the extension is not presented as prospectively frozen before the initial findings.

## Hosted and cross-level evidence

The hosted overview retains all seven deployments, five domains and three levels, using the frozen formatting normalizer. Revised Opus Python wording remains explicit; the display is not a common-prompt ranking. Its collision summary pools correct pairs, whereas the training comparison weights eligible prompts equally. These are related measurements with different aggregation weights.

The original-protocol hosted Graph figure, temperature comparisons, retries, and full prompt-specific tables remain in the supplement. They identify neither an RLVR training cause nor a model-size effect. Temperature can improve observed breadth in the measured no-reasoning configuration; no general optimum is claimed.

Level-2 comparisons hold prompts and the prescribed protocol fixed within each level. Different terminal populations, prompts, syntax and generation budgets prevent a pure cross-level difficulty interpretation. The Methods construction figure presents historical frozen Qwen2.5-0.5B development admission for Levels 1–2. The experiment figure separately presents terminal training means. The planned five-level, four-model held-out grid remains incomplete and does not appear as complete evidence.

## Retained evidence and validation

All original example, benchmark, method, endpoint, training-trajectory, comparator, fixed-bank, hosted, and local prompt-intervention figures remain source-bound. The numerical checks for those original figures were preserved when the editorial layout changed. Missing training checkpoints remain gaps; teacher-forced exemplar scores remain surrogates rather than exact mode probabilities; the local prompt intervention retains its omitted hosted panel disclosure.

Every formal statement/proof block is preserved byte-for-byte in each manuscript. The [restoration audit](audits/figure_restoration_20260911/) records prior files, block hashes, restored placement, figure checks, and both builds. The earlier reorganization remains archived separately. Workshop validation retains official style, anonymity, a four-page main, complete copied inputs, and compiled-source/hash verification. The ICLR check retains its nine-page maximum and independently checks the rendered references boundary and every main caption.

## Sampling-budget follow-up

The second appendix control evaluates 64 fresh responses per selected problem
and wording, with 16 problems in each of six domain/level cells. Its 110,592
responses cover the same 25 local checkpoints as the first control; all 25 jobs
completed successfully, and every draw and strict grade is authenticated. One
raw Python grade was independently corrected; formatting normalization adds no
successes. The original grades remain intact.

Exact rarefaction separates pass@k, distinct@k, and additional modes through
64; preassigned prefixes remain a labeled sensitivity. Correct-draw rarefaction
and collision use explicit own/joint eligibility, equal-seed means, and
certified-support uniform bounds. Python/MathIR have five seeds, Pantry two.
The full panel shows both persistent concentration and additional late
discovery; it supplies no completed frontier result at this publication stage.
The hosted cohorts are collecting under their separately frozen native settings.
