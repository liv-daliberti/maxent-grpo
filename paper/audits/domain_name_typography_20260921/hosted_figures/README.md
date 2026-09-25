# Hosted and discovery figure typography

Updated eight existing figure families: hosted verified breadth, GPT all-level 512-draw sampling, frontier level grid, four discovery curves/correct-budget figures, and local prompt ablation. Each PDF and PNG was rendered directly from its existing stored figure/report JSON. No analyzer main routine, model, training, grader, bootstrap, or new statistical calculation ran.

All 46 benchmark-domain font spans use the embedded DejaVu Sans Mono face. The check includes `Python` split across six rotated character spans in each discovery figure, as well as the 18 domain names embedded in prompt-ablation tick labels. All visible numeric tokens match the original PDFs. Retained plot records, point values, intervals, and descriptive measurement fields are unchanged. The plotting arrays and limits also remain identical before and after each typography call.

Three direct plotting functions were updated to apply the shared figure helper. Mixed prompt-ablation tick strings receive domain mathtext before constructing their tick formatter, so later layout does not restore the proportional-font text. Direct discovery/prompt plotters run in a context containing Matplotlib's original default settings; this prevents unrelated style state from changing colors, tick spacing, or other fonts.

Figure records update only output paths/hashes and current renderer/source bindings. The existing hosted-level `cohort_builder` and all-level sampling `validation_dependency` hashes were refreshed after the plotting-module changes. Other measurement fields are unchanged. `validation.json` contains embedded-font evidence, numeric-token comparisons, and data-preservation checks. `before/` and `patches/` preserve the three narrow plotting-function edits. The figure owner's common backup contains the original assets.

`render.py` is a reproducible rendering-only driver. It uses existing publication records and the saved asset backup under `/tmp/paper-domain-typography-20260921/before/`; it does not regenerate analyses. Figure compilation and final document layout remain root's release checks.
