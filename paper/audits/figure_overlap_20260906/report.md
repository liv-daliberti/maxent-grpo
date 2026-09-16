# Whole-paper figure overlap audit — 2026-09-06

All 13 numbered figures were inspected as standalone 1500-pixel PDF renders and on every figure-bearing page of the existing full and workshop PDFs. The existing workshop crops on Figures 2 and3 introduce no clipping. Two layout issues were fixed; all other figures remain unchanged. Final rebuilds are coordinated by the parent agent after the concurrent prose edits.

Figure 4’s Countdown/Python labels had only 0.20 pt separation. Reducing only the x-tick font from 6.6 to 6.0 pt increases the minimum gap to 2.90 pt while preserving all labels, cells, values, seed exclusions and color encodings. The source already contained the workshop’s 5 pt panel-B inset and omitted the duplicate footnote (seed coverage is explicit in both captions); both mirrors now use that existing layout.

Figure 12B’s capacity 16 annotation overlapped the title by 4.24 pt vertically and extended outside the axes. The annotation now uses an axes-relative vertical coordinate, remains fully inside the axes, sits 5.03 pt below the title, and is inset 3 pt from its dashed reference line. No curves, intervals, values, axes limits, or labels changed.

Both affected JSON data files are byte-identical to their backups. Figure 5 PDF/JSON remains byte-identical to the completed Qwen2.5-3B addition; the incomplete MaxRL 3B track remains excluded. All 13 primary/workshop figure PDFs now match. No TeX or manuscript builds were changed by this audit task.

| Figure | PDF stem | Full/workshop page in audited PDFs | Finding |
|---|---|---|---|
| 1 | `modecollapse_story` | 1/1 | No text overlap or clipping observed. |
| 2 | `modebench_examples` | 4/2 | No text overlap or clipping observed. |
| 3 | `verified_support_story` | 5/2 | No text overlap or clipping observed. |
| 4 | `experiment1_retention_comparator_matrix` | 7/3 | Countdown/Python x-ticks nearly touched; reduced only x-tick font 6.6 → 6.0 pt. Minimum gap 0.20 → 2.90 pt. Preserved existing workshop panel-B 5 pt inset and used its layout in both mirrors. |
| 5 | `e118_all_scale_factorial_progress` | 8/4 | No overlap. Completed Qwen2.5-3B Dr.GRPO/ReplayDr.GRPO row and all existing rows preserved byte-for-byte in PDF. |
| 6 | `modebench_level_admission` | 9/4 | No overlap. Root made caption model Qwen2.5-0.5B-Instruct explicit for both panels; plot unchanged. |
| 7 | `sustained_auc_effects_qwen05b` | 39/37 | No text overlap or clipping observed. |
| 8 | `direct_comparator_endpoint_effects` | 40/38 | No text overlap or clipping observed. |
| 9 | `direct_baseline_learning_curves_static_strip` | 41/39 | No text overlap or clipping observed. |
| 10 | `baseline_collapse_precheck` | 45/43 | No text overlap or clipping observed. |
| 11 | `verified_support_discovery_two_scale_effects` | 47/45 | No text overlap or clipping observed. |
| 12 | `replay_mechanism_telemetry_qwen05b` | 54/53 | Panel B capacity 16 annotation overlapped title; placed inside axes, 5.03 pt below title and 3 pt left of dashed capacity line. |
| 13 | `e118_scale_extensions_appendix` | 56/55 | No text overlap or clipping observed. |

Visual evidence: `before_rendered/figure_01.png` through `figure_13.png`, independent workshop Figure 4 at `before_rendered/workshop_figure_04.png`, fixes at `after_rendered/figure_04.png` and `figure_12.png`, and all embedded pages/contact sheets in `embedded_pages/`. Exact numerical geometry and hashes: `geometry_before.json`, `geometry_after.json`, `validation.json`. Reproducer: `render_layout_fixes.py` (renders frozen data only).
