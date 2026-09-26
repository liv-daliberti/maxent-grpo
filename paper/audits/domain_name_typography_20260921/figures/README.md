# Domain-name typography in figure assets

All **166 domain-name occurrences in 29 figures** now use monospace. The audit covers all **40 figure PDFs** in the incoming Overleaf archive; the other eleven contain no domain-name labels. The domain words are Graph, Countdown, Python, MathIR, PantryPlan, and the existing abbreviation Pantry.

## Rendering change

`ops/paper_domain_figure_typography.py` applies DejaVu Sans Mono to standalone domain names, preserving their size, color, weight, position, and rotation. Mixed labels retain their surrounding font and use mathtext `mathtt` only for the domain name. The helper is idempotent and skips existing math spans. `paper_style.save` applies it centrally; direct renderers call it before layout or saving.

The local prompt-ablation plot formats domain spans before constructing its fixed tick formatter, so a later draw cannot restore proportional text. Direct discovery and prompt-ablation plotting use the original Matplotlib defaults in an isolated context. This prevents another renderer's global style from changing ticks or other unrelated labels.

All figures were rendered from retained numerical records or deterministic formula/example inputs. No model calls, training, grading, bootstrap sampling, or experimental-analysis pipelines were run. For scripts that also perform analysis, only their direct plotting functions were called.

## Validation

- `inventory.json`: initial font evidence and script references for all forty PDFs. All 166 original domain occurrences used proportional DejaVu Sans.
- `after_inventory.json`: all forty live PDFs; 166/166 domain occurrences now resolve to monospace, with zero remaining proportional occurrences. Four rotated Python labels are reconstructed from adjacent same-font glyph objects because Poppler writes mathtext one character at a time. A separate case-insensitive text scan found no unaccounted lowercase domain names.
- `measurement_validation.json`: all 29 changed PDFs preserve their complete extracted numeric-token multisets. Non-binding figure metadata is unchanged. The hosted subset additionally verifies full plot data arrays and axis limits in `../hosted_figures/validation.json`.
- `binding_validation.json`: all 52 available renderer/style and PDF/PNG output bindings checked successfully. Two training-curve renderer hashes and the withdrawal figure's result-record hashes were refreshed.
- Representative concentration, scale-extension, replay-effect, and withdrawal plots were visually inspected; domain labels fit. Fixed-canvas dimensions are unchanged. Tight-cropped plots have small bounding-box changes from the new glyph widths; no measured values or uncertainty intervals were changed.
- `reference_kl_plane.pdf` differed from the archive before this task and changed independently during it. It contains no domain labels and was not edited by this figure-typography work. Its numeric-text difference is excluded from the 29-figure preservation claim.

## Figure inventory and retained-data rendering paths

| Figure stem | Domain labels | Generator | Rendering path |
| --- | ---: | --- | --- |
| `baseline_collapse_precheck` | 5 | `ops/exp_scaling/plot_paper_baseline_collapse_precheck.py` | `render(existing payload, output)`; central `paper_style.save` |
| `concentration_levels` | 5 | `ops/plot_paper_concentration_levels.py` | Existing source; height 6.6, limits −90/+40 |
| `concentration_story_all_scales` | 5 | `ops/plot_paper_concentration_story.py` | Existing source; all scales, height 6.0, limits −130/+40 |
| `concentration_story_resampled` | 5 | `ops/plot_paper_concentration_levels.py` | Existing source; Qwen3B, Level 1, width 3.15, limits −80/+40 |
| `decoding_objection` | 5 | `ops/plot_paper_decoding_objection.py` | Plot two retained decoding-result JSONs; central save |
| `direct_baseline_learning_curves_pass8` | 5 | `ops/exp_scaling/plot_paper_aligned_domain_strips.py` | `render_ucpo(snapshot=existing figure record, metric="pass8")` |
| `e118_scale_extensions_appendix` | 15 | `ops/exp_scaling/plot_paper_e118_all_scale_progress.py` | `render_appendix_figure(existing record, existing absolute means)` |
| `factorial_training_curves_pass8` | 5 | `ops/exp_scaling/plot_paper_training_curves.py` | `build_figure` from dated 20260919 snapshot, Level 1/pass@8 |
| `factorial_training_curves_pmd` | 5 | `ops/exp_scaling/plot_paper_mode_diversity_curves.py` | Existing diversity-curve payload, Level 1 |
| `frontier_level_grid` | 5 | `ops/plot_paper_frontier_level_grid.py` | Retained source, central save; hosted-agent audit |
| `gpt56_all_levels32_sampling_budget` | 5 | `ops/plot_paper_gpt56_all_levels32_sampling.py` | Existing curve estimates only; hosted-agent audit |
| `hosted_verified_breadth` | 6 | `ops/plot_paper_hosted_breadth.py` | Existing hosted display record; hosted-agent audit |
| `level2_factorial_training_curves` | 5 | `ops/exp_scaling/plot_paper_training_curves.py` | `build_figure` from dated 20260919 snapshot, Level 2 |
| `level2_training_curves_pmd` | 5 | `ops/exp_scaling/plot_paper_mode_diversity_curves.py` | Existing diversity-curve payload, Level 2 |
| `mode_diversity_families_appendix` | 5 | `ops/plot_paper_mode_diversity_levels.py` | Existing base-grid payload, cross-family panels |
| `mode_diversity_level_construction` | 5 | `ops/plot_paper_mode_diversity_levels.py` | Existing base-grid payload, scale rows |
| `mode_diversity_levels_appendix` | 5 | `ops/plot_paper_mode_diversity_levels.py` | Existing base-grid payload, Qwen levels |
| `modebench_discovery_correct_budget_frontier` | 3 | `ops/analyze_modebench_discovery_curves.py` | Direct `plot_figure` on retained report; no analyzer main |
| `modebench_discovery_correct_budget_local` | 3 | `ops/analyze_modebench_discovery_curves.py` | Direct `plot_figure` on retained report; no analyzer main |
| `modebench_discovery_curves_frontier` | 3 | `ops/analyze_modebench_discovery_curves.py` | Direct `plot_figure` on retained report; no analyzer main |
| `modebench_discovery_curves_local` | 3 | `ops/analyze_modebench_discovery_curves.py` | Direct `plot_figure` on retained report; no analyzer main |
| `modebench_examples` | 5 | `ops/plot_paper_modebench_examples.py` | Existing example inputs; font-only artist changes |
| `modebench_prompt_ablation_local` | 18 | `ops/analyze_modebench_prompt_ablation.py` | Direct `make_figures` on retained report; no analyzer main |
| `reference_kl_knee` | 5 | `ops/plot_paper_reference_kl_knee.py` | Existing KL comparison JSON; central save |
| `replay_bank_decomposition` | 5 | `ops/plot_paper_replay_bank_decomposition.py` | Existing measured JSON and deterministic formula curves |
| `replay_factorial_effects` | 10 | `ops/plot_paper_reorganized_results.py` | Retained factorial figure records |
| `replay_key_weighting` | 5 | `ops/plot_paper_reorganized_results.py` | Existing e120 breadth JSON |
| `replay_level2_effects` | 5 | `ops/plot_paper_reorganized_results.py` | Existing Level-2 contrast JSON |
| `withdrawal_pmd_agreement` | 5 | `ops/build_paper_portfolio_withdrawals.py` | Direct `render_agreement_figure(existing results, output)` |

The Python scripts saved here document this session's audit and rendering calls. `session_render_commands.py` uses the session backup directory to keep input records fixed; the table above gives the reusable render entry points. Temporary backups and previews are under `/tmp/paper-domain-typography-20260921/`.
