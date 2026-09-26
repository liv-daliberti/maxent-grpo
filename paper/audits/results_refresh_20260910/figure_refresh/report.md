# Figures 5 and 6 — September 10 refresh

Figure 5 and its appendix companion now bind to the final September 10 endpoint census, SHA-256 `a27e6762a3f05d6f3fa6b5f6659cf400d4a929a67db892c69f2f6c96c424a8e3`. The plotted completed tracks, both PNGs, and the generated Qwen-3B Python table are unchanged. The JSON sidecars include newly admitted Qwen-3B MaxRL pairs: Countdown 2, Graph 5, MathIR 3, Pantry 1, and Python 5. The Qwen-3B cross-domain MaxRL track remains incomplete.

Figure 6 uses an independent full-checkpoint snapshot collected September 10, 16:52:52–16:55:40 UTC. Its E118/E119 ledger hashes match the canonical census, and all 85 admitted Level-2 terminal cells match the census exactly. The admission data and complete Graph example remain fixed. The dated comparison snapshot is retained in `../figure6/modebench_level_comparison_snapshot.json`, SHA-256 `11f3b57cf0b3ee0933694f115c3c70d5e2d1e0c92aafe4e5abab14c3c71c79a0`.

The matched interim comparison has 22 domain–seed cells, up from 21. Pantry contributes seed 43 at shared step 672 (1.75 passes) and seed 45 at actual step 0. Each other domain contributes all five seeds at step 3072. Both levels and all four methods use the same checkpoint within every selected cell. Each domain receives equal weight after averaging its available matched seeds.

| Level | Replay comparison | Change in pass@8 | Change in distinct@8 |
|---|---|---:|---:|
| 1 | ReplayDr.GRPO − Dr.GRPO | +0.321328 | +0.934453 |
| 1 | ReplayMaxRL − MaxRL | +0.287539 | +0.901992 |
| 2 | ReplayDr.GRPO − Dr.GRPO | +0.244102 | +0.418203 |
| 2 | ReplayMaxRL − MaxRL | +0.121523 | +0.238945 |

These are descriptive interim equal-domain means, with no new interval or pooled terminal-effect claim. The manuscript statement that replay improves both interim means at both difficulty levels remains supported. Figure 6's caption/prose must use 22 cells and the two exact Pantry seed/checkpoint pairs. Figure 5's displayed numeric effects require no revision.

Validation: 18 existing focused tests passed; `validate_figures.py` passed ledger, census, selection, terminal-value, fixed-reference, and prior-artifact checks. Figure 6 was visually inspected; labels and markers are unclipped. Both unchanged Figure 5 PNGs were verified by byte equality. All 11 prior figure/result artifacts are retained under `before/` with hashes in `before_sha256.json`. No generator, active controller, or pinned helper source was edited.

Reproduction from retained inputs:

```bash
python ops/exp_scaling/plot_paper_e118_all_scale_progress.py --endpoint-audit paper/audits/results_refresh_20260910/latest_endpoints.json
python ops/exp_scaling/plot_paper_modebench_levels.py --snapshot paper/results/modebench_level_comparison_snapshot.json --output paper/figures/modebench_level_admission
python paper/audits/results_refresh_20260910/figure_refresh/validate_figures.py
```
