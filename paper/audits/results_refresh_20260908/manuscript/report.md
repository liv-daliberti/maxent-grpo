# September 8 manuscript and figure refresh

The dated integrity-checked census admits 124/150 E118 endpoints (59 matched pairs; 11 complete blocks), 76/100 E119 endpoints (three complete blocks), and 33/45 E120 endpoints (six complete blocks). All six completed E120 blocks pass the persisted mechanism audit. The September 4 primary E120 artifact is unchanged.

Python and MathIR newly complete the Level-2 factorial. All four new replay effects have positive co-primary means, with unadjusted paired Student-t intervals including zero. The paper reports these estimates without promoting the incomplete Countdown or PantryPlan blocks.

Figure 5's audit now includes Qwen3B Graph seeds 70, 72, 73 and MathIR seed 74; only the completed Python block supports five-seed E118 inference at that model scale. The main plotted tracks retain their completed-scale gate. Its breadth axis now extends to 1.8 so the seed maximum 1.692578125 is visible.

Figure 6 uses the refreshed, independently checked 200-cell coverage inventory. Graph/Python/MathIR use all five seeds at pass 8. Countdown uses steps 3072, 3072, 2592, 2304, 2304; PantryPlan retains its sole common initial evaluation (seed 45, step 0). The comparison remains an interim equal-domain summary over 21 domain-seed pairs.

Reproduce the frozen report and figures:

```bash
python ops/exp_scaling/build_paper_latest_results.py --date 2026-09-08 --from-audit paper/audits/results_refresh_20260908/latest_endpoints.json
python ops/exp_scaling/plot_paper_e118_all_scale_progress.py --endpoint-audit paper/audits/results_refresh_20260908/latest_endpoints.json
python ops/exp_scaling/plot_paper_modebench_levels.py
python ops/check_paper_current_contract.py
```

The source contract passes and 29 focused tests pass. Parent/workshop compilation and packaging are handled by the coordinating agent.
