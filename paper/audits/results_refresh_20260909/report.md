# September 9 current-results integration

The current source-admissible census contains 126/150 E118 endpoints (61 matched pairs; 11/15 complete pair blocks), 83/100 E119 endpoints (four complete four-method domains), and 35/45 E120-R1 endpoints (six complete weighting blocks). The census was collected from 17:34:45 to 17:43:25 UTC.

The new readable report and CSV/JSON cover all 34 planned contrasts, including 30 with observed terminal pairs and 25 complete five-seed contrasts. The standalone two-panel forest plot displays correctness and extra correct modes with exact paired denominators; partial blocks have no interval. Both paper appendices now contain all current E118/E119/E120 paired effects and a separate E119 four-arm table of estimator effects and replay interactions. Main results and coverage text, Figures 5 and 6, and the two READMEs are current. The workshop PDF and standalone source ZIP were rebuilt.

Level-2 Countdown now completes the fourth full factorial. Its ReplayMaxRL extra-mode effect is +.016796875 with descriptive paired 95% interval [.001117734, .032476016]; correctness and raw distinct-mode intervals include zero. Graph remains the clearest harder-task result, with positive correctness and breadth replay effects under both estimators. Its negative interaction means the marginal replay benefit is smaller under MaxRL. Current Qwen-3B E118 prefixes are Graph n=4, Python n=5, MathIR n=1, Pantry n=1, and Countdown n=0. New E120 Qwen-3B Graph pairs favor uniform weighting on breadth; partial Falcon Pantry pairs favor frequency weighting.

The September 4 E120 primary input is byte-for-byte unchanged. No missing run is imputed, no partial block receives a five-seed interval, no outcome-based checkpoint selection was added, and no experiment job or protocol was changed. The initial working tree was already extensively modified; relevant pre-update paper files were preserved in before/. The new table contract reproduces every table from its frozen source and checks factorial effects on their all-four-arm intersection.

Validation completed:
- 50 relevant regression tests passed (targeted-tests.log).
- All digest statistics and factorial contrasts independently reconciled with the dated audit, including seed-level effects, interval gates, source hashes, and deterministic reproduction.
- Full paper: evidence contract, all five frozen prompt checks, LaTeX/BibTeX build, and all 315 prose line-fill checks passed (paper-build-r2.log); 65 total pages, with conclusion on main page 9.
- Workshop: four content pages, references starting on page 5, six main/eight supplementary figures, source hashes, compiled receipt, anonymous style, no unresolved references or overfull boxes, and standalone ZIP checks passed (workshop-build-r2.log and workshop_validation.json).
- New full-paper tables on pages 62–64, workshop page 4 and supplementary tables, and the standalone forest plot were visually inspected. A one-word caption widow in the first parent build was shortened; both final paper versions use the corrected wording.

Entry points:
- paper/results/current_campaign_results_20260909.md
- paper/results/current_campaign_results_20260909.csv
- paper/figures/current_campaign_results_20260909.pdf
- paper/main.pdf
- paper/mathai2026/main.pdf
- paper/mathai2026/mathai2026-source.zip

Reproduction uses build_paper_latest_results.py --from-audit for the frozen census, build_paper_current_campaign_results.py --snapshot paper/results/latest_results_20260909.json --figures for the digest, build_paper_level2_factorial_contrasts.py --date 2026-09-09 for factorial estimates, plot_paper_e118_all_scale_progress.py --endpoint-audit for Figure 5, and the retained Level-1/Level-2 snapshot for Figure 6. Then make -C paper and make -C paper/mathai2026 bundle build the manuscripts. No external publication or repository commit was performed.
