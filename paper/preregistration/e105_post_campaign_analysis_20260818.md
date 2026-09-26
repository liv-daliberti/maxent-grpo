# E105 all-cell post-campaign analysis transaction

Frozen before E105/E109 submission and before any repaired-v6 long-horizon
outcome exists. This is analysis automation only and changes no scientific
configuration or registered estimand.

After E105 submits its 75 treatment cells, it submits one CPU-only job with an
`afterany` dependency on those 75 jobs and all 15 E109 repaired-parser Python
ReplayDr.GRPO controls. The analysis job uses the builder and dual renderer
already bound into the E105 ledger. The builder requires the exact 75-cell
treatment grid, exact 15-cell repaired Python grid, all 17 checkpoints, four
sampled draws per checkpoint, deterministic greedy evaluations, and exact
paired comparator bindings. Therefore any failed, incomplete, missing, or
non-finite cell makes the analysis fail rather than emit a selected subset.

On success it writes the complete machine-readable paired result, the terminal
3-by-5 forest, and the normalized trajectory-AUC 3-by-5 forest. PointMaze is
excluded. No outcome is read before all 90 new jobs reach a terminal scheduler
state; the data-completeness checks, rather than scheduler labels, determine
whether an official result can be emitted.

