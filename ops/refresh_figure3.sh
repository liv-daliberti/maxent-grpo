#!/usr/bin/env bash
# Rebuild Figure 3 from whatever grid cells have completed, then recompile the
# paper. Safe to run repeatedly while collection is still in flight.
set -euo pipefail
ROOT=/n/fs/similarity/maxent-grpo
PY=$ROOT/var/seed_paper_eval/paper310/bin/python
MF=$ROOT/artifacts/modebench_base_level_grid_multifamily_20260914
# Re-pinned after the PythonFactors size-cap re-grade; see regrade_python_factors.py.
BASE=$ROOT/artifacts/modebench_base_level_grid_20260911/status_updates/20260912T152423Z/figure_source.json

cd "$ROOT"
$PY ops/build_multifamily_figure_source.py | tail -1
rm -f paper/results/mode_diversity_base_grid.json
$PY ops/build_mode_diversity_payload.py --source "$BASE" --source "$MF/figure_source.json" | tail -1
$PY ops/plot_paper_mode_diversity_levels.py > /dev/null
# Keep the \MDcells/\MDpmdcorr macros in step with the payload the figures use.
$PY ops/build_mode_diversity_table.py > /dev/null
cd "$ROOT/paper"
for _ in 1 2; do pdflatex -interaction=nonstopmode -halt-on-error main > /dev/null 2>&1; done
cd "$ROOT"
$PY ops/report_grid_trends.py
echo
echo "receipts: $(ls "$MF"/receipts/*.json 2>/dev/null | wc -l)/280   main.pdf rebuilt"
