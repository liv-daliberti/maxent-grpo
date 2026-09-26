# Primary-method training-curve freeze — 2026-09-11

This audit supports the appendix accuracy (`pass@8`) and correct-mode
(`distinct@8`) training figures in both papers. It contains all 400 registered
primary-method cells: Dr.GRPO, ReplayDr.GRPO, MaxRL, and ReplayMaxRL at three
Level 1 model scales, plus all four methods at Qwen2.5-0.5B on Level 2.

`coverage_snapshot.json` retains normalized four-draw records, source-prefix
SHA-256 hashes, exact source line numbers, missing checkpoints, and conflicts.
`selection_summary.json` gives the admitted population and missing-checkpoint
inventory. The final rendering input is
`paper/results/training_curve_snapshot_20260911.json`.

Terminal populations are fixed to the copied September 11 endpoint census,
core endpoints, and main Figure 5 data in this directory. Level 1 curves use
each objective pair's own per-domain terminal seed intersection: 74 Dr.GRPO
pairs and 67 MaxRL pairs. The 3B MaxRL counts are Graph 5, Countdown 3,
Python 5, MathIR 3, and Pantry 1. Level 2 has 20 complete four-arm domain/seed
blocks, plus the Pantry Dr.GRPO pair at seeds 43 and 46. Other observed
histories remain separate descriptive series with no paired effect.

The registered grid is steps 0–3072 in increments of 192 (training passes
0–8). Four complete fixed-seed K=8 draws at temperature 1 on 128 prompts are
required jointly for both metrics. Four draws are averaged within each seed.
A primary mean requires the entire fixed cohort; the renderer additionally
requires both members of an objective pair at each displayed checkpoint.
Missing or conflicted checkpoints remain gaps, with no interpolation or
forward filling. Some evaluators additionally wrote observations at 96-step
intervals; these extra points are omitted from the registered figure grid.

Source paths are authorized by the copied terminal census for each scientific
cell. This preserves valid pre- and post-continuation observations together;
a promoted scheduler job ID does not erase earlier training evidence.
Identical repeated draws are deduplicated, and conflicting records invalidate
the whole checkpoint without choosing a retry. Sources created after the
frozen census are excluded and identified. The existing all-step Falcon
Countdown ReplayDr.GRPO seed-59 exclusion is retained and hash checked.

The exact 200-cell Qwen2.5-0.5B coverage from the September 11 Figure 6 audit
is reused after source hash and source-set checks; the additional 200
larger-model cells are frozen from their census-authorized log prefixes.
All admitted terminal observations are checked exactly against the census.
Level 1 terminal seed values also reconstruct the main Figure 5 values.
Normal paper builds only read the final frozen JSON and never poll live jobs.

To recompose from this audit without rereading run logs:

```sh
python ops/exp_scaling/build_paper_training_curve_snapshot.py --reuse-coverage
```
