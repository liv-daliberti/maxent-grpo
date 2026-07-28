# E32 visualization-only amendment: confidence bands

**Status: FROZEN — requested after launch on 2026-07-22**

This amendment changes only the display of uncertainty in E32-consuming
figures. It does not change training, evaluation, metrics, exclusions, curve
stitching, or any stored trace.

- Remove the plotted raw-draw dots and min/max whiskers.
- Retain all four raw fixed K=8 draws in the machine-readable curve artifacts
  and evaluation trace JSONL.
- Continue to show every unsmoothed training-seed trajectory as a thin line.
- Continue to draw the heavy mean only where all three training seeds exist.
- For mean@8, pass@8, coverage@8, and distinct@8, average the three training
  seeds separately within each of the four fixed evaluation draws. Around the
  mean of those four draw-level means, show the two-sided 95% Student-t
  interval with three degrees of freedom: mean ± 3.182446 × SE.
- The interval represents Monte Carlo uncertainty from the four fixed K=8
  evaluation draws in the reported three-seed mean. It is not labeled or
  interpreted as uncertainty over training seeds.
- Clip intervals only to each metric's mathematical support. Pass@1 remains a
  deterministic greedy metric and receives no evaluation-draw interval.
- No smoothing or interpolation is applied to the underlying trajectories.
