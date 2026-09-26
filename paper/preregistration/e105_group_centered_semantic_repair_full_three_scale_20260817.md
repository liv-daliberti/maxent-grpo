# E105: full three-scale evaluation of the score-function repair

**Frozen with E104 and before any repaired model was trained or evaluated on
2026-08-17.**

E105 is the full follow-up defined in the E104 protocol. It can be released
only when `var/artifacts/e104_group_centered_semantic_repair_gate.json` has the
registered passing schema, names the identical frozen runtime snapshot, and
states that no outcome metric was inspected.

The sole treatment is the v6 sampled-group-centered semantic score at fixed
`eta = 0.10` on top of uniform verified-likelihood ReplayDr.GRPO at fixed
weight 0.10. The legacy predictor-centered estimator and adaptive RMS
controller are off. All other objectives and proposal mechanisms are off.

The 75 cells are the Cartesian product of five static domains, five seeds, and
three model scales:

- Qwen2.5-0.5B-Instruct, seeds 43--47;
- Falcon3-1B-Instruct, seeds 55--59; and
- Qwen2.5-3B-Instruct, seeds 70--74.

Domains are Graph Coloring, Countdown, Python Factors, MathIR, and PantryPlan.
PointMaze is excluded. Each run uses the paired replay cohort's data, prompt
surface, optimizer, decoding, and placement, with 384 prompts, eight passes,
3,072 optimizer updates, group size 16, and evaluations every 192 updates.

The primary estimands are paired seed differences against the registered
ReplayDr.GRPO cell within model and domain at pass 8: sampled pass@8 and mean
distinct correct modes@8. Greedy pass@1, sampled mean correctness@8, excess
multiplicity, and pass-0--8 trapezoidal AUC are secondary. Every cell and every
registered checkpoint is shown; no best-checkpoint or domain filtering is
allowed. Breadth gains accompanied by correctness loss are reported as
tradeoffs. The fixed method succeeds as a general extension only if breadth
improves in most model/domain families without a systematic correctness loss;
otherwise the heterogeneity or negative result is reported directly.

