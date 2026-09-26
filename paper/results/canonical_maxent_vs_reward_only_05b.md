# Canonical MaxEnt versus matched reward-only training at 0.5B

**Status: frozen post-hoc paired exploratory result (2026-07-20).**

This record freezes the five-pass sampled endpoints for paired training seeds
43--45. The reward-only canonical Dr.GRPO controls were specified after the
MaxEnt treatment outcomes were visible, so the comparisons are post-hoc and
exploratory. The controls were frozen before their own outcomes. The JSON
companion records source paths and SHA-256 hashes, reward-only seed endpoints,
every seed-paired difference for all five sampled metrics, and the mapping from
paper method names to source-arm names.

The source arms are `grpo` (reward-only), `maxent` (fixed),
`maxent_control` (proportional), and `maxent_dual` (dual).

## Endpoint means and paired differences

Parentheses contain the paired mean difference from reward-only canonical
Dr.GRPO. Probability metrics are proportions; `distinct@8` is a mode count.

| Domain | Method | pass@8 | mean@8 | coverage@8 | distinct@8 | pass@1 |
|---|---|---:|---:|---:|---:|---:|
| Graph coloring | reward-only | .344 | .344 | .056 | .344 | .344 |
| | fixed | .913 (+.569) | .370 (+.026) | .360 (+.304) | 2.177 (+1.833) | .465 (+.122) |
| | proportional | .892 (+.549) | .357 (+.013) | .338 (+.282) | 2.042 (+1.698) | .361 (+.017) |
| | dual | .899 (+.556) | .369 (+.026) | .352 (+.296) | 2.132 (+1.788) | .497 (+.153) |
| Countdown | reward-only | .503 | .456 | .117 | .503 | .466 |
| | fixed | .609 (+.107) | .348 (-.108) | .255 (+.138) | 1.120 (+.617) | .500 (+.034) |
| | proportional | .651 (+.148) | .440 (-.017) | .287 (+.170) | 1.263 (+.760) | .552 (+.086) |
| | dual | .651 (+.148) | .442 (-.014) | .267 (+.150) | 1.156 (+.654) | .539 (+.073) |

## Interpretation boundary

- Graph-coloring `pass@8`, coverage, and distinct-mode gains are positive for
  every paired seed under all three MaxEnt rules.
- Countdown proportional and dual gains in `pass@8`, coverage, distinct modes,
  and greedy `pass@1` are positive for every paired seed.
- Three seeds support paired effect sizes and directional replication, not
  precise population inference.
- Exact held-out full-support terminal audits remain separate from these
  sampled endpoints.
