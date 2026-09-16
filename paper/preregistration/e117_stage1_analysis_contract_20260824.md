# E117 successor Stage-1 analysis contract

Frozen: 2026-08-24, while E117-R1 has no training outcome and before any
Stage-1 seed, evaluation split, job, or endpoint exists.

Status: analysis-only development contract. This does not authorize a Stage-1
launch. Submission remains conditional on a passing E117 mechanism audit and a
separate frozen execution protocol. PointMaze remains excluded.

## Statistical object

The primitive sampled endpoint is the paired vector
`(pass@8, raw distinct correct modes@8)`. Correctness-adjusted breadth is
derived row by row as `raw distinct@8 - pass@8`; it is not a third independent
coordinate. This keeps an accuracy rescue visible even when it reduces excess
multiplicity per solved prompt.

The exact C/P/F grid must contain three new paired training seeds, at least 16
registered common-random-number K=8 evaluation draws, all registered fixed
checkpoints from step zero through the eight-pass terminal horizon, and all
four E117 sentinels. Missing, duplicate, extra, non-finite, or out-of-range
rows fail the build. Step-zero primitive endpoints must agree exactly across
C/P/F within sentinel, seed, and draw.

For each seed and evaluation draw, compute `P-C` and `F-P` before averaging.
Compute terminal effects directly and normalized AUC effects by trapezoidal
integration over the complete fixed grid divided by its full horizon. No
checkpoint, draw, K, temperature, domain, or seed selection is allowed.

## Two uncertainty axes

For every endpoint, contrast, sentinel, and terminal/AUC summary, retain all
paired seed-by-draw effects and report:

- the effect estimate, averaging registered draws within seed and then the
  three paired training seeds;
- training-seed SE across the three draw-averaged paired seed effects; and
- evaluation Monte Carlo SE across the common-draw mean paired effects after
  averaging the three training seeds.

These two SEs answer different questions and are never pooled into one sample
size, confidence interval, or p-value. Per-seed evaluation MC SEs are retained.
Three training seeds make this a screen, not population-level confirmation.

## Frozen development gates

For a component within a sentinel, terminal and normalized-AUC adjusted breadth
must each be strictly greater than +0.05, exceed two corresponding evaluation
MC SEs, and be positive in at least two of three paired training seeds. The
candidate arm must have terminal pass@8 versus C at least -0.03 on the
three-seed mean and at least -0.10 in every paired seed.

The proposal component compares P with C and applies correctness safety to
P-C. The semantic component compares F with P but applies correctness safety
to F-C. A broad development successor requires the same component to be
actionable in at least three of four sentinels. An actionable Countdown-only
result is labeled a domain-specific candidate, with Graph retained as its
registered negative boundary. All other patterns do not advance.

The executable contract is
`ops/exp_scaling/e117_stage1_statistics.py`. Its result records the primitive
coordinates, derived algebra, both uncertainty axes, raw paired effects, each
elementary gate, and the scope decision. A future launcher must freeze new
seeds, evaluation draws, split digest, checkpoint grid, placement blocks,
source snapshot, and complete C/P/F exports before submission.

