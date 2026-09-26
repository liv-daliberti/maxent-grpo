# E112-R1 two-scale exploratory public disclosure

Recorded: 2026-08-28, after the author explicitly requested that the completed
Qwen2.5-0.5B and Falcon3-1B E112-R1 panels be included in the paper without
waiting for the incomplete Qwen2.5-3B panel.

## Status and deviation

This is an author-requested, post-look public disclosure, not a prospective
confirmatory analysis. The registered E112 analysis required all 75 cells.
Private exploratory looks at 14, 33, and 50 terminal cells broke continuous
confirmatory outcome blindness. The 50-cell membership was nevertheless frozen
from completion markers before its newly available endpoints were read.

The public result must disclose those repeated looks, the incomplete third
scale, and the historical-comparator estimand. It must not be described as the
registered 75-cell confirmatory result, a successful all-three-scales result,
or an isolated effect of the v7 semantic term.

## Immutable membership

Use exactly
`var/artifacts/e112r1_private_interim_unblinding_freeze_20260826_50.json`:

- 25 Qwen2.5-0.5B cells: five domains by seeds 43--47;
- 25 Falcon3-1B cells: five domains by seeds 55--59; and
- zero Qwen2.5-3B cells.

No endpoint that became terminal after the 50-cell freeze may enter this
analysis. Qwen2.5-3B remains an incomplete registered scale and receives no
mean, interval, scale decision, or efficacy interpretation here.

## Estimand and summaries

Pair each frozen E112-R1 cell to the ReplayDr.GRPO comparator bound in the
released ledger. The estimand is the bundled E112-R1 request path and source
snapshot minus its registered historical ReplayDr.GRPO comparator. Because the
request/source provenance differs, it is not an isolated semantic-v7 effect.

Use the registered fixed 128-prompt evaluation bank, four common sampled-K
draws, pass 8, and the 17-checkpoint grid from update 0 through 3,072. For each
complete model/domain family, report all five paired training-seed effects, the
unweighted mean, and the two-sided 95% Student-t interval (df=4) for:

1. terminal sampled pass@8;
2. terminal correctness-adjusted breadth, `distinct@8 - pass@8`;
3. normalized trajectory AUC for sampled pass@8; and
4. normalized trajectory AUC for correctness-adjusted breadth.

Keep evaluation-draw Monte Carlo variation separate from training-seed
variation in the machine-readable result. Do not pool domains or models.

For descriptive continuity with the registered rule, report each completed
scale's unweighted 25-cell mean terminal pass@8 effect, mean terminal adjusted-
breadth effect, and number of positive domain-family breadth means. A completed
scale satisfies that descriptive rule when mean breadth is positive and mean
pass@8 is non-negative. Do not evaluate the registered 8-of-15 general
criterion or the all-three-scales criterion.

## Claim boundary

The paper may report these two complete panels as exploratory evidence. It
must state that:

- the disclosure followed repeated author-requested looks;
- Qwen2.5-3B is incomplete and excluded;
- the contrast is bundled and historical rather than component-isolated;
- family intervals are descriptive paired uncertainty, not multiplicity-
  adjusted confirmatory tests; and
- no E112 scheduler, training, comparator, or model-selection decision may be
  based on these outcomes.
