# Completed E121 result integration — September 9, 2026

All five registered Qwen2.5-0.5B-Instruct ReplayDr.GRPO Graph runs (seeds 43–47,
Slurm 31040762–31040766) completed. This result is a descriptive fixed-bank
mechanism audit, not a new efficacy comparison or an exact mode-survival guarantee.

The independent integrity audit passes all five runs. It matches every observed
prompt/outcome fingerprint to the independent frozen-bank population counts and
checks complete round-robin cycles, unchanged membership and fresh counts,
finite aligned scores, and at least two observations per identity. There are
1,575 seed–prompt pairs, 3,406 frozen identities, and 29,041 identity observations.
Every identity has 8–9 scheduled visits; all have finite mean and sequence scores
at every visit. There are 2,689 actual post-freeze updates 384–3072 per seed.
The step 3073 terminal logging record carries forward update 3072 telemetry;
it is checked for equality and excluded, rather than counted as another visit.

## Registered descriptive result

All five per-seed median changes in mean token log probability are positive
(+0.024 to +0.043 nat/token). The pooled median is +0.038373, with hierarchical
95% interval [+0.025363,+0.047822]. The pooled 10th percentile is −0.256294
[−0.305284,−0.218357], and 89/3,406 identities (2.6130%) lose more than 0.5 nat/token
[1.5352%,3.7730%]. The worst final and intermediate change is −1.328338 nat/token.
The per-seed threshold percentages range 1.42–3.58%.

Sequence log scores are separately reported because exemplar lengths vary
(4–100 tokens). The pooled sequence median change is +0.154280 nat/sequence
[+0.104235,+0.198450]. The 10th percentile is −1.046803, 21.9906% decline by more
than 0.5 nat/sequence, and the worst change is −5.313353 nat/sequence.

All identities and their complete scheduled score trajectories are included in
[the frozen result JSON](../../results/e121_fixed_bank_survival.json). All five
seed summaries and the pooled ECDF are shown in both paper appendices.

## Analysis implementation

The builder uses 10,000 bootstrap draws with NumPy default_rng(121), preserving
all identities of each sampled prompt cluster. For each draw it first resamples
prompts separately within each original seed, then samples five seeds with
replacement; a repeated seed reuses that seed's within-draw prompt sample.
Pooled summaries give each identity equal weight. Quantiles use linear
interpolation and intervals use 2.5th/97.5th percentiles. The preregistration fixes
the hierarchy, draw count, and RNG seed; the exact pooling, quantile convention,
and displayed interval estimands are transparent reporting choices.

Reproduce from this checkout:

```sh
python ops/exp_scaling/audit_e121_fixed_bank_survival.py
var/seed_paper_eval/paper310/bin/python ops/exp_scaling/build_paper_e121_survival.py
make -C paper
make -C paper/mathai2026 bundle
```

The builder gates on the independent integrity receipt and verifies SHA256 of
every metrics file, completion receipt, ledger, and registration before reading
the selected observations. The result JSON also records its builder hash and
NumPy version. Publication sources and asset hashes are bound by the workshop
snapshot and its isolated build receipt.

## Scope and provenance

Finite teacher-forced scores establish complete score monitoring. They do not
establish a material probability of sampling a canonical mode. Scores refer to
one fixed surface exemplar per retained canonical identity. The modest positive
medians coexist with negative tails; this is not uniform likelihood improvement.
There is no matched frozen-bank no-replay control, so these changes do not
identify a causal effect of replay. The result does not validate the theorem's
numerical lower bound or an exact neural-network guarantee. Intermediate drops
refer to scheduled replay visits, not every intervening update.

The September 8 scheduling amendment removed E120 dependencies and moved the
jobs to non-PVL node204 with user authorization; it did not alter the scientific
recipe. All five completed in one continuous attempt. Resume checkpoints were
automatically pruned on success. The independent counts and complete scheduling
cycles establish coverage despite that cleanup. Fingerprints were downcast in
routine logging; matching full-bank cardinalities detects any within-run identity
collision relevant to this population. See the [independent audit](../e121_20260909/integrity.md)
for exact provenance and implementation limitations.

## Independent verification

The separate statistics review recomputed every per-seed and pooled endpoint
from the raw logs with standard-library statistics (agreement within 1e-12).
A multiplicity-weight bootstrap and weighted order-statistic implementation
independently reproduced all 10,000-draw interval bounds exactly. The receipt is
[the independent statistics audit](../e121_20260909/independent_statistics.json).
The 14 coverage-auditor tests and 13 statistical-analysis tests passed.
