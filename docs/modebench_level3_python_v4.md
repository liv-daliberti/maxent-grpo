# Python v4: even anchors with fixed small-factor components

This revision tests whether all-even inputs provide the high pass@8 needed for
calibration while preserving the original PythonFactors problem and exact
canonical support. Every even input admits constant output `2`; the model's
probability of finding that program remains an empirical question. No difficulty
match is claimed before the unchanged four-draw development fitting and fresh
held-out confirmation.

The four frozen presets each sample **four distinct cases uniformly conditional
on exact support**, using integer profile-capacity tickets and uniform samples
within each divisor-count profile:

| Difficulty | Exact case catalog |
| --- | --- |
| 0 | Even integers in [48,192], union {4,16,36} |
| 1 | Even integers in [512,1000], union {4,16,36,64} |
| 2 | Original Python v3 difficulty 0: [48,192], at least two proper divisors, smallest proper divisor ≤5 |
| 3 | Original Python v3 difficulty 3: [512,1000], at least two proper divisors, smallest proper divisor ≤5 |

The even catalogs admit input 4, which has exactly one proper divisor. The
original verifier permits this: it constrains the total product of divisor
counts, not the count for each input. All cases remain within 4–1000. The original
prompt, four-input shape, verifier, canonical output vectors, and format budget
remain unchanged. The two existing certified programs run through the original
external Python worker for every generated row.

The low core values provide divisor-count classes needed for some support cells.
Under the raw support-conditioned law and exact development histogram, 77.33%
of difficulty-0 rows and 84.16% of difficulty-1 rows contain no low core value.
Expected low-core input counts are 0.236 and 0.173 per row. These are analytic
proposal-law quantities before identity exclusions. The two even catalogs can
intersect only on the four-case set {4,16,36,64}, whose support is 105; support
105 is absent from every reference split, so the two anchors are disjoint on
all required cells.

The hypothesis comes from complete previous development pools, analyzed using
fixed broad parity and support features. Across the four preceding small-factor
windows, the all-even subgroup had 38 prompts with pass@1/pass@8 of
0.41283/0.96711; other parity patterns had 474 prompts at 0.32496/0.76477.
Adjusting jointly for pool and support in the shared cells retained the pass@8
association (0.97299 versus 0.81002). Per-pool all-even results were:

| Prior window | Prompts | pass@1 | pass@8 |
| --- | ---: | ---: | ---: |
| [48,192] | 10 | 0.290625 | 1.000000 |
| [48,384] | 9 | 0.343750 | 0.861111 |
| [256,1000] | 11 | 0.613636 | 1.000000 |
| [512,1000] | 8 | 0.367188 | 1.000000 |

These small structural groups motivate fresh catalogs; their outcomes do not
select individual rows. The earlier 11-row minimum-case≥48 subgroup was also
exploratory: support composition explained part of its apparent pass@8 increase,
and none of the fresh broader-window rows directly retested all four cases in
[48,96]. An all-odd alternative was mathematically feasible but had poorer
pass@8 in the completed broad feature diagnostic. No alternate weights, row
ordering, hash seeds, or individual model outcomes enter this generator.

The standalone prior-evidence record, including receipt/source hashes, complete
support-cell tables, and analytic core frequencies, is
`var/artifacts/modebench_level3_python_v4/development_rationale.json`.

Four fresh 128-row pools were published under
`var/data/modebench_level3_calibration_v7/pools/python_factors`. The fixed seeds
are 6,937,100; 6,938,100; 6,939,100; and 6,940,100, following
`SEEDS[python_factors] + 600000 + 1000 * difficulty`. A scoped override invokes
the new builder; existing materializer routing is unchanged. The original v3
source and old candidate pools remain preserved.

The CPU audit at
`var/artifacts/modebench_level3_python_v4/capacity_audit.json` records all 60
required support-cell capacities before and after the new pilots, exclusions,
source/pool hashes, seeds, and exact support histograms. All four presets passed
384/128/128 builds after excluding all historical and candidate identities,
including the new pilots. The 2,560 capacity rows are mutually disjoint across
presets and splits and preserve per-cell quota prefixes. Their 5,120 witness
validations called the original external worker; including the 512 pilot rows,
6,144 witness calls succeeded. These in-memory split builds are capacity evidence,
not finalized training or confirmation data.

Validation: 32 tests passed across the new v4 and preserved v3 suites. The v4
tests check exact integer-ticket uniformity, frozen catalogs and v3 parity,
quota/multiplier prefixes, input 4, original external witnesses, exact capacities,
and bounded failure without a fallback. No GPU evaluation or fitting was run.

Reproduce CPU checks with:

```bash
var/seed_paper_eval/paper310/bin/python -m pytest -q tests/test_modebench_level3_python_v4.py tests/test_modebench_level3_python_v3.py
var/seed_paper_eval/paper310/bin/python ops/exp_scaling/audit_modebench_level3_python_v4.py --output /tmp/python_v4_capacity_recheck.json
```

The initial audit additionally used `--materialize-development`. Existing pools
and artifacts are never overwritten. Later candidate pools can change the
exclusion set and the resulting available-capacity counts on a recheck.
