# Python v5: fixed parity and factor-7 development laws

The corrected four-draw development fit of Python v4 failed the unchanged pass1
gates. The full-pool optimum was `[0, 0, .95, .05]`; its sole selected development
set scored pass1 0.31396484375 and pass8 0.81640625 against baseline
0.237060546875 and 0.845703125. The failed recipe remains immutable at
`var/artifacts/modebench_level3_v2/recipes/python_factors.json`.

V5 registers four fresh, fixed proposal laws. Every row retains the original
four-input prompt, legal lambda syntax, external Python verifier, canonical
return-vector identity, and exact product of proper-divisor counts.

| Tier | Eligible case sets |
| --- | --- |
| 0 | Four distinct values in 48–192, each with at least two proper divisors and smallest proper divisor at most 5. This preserves the previous d2 law as an explicit anchor. |
| 1 | The same catalogue, conditioned on exactly two odd and two even cases. |
| 2 | The same two-odd/two-even composition with the fixed wider catalogue 48–384. |
| 3 | Exactly one value in 4–1000 whose smallest proper divisor is 7, plus three distinct values in 48–384 whose smallest proper divisor is at most 5. Every value has at least two proper divisors. |

Each preset is also conditioned on the requested exact support. The generator
uses integer tickets proportional to each joint divisor-count/composition
profile's combinatorial capacity, then samples uniformly within that profile.
Every eligible unordered four-case set therefore has equal proposal mass.
Separate per-support RNG streams and identity rejection preserve quota prefixes.
The laws have no outcome-dependent row selection, alternate seeds, or fallback.

The complete corrected development diagnostic is
`var/artifacts/modebench_level3_v2/python_v5/development_diagnostic.json`.
In the small-window d2 pool, the exactly-two-odd stratum had 44 rows with
pass1/pass8 0.25923295454545453 / 0.8068181818181818. On the 19 observed support
cells, covering 109 of the 128 target rows, support reweighting gave
0.249689 / 0.778937, compared with 0.308773 / 0.811927 for all d2 rows on those
same cells. The 13 missing support cells prevent treating this as a forecast for
the full new pool. The large-window d3 two-odd stratum did not show the same
pass1 benefit. GCD grouping did not support an additional common-factor rule.

The exact `2/3/5` conditional chain accounted for 991 of 1,291 successful d2
attempts. The factor-7 component tests whether defeating that repeated shortcut
lowers pass1; a broader finite conditional chain or a case-specific mapping can
still solve every instance. Calls and loops remain forbidden. These observations
motivate prospective development laws; they do not establish that v5 will pass.
No legacy confirmation outcomes enter this revision.

The immutable prospective registration is
`var/artifacts/modebench_level3_v2/python_v5/candidate_protocol.json`, SHA256
`a41f4363ca7d818e1575fb74cc5f3feaceb4c8eca9dfaa3a0d69b58f21d5fa75`.
It pins 626 source/evidence files, including every file in the inherited 503-file
Graph-v7 seal. Development seeds are `8337100 + 1000*tier`; evaluator draw labels
remain `6328000` through `6328003`.

The standalone materializer authenticates all historical Python datasets and
all previous candidate pools. It registers before construction and publishes
only the four new 128-row development pools under
`var/data/modebench_level3_calibration_python_v5/pools/python_factors` after every
capacity check passes. Full 384/128/128 constructions for each preset use
`8437100 + 1000*tier + 10000*split_index`, are globally disjoint from history,
all candidate pools, and one another, and retain two original external grader
witnesses per row. Only their hashes and checks are kept as capacity evidence;
they are not finalized training or confirmation data.

The resulting certificate is
`var/artifacts/modebench_level3_v2/python_v5/structural_audit.json`.
CPU tests cover exact composition, integer-ticket uniformity, original witnesses,
quota prefixes, exclusions, bounded exhaustion, immutable registration order,
and mutation during construction:

```bash
PYTHONDONTWRITEBYTECODE=1 var/seed_paper_eval/paper310/bin/python -m pytest -q \
  tests/test_modebench_level3_python_v5.py
```

Existing registration and pool paths are never overwritten. Fitting, model
sampling, finalization, and confirmation use separately guarded workflows.
