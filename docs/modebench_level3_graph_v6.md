# Graph v6: shared visible color development proposals

This revision tests a structural hypothesis: repeated constraints against one
known color may let Qwen 3B answer more consistently, increasing pass@1 relative
to pass@8. Mixing these proposals with larger coupled graphs may supply a useful
calibration range. No difficulty match is claimed; frozen development fitting
and independent confirmation remain required under the unchanged protocol.

Graph v5 remains preserved. Its full-pool pass@1/pass@8 rates for difficulties
0–3 were approximately 0.2473/0.6211, 0.2026/0.5723, 0.1157/0.4082, and
0.0891/0.3398, versus the Level 1 Qwen 0.5B baseline 0.1909/0.3828. The corrected
fit forecast was approximately 0.154/0.457, but its selected development set was
0.166/0.482 and failed the pass@8 gate. This revision changes the proposal law;
it does not retry the failed recipe's weights, row order, or hash seed.

| Difficulty | Vertices | Hidden vertices | Proposal |
| --- | --- | --- | --- |
| 0 | 5 | 2 for support 4, 6, 9; otherwise 3 | Shared visible color, with explicit support 5 and 9 exceptions |
| 1 | 6 | 2 for support 4, 6, 9; otherwise 3 | Shared visible color, with explicit support 5 and 9 exceptions |
| 2 | 6 | 3 | Exact v5 coupled proposal law |
| 3 | 7 | 4 | Exact v5 coupled proposal law |

For the shared-color law, uniformly sample the hidden vertex labels, choose one
visible color uniformly from 1, 2, 3, and give every visible vertex that color.
There are no edges within the hidden set or within the visible set. Each
hidden-visible edge is independently present with probability 0.5. Condition
only on the required exact support and semantic identity exclusions. The
sampler never prefers color 1 or balances accepted rows by color.

Support 5 uses the exact v5 coupled proposal: independent hidden choices have
sizes in {1, 2, 3}, whose products cannot equal 5. Support 9 uses the exact v5
independent proposal with independently sampled visible colors. With two hidden
vertices, monochrome support 9 would require the empty graph, leaving only 30
labeled n=5 identities or 45 labeled n=6 identities before exclusions. The
explicit alternative avoids that capacity bottleneck.

The generator retains fixed per-row, per-support-cell random streams. Increasing
quotas extends the existing cell prefixes; it does not change proposal laws or
select a new ranking. There is no topology fallback, quota rank cutoff, or
selection using model outcomes. The original graph prompt, verifier, and full
coloring canonicalization remain unchanged.

For n=5, the complete shared-color identity capacities before exclusions are
1,470 at support 4; 420 at support 6; 810 at support 8; 810 at support 12; and
270 at support 18. For n=6 they are 10,125; 1,350; 20,580; 8,820; and 1,260,
respectively. The audit exhaustively enumerates these capacities, reports the
remaining counts after all historical and candidate-pool exclusions, and checks
equal proposal capacity for each visible color before exclusions.

The four fresh 128-row development pools are under
`var/data/modebench_level3_calibration_v6/pools/graph_coloring`. Their fixed seeds
are 6,827,100; 6,828,100; 6,829,100; and 6,830,100, following the rule
`SEEDS[graph_coloring] + 500000 + 1000 * difficulty`. A scoped override in the
new CPU audit script invokes the v6 builder; the existing materializer routing
is unchanged. These pools preserve the exact Level 2 development histogram.
Support 5 and 18 do not occur in that reference histogram, so the separate
384/128/128 capacity checks and unit tests exercise those cells explicitly.

The audit at
`var/artifacts/modebench_level3_graph_v6/structural_audit.json` records hashes,
all exclusions, exact capacities, original-grader witness counts and canonical
support, per-cell quota-prefix checks, and 384/128/128 generated rows for each
of all four presets. These in-memory capacity rows are mutually disjoint across
presets and splits and also exclude every existing development candidate pool,
including v6. They are capacity evidence, not finalized or confirmation data.
Old sources, old pools, and the root routing are hash-checked before and after.
No GPU evaluation, fitting, or confirmation is performed by this script.

Reproduce the CPU checks with:

```bash
var/seed_paper_eval/paper310/bin/python -m pytest -q tests/test_modebench_level3_graph_v6.py tests/test_modebench_level3_graph_v5.py
var/seed_paper_eval/paper310/bin/python ops/exp_scaling/audit_modebench_level3_graph_v6.py --output /tmp/graph_v6_structural_recheck.json
```

The initial run additionally used `--materialize-development`. Existing pools
and audit artifacts are never overwritten. Rerunning the audit after more graph
pools are added may change the exclusion set and available capacity.
