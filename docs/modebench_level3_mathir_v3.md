# MathIR v3: fixed numeric binding laws

This revision tests whether bounding the right-hand constant in the original
one-sided equation reduces concentrated success while retaining pass@8.
It changes the numeric proposal distribution. It does not retry the failed
v2 recipe's mixture weights, row ranking, or hash seed. No capability match is
claimed before fresh development evaluation and independent confirmation.

The complete v2 d0 development pool had 128 prompts and four independent n=8
draws per prompt: pass@1 was 0.101074 and pass@8 was 0.292969. A broad diagnostic
stratum with `|c| <= 12` contained 44 prompts, with rates 0.058239 and 0.295455;
the remaining 84 prompts had rates 0.123512 and 0.291667. The threshold is the
original offset bound. All ten prompts with pass@1 greater than 0.5 belonged
to the larger-constant stratum. Other coefficient-sign and magnitude strata
were also inspected, so this is an exploratory hypothesis rather than a
preregistered statistical finding.

The 44-prompt stratum's approximate normal 95% intervals across prompts were
[0.023885, 0.092592] for pass@1 and [0.162858, 0.428051] for pass@8. The
small-minus-large pass@1 difference interval was [-0.131394, 0.000848], and the
pass@8 difference interval was [-0.158439, 0.166015]. Action-position imbalance
may also explain part of the observed association. The narrower `|c| <= 6`
preset is a new structural hypothesis; it was not selected from evaluated
rows at that threshold.

| Preset | Equation and law |
| --- | --- |
| 0 | Original `a*x + b = c`; condition original bindings on `|c| <= 12` |
| 1 | Original `a*x + b = c`; condition original bindings on `|c| <= 6` |
| 2 | Original `a*x + b = c`; retain the unconditioned original binding law |
| 3 | Retain v2's `a*(x + b)/e = (c - d*x)/f` and its binding law |

For presets 0–2, independently sample a nonzero integer solution and nonzero
`a` from -9 through 9, and `b` from -12 through 12; set `c = a*solution + b`.
Condition only on the declared constant bound and ordinary semantic identity
exclusions. The respective exact identity inventories are 2,584, 1,428, and
8,100. Preset 3 retains nonzero coefficients from -29 through 29 and rejects
`a*f + d*e = 0`. There is no numeric bound expansion, topology fallback, or
quota-dependent alternate law.

All presets retain six randomly permuted actions A–F, a maximum of four
actions, the original prompt, the `linear-menu-v1` reference schema, and the
original grader and canonicalizer. Menu permutations use a separate random
stream from bindings and their rejection loop. Seeds depend only on the
profile, caller seed, preset, support cell 5, row index, and stream label.
Increasing a quota extends the existing prefix. Menus never depend on model
outcomes, and changing exclusions cannot select a preferred menu for a row.
Original family names remain part of the identity, so the new profile cannot
bypass earlier dataset or candidate-pool exclusions.

Each preset has exactly five canonical symbolic state paths. The verifier
normalizes formal expressions without substituting numeric binding values;
therefore zero offsets or numeric coincidences cannot merge symbolic paths.
Every one-sided multiplicative argument is nonzero `a`. The rational preset's
conditions also keep `a/e` and `a/e + d/f` nonzero. Exhaustive template
enumeration and boundary checks establish the support count; original-grader
witnesses check concrete generated rows. No fresh symbols or additional action
IDs are introduced: they would violate the unchanged schema or guided grammar.

Fresh 128-row development pools are published under
`var/data/modebench_level3_calibration_v7/pools/mathir` using
`SEEDS['mathir'] + 600000 + 1000 * preset`. Their metadata hashes the new v3
module. Before publication, the audit prepares all four pools and checks the
narrow preset's remaining inventory. After historical and all new candidate
pool exclusions, 1,142 unused `|c| <= 6` identities remain, exceeding the
required 640-row capacity.

The CPU audit at
`var/artifacts/modebench_level3_mathir_v3/structural_audit.json` records the
published pools' original-grader witnesses, exact support histograms, context
budgets, hashes, and historical/pool exclusions. It also builds fresh
384/128/128 capacity rows for each of all four presets, preserving the exact
Level 2 train/dev/eval support histograms. Capacity rows are disjoint within
each preset and from every historical or published candidate pool. The audit
reports cross-preset overlap separately; independent capacity demonstrations
do not consume one another's inventory. These rows stay in memory and are not
finalized training or confirmation sets.

Old sources and pools are hash-checked before and after. Root generator
routing, evaluator code, and runtime source are unchanged by this revision.
The frozen interface remains `level2_qwen_r5`, `hybrid_solver_v4`, and
`domain_legal_v1`, with 192 output tokens and a 2,048-token context. No GPU
inference, fitting, or confirmation scoring occurs in this script.

The completed initial audit preserves its exact source snapshot as
`var/artifacts/modebench_level3_mathir_v3/audit_source_initial.py`. A subsequent
audit-helper review added explicit within-pool uniqueness verification when
reloading published pools and extended protected hashes to the imported
original prompt/sampler/spec modules. The targeted regression and the
`audit_reload_supplement.json` receipt check these additions on all four
frozen pools, including rejection of an intentionally duplicated row. The
generator, pool files, and certificates remain unchanged.

```bash
var/seed_paper_eval/paper310/bin/python -m pytest -q tests/test_modebench_level3_mathir_v3.py tests/test_modebench_level3_mathir_v2.py
var/seed_paper_eval/paper310/bin/python ops/exp_scaling/audit_modebench_level3_mathir_v3.py --output /tmp/mathir_v3_structural_recheck.json
```

Initial publication additionally uses `--materialize-development`. Existing
pools and audit artifacts are never overwritten. Later candidate pools can
change the exclusion inventory reported by a recheck.
