# Level 4 (Qwen 7B) and Level 5 (Qwen 14B)

Levels 3, 4, and 5 share one target convention: match the measured initial
success rates of Qwen2.5-0.5B-Instruct on Level 1. Level 4 uses frozen
Qwen2.5-7B-Instruct and Level 5 uses frozen Qwen2.5-14B-Instruct. Each scale
requires separate calibration and fresh, disjoint problems. Qwen 0.5B on
Level 3 is not the target.

**Final status (2026-09-15): Level 5 is admitted; Level 4 is closed at four of
five domains and is not admitted.** Both levels bound their five sources in
`var/data/modebench_scale_release_v2` before any held-out outcome existed, and
the single confirmation array `31290252` then produced all ten receipts. Nine
domains passed; Level 4 MathIR failed at 1.07 of tolerance on pass@1. Level 5
carries `admission.json` with `difficulty_matched: true`. Level 4 was closed
rather than rebuilt, by the user's decision, with its cause established and
three independent mechanisms recorded below that prevent an in-place fix. Both
levels' ten datasets are frozen at 384/128/128 and usable from their source
roots; admission is a statement about difficulty matching, not about whether the
rows exist. The registered sampling seeds, difficulty gates, and neutral Level 3
default are unchanged throughout.

Ten-domain outcome, superseding every build-state table below it:

| Level | Domain | Source | Development fit | Frozen data | Held-out confirmation |
| --- | --- | --- | --- | --- | --- |
| 4 | Countdown | original | Passed | Complete | Passed (0.20 of tolerance) |
| 4 | MathIR | original | Passed | Complete | **Failed (1.07 of tolerance)** |
| 4 | Graph Coloring | r2 | Passed | Complete | Passed (0.34 of tolerance) |
| 4 | Pantry | r2 | Passed | Complete | Passed (0.81 of tolerance) |
| 4 | Python Factors | r5 | Passed | Complete | Passed (0.32 of tolerance) |
| 5 | Countdown | original | Passed | Complete | Passed (0.43 of tolerance) |
| 5 | MathIR | r2 | Passed | Complete | Passed (0.92 of tolerance) |
| 5 | Graph Coloring | r5 | Passed | Complete | Passed (0.56 of tolerance) |
| 5 | Pantry | r4 | Passed | Complete | Passed (0.49 of tolerance) |
| 5 | Python Factors | r2 | Passed | Complete | Passed (0.71 of tolerance) |

Margins are the worst of the two gates as a fraction of its tolerance. Level 4
therefore has five built datasets and four confirmed domains; Level 5 has five
of each and is the admitted level.

The Level 4 Python fourth revision failed its fixed fit; the fifth passed all
six gates at 17:42 UTC on September 14 with weights `[0, 17, 3, 0]` and selected
development pass@1/pass@8 of 0.209716796875/0.7578125 against targets
0.2109375/0.76953125. Its freeze certificate was published at 18:34 UTC and
passed independent review: 384 train, 128 dev, and 128 test rows. Both failed
recipes are preserved and are not retried.

**Level 4 release-path blocker.** The v1 composite controller cannot carry
Level 4 to confirmation, for two independent reasons. Its Level 4 source
manifest in `var/data/modebench_scale_release_v1` names Python revision 2, the
failed original binding, and that manifest's digest is pinned by four sealed
records; rewriting it would wedge the controller with no rollback. Separately,
`_sweep` builds one confirmation stage from all ten level/domain sources and
returns `needs_new_development_revision` while any revised fit fails, so Level 4
would wait on Level 5 Graph and Pantry even with a correct manifest. A rebind
adapter and its tests were written, checked, and then
[refused unpublished](../artifacts/modebench_scale_level4_source_rebind_refused_20260914.json)
at 00:27 UTC on September 15; nothing was written. The registered alternative is
the Level 4-first family, which binds all five Level 4 sources into a fresh
`modebench_scale_release_v2` root and never touches the v1 seal lattice. Those
helpers currently pin Python revision 4 and need an r5 successor before they can
run. The final combined `execution_provenance.json` and the campaign admission
still require both levels; a Level 4 level-admission does not.

**Level 5 next step.** Both failed domains were diagnosed at 22:07 UTC on
September 14. Graph r2 brackets both metrics marginally but admits no joint
grid mixture: tiers 0-2 cluster near pass@1 0.35-0.40 and tier 3 falls to 0.110,
and the only real difficulty step, hidden vertices 3 to 4, produces dead rows.
Graph r3 destroyed that gradient outright, with three measured inversions, so
the screening pilot branches from r2. Pantry r2 is fully monotone with no
inversions and simply does not reach far enough: its hardest tier is pass@1
0.148 / pass@8 0.336 against targets 0.059 / 0.260. An initial claim that
Pantry could be extended by activating the ladder its law already declares was
[withdrawn](../artifacts/modebench_scale_level5_pantry_knob_correction_20260914.json):
`difficulty = 0` is load-bearing, pinning `step_g` to 25, the only step dividing
the fixed availabilities, and generation, the declared profile and the
structural audit are welded together. Pantry therefore uses a new sealed
candidate law, `ops/exp_scaling/modebench_scale_pantry_r3_candidates.py`.
Screening pools for both domains were generated on CPU into
`var/data/NONPRODUCTION_modebench_scale_level5_pilot_20260914`, 48 rows per tier
for Graph and 59 for Pantry, with 428 burned identities that must be excluded
before any production generation. Their 14B scoring array `31284470` was
submitted at 01:16 UTC on September 15.

**The Graph screening pilot failed and its hypothesis is rejected.** Raising
vertices at hidden vertices 3 is not the missing knob: the measured ladder is
not even monotone, with 8 vertices easier than 7 on both metrics, and 0 of 1,771
grid-20 mixtures satisfy both tolerances at a best normalised error of 1.077,
against 1.182 for r2 and 2.902 for r3. Every tier is again more bimodal than the
target, with heterogeneity gaps of 0.346 to 0.495 against the target's 0.282. The
[outcome record](../artifacts/modebench_scale_level5_graph_pilot_outcome_20260915.json)
also reports that the pilot rows carry a `scale_candidate_profile` one rung off
from how they were built: the generator patches the module global
`GRAPH_PRESETS`, but `PROFILES` is computed from it at import time. The rows and
the measurement stand, verified against the graphs in the frozen pools, but the
recorded profile must not be carried into a production registration.

That record's per-cell assignment search was then
[corrected](../artifacts/modebench_scale_level5_graph_per_cell_correction_20260915.json).
The pilot repeated one configuration r2 had already measured at production size,
6 vertices with 3 hidden in random order, and treated as a replication it shows
48-row per-cell estimates are unusable: support 9 moved from 0.521 to 0.031 and
support 12 from 0.481 to 0.707, while the pool means agreed to within 0.05. On
the pooled r2-plus-pilot evidence, seven distinct configurations at up to 146
rows per tier, per-cell grading reaches a best normalised error of 0.706, a real
improvement over every tier-shaped mixture but still the best of 32,768
assignments scored on the estimates that chose it. Five-vertex configurations are
systematically less bimodal than six-to-eight-vertex ones at the same cell, which
is the direction the target needs; larger graphs buy lower pass@1 mainly by
creating dead rows. The next Graph step is therefore a candidate law that grades
structural difficulty per answer-mode-count cell, measured at production pool size
on the cells that carry weight, rather than a third tier-shaped revision or
another 48-row screen.

That law has a
[necessary condition](../artifacts/modebench_scale_level5_graph_r4_design_criterion_20260915.json)
to satisfy, checkable on a small pool before any ladder is generated. Pass@8
cannot exceed the fraction of prompts that are not dead, so a configuration used
near the target pass@1 must keep its dead fraction well below 0.4375. Every knob
measured so far fails it in the same way: each lowers pass@1 by killing prompts
rather than by making them intermediate. Among cells of at least eight rows whose
pass@1 falls in [0.10, 0.30], dead fractions run 0.22 to 0.67 and the fraction of
prompts with per-sample success between 0.05 and 0.5 never exceeds 0.35. Support 4
is the cell to attack: it carries 54 of 128 dev rows, and across all seven measured
configurations its dead fraction never falls below 0.21, reached only at pass@1
0.290. Low-bimodality regimes do exist — six vertices with three hidden at support 5
gives pass@1 0.102 with pass@8 0.500, a gap of 0.075 — but they sit in cells with
almost no dev weight.

**The Pantry screening pilot reached the target region.** Its measured ladder is
pass@1/pass@8 of 0.1896/0.3898, 0.0742/0.2161, 0.0048/0.0339 and 0.0117/0.0508,
against a target of 0.0586/0.2598. Tier 1 lands inside both tolerances on its own,
and 392 of 1,771 grid-20 mixtures satisfy both, where r2 had none because every
tier was strictly easier than the target on both metrics. The 0.115 move in pass@1
is well beyond the roughly 0.05 pool-to-pool noise, so the new candidate law — menu
size, headroom, forbidden-tag probability and interval width all graded by tier,
with `available_g` derived from `step_g` — does what activating r2's dead ladder
could not. Its rows also carry a truthful per-tier profile, unlike the Graph pilot.

Two things temper that, both in the
[outcome record](../artifacts/modebench_scale_level5_pantry_pilot_outcome_20260915.json).
The result is pool-level arithmetic: Pantry spreads 59 rows over 28 answer-mode-count
cells, a median of two rows per cell, and 14 of 28 cells are fully dead at tier 1,
while the fitter forecasts per cell and reweights by the real dev and eval histograms.
And the ladder has a defect. Pass@1 falls from 0.0742 at tier 1 to 0.0048 at tier 2,
a factor of fifteen across one rung, with the target sitting inside that gap; tiers 2
and 3 are dead at dead fractions 0.90 and 0.92 and invert against each other. Every
feasible mixture is therefore tier-1 dominated, and if tier 1 lands slightly high on
193-row production pools there is no intermediate rung to mix toward. The
recommendation is to re-cut the tier table to place rungs between the current tiers 1
and 2 before registering r3, which changes the piloted ladder and so needs a decision
first.

## 2026-09-15: four graph revisions, a governing variable, and a registered Pantry

**Corrected framing.** Earlier entries describe the target as "requiring" a
heterogeneity gap of 0.282. That is the band centre, not a requirement. At the
target pass@1 of 0.207763671875 the homogeneous pass@8 is 0.8448 and the gate
admits pass@8 in [0.4825, 0.6425], so the acceptable gap band is [0.202, 0.362].
Claims elsewhere that a 0.40 gap makes a construction unusable overstate a
0.04 shortfall.

**Graph: four revisions, one working knob.** Feasible grid-20 mixtures and best
normalised error, all on the same footing:

| design | mechanism | feasible | best error |
| --- | --- | ---: | ---: |
| r2 | one (vertices, hidden, order) preset per tier | 0 | 1.182 |
| r3 | skeleton catalog, path-multiplier tiers | 0 | 2.902 |
| pilot 1 | more vertices at hidden 3 | 0 | 1.077 |
| r4 | preset per answer-mode-count cell | 28 | 0.942 |
| r5 | fourth hidden vertex and distractor edges | 97 | 0.863 |
| r6 | forced vertices and hidden-edge bands | 0 | 2.734 |
| **r7** | **hidden count per cell by answer density** | **336** | **0.550** |

Four knobs measured inert: the path multiplier, shown-shown distractor edges
(+0.003 on pass@1), forced hidden vertices, and the hidden-edge band. The reason
is that none of them changes what governs difficulty here. Across twelve
configurations spanning three laws, pass@1 collapses onto two levels set entirely
by hidden-vertex count through the answer density, exact support over three to
the hidden power: 0.3639 with a standard deviation of 0.019 at three hidden
vertices and 0.1053 with 0.006 at four. Density is pinned by the registered
support histogram and by the hidden count being an integer, which quantizes the
reachable level and strands the 0.208 target between 0.36 and 0.11. r7 assigns
the hidden count per cell from a declared target density, which makes the
weighted mean tunable; its measured ladder is monotone at 0.2965, 0.1725, 0.1419
and 0.1283 and the density model predicted those to within 0.039.

Two predictions of mine failed and are recorded as failures. Equalising cell
density did not reduce the gap: r7 tier 1 has the lowest density spread at 2.2x
and the highest gap at 0.330, while tier 3 at 3.0x spread has 0.312. And ranking
revisions by best reachable error selected the wrong branch on 2026-09-14,
because that statistic is dominated by the mean mismatch, which mixing corrects,
and hides dispersion, which mixing cannot.

**Pantry r4 is registered and its dev pools are materialized.** The re-cut ladder
measured 0.1105/0.2906, 0.0734/0.2344, 0.0352/0.1531 and 0.0129/0.0781 against a
0.0586/0.2598 target: monotone in both metrics, bracketing on both, no dead rung,
tier 1 inside both tolerances alone, and 1,024 of 1,771 feasible mixtures at best
error 0.367 against r3's 392 at 0.533 and r2's 0 at 2.226. The user delegated the
two blocking decisions; both are recorded with their justification in the
[decision record](../artifacts/modebench_scale_level5_pantry_r4_registration_decision_20260915.json).

The Level 4 admission gate in the Sep-13 registration scripts was dropped rather
than satisfied, because it is precedence only: it validates Level 4's admission
schema and row counts and asserts that Level 4 must precede Level 5 preparation,
and checks nothing about Pantry's own science. Registration used the native
`modebench_scale_domain_revision.register` API, which the Sep-13 scripts wrap, so
no third design was introduced and no sealed record was rewritten. The measured r4
law was registered as `level5_pantry_r4` in preference to the reviewed but never
scored `modebench_scale_pantry_menu_r3_candidates`; that law, its review, its
qualification manifest and the reserved `level5_pantry_r3` name all remain
untouched and available.

**The campaign protocol no longer authenticates against the live tree.**
`src/oat_drgrpo/templates.py` has drifted from its pin. The diff is 73 insertions
and zero deletions, adding two Level 2 template functions and a neutral Level 3
Python template without modifying any existing one, so no prompt the campaign
renders has changed. This drift was already identified on 2026-09-12 and the
frozen source view addresses it: its single mapping binds the pinned file over
the live one. Registration, pool materialization and dev scoring all run inside
that view, and `templates.py` was neither reverted nor re-pinned.

**The exclusion snapshot does not see the pilots by default.**
`materialize_modebench_scale.history` globs `var/data/modebench*`, which does not
match the `NONPRODUCTION_modebench_*` pilot roots, and globs
`**/pools/<domain>/*.jsonl` while the pilots use `<root>/<domain>/pools/`. Either
alone would have hidden every burned pilot identity from production generation.
The 556 burned Pantry pilot rows were staged into
`level5_pantry_r4/burned_pilots/pantry/`, which `exclusions()` collects through
its documented hook for explicit local rows. Verified afterwards: all 556 present
in the snapshot, all eight files pinned, and 928 production rows with zero
collisions and zero cross-tier duplicates. Any future production generation for
Graph must apply the same control; four graph pilots are now burned.

Dev scoring was job `31289208`, four 232-row tiers on Qwen2.5-14B.

**Pantry r4 passed all six gates and froze.** Its measured ladder is
0.1362/0.3254, 0.0555/0.2037, 0.0471/0.1724 and 0.0155/0.0625 against a
0.0586/0.2598 target: monotone in both metrics, bracketing on both, no dead rung.
The fit selected weights `[7, 11, 1, 1]`, a genuine four-tier blend, with
forecast pass@1/pass@8 of 0.0759/0.2250 on dev and 0.0771/0.2288 on eval, and a
selected-dev pair of 0.0906/0.2578 whose pass@8 sits 0.002 from target. Its worst
gate margin is 80 percent of tolerance against Graph's 59. The dataset is frozen
at 384/128/128, `frozen_pending_heldout_confirmation`.

**Level 5 therefore has five built domains**: countdown from the campaign, mathir
r2, python_factors r2, graph_coloring r5 and pantry r4. Level 4 already had five.

**A concurrent source edit destroyed a scoring run and broke campaign
authentication.** `src/oat_drgrpo/args.py` was edited while Pantry's dev scoring
was in tier 3. The evaluator's `code_identity()` globs `src/oat_drgrpo/*.py`, so
the identity every receipt pins changed mid-run and the tier-3 receipt was
refused after all 117 of its batches had been written. Worse, the campaign
protocol pins that file at `a379e57a`, a version present in no commit, stash,
reflog or loose object, so `original.authenticate` failed for every level and
every stage. The content was recovered from the Isilon snapshot at
`.snapshot/20260915-12:00-Q4H-PROJ`, preserved read-only at
`artifacts/modebench_scale_source_drift_20260915/args.scale_pinned.py`, and the
working tree was left as the author had it.

The fix is
[a snapshot source view](run_modebench_scale_snapshot_view_20260915.py) that
supersedes the single-file frozen view. It binds `src/oat_drgrpo` as a
*directory*, which freezes the glob so a newly created module cannot enlarge the
identity set either, plus the five `ops/` modules in the set, with `templates.py`
and `args.py` carrying protocol-pinned content. Inside it all 908 protocol pins
authenticate. Because the three earlier receipts recorded `a379e57a`, they
validated again inside the view and only the tier-3 receipt had to be written, so
the three hours of sampling were not lost after all. Every production step from
here runs inside this view.

**Both source manifests are bound in `release_v2`.** Level 4 at `2e90606b`
(countdown and mathir carried; graph r2, pantry r2, python r5) and Level 5 at
`bb5a933c` (countdown carried; graph r5, mathir r2, pantry r4, python r2). The
two blocking decisions were delegated by the user and are recorded with their
justification in the
[binding decision](../artifacts/modebench_scale_level5_pantry_r4_registration_decision_20260915.json)
and the binding record. `bind_level` verified all ten original campaign fits
reproduce, that each revision map covers exactly its failed originals, and that
no heldout receipt existed anywhere, which is the anti-cherry-picking guarantee:
the source choice was committed before any test score was observable.

**Staging exclusions inside a revision root breaks disjointness verification.**
The burned pilot pools were staged under `<root>/burned_pilots/<domain>/` so
`exclusions()` would collect them, which worked: production pools came out with
zero collisions. But `discover_sources` globs `var/data/modebench*`, so those
files then counted as sources required to be disjoint from every frozen split at
*both* levels, and five Level 5 pilot rows duplicate problems in the Level 4
graph dataset frozen on 2026-09-13, before those pilots existed. The Level 5 r2
exclusion snapshot the pilots branched from predates that freeze, which is how
the collision was possible. No released data is affected. The pilots were moved
to `artifacts/modebench_scale_level5_burned_pilots_20260915/`, outside the search
root; the exclusion snapshots already hold the identities as immutable lists, so
nothing became un-excluded.

**Held-out confirmation completed on 2026-09-15**: array `31290252`, ten cells,
4096 attempts per domain. Nine of ten domains passed both gates; Level 4 MathIR
failed at 1.07 of tolerance on pass@1. All ten were audited through
`audit_source`, replaying 40,960 attempts through the original graders with
bootstrap intervals.

**Level 5 is admitted.** `var/data/modebench_scale_release_v2/level5/admission.json`
carries `difficulty_matched: true` for all five domains at 384/128/128, with
dataset symlinks and a README, and 1,389 files pinned. Countdown is carried from
the campaign; graph_coloring uses r5, mathir r2, pantry r4 and python_factors r2.
Worst gate margins were 0.43, 0.56, 0.92, 0.49 and 0.71 of tolerance. The base
grid subsequently reproduced the 14B confirmation values through an independent
evaluator and seed schedule to within 0.0005 on python, 0.003 on graph, 0.005 on
pantry and 0.006 on countdown.

One caveat is pinned into the grid registry rather than left implicit: Level 5
MathIR cleared at 0.92 of tolerance with a bootstrap of [+0.0125, +0.0635] that
excludes zero, so that split is reliably a little easier than its target and
could as plausibly have failed.

**2026-09-16: the Level 4 MathIR target is not reachable at 7B, and the last
confirmation was not spent.** The 2026-09-15 closure below said a reachable
ladder existed, on the evidence of one 96-row pilot pool. It does not. A
recalibration was registered against
`modebench_scale_mathir_structural_candidates`, the law that produced the
*admitted* Level 5 MathIR dataset, its development pools were scored on 7B, and
the sealed fitter searched all 1,771 mixtures. The best available fit,
`[4, 9, 0, 7]`, forecasts 0.0673/0.1932: **+0.59 of tolerance off centre on
pass@1, against the +0.48 of the fit that already failed**, leaving 1.11 standard
errors to the gate instead of 1.41, and an ex-ante 75 percent chance of clearing
a 128-row confirmation against the original's 85 percent. The domain has one
confirmation left; it was not spent on a worse bet than the one that lost.
Nothing was frozen and the eval draw labels are unused, so the allowance remains
available if a genuinely different construction ever appears.

The cause is structural and is the finding worth keeping. The target pairs
pass@1 0.0437 with pass@8 0.2402 — a heterogeneity gap of 0.060, success spread
fairly evenly over prompts. Every 7B MathIR construction measured runs a gap of
0.169 to 0.398, concentrating success on a few prompts. Lifting pass@8 into its
band therefore requires making problems easy, which drives pass@1 through its
ceiling. Three difficulty mechanisms now fail in exactly this way — family
choice, binding magnitude, and denominator structure — and the same law's
difficulty ordering *scrambles* between scales (t0>t1>t3>t2 at 14B spanning
0.0892 on pass@1; t2>t1>t0>t3 at 7B spanning 0.0254, about 1.5 standard errors),
which says the knob stops biting because 7B sits near its floor. The production
ladder's tier 3 makes the bind concrete: pass@1 0.0469, within 0.08 of tolerance
of target, with pass@8 0.1504, below the 0.1602 floor.

This is a statement about the model–task pair, not about the data. The same law
is difficulty-matched at 14B and is part of the admitted Level 5 release. Full
evidence in the
[outcome record](../artifacts/modebench_scale_level4_mathir_recalibration_outcome_20260916.json).

**Level 4 is closed at four of five, not admitted, with the cause established.**
The decision to stop rather than rebuild was taken by the user on 2026-09-15 and
is recorded in the
[closure record](../artifacts/modebench_scale_level4_admission_closure_20260915.json).

The cause is a construction error, not a marginal confirmation. Level 4 MathIR's
target is 0.0437 and the campaign's four MathIR families span measured
development pass@1 of 0.0488 to 0.1440, so the target lies below the entire
ladder. The fitter selected the hardest available mixture, `[0, 0, 17, 3]`, and
still forecast 0.0630, already +0.0193 off centre inside a 0.04 tolerance; an
unexceptional +0.0235 held-out delta then carried it past the gate. Across the
ten confirmed domains the held-out minus forecast delta has mean +0.0043 and
median +0.0020, with three domains landing harder than forecast, so the fitter is
not systematically optimistic. MathIR is the exception at both levels and the
only domain whose deltas share a sign across them. It is also the only domain
with a single calibration cell, against 7 for countdown and graph, 60 for python
and 109 for pantry, so no residual from other cells can reveal a misplaced
ladder.

Three mechanisms prevent an in-place fix, and none was circumvented. `bind_level`
refuses a source choice once a heldout receipt exists for the domain or its
parent, which is the anti-cherry-picking guard. `prepare` hardcodes the candidate
module, so the fresh campaign root `var/data/modebench_scale_v2`, prepared the
same day and left unused, pinned the identical four families and would reproduce
the same failure. And `revision.register` requires a failed development fit, which
Level 4 MathIR does not have, so the mechanism that would register a bracketing
ladder is unavailable to it. A reachable ladder does exist: the magnitude pilot
measured the origin family alone at 0.0449, below the current floor. Reopening
this would need a prospectively registered change to the MathIR families or to
the target convention, plus isolated source views so the Level 5 admission stays
verifiable.

**Both re-cuts were built and piloted on 2026-09-15.** Pantry r4
(`ops/exp_scaling/modebench_scale_pantry_r4_candidates.py`) keeps tiers 0 and 1 at
exactly their measured r3 settings and replaces the two dead rungs with milder steps,
holding the tight-interval law off and the forbidden-tag probability fixed so that
only headroom and menu size move below tier 1. It is derived mechanically from r3 and
a test asserts the two files are identical outside the docstring, schema and four tier
constants. Graph r4
(`ops/exp_scaling/modebench_scale_graph_r4_candidates.py`) grades a preset per
answer-mode-count cell, and builds each row's profile from the preset that actually
generated it, so the 20260914 mislabelling cannot recur. Its pools verify with zero
profile mismatches against 192 mislabelled rows before.

**Graph r4 is the first feasible graph design, and still too thin to use.** Its
measured ladder is 0.2510/0.4974, 0.2165/0.4297, 0.3174/0.5286 and 0.2142/0.4427, giving
28 of 1,771 feasible mixtures at a best normalised error of 0.942, against 0 feasible
for r2 (1.182), r3 (2.902) and the first pilot (1.077). A 6 percent margin is inside the
pool-to-pool noise, so it is not a design to take to a production fit on its own. The
[outcome record](../artifacts/modebench_scale_level5_graph_r4_outcome_20260915.json)
also validates per-cell composition as a predictive tool and bounds when it can be
trusted: tiers whose support-4 preset came from 62 production rows were predicted to
within 0.017 and 0.023 on pass@1, while the tier that took support 4 from 20 pilot rows
missed by +0.079 and broke monotonicity.

**Per-cell grading did not move the heterogeneity gap, which corrects an earlier
claim.** r4's gaps are 0.404, 0.428, 0.424 and 0.412, indistinguishable from r2's 0.32
to 0.40. Equalising cell means left the gap unchanged because much of the dispersion is
within cell, not between: dead fractions inside individual r4 cells are 0.38 to 0.44.
The earlier statement that between-cell spread dominates the gap was too strong.

**Ranking revisions by best reachable error selected the wrong branch.** r3 was rejected
on 2026-09-14 at error 2.902 in favour of r2 at 1.182, and the pilot, then r4, were built
on that choice. But r3 carries the lowest heterogeneity gaps ever measured for this
domain, 0.329, 0.333, 0.289 and 0.260, falling with its path multiplier, and its tier 3
is below the 0.282 the target requires. r3 was never too bimodal; it was uniformly too
easy, with all four tiers between pass@1 0.32 and 0.38 and no downward range. Best
reachable error is dominated by the mean mismatch, which mixing corrects, and hides
dispersion, which mixing cannot. Shift an r3-style configuration to the target pass@1
holding its gap anywhere in 0.26 to 0.32 and pass@8 lands at 0.525 to 0.585, inside the
tolerance.

**Graph r5 branches from r3.** `ops/exp_scaling/modebench_scale_graph_r5_candidates.py`
keeps r3's enumerated skeleton catalog, which is the source of its uniformity, and adds
two knobs as a 2x2 factorial at the fixed multiplier 64: a fourth hidden vertex, which
grows the answer from three digits to four and the hidden search from 27 to 81
assignments while leaving every prompt reachable, and up to three shown-shown edges,
which are satisfied by construction because the shown vertices carry distinct colours
and therefore change parsing load without touching the support or the answer. Tier 0
reproduces r3 tier 3 against a byte-identical catalog, so it is a control on a measured
point. Twenty-two structural tests pass and all 28 cells pass capacity at both hidden
counts. Its screening job is `31285344`.

**Replication is configuration-dependent, which qualifies the noise concern.** The
pantry tier-0 configuration has now been measured three times on independent pools at
0.1476 (146 rows), 0.1896 (59 rows) and 0.1105 (80 rows), a spread of 0.079 against a
0.04 gate. The tier-1 configuration, which is the rung the fit depends on, measured
0.0742 (59 rows) and 0.0734 (80 rows), agreeing to 0.0008. Recorded standard errors
across 84 development receipts explain the difference: the median pass@1 standard error
is 0.0122, so the gate is 3.3 standard errors wide, but the graph and pantry revisions
run 0.021 to 0.029, leaving their gates only 1.4 to 2.0 standard errors wide. The
domains that keep failing are the dispersed ones, and dispersion inflates both the
heterogeneity gap and the standard error. Held-out confirmation uses 128 fresh eval
rows, so for those two domains a correctly calibrated dataset still carries a real
chance of failing confirmation. The registered tolerances are unchanged here; this is
recorded as a protocol question, not a calibration one.

**Recurring nonzero exit statuses are a wrapper artifact.** The frozen-view
PRoot wrapper returns a nonzero status after its guest has exited 0, and the
production workers `exec` that wrapper as their final command, so its status
becomes the job status. A NONPRODUCTION 0.5B probe reproduced it directly:
tensor-parallel size 1 returns 1 and size 2 returns 143, with the guest printing
its exit-0 marker in both cases. That matches every `FAILED 1:0` and
`FAILED 143:0` recorded below. The
[finding](../artifacts/modebench_scale_proot_teardown_exit_finding_20260915.json)
changes no scientific decision and relabels no recorded scheduler status; each of
those jobs already carries its own complete saved-output and original-grader
audit.

At 16:20 UTC on September 13, Level 4 readiness was:

| Domain | Development fit | Frozen data | Held-out confirmation |
| --- | --- | --- | --- |
| Countdown | Passed | Complete | Pending |
| MathIR | Passed | Complete | Pending |
| Graph Coloring | Revised fit passed | Complete | Pending |
| Pantry | Revised fit passed | Complete | Pending |
| Python Factors | Third revision failed; fourth scored and independently audited, fit pending | Not ready | Not ready |

Graph and Pantry were frozen successfully at 18:22 UTC. Their saved JSONL and
Hugging Face splits passed identity and disjointness checks, bringing Level 4
to four built domains. These are still pending held-out confirmation.

Python needs another development revision before Level 4 can be released.
Its selected development pass@1 was 0.26806640625 against target 0.2109375
(delta +0.05712890625, exceeding the 0.04 tolerance); pass@8 was 0.64453125
against target 0.76953125 (delta -0.125, exceeding the 0.08 tolerance). Both
whole-pool forecast gates also failed after all 1,771 fixed mixtures. The failed
recipe is preserved; it is not retried or admitted. No held-out outcomes have
been observed. The earlier conditional 2–4 hour Level 4 estimate no longer
applies; another Python calibration round and final validation remain.

The harder Python revision uses five or six factor cases, including balanced
semiprimes and large prime squares, while retaining the native prompt and
verifier. It passed 52 structural tests, independent review, and an actual
native check of eight scratch problems with 16 solution witnesses. Its
reviewed production preparation completed at 20:40 UTC: four 193-row development
pools, 772 problems total, under the existing numerical targets. One six-input
example uses `[284, 556, 667, 841, 880, 961]`, including 667 = 23 × 29 and
961 = 31². The exact temporary diagnostic evidence is now available on shared
storage and is staged and checked on workers. The complete node202 preflight
passed at 21:01 UTC. **7B Python development job `31260226` finished all four
tiers and 400 batches at 23:15 UTC on September 12.** All 24,704 saved attempts
are present. The evaluator returned zero, while Slurm recorded `FAILED 143:0`;
the cause of the enclosing failure is unknown. The exact observed reconciliation
has passed review, structural validation and the complete original-grader audit.
The fixed difficulty fit subsequently completed and failed. Four Level 4 domains remain built; all five held-out
confirmations are still pending. The session is now on soak. A new, reviewed
CPU portability helper restored the exact historical diagnostic without changing
the sealed sources or recorded execution identities. The frozen source view and
local dependencies passed qualification. The read-only check passed for all four
output sets and native prompts at 03:53 UTC on September 13. All 24,704 original-
grader checks passed and the certificate was published. The audit wrapper exited
with code 143 afterward; that outcome is preserved separately. An independent
read-only verification passed with exit 0 at 04:01 UTC. The fixed fit completed with exit 0 at 04:30 UTC and failed all six gates.
Its selected pass@1/pass@8 were 0.0048828125/0.029296875, far below the unchanged
0.2109375/0.76953125 targets. A separate read-only verification passed. This
negative recipe is preserved; no further audit or fit of this revision is planned.

A fourth Python revision has completed development scoring. It retains five or six inputs but
uses balanced semiprimes with least factor 7 or 11 and appended squares 49 and/or
121. The four profiles are prospective hypotheses, without a claimed difficulty
ordering. Its one fresh exact-capacity check passed all 60 support cells after
excluding historical identities and projections, including all 772 failed r3
problems. Native qualification passed at 04:59 UTC: eight fresh scratch problems and
16 original verifier checks, with exit 0. Those scratch identities and prompts
were excluded during production generation. Full preparation completed with
exit 0 at 05:10 UTC: four 193-row pools, 772 problems total, with unchanged
revision-four draw labels and numerical targets. Fresh 7B scoring job `31261726`
ran on node203 from 05:19:21 to 07:12:51 UTC. All four receipts and 400 batch
files exist, and the evaluator recorded exit 0 at 07:12:50.911653 UTC.
Slurm recorded `FAILED 143:0` for the allocation and batch step, with the
external step completing at 0. The cause is unknown. The exact terminal capture
and logs are preserved. The user explicitly approved auditing these saved
results and using them for calibration if all checks pass. Structural checks,
all 24,704 original-grader checks and the complete certificate passed. The
CPU audit wrapper recorded exit 143 after publishing the certificate; that
operational failure and its unknown cause remain recorded. A separate unchanged
read-only verification passed with exit 0 at 16:19:50 UTC, authenticating the
5,135-file certificate with no new grader calls. The fit adapter is being
reviewed to bind only this exact observed evidence before the unchanged fixed
calibration fit. All final held-out confirmations remain pending.
The latest conditional 1–3 hour estimate no longer applies; several more hours
are needed at minimum, with no reliable completion time until calibration passes.
The optional 0.17/0.65 target
proposal remains unregistered; it does not block this existing-target revision.
No replacement release source manifest or held-out evaluation has started.
The [structural design report](../artifacts/modebench_scale_python_harder_case_design_draft_20260912.md)
explains how five or six inputs preserve the registered solution-mode counts.
The harder revision failed measured calibration; the new lower-prime revision
still requires its audited fixed-fit decision.

Original 7B development is complete: Countdown and MathIR passed their fits;
Graph Coloring, Python Factors, and Pantry require the fresh registered
revisions below. At 14B, Countdown passed its development fit; Graph Coloring,
Python Factors, and MathIR require their registered revisions. The original
14B Pantry recovery completed all original generation and passed the saved-attempt
audit, but its scheduler job ended with status 1 after the last receipt was saved;
the exact exit cause is unknown. Its full fit narrowly failed the selected
development pass@8 gate, so Pantry also requires a fresh 14B revision. An additive
execution reconciliation preserves the failed scheduler evidence and validates
the complete outputs. Both source manifests are published, the three passing
original datasets are frozen, and all 28 revised development pools are built.
Revised array `31254520` saved all four scoring receipts for 7B Graph,
Pantry, and Python Factors, plus 14B Graph and MathIR. All five domains have
completed original-grader reconciliation of their saved attempts. Their actual
failed scheduler statuses remain recorded, including `FAILED 143:0` for 7B
Python Factors; these audits do not imply a passing development fit. At 18:38 UTC,
the two prescribed revised Level 5 fits completed: MathIR passed all six gates
with weights `[2, 17, 0, 1]`; Graph failed four of six gates with weights
`[10, 0, 0, 10]` and requires another development revision. Both decisions are
preserved. These fits used saved receipts and performed no new model scoring
or grading. The three revised Level 4 fits are recorded above.
MathIR was frozen successfully at 19:07 UTC and passed a separate read-only
verification: 384 train, 128 dev, and 128 test rows. Together with Countdown,
two Level 5 domains are now built, pending held-out confirmation. A fresh
Graph topology revision was registered and its four 146-row development pools
were generated and verified at 19:31 UTC. Its first scoring job, `31259795`,
failed before GPU probing or model execution because a historical ledger check
compared filesystem device numbers across hosts. The saved failed attempt and
all unscored pools are preserved. An additive correction passed the complete
read-only verification on node202 at 19:45 UTC. Recovery job `31259860` was
submitted at 19:51 UTC and completed all four receipts and 304 batches by
19:57 UTC. The evaluator sidecar records return code 0; Slurm records
`FAILED 1:0`, whose exact cause remains unknown. The original-grader audit of
all 18,688 saved answers and separate read-only verification completed
successfully at 20:19 UTC. The fixed Graph r3 fit completed and verified at
20:58 UTC; all three pass@1 gates failed, while all three pass@8 gates passed.
Its selected rates were 0.334228515625/0.634765625 with weights `[0, 20, 0, 0]`.
The failed decision is preserved; Graph needs another development revision.

At 13:28 UTC, node105 unexpectedly rebooted and the 14B Pantry cell ended
`NODE_FAIL`, with 409 of 464 batches and three of four receipts saved. The wash
controller recorded an `execution_failed` terminal at 13:29 UTC. A recovery plan
preserves the same A5000 GPU type, model, sampling settings, and registered seeds.
The unstarted final Python task was held and cancelled at 15:48 UTC so the two
unfinished cells can run in one recovery array. A reviewed accounting adapter
preserves Slurm's literal `None` start time for that cancelled task. Recovery
array `31258973` was submitted at 16:06 UTC, but the cluster rewrote its requested
partition to `mltheory`, leaving both cells pending on the unavailable node.
Both unstarted cells were held and then cancelled with recorded evidence at
16:59 UTC. Corrected recovery array `31259131` was submitted at 17:02 UTC using
the existing `allcs` account and `cs` partition. Its Pantry cell saved all 464 batches and four receipts, then ended with
`FAILED 1:0` at 17:37:54 UTC; that status and the unknown exact exit cause remain
recorded. Python Factors started on node203 at 17:37:55 UTC with two verified
A5000 GPUs and completed all 400 batches and four receipts before ending
`FAILED 143:0` at 20:01:50 UTC; its exact exit cause is also unknown.
The complete eight-receipt recovery audit checked 54,400 saved answers and
published its certificate at 20:20 UTC. The audit wrapper then returned 143,
and a separate read-only verification passed at 20:41 UTC. The two prescribed
Pantry/Python fits completed and verified at 20:49 UTC. Python passed all six
gates with weights `[0, 0, 16, 4]` and selected development pass@1/pass@8 of
0.226318359375/0.73828125. Its native dataset freeze completed at 20:56 UTC,
and separate read-only verification passed: 384 train, 128 dev, and 128 test
problems. Level 5 now has three built domains, pending held-out confirmation. Pantry
failed five of six gates with weights `[0, 0, 0, 20]` and selected development
rates of 0.171142578125/0.3671875 against targets 0.05859375/0.259765625. Pantry
needs a harder revision. Both fixed decisions and all failed execution evidence
are preserved; no audit or fit is being repeated. The exact models, sampling settings, seeds,
original plans, saved outputs, and failed execution evidence remain preserved.
Fresh held-out confirmation remains pending. The neutral Python Level 3
default is retained; the scale recovery uses an isolated copy of its original
registered source. The user has authorized wash.cs.princeton.edu for the
continuation; GPU evaluations run in their recorded Slurm allocations.

A level is admitted only after all five domains pass fresh held-out
confirmation. Its eventual decision will be
`var/data/modebench_scale_release_v2/<level>/admission.json` with
`difficulty_matched: true`. A source manifest, passing development recipe,
or frozen dataset directory alone does not establish admission.

| Domain | Fixed Level 1 pass@1 | Fixed Level 1 pass@8 |
| --- | ---: | ---: |
| Countdown | 0.0126953125 | 0.0859375 |
| Graph Coloring | 0.207763671875 | 0.5625 |
| Python Factors | 0.2109375 | 0.76953125 |
| MathIR | 0.043701171875 | 0.240234375 |
| Pantry | 0.05859375 | 0.259765625 |

The targets are the completed historical independent Level 1 control receipts
used for the admitted Level 3 V3 release. They remain explicitly historical
fixed targets. This extension is not fresh two-sided statistical equivalence.
The gates remain absolute differences of at most 0.04 for pass@1 and 0.08 for
pass@8, evaluated without rounding. Distinct verified modes@8 is reported,
not fitted. Confirmation includes prompt bootstrap intervals conditional on
the fixed targets and original-grader replay of every generated attempt.

Each domain has **384 train, 128 dev, and 128 test rows**. The repository's
standard name for the test split is `eval`, and its Hugging Face dataset key
is `multi_answer`. Training uses the key `train`. Dataset directories and
JSONL copies are frozen after a passing development recipe, before held-out
confirmation; they become part of a released level only after admission.

The composite release collects a canonical source for each
domain. It retains an original domain whose development fit passes. A domain
whose original fit fails must use an explicitly registered fresh revision.
This choice is fixed in `source_manifest.json` before any held-out outcomes
are observed; it does not select among sources using test scores.

| Source kind | Canonical source root | Selection rule |
| --- | --- | --- |
| `campaign_v1` | `var/data/modebench_scale_v1` | Carry the passing original development fit. |
| `domain_revision_v1` | An explicit root under `var/data/modebench_scale_domain_revisions_v1`, such as `level4_graph_coloring_r2` | Use a registered revision for a failed original fit. |

Each canonical source keeps its own protocol, four development pools, recipe,
frozen train/dev/eval datasets, and confirmation evidence. The revision root
is not inferred from the example name: the level's sealed source manifest
records the exact absolute `source_root` and `source_kind` for every domain.
Original failures remain preserved in their original campaign.

The following fresh revisions are registered. Their protocols and candidate
laws are sealed. The wash continuation generates their calibration pools and
frozen datasets; released datasets require the final admissions:

| Level | Domain | Revision directory under `var/data/modebench_scale_domain_revisions_v1` |
| --- | --- | --- |
| 4 | Graph Coloring | `level4_graph_coloring_r2` |
| 4 | Python Factors | `level4_python_factors_r2` |
| 4 | Pantry | `level4_pantry_r2` |
| 5 | Graph Coloring | `level5_graph_coloring_r2` |
| 5 | Python Factors | `level5_python_factors_r2` |
| 5 | MathIR | `level5_mathir_r2` |
| 5 | Pantry | `level5_pantry_r2` |

The Pantry revision retains six ingredients and the original family tables,
constraints, prompts, and verifier. Its four candidate laws set available
amounts to 50 g, 75 g, or 100 g, with a fresh original-tier-0 law as the
fourth candidate. A joint capacity check for both levels passed 5,952 fresh
rows and 113,880 original-verifier witness checks, covering every required
family/solution-count cell and full train/test quotas. The graph candidate laws
also passed joint capacity checks for both levels. The Python candidate laws
passed a joint check of 5,640 disjoint scratch rows across both levels, with
11,280 original verifier witnesses. These checks establish that fresh disjoint
problems can be generated; model scoring must establish difficulty.

The 14B MathIR revision uses four denominator pairs in
`a*x/D1 + b/D2 = d*x + c`: `(e,f)`, `(a+e,f)`, `(e,b-f)`, and
`(a+e,b-f)`. It retains the original prompt renderer and verifier, six actions,
a four-step limit, and exactly five canonical answer modes. Full quotas for
all four tiers passed capacity validation on 2,560 globally disjoint scratch
rows, with 12,800 original verifier witness checks and native prompt budgets.
These structural checks do not establish model difficulty; fresh calibration
and held-out confirmation are still required.

The release layout, as it now stands:

```
var/data/modebench_scale_release_v2/
  level4/                        # Qwen2.5-7B-Instruct
    source_manifest.json         # source choices; does not imply admission
    source_manifest.sha256.json
    release_status.json          # four of five confirmed; NOT an admission
    README.md
                                 # no admission.json: MathIR failed its gate
  level5/                        # Qwen2.5-14B-Instruct
    source_manifest.json
    source_manifest.sha256.json
    admission.json               # all five held-out domains passed
    README.md
    dataset/<domain>/            # symlink to its canonical source dataset
      train/                     # DatasetDict key: train
      dev/                       # DatasetDict key: multi_answer
      eval/                      # test; DatasetDict key: multi_answer
      train.jsonl, dev.jsonl, eval.jsonl
```

Two files the earlier plan reserved are absent and will stay absent. The
campaign-wide `admission.json` and `execution_provenance.json` both require
*both* levels to be admitted, and Level 4 never will be. Neither is a
precondition for using Level 5: level admission is its own decision, and it is
complete. Level 4's five datasets are loadable from the per-domain
`source_root` entries in its `source_manifest.json`, with the standing caveat
that MathIR is not difficulty-matched.

Admission requires reproducible passing development fits, unchanged frozen
384/128/128 splits, original-grader replay of all 4,096 held-out attempts per
domain, both metric gates in every domain, and current cross-source identity
and prompt disjointness. The admission record pins the source evidence and
lists the canonical path of every split. Level admission is separate from
the campaign-wide record, which requires both levels to pass. The combined
extension also requires `execution_provenance.json` with schema
`modebench_scale_supplemental_execution_provenance_v1` and status `verified`.
This supplemental record authenticates the initial execution plan, synthetic
qualification and applied operational amendment, both level admissions, and
the campaign admission. It supplements the scientific admission decisions.

Load the train/test splits from the canonical paths in the level admission,
which is the record that says the split was difficulty-matched. Run this from
the repository root; it refuses any level that lacks a complete admission, which
is how it declines Level 4:

```python
import json
from pathlib import Path
from datasets import load_from_disk

release = Path("var/data/modebench_scale_release_v2")
level = release / "level5"
admission_path = level / "admission.json"
if not admission_path.is_file():
    raise RuntimeError(f"{level.name} has not been admitted.")
admission = json.loads(admission_path.read_text())
expected_domains = {"countdown", "graph_coloring", "python_factors", "mathir", "pantry"}
if (admission.get("schema") != "modebench_scale_composite_level_admission_v1"
        or admission.get("level") != level.name
        or admission.get("difficulty_matched") is not True
        or admission.get("test_split") != "eval"
        or set(admission.get("domains", {})) != expected_domains):
    raise RuntimeError(f"{level.name} has no complete matching level admission.")

splits = admission["domains"]["countdown"]["splits"]
train = load_from_disk(splits["train"]["path"])["train"]
test = load_from_disk(splits["eval"]["path"])["multi_answer"]
```

`<release>/level5/dataset/<domain>/train` and `eval` are convenience symlinks to
the same canonical data; keep the canonical source directories in place when
using them. Each row contains `problem` and its executable `answer`
specification.

Level 4 has no admission record, so this loader refuses it by design. Its rows
are reached through `level4/release_status.json` instead, which carries the same
per-domain split map plus each domain's measured held-out verdict, and whose
`difficulty_matched` is `False`. Read a Level 4 split from
`release_status.json["domains"][domain]["splits"][split]["path"]`, and carry the
per-domain `difficulty_matched` forward with any number taken from it: four
domains confirmed, MathIR not difficulty-matched at 1.07 of tolerance.

**Why the tolerance was not simply re-registered.** Widening the gate to admit
the measured MathIR value was considered on 2026-09-15 and is not available.
The tolerances live in `var/data/modebench_scale_v1/protocol.json` and in
`materialize_modebench_scale.TOLERANCES`, which are pinned by 16 and 17 sealed
records respectively, including both source manifests and the Level 5
admission's 1,389 file pins; editing either invalidates the level that *is*
admitted. And it would not change the verdict regardless: the five held-out
audits are write-once through `atomic_new`, which links onto its destination and
so refuses an existing path, and `audit_source` re-derives any published audit
and requires it to match. Changing a recorded decision would mean deleting a
published audit. A different MathIR verdict requires fresh, unobserved held-out
rows, which means a new dataset, not a new threshold.

The split histograms match the admitted Level 3 release exactly, including
Pantry's joint family/solution-count composition. Development pools cover
all dev/test cells and any training-only cells. Historical identities and
exact prompts are excluded, including earlier failed calibration pools.
Level 4 and Level 5 also exclude one another. Shared development candidate
laws vary operand magnitude, graph size, factorization structure, rational
equations, and Pantry menus/constraints. Their four tiers are hypotheses;
model scores determine mixtures separately for each scale.

The fitter ranks all 1,771 grid-20 mixtures by complete-pool forecasts under
both dev and test solution-count histograms. It then scores one deterministic
dev selection. Neither selected-row residuals nor test outcomes rank mixtures.
All six forecast/selection gates must pass before generating train/test data.
A failed fit or confirmation is retained and requires a prospective new
revision; changing seeds or choosing another subset is not an automatic retry.

The existing prompts, answer syntax, canonicalizers, and executable verifiers
are retained. Evaluation uses native chat templates, FP16, temperature 1,
top-p 1, 192 output tokens, and four independent draws of eight samples per
problem. Hashed per-prompt/draw seeds avoid overlapping vLLM child streams.
The new evaluator has its own receipt schema, source pins, and explicit
runtime settings. Graph context is 1,024 tokens; other domains use 2,048.
Historical Level 3 code and evidence are unchanged.

The applied v2 operational amendment allows two concurrent cells in parent
development array `31243495`, using at most four GPUs. After synthetic
qualification `31245818` passed, only the previously untouched 14B cells
5–9 changed their host-memory requests from 60 to 59 GiB. The running 7B
cell retained 60 GiB. Model identities, numerical settings, task commands,
batch sizes, seeds, and token budgets are unchanged. The qualification
observed a 21.50 GiB job-memory peak; its visible enforced limit was 177 GiB.
It establishes neither a 59 GiB hard ceiling nor a cold-cache result.

Models are pinned to official checkpoint snapshots:

- [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct): `a09a35458c702b33eeacc393d103063234e8bc28`.
- [Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct): `cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8`.

Run the additive tools with `var/seed_paper_eval/paper310/bin/python -B`:

```bash
python -B ops/exp_scaling/materialize_modebench_scale.py prepare
python -B ops/exp_scaling/materialize_modebench_scale.py pools --level level4 --domain graph_coloring
python -B ops/exp_scaling/launch_modebench_scale.py --help
python -B ops/exp_scaling/fit_modebench_scale.py fit --level level4 --domain graph_coloring
python -B ops/exp_scaling/materialize_modebench_scale.py freeze --level level4 --domain graph_coloring
python -B ops/exp_scaling/fit_modebench_scale.py confirm --level level4 --domain graph_coloring
```

These commands require the preceding stage's real authenticated evidence.
`prepare` uses a fresh campaign directory and requires complete local model
snapshots. Pool generation and freezing refuse overwrites. The launcher seals
inputs before submitting an array and records submission intent/results so an
interrupted client cannot silently duplicate jobs. Test evaluation requires
passing reproducible recipes and sealed test rows. The requested train/test
work builds datasets; it does not launch treatment training runs.

The original scientific functions remain in
`ops/exp_scaling/continue_modebench_scale_composite.py`. Current orchestration
uses individually recorded actions from spin, each under the existing shared
controller lock, with the unchanged scientific code inside the
[frozen source view](../artifacts/run_modebench_scale_frozen_view_20260912.py).
The [Python development transport](../artifacts/modebench_scale_level4_python_r3_transport_v2_20260912.py)
submitted the current single-cell Level 4 calibration. No replacement watcher
has been started. The canonical neutral Python Level 3 source remains in place.

The earlier [wash continuation](../artifacts/continue_modebench_scale_composite_wash_v2_20260912.py)
and all its records are preserved. Its recorded terminal result was
`execution_failed` after revised Pantry hit `NODE_FAIL`; its remote operating
system process state remains unverified. It is historical evidence, not the
current route. The original Pantry recovery retains `FAILED 1:0` and an unknown
exit cause, alongside its complete saved-output and original-grader evidence.
The later revised Pantry/Python recovery likewise retains its actual
`FAILED 1:0` and `FAILED 143:0` outcomes. Full scientific-output reconciliation
does not relabel those scheduler outcomes as successful.

Level 4 proceeds through the complete Python development audit and fixed fit,
then a passing Python freeze and a new five-source binding. The
[Level 4 release provider](../artifacts/continue_modebench_scale_level4_first_release_20260912.py),
[five-domain transport](../artifacts/modebench_scale_level4_first_transport_20260912.py),
and [native confirmation/admission helper](../artifacts/complete_modebench_scale_level4_first_20260912.py)
have passed component review. They are prospective and still require final
reviews tied to actual prerequisite evidence before execution. The planned
heldout array allows three simultaneous cells, each with two A5000 GPUs, six
CPUs, and 60G memory. Its confirmation audit checks actual allocation overlap
as well as the requested limit. Scientific arguments and numerical settings
per cell remain unchanged. Durable claims prevent automatic duplicate work.
All five sources must be fixed before any heldout scoring for that level, and
all five heldout gates must pass before its admission and dataset links.

Final execution verification must cover the historical schedule and original
recovery, the recorded wash execution, all actual revised development and
recovery jobs, the individual fit/freeze actions, the final heldout jobs, both
level admissions, and the combined admission. It must preserve literal failed
statuses and their complete audited evidence. Read-only verification uses the
saved native grader audits; it must not call the native confirmation entrypoint
again because that would repeat grading for revised domains.

The historical v2/v3 verifiers and the reviewed
[v4 verifier](../artifacts/verify_modebench_scale_frozen_execution_v4_20260912.py)
are preserved. They do not cover the current sequence of individual actions
and separate Level 4 completion. V4 expected a new admitted spin watcher and
one combined ten-cell confirmation array; neither has occurred. An additive
final verifier must bind the actual completion path before publishing the
execution record. None of those historical verifiers is an available final
certificate for this build.

The new [recorded-history component](../artifacts/verify_modebench_scale_recorded_history_20260912.py)
has passed 52 tests and independent component review. Its actual frozen-view
read-only integration passed at 23:03 UTC, covering 6,827 saved files. It preserves the original failed job states
and joins them to existing saved-output audits. The
[MathIR static adapter](../artifacts/verify_modebench_scale_level5_mathir_static_freeze_20260912.py)
passed 56 tests and actual frozen-view verification at 22:58 UTC, returning the
original freeze certificate unchanged. The
[combined-admission publisher draft](../artifacts/publish_modebench_scale_combined_admission_draft_20260912.py)
passed 47 tests and component review. It requires both actual level admissions
and all ten saved audits before publication; its final activation review is
still absent. These components do not establish either level's readiness.

At 22:11 UTC, the independently reviewed [destination registration](../var/artifacts/modebench_scale_final_release_destination_20260912/registration.json)
completed and passed a separate frozen-view verification. It directs the future
execution record to `modebench_scale_release_v2/execution_provenance.json`,
while retaining unchanged validation of the historical schedule against its
original v1 destination. This metadata action publishes no dataset or admission.
Level 4 can still be admitted independently after its own five passing checks.

The combined extension is complete only when `execution_provenance.json` has
schema `modebench_scale_supplemental_execution_provenance_v1`, status `verified`,
and both level admissions are present. None of this dataset work launches
treatment training runs.
