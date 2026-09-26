# Current figure coverage audit

This audit covers the seventeen figures compiled in both manuscripts at the
**September 11, 2026** cutoff. The exact placement and builder inventory is in
[FIGURE_MANIFEST.md](FIGURE_MANIFEST.md). The superseded August 26 audit is
preserved under
[audits/figure_refresh_20260911/organization_before/paper/FIGURE_DATA_AUDIT.md](audits/figure_refresh_20260911/organization_before/paper/FIGURE_DATA_AUDIT.md).

## Model and level coverage

| Evidence family | Qwen2.5-0.5B | Falcon3-1B | Qwen2.5-3B | Where it appears |
|---|---|---|---|---|
| Level 1, ReplayDr.GRPO vs Dr.GRPO | All five domains n=5 | Four domains n=5; Countdown n=4 | All five domains n=5 | Figures 4/5 and retained supporting figures, according to their estimands. |
| Level 1, ReplayMaxRL vs MaxRL | All five domains n=5 | All five domains n=5 | Graph 5, Countdown 3, Python 5, MathIR 3, Pantry 1 | Figure 5's two shared three-model panels; domain-resolved three-model appendix. |
| Level 2, four-method factorial | Graph, Countdown, Python, MathIR n=5; Pantry incomplete | No result in this factorial | No result in this factorial | Figure 6 terminal comparison and domain companion; complete paired tables in both appendices. |
| Level 3 | No terminal result for this update | No terminal result for this update | No terminal result for this update | No efficacy marker. |

Level 1 covers the registered three-model core, with the single persistent
Falcon endpoint exclusion. The nearly complete Level-2 set means four of five
domains at Qwen2.5-0.5B; it does not mean three-model Level-2 completion.

E118 has **138/150 admitted endpoints**, **67 MaxRL replay pairs**, and
**12/15 complete paired blocks**. Its four-arm intersection is smaller where
the corrected Dr.GRPO replay comparator is incomplete. Figure 5 now includes
Qwen-3B MaxRL and ReplayMaxRL in the same two main panels as the other models.
The dashed Qwen-3B track is a descriptive equal-domain mean: each domain uses
its own available paired seed set (5/3/5/3/1), and those five means receive
equal weight. It has no common seed across all domains, so it displays neither
an interval nor individual-seed cross-domain paths. The appendix retains every
domain's paired observations and exact n. Completed tracks continue to use
common paired seeds across domains.

E119 has **88/100 admitted endpoints** and **four complete five-seed domain
factorials**. PantryPlan terminal arm counts are 3 Dr.GRPO, 3 ReplayDr.GRPO,
1 MaxRL and 1 ReplayMaxRL. Dr.GRPO/replay has two paired seeds (43,46);
MaxRL/replay has zero. Figure 6's main comparison uses the four completed
domains at pass 8, identically across levels and all four methods. Pantry's
incomplete coverage is shown separately instead of entering a mixture of
terminal and intermediate checkpoints.

E120-R1 has **43/45 admitted treatment endpoints** and **7/9 complete paired
blocks**. Qwen-3B Graph and Pantry each have four pairs; the paired seeds are
70–73 and 71–74 respectively. The original Qwen2.5-0.5B primary analysis
continues to use its frozen September 4 input. Current partial extensions are
reported in the dated tables without five-seed intervals.

## Figure-by-figure decisions

| Figures | Refresh decision | Retained evidence boundary |
|---|---|---|
| 1: Graph illustration | Keep the existing mechanically selected example and source set; align inventory with its Dr.GRPO/ReplayMaxRL labels. | Adding completed runs does not retrospectively select a new illustrative prompt. |
| 2–3: benchmark/method | Rebuild from their current code-native diagrams. | These encode verifiers and the algorithm, not run coverage. |
| 4: retention/alternatives | Rebuild after current endpoint sidecars and Figure 5 are ready. | 74 admissible primary pairs; the Falcon exclusion remains active. |
| 5 and MaxRL appendix | Refresh from the dated endpoint audit; show all three models including Qwen-3B MaxRL/ReplayMaxRL in both main panels, with a three-model domain appendix. | Qwen-3B's dashed MaxRL track is an equal-domain descriptive mean of domain-specific paired means (n=5/3/5/3/1), without an interval or across-domain seed paths. |
| 6 and companion | Separate complete terminal comparison from incomplete Pantry coverage; preserve admission as its own measurement. | Four common completed domains, five common seeds, four methods, both levels. |
| Primary training trajectories | Add accuracy and mode plots at every Level-1 model scale and both metrics for Level 2. | Four methods; exact main-figure paired terminal cohorts, no changing-seed mean, explicit missing checkpoints. |
| Direct-comparator forest and trajectories | Keep audited endpoints; extend UCPO/RLEP trajectories to both metrics at 0.5B and 1B only. | GRPO 75 pairs, UCPO 50, sparse RLEP-Dr 47 including Falcon Python n=2. The primary four-method trajectory figures cover all three scales. |
| Baseline collapse precheck | Rerender the complete retained 150-run record. | Per-arm pass-0 values and the initial-breadth floor remain explicit. |
| E121 fixed-bank scores | Rerender retained per-identity score records. | Five seeds, one model/domain; scores are not canonical-mode probabilities or a causal replay estimate. |

## Reproduction and source separation

- `results/latest_results_20260911.json` and the dated endpoint audit bind the
  current E118/E119/E120 census and paired seed intersections.
- `results/training_curve_snapshot_20260911.json` binds primary trajectories
  on both metrics to the same terminal populations and audited checkpoint draws.
- `results/modebench_level_comparison_snapshot.json` binds Figure 6's admission,
  terminal comparisons and checkpoint-availability provenance.
- Each empirical figure has an adjacent JSON sidecar or a named result/audit
  record. The retained rendering helper reads these records and checks that
  its input bytes did not change.
- Both papers compile the same named PDF set. The workshop synchronization
  helper copies referenced figures/tables and provenance, preserves replaced
  files and snapshots, and records new source hashes before compilation.
- Old figures, dated reports and audits remain on disk. They are excluded from
  the active figure target and do not silently overwrite current figures.

The old `make figures` recipe omitted current Figures 3–6 and invoked an older
story generator that reused Figure 1's filename. The new target lists the full
compiled set and uses the intended Figure 1 builder. The obsolete all-campaign
`terminal-results` recipe now delegates to the dated report and retained
figure workflow. Collection of new campaign observations is explicit.

## Streamlined evidence presentation

The September 11 editorial pass removes duplicate status/seed tables and three
secondary figures from the compiled manuscripts. It changes no endpoint,
training-curve snapshot, model population, or statistical analysis. Current
results are consolidated in one empirical section. The frozen Qwen-0.5B
weighting estimand and its uncertainty remain explicit; all unfavorable and
inconclusive effects remain available. UCPO/RLEP coverage denominators refer
to the selected 0.5B/1B scope (50/50 and 47/50). The full prior manuscripts
and proof extensions are retained in `audits/streamlining_20260911/`.

## Hosted breadth display

Figure 7 has 105 cells: seven deployments, five domains and three levels, with
per-response accuracy and distinct@8 shown separately. Each cell retains all
128 prompts and 1,024 responses. The frozen formatting normalizer is shared
across the display. Only Opus 5 Python uses its full revised-wording cohort
(3,072 responses, 3,069 normalized correct); no response is filtered by outcome.
The graphic is descriptive across the stated formulations. Original scores,
refusals, normalization differences and prompt contrasts remain in the appendix.
The figure sidecar binds every cell to its source fields and complete counts.
