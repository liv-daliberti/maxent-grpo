# MATH-AI NeurIPS 2026 workshop version

**Mode Collapse in RLVR & ModeBench**

[main.pdf](main.pdf) is the anonymous submission: four pages of main content,
then references and supplementary material. The official workshop style remains
unchanged. [main.tex](main.tex), [preamble.tex](preamble.tex), and
[appendix.tex](appendix.tex) are the three manuscript roots.

## Scientific scope and organization

The four-page main follows measurement, observed training concentration,
conditional mechanism, and matched replay intervention. Its three figures are
shared with ICLR: concentration changes, the objective-by-replay factorial,
and uniform-versus-frequency key weighting. The hosted overview and
within-Level-2 effect figure remain supplementary, with their main-text findings
and limits stated briefly.

The twenty supplementary figures retain every original scientific display.
The appendix begins with the claim/evidence map, measurement, and full
conditional theory, followed by benchmark construction, prompts, algorithm,
controlled results, longitudinal diagnostics, hosted observations, prompt
interventions, related work, and reproducibility. All eighteen formal
statement/proof blocks match the pre-reorganization manuscript exactly.

Level-1 primary comparisons cover Qwen2.5-0.5B, Falcon3-1B, and Qwen2.5-3B.
All five accuracy (`pass@8`) and verified-mode (`distinct@8`) trajectory figures
remain: the four primary methods across all three scales, the Level-2
Qwen2.5-0.5B factorial, and UCPO/RLEP comparisons at 0.5B and 1B only.
Fixed terminal cohorts, unavailable checkpoints, and partial histories remain
explicit. The 3B MaxRL comparison is complete, with five paired seeds in all five domains.

Results use the frozen September 11 endpoint census: 74 core replay pairs,
75 MaxRL replay pairs (including all 50 Qwen2.5-3B endpoints), four complete
Level-2 domains (80/80 endpoints; PantryPlan remains at 10/20), and eight
complete weighting-ablation blocks. The original Qwen-0.5B weighting analysis remains
frozen. The [figure manifest](../FIGURE_MANIFEST.md) and
[evidence audit](../FIGURE_DATA_AUDIT.md) give exact populations.

Duplicate completion updates and seed tables, the old bundled semantic
experiment, AUC sensitivity figure, telemetry figure, and extended theoretical
arguments are preserved in the [streamlining archive](../audits/streamlining_20260911/README.md).
The fixed-semantic baseline, weighting control, source exclusion, and
fixed-bank score deterioration remain in the active paper.

## Build and source bundle

```sh
make -C paper/mathai2026 bundle
```

The build runs in a temporary directory and promotes a PDF only after checking
four content pages, all 23 figures, references on page five, anonymous official
style, source hashes, and absence of unresolved references or overfull boxes.
A failed build preserves the last validated PDF and saves `main.failed.log`.
The source ZIP is independently checked before replacement.

Before rebuilding after intentional edits, synchronize figures, nested inputs,
and required numerical provenance from the parent:

```sh
python ops/sync_paper_workshop_assets.py --date 2026-09-11 \
  --audit-directory paper/audits/narrative_reorganization_20260911/workshop_next \
  --reason 'Synchronize the reorganized papers while preserving frozen numerical evidence' --apply
```

Choose a new audit directory for each later synchronization. The updater
preserves prior bindings and replaced assets; the current source package binds
active referenced assets and scientific snapshots. Archived experiments remain
available in the research archive without entering the active submission ZIP.

Upload `mathai2026-source.zip` to Overleaf and select `main.tex` as the main
document. Earlier editorial history, template notes, and revision records are
preserved in [the previous README](../audits/streamlining_20260911/before/paper/mathai2026/README.md).

## Hosted deployment comparison

The supplementary hosted overview shows all seven deployments, five domains
and three levels. Its main-text summary follows the controlled replay results. Frozen formatting normalization applies to every cell; only
Opus 5 Python uses the separately evaluated revised task wording. All eight
responses per prompt count, so the display does not select successful draws.
The original cohorts, native refusal counts, strict/normalized results and
prompt comparisons remain in the appendix. The source record and figure
sidecar bind both evidence sets and each displayed denominator.

The GPT-5.6 Sol temperature curve is a supplementary figure here,
and is also supplementary in the long paper.
All four temperature conditions use reasoning `none` and retain 960 draws each.
Its two panels plot empirical `pass@8` against mean `distinct@8`, overall and
by level. `pass@8` is the fraction of prompts with at least one correct answer
in their eight saved draws; failures remain included. The normalized aggregate
peaks among sampled temperatures at 1.5 (72.50% `pass@8`, 1.050 modes), an
observed result rather than an established optimum. The original
medium-reasoning reference remains separate and unconnected (97.50%, 1.550).
The source is `GPT56_PASS8_FRONTIER.{json,md}` in
`artifacts/frontier_temperature_20260911/`; the per-response report is preserved.
The appendices also include Grok/Kimi temperature sensitivity and the separate
three-request Python retry diagnostic, without changing the fixed-draw main
comparison.

## Claim and theory alignment

Both papers share a claim-by-claim evidence map, the success–breadth lemma, and the complete conditional theory. The [correction and verification record](../audits/claim_theory_alignment_20260911/README.md) explains the assumptions and the construction-reserve versus terminal-test distinction in the Level-2 comparison.

## Matched prompt-hint control

The supplement includes the completed 27,648-response local prompt ablation:
initial Qwen2.5-0.5B-Instruct plus 24 archived Dr.GRPO/ReplayDr.GRPO checkpoints,
Python factors/MathIR/Pantry, and Levels 2 and 3. Original and neutral wording
use identical problems and within-model sampling settings. The new figure and
tables report all cells, paired pass@8 and distinct@8 effects, additional-mode
effects, and the smaller two-seed Pantry scope. Level 3 is transfer evaluation.

The frontier panel remains uncollected pending an API credential; its omission
and the incomplete overall experiment are explicit in the appendix and result
JSON. The copied local result and PDF/PNG/JSON figure companions are bound by
`snapshot.json`. Standalone source builds validate these files and exactly
twenty-three figures, with twenty in the supplement; they require no repository
inputs or model calls. The parent build additionally reconstructs the statistics.

The [completed reorganization record](../audits/narrative_reorganization_20260911/README.md) records figure placement, source preservation, both builds, and visual review.
