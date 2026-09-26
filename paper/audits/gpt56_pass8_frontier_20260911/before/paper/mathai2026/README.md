# MATH-AI NeurIPS 2026 workshop version

**More than One Way to Skin a Cat: Preserving Verified Modes in RLVR**

[main.pdf](main.pdf) is the anonymous submission: four pages of main content,
then references and supplementary material. The official workshop style remains
unchanged. [main.tex](main.tex), [preamble.tex](preamble.tex), and
[appendix.tex](appendix.tex) are the three manuscript roots.

## Scientific scope and organization

The main paper presents ModeBench and the objective × replay comparison.
Its seven main figures are shared with the long paper. The supplement contains eleven
figures and follows benchmark/protocol, algorithm, current results, mechanism
checks, core proofs, and reproducibility. It supplies a concise formal metric
primer and related work without repeating the long main paper's results story.
The hosted-model observation follows the three training results. Figure 7
shows per-response accuracy and verified modes at each level, using frozen
normalized grading and the complete revised Opus 5 Python condition. The
caption identifies that condition; original scores and prompt details remain
in the supplement.

Level-1 primary comparisons cover Qwen2.5-0.5B, Falcon3-1B, and Qwen2.5-3B.
All five accuracy (`pass@8`) and verified-mode (`distinct@8`) trajectory figures
remain: the four primary methods across all three scales, the Level-2
Qwen2.5-0.5B factorial, and UCPO/RLEP comparisons at 0.5B and 1B only.
Fixed terminal cohorts, unavailable checkpoints, and partial histories remain
explicit. Figure 5's partial 3B track is descriptive.

Results use the frozen September 11 endpoint census: 74 core replay pairs,
67 MaxRL replay pairs, four complete Level-2 domains, and seven complete
weighting-ablation blocks. The original Qwen-0.5B weighting analysis remains
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
four content pages, all 18 figures, references on page five, anonymous official
style, source hashes, and absence of unresolved references or overfull boxes.
A failed build preserves the last validated PDF and saves `main.failed.log`.
The source ZIP is independently checked before replacement.

Before rebuilding after intentional edits, synchronize figures, nested inputs,
and required numerical provenance from the parent:

```sh
python ops/sync_paper_workshop_assets.py --date 2026-09-11 \
  --audit-directory paper/audits/streamlining_20260911/workshop_sync \
  --reason 'Streamline both supplements while preserving the frozen primary evidence' --apply
```

Choose a new audit directory for each later synchronization. The updater
preserves prior bindings and replaced assets; the current source package binds
active referenced assets and scientific snapshots. Archived experiments remain
available in the research archive without entering the active submission ZIP.

Upload `mathai2026-source.zip` to Overleaf and select `main.tex` as the main
document. Earlier editorial history, template notes, and revision records are
preserved in [the previous README](../audits/streamlining_20260911/before/paper/mathai2026/README.md).

## Hosted deployment comparison

Figure 7 shows all seven deployments, five domains and three levels after the
training results. Frozen formatting normalization applies to every cell; only
Opus 5 Python uses the separately evaluated revised task wording. All eight
responses per prompt count, so the display does not select successful draws.
The original cohorts, native refusal counts, strict/normalized results and
prompt comparisons remain in the appendix. The source record and figure
sidecar bind both evidence sets and each displayed denominator.

The GPT-5.6 Sol temperature curve is a supplementary figure here (page 30),
with a brief main-text finding; it is Figure 8 on page 9 of the long paper.
All four temperature conditions use reasoning `none` and retain 960 draws each.
The original medium-reasoning reference remains separate and unconnected.
The appendices also include Grok/Kimi temperature sensitivity and the separate
three-request Python retry diagnostic, without changing the fixed-draw main
comparison.
