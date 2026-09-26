# Frozen 0.5B level comparison: five domain panels

Figure 3 now appears in Methods/benchmark construction (ICLR Section 2.2,
page 3; workshop page 2). Five side-by-side panels show Graph, Countdown,
Python, MathIR and Pantry. The horizontal axis is pass@8; the vertical axis
is mean distinct verified modes per eight draws. A filled circle represents
one measured level of frozen Qwen2.5-0.5B-Instruct; color identifies the level.

Each panel uses the exact corresponding Figure 2 domain tint, imported from
its `DOMAIN_PANEL` mapping. The figure sidecar binds that palette source,
native receipts, frozen validator/runtime, reconstructed metrics and PDF/PNG.
No admission-band shading or historical development markers remain.

## Data freeze and continuing collection

The immutable publication source contains eight complete cells (32,768 native
responses): all five Level-1 domains plus Level-2 Graph, Countdown and Python.
Each cell has 128 held-out prompts and four independent groups of eight draws
at temperature 1. Groups are scored separately before averaging; failed groups
remain included. Missing cells have no plotted values and are labeled pending.
See `plotted_points.csv` for the exact coordinates. These held-out measurements
are separate from the historical development admission gate.

The broader four-model collection continues. At 02:24 UTC, 16/100 cells were
complete and authenticated: nine 0.5B and seven 14B cells, 65,536 responses.
0.5B Level-2 Pantry and 14B Level-2 Python were running; 3B and 7B Level-1–3
jobs were queued. Forty Level-4/5 cells await dataset admission. The later
collection status is retained separately from the eight-point figure freeze.
A later figure refresh should create a new immutable source index, update its
coverage/caption, and rebuild from the same native-validation pipeline.

## Verification

Twelve construction-integrity tests passed, covering native summary tampering,
canonical-set recomputation, the exact plotted coordinates, model scope,
missing-cell handling, panel offsets and source/output bindings. The full
scientific paper contract passed, including reconstruction of the unchanged
480-prompt, 19,200-response GPT temperature study.

The final ICLR PDF was compiled in isolation after recording every local TeX,
figure, bibliography and style input. Inputs remained unchanged through the
build and validation. All eight main figures fit within nine main pages;
References starts on page 10. All 354 prose blocks pass the final-line check.
The workshop independently passes four main pages, eight main and nineteen
supplementary figures, source/output hashes, references and overfull-box checks.
Its standalone ZIP and both ICLR review PDFs are refreshed.

`publication_verification.json` binds the final source inputs, output PDFs,
figure companions and collection status. Build/test logs and reviewed pages
are retained here. `appendix_line_fill_before/` preserves three wording-only
caption/prose fixes for a concurrently added appendix; its numerical JSON and
artwork were verified unchanged. The superseded Level-1/2 admission plot and
manuscripts are preserved in the neighboring construction-restoration audit.
