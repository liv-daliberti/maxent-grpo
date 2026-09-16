# September 11 paper figure refresh

Both manuscripts use the same 14 compiled figures: six main figures and eight supplementary figures. Figure 5 puts Qwen2.5-0.5B, Falcon3-1B and Qwen2.5-3B on the same two main axes, each with MaxRL/ReplayMaxRL and Dr.GRPO/ReplayDr.GRPO tracks. The Qwen-3B MaxRL track averages the five domain-specific paired means equally (Graph/Countdown/Python/MathIR/Pantry n=5/3/5/3/1). Its dashed descriptive track has no shared cross-domain seed, seed paths or interval. The appendix covers all three scales by domain on both metrics.

Figure 6 retains frozen admission and compares the four complete terminal domain factorials across both levels and all four methods. Its standalone domain companion shows available partial arm means with explicit non-paired interpretation. The illustration remains mechanically selected from its original source set; benchmark/method diagrams and all supporting figures were regenerated with their scientific scopes intact.

The dated endpoint census admits E118 138/150 endpoints and 67 pairs, E119 88/100 endpoints with four complete domain factorials, and E120-R1 43/45 endpoints. The historical E120 primary input retains SHA-256 `92304ed9ac70f6ebc4dd38e12801750175b96703bf657459dafed3c390bf2bab`.

## Reproduction and review

- `make -C paper figures` renders the current compiled set; supporting figures read retained numerical JSON.
- `make -C paper latest-results` regenerates the dated report/tables from the retained audit. Live collection is a separate target.
- `ops/sync_paper_workshop_assets.py` preserves replaced assets and snapshot history before binding the two manuscripts.
- Figure inventory and model/level coverage are in `paper/FIGURE_MANIFEST.md` and `paper/FIGURE_DATA_AUDIT.md`.
- Final focused scientific-integrity suite: 59 tests passed. It covers endpoint admission, paired subsets, missing draws, terminal cohort selection, descriptive weighting, and comparator coverage.
- Figure 5, its three-model appendix, Figure 6, its companion, Figure 4, and the workshop main results page were visually reviewed. All 14 shared PDFs match byte-for-byte between manuscripts.
- Final long-paper build passed the evidence contract, domain-prompt checks, and all 320 paragraph layout checks. Final workshop PDF and source ZIP passed the four-page, 14-figure, reference, anonymity, source-hash and compiler checks.
