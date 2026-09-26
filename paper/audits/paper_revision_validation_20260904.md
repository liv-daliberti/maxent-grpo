# Paper revision validation — September 4, 2026

The workshop title is **More than One Way to Skin a Cat: Preserving Verified Modes in RLVR**. The canonical-key explanation now links directly to Appendix B.1, where each domain's identity and alias rules are explained. The original ICLR title is retained.

Both papers follow the same argument: observable loss of sampled alternatives; executable identity; replay's role and conditional theory; held-out gains; a stronger fresh-objective comparison; and a controlled harder-task result. Repeated closing recaps were removed from the full version to make room for the direct weighting ablation and explicit evidence boundaries.

## Completed checks

- Workshop `make bundle`: four main-content pages, six main figures, seven supplementary figures, references beginning on page 5, anonymous mode, unchanged official style, resolved references, and no overfull boxes. All four main pages were rendered and visually inspected. The source archive contains 42 files. A fresh extraction into an independent temporary directory also passed `make all`, including the four-page and asset checks.
- ICLR full `make` with the temporary datasets-capable interpreter: frozen-domain prompt verification, scientific evidence contract, bibliography and cross-reference resolution, and final-line fill all pass. The conclusion is on page 9; 174 natural-prose blocks satisfy the 50% final-line minimum. The last main page was visually inspected.
- Focused source/reporting regressions: **27 passed** across `test_paper_core_endpoint_integrity.py`, `test_e118_integrity_seed_tracks.py`, and `test_paper_partial_block_reporting.py`.
- The model-size theorem section is byte-identical between versions and both main links resolve. The [numerical verifier](verify_model_size_theorem_20260904.py) checks 30 logit cases and 24 parameter-replication comparisons; no monotonicity or bound violation occurred. The largest residual is about 1.27e-10; [machine results](verify_model_size_theorem_20260904.json) and [text output](verify_model_size_theorem_20260904.txt) are retained.
- `git diff --check` passes for the changed manuscripts and result-reporting code.

A broader, nonrequired grid-style test also ran during the source-reader work. `test_paper_grid_contract.py::test_graph_generators_do_not_request_one_axis_grids` fails on an existing `axis.grid(axis="x")` call in `plot_e118_cross_domain_only_preview.py`, which this revision does not edit or use for the compiled figures. This is separate from the passing paper build and targeted scientific-integrity tests.

## Scientific correction and limits

The [evidence audit](story_evidence_audit_20260904.md) and its source records document why one ambiguous Falcon Countdown ReplayDr.GRPO run is excluded. Corrected primary results contain 74 admissible pairs: 14 complete five-seed blocks and one four-seed block. All ten complete MaxRL replay pair blocks remain intact. The partial Falcon Dr.GRPO macro uses the same four seeds across all domains, with descriptive means and no five-seed interval. The workshop snapshot preserves superseded asset hashes and records the reason for each correction.

The [supplemental reported-comparator audit](comparator_source_integrity_20260904.md) also passed: 281 admitted source files and 356 requested checkpoints, including all plain-GRPO initial/terminal pairs and the other reported comparator inputs. It found no additional duplicate, missing-data, or source-binding issue and required no artifact regeneration.

No raw experimental observations, registered stopping rules, or training runs were changed. No new model evaluations were selected. The papers distinguish the conditional mathematical mechanism from a parameter-count law, endpoint improvements from individual-mode survival, and completed results from unfinished experimental progress.

## Final artifact hashes

| Artifact | SHA-256 |
|---|---|
| `paper/main.tex` | `10278cfdb5c84ed57c69a5a1d1327f0e624a4692be15ed1dcd0a3dc07bd21489` |
| `paper/main.pdf` | `c44beca1686173b4929a58d820b18ea020c7085f5c9e71bcf4542640409b497e` |
| `paper/mathai2026/main.tex` | `f435872d388b00216d738839834b705ace884e8293aafdfe79a0a16a5d4dd7c3` |
| `paper/mathai2026/appendix.tex` | `cb0697879f642667f317125e194b812e35683c08b5abc737d6558699cd06f616` |
| `paper/mathai2026/main.pdf` | `30887c6db38707199ce0c246286ef5498f87e17061de8ae5cffb3106c59edf95` |
| `paper/mathai2026/mathai2026-source.zip` | `b88af2ec5c8c9d56f97480a0cdee931d515836d2e3cc8341bbcf51fc0f893574` |
