# September 10 manuscript results revision

The user requested integrating the additional experiment results. Both the ICLR manuscript and the distinct four-page MathAI workshop manuscript now use the September 10 endpoint census and refreshed Figures 5 and 6.

The census adds 19 admitted terminal endpoints relative to September 9: E118 gains 11 (137/150), E119 gains 2 (85/100), and E120 gains 6 (41/45). E118 now contains 12 complete MaxRL/ReplayMaxRL blocks and 66 matched pairs; the common four-arm intersection contains 11 complete blocks and the retained four-seed Falcon Countdown block. E119 still has four complete domain factorials. E120 now has seven complete, mechanism-validated blocks.

The newly complete Qwen-3B Graph result adds .163 extra modes with a descriptive paired 95% interval [.021, .305]; its correctness and raw-support intervals include zero. E119 Pantry contributes its first descriptive ReplayDr.GRPO pair, seed 46. The complete Falcon Pantry weighting contrast is inconclusive: uniform-minus-frequency extra modes are -.146 [-.564, .414], replacing the earlier three-seed directional mean. Qwen-3B weighting extensions retain their exact partial seed counts without intervals.

Figure 6 now has 22 matched domain-seed cells. Pantry contributes seed 43 at step 672 and seed 45 at step 0, each matched across both levels and all four methods; the other four domains contribute five terminal seeds each. No incomplete Qwen-3B MaxRL aggregate was added. The original September 4 E120 primary analysis and all prior admitted endpoint values remain unchanged. Level-3 jobs supply no terminal evidence to this update.

## Validation

- Dated data: 69 contrasts and 276 metric summaries independently checked; all 14 generated outputs reproduce byte-for-byte. See `../data_generation_validation.json` and `../data_reproduction_verification.json`.
- Numerical manuscript review: `../manuscript_numeric_review.json`.
- Reporting, endpoint integrity and frozen-primary tests: 39 passed (`reporting_tests.log`). Figure tests: 18 passed (`../figure_refresh/validation.json`).
- Main PDF: required scientific contract, dataset prompt checks, TeX/BibTeX and rendered line-fill checks pass (`paper_build.log`). Final log has no undefined references, multiply defined labels or overfull boxes. Pages 60, 62 and 64 were visually reviewed.
- Workshop: validated four-page main text, six main figures, eight supplementary figures, references on page 5, anonymous style, current source/PDF hashes and a rebuilt 93-file source ZIP. Root visually reviewed main-text page 4; workshop-specific receipts record the appendix review.
- Pre-edit source copies are under `before/`; final parent output hashes and checks are in `parent_outputs_sha256.json` and `parent_validation.json`. No training configuration or scheduler operation was changed during this paper update.

## Current artifacts

- Main manuscript: `paper/main.pdf`.
- Workshop manuscript: `paper/mathai2026/main.pdf`.
- Rebuilt standalone workshop source: `paper/mathai2026/mathai2026-source.zip`.
- Readable numerical report: `paper/results/current_campaign_results_20260910.md`.

The older `mathai2026-overleaf.zip` remains historical; use the rebuilt source ZIP for this revision. Reproduction commands in the paper README and figure manifest read the retained dated audit. A new live collection must use a new analysis date.
