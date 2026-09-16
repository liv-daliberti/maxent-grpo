# Frontier evaluation continuation — September 11, 2026

Resumed the saved **Test mode effects on ChatGPT** thread (01a08f34-b9f6-7c12-b452-1ceb5d49d1f1). The original thread continued the manuscript builds while this continuation independently verified results and layouts. No new API calls were made here.

## Completed deliverables

- Long paper: `paper/main.pdf`, with nine main pages; all seven models in Figure 7 on page 8 and the GPT temperature curve in Figure 8 on page 9.
- Main-only reader copy: `paper/main-body.pdf`, refreshed from the first nine pages of the validated PDF, with matching text and a passing structural check.
- Workshop: `paper/mathai2026/main.pdf`, four main pages, all seven models in Figure 7; GPT temperature curve in the appendix (Figure 17, page 30).
- Workshop source bundle: `paper/mathai2026/mathai2026-source.zip`, current validated source and assets.

The continuation separated the remaining overlapping temperature labels and repaired the figure inventory's Markdown and counts. Original-thread fixes to captions, TeX escaping, interval explanations and build metadata were preserved. Final rendering and both source bindings include the figure correction.

## Evidence and interpretation

The four-temperature GPT curve retains all 3,840 responses, uses reasoning `none`, and keeps the earlier medium-reasoning reference unconnected. Normalized accuracy falls from 55.52% to 50.21% as temperature rises from 0.5 to 2.0; distinct correct modes per eight draws rise from 0.775 to 1.042. Paired pointwise 95% intervals are +0.267 modes [0.183, 0.350] and −5.31 accuracy percentage points [−8.33, −2.29]. This is a sampling tradeoff within the no-reasoning condition, not a temperature sweep of the original medium-reasoning condition. Strict accuracy has a smaller, uncertain endpoint difference; the grading convention must remain explicit.

The complete revised Opus Python cohort remains the main-display condition. Three later retries recovered valid responses but no additional modes; selected-set correctness is conditional on the retry rule. Original failures and every retry remain saved separately.

## Validation

105 focused tests passed. The current-paper evidence contract, nine-page bound, 255-block prose line-fill check, workshop submission validation, and standalone source packaging passed. An independent audit reproduced 640 GPT estimates/intervals and verified 91 source bindings; retry occupancy also matched. See `receipt.json` for delivered file hashes.

No required manuscript or evaluation work remains for this handoff. New experiments or stronger causal claims would be a separate task.
