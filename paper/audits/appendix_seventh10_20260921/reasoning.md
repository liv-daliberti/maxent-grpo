# Reasoning-control boundary review

Edited `hosted_reasoning_off_20260912_main.tex`, the complete connected appendix fragment, and the prose/caption rendering in `ops/build_paper_hosted_reasoning_off.py`. The incoming page 88 began this material; after reflow it occupies pages 89–90. Both numerical tabular environments are byte-identical. The publication JSON changes only its renderer digest; the PMD result record is unchanged.

## Corrections

- Removed admission/completion, reused-audited-response, source-record, authentication, and retention narration. Described the actual first-32-per-cell population, outcome-independent selection, two sets of responses, original Python wording, prompt controls, output cap, and provider configurations directly.
- Kept the material timing limitation: the medium and disabled responses come from different periods, default sampling settings can vary over time, and corresponding sample indices are not shared random seeds. Provider reasoning labels do not establish the absence of hidden computation or matched compute.
- Both captions distinguish unconditional prompt success/raw distinct-mode counts from paired conditional diversity. Success and raw modes average equally across the balanced domains/levels. PMD pools jointly eligible prompts and weights each such prompt equally; neither level PMD nor overall PMD is an equal-domain macro. The overall PMD is not a simple mean of the three level entries.
- Level PMD uses 65–159 jointly eligible prompts; overall PMD uses 245–458. The thirty-prompt threshold applies to the displayed level/overall population, not to every domain separately. Joint eligibility does not fix overall accuracy or represent prompts only one condition solves twice.
- Replaced categorical claims of unchanged diversity with the actual small positive estimates: Opus 4.8 rises from .202 to .209 overall, with level changes approximately .0002–.0142. GPT-5.4 rises by .009 at Level 1 and falls at Levels 2/3, so the old assertion that no model changes direction across levels was false. These are descriptive point estimates without uncertainty intervals.
- Preserved the overall decreases in pass@8 and distinct@8 for all five displayed deployments. PMD decreases for the two GPT deployments and increases for Grok and Kimi; the shared decrease in raw mode counts does not imply a shared increase in conditional concentration.
- DeepSeek has 3,839 responses and one seven-response Level-1 PantryPlan prompt. The missing request has an unknown outcome, not a failed score. Its strict-grading PMD is .282/.369 on 437 jointly eligible prompts, difference +.087. Independent review found that the incomplete prompt has only one correct answer and is excluded from this estimate. The text now distinguishes incomplete cohort coverage from the actual complete-response groups entering the conditional estimate.
- Opus 5's thinking block and forty thinking tokens under an explicit disabled-thinking request are a control violation. No validated reasoning-disabled contrast is claimed for that response cohort.

## Validation

Rebuilt all 300 model × grading × condition × domain × level success/mode aggregates from existing prompt counts. Recomputed the complete paired PMD record from existing per-response grades and keys; it matches exactly, including eligibility, weights, values and exclusions. Both final fragments exactly match their renderer. All table numbers, source estimates and intervals are unchanged. No new model responses, experimental runs or bootstrap draws.

An independent read-only review checked the estimator, populations, directions, provider settings, uncertainty and interpretation, and prompted the DeepSeek eligibility clarification above. `reasoning_validation.json` records the final checks and hashes.
