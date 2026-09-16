# GPT-5.6 Sol: expanded 480-prompt temperature figure

The approved expansion retains the original 120 prompts and 4,800 responses,
adds 360 prompts and 14,400 responses, and uses the same 480 prompts at
T=0,0.5,1,1.5,2 with eight draws each. All 19,200 native responses authenticate.
The new prompts extend the original outcome-independent hash ranking to 32
prompts in each of 15 domain/level cells. Both strict and frozen formatting-normalized
grades retain zero-success groups and unsuccessful responses.

The normalized overall pass@8 values are 65.625%,72.2917%,75.2083%,79.7917%,79.375%;
distinct@8 is 0.672917,0.839583,0.9625,1.125,1.145833. T=1.5 has the highest observed
pass@8 and T=2 the highest observed breadth. Paired 20,000-replicate whole-prompt
bootstrap intervals for T=2 minus 1.5 include zero on both metrics. Separate
original 120/additional 360 comparisons remain in the numerical appendix.

The deployment rejected T=2.5 and non-default temperatures with medium reasoning.
Historical medium measurements cover only 120 prompts and are excluded from the
expanded figure and tables. Previous measurement files and reports are preserved.

Authoritative data and frozen analysis are under
`artifacts/frontier_temperature_20260911/`, with the suffix `EXPANDED480` and
`gpt56_expanded480_analysis_code_v2`. The additive experiment verification record is
`prompt_expansion_32_per_cell/final_verification_20260912T010953Z/verification.json`.
It records 19 passing analysis tests and independent inventory/arithmetic checks.
The renderer has 24 passing tests, including rejection of a missing-zero grid with
falsely adjusted response counts. This directory retains tests and build evidence.

Both manuscript captions, numerical discussions, appendix tables, and the temperature figure use
the expanded data. Figure labels were inspected after rendering. The workshop
PDF and standalone source bundle pass all submission checks at four main pages.
The longer-paper build is tracked in `build_coordination.json`; its final validation
will be recorded alongside this audit after completion.

The separate all-model/all-level Qwen figure remains in collection. L1–L3 use
admitted historical datasets; L4/L5 cells await admitted datasets. No pending or
prospective dataset is represented by an invented measurement.

Final publication verification is retained in `publication_verification.json`. The expanded native-data and current-paper evidence contracts pass; the refreshed long PDF has nine main pages, the workshop PDF has four, and their temperature assets match. The main-only PDF and workshop source bundle are refreshed. The separate Methods grid remains pending; the latest authenticated snapshot contains 9/100 cells.
