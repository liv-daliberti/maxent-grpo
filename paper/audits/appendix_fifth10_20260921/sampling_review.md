# Second review: K.4 and Figures 32–34

Read-only review of K.4 in main.tex found no remaining material error in the checked interpretation. Source: paper/results/modebench_discovery_curves_20260911.json; aggregation implementation: ops/analyze_modebench_discovery_curves.py.

- Twenty-five local checkpoints contribute 110,592 responses. Strict and normalized per-prompt records are identical for every local checkpoint.
- The 76 jointly eligible Level-2 Re:Dr MathIR groups at m=8 contain 16 unique problems reused across five checkpoints; per-seed counts are 16, 15, 15, 15, 15 for seeds 43–47. Both wordings have R8=1. DrGRPO's corresponding count is 40, so keeping the 76 sentence attached specifically to Re:Dr is essential; the current wording does.
- The only trained MathIR group with more than one observed correct key is original-wording Level-3 DrGRPO at seed 46, row 60, with mode counts [28, 9]. Its equal-seed collision aggregate is 0.9802894016425497. The other seven trained MathIR level–wording–method cells have collision 1.
- Neutral DrGRPO Python has zero correct responses from 5,120 at Level 2, and three from 5,120 at Level 3. The three successes occupy three separate checkpoint–problem groups, each with one success; no eligible correct pair exists.
- Collision pools pairs within each checkpoint and then averages checkpoint ratios equally. PMD averages eligible prompts within checkpoint and then checkpoints equally. Own-wording and joint-wording populations are correctly distinguished. The revised text does not claim matched accuracy from fixed-correct-draw conditioning.

One material caption ambiguity was corrected with root authorization: dotted uniform references in Figures 32 and 34 are computed only for the strictly eligible paired population. Normalized crosses can have a different eligible population, and the plot does not draw a corresponding normalized reference. Both captions now specify strict grading in both wordings and the same prompt/checkpoint weighting as the strict curves.

Only that sentence was changed twice in the TeX fragment and once in its shared `discovery_caption` emitter. Rendering the saved report gives the exact current fragment. All tabular environments, display equations, labels, statistical functions, and plot functions are unchanged. Source report SHA256 remains 52253cf29925a35408350e2c97f0c689c2aa7e9672bf1ddd63a63acd1df10483. Before copies of the emitter and fragment are saved here.
