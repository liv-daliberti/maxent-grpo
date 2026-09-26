# Fourth appendix prose and caption pass — 2026-09-21

## Scope

Reviewed the next ten pages after the third pass: incoming PDF pages **48–57**. Concurrent additions elsewhere in the manuscript shifted this content to **49–58** in the final PDF. The span covers the baseline/control summaries, fixed-bank exemplar trajectories, prompt-hint ablation, decoding sweeps, and the beginning of the sampling-budget ablation. The connected hosted-discovery interpretation and companion discovery captions were cleaned together. The next consecutive pass should start on final PDF page **59**, with the all-level GPT sampling experiment and the remaining discovery figures/tables.

Removed editorial/process narration, retrospective chronology, post-hoc headings, registration and archive references, dataset-admission history, source-artifact defenses, and figure text about reproduction history. Retained actual experimental design, populations, missing estimates, uncertainty, and limitations. Figure captions now begin with a bold supported finding, followed by setup, visual encodings, and the statistical information needed to read them. Changed embedded figure text and corresponding generators are synchronized.

## Corrections and qualifications

### Baselines and fixed-bank trajectories

- Tables 20 and 21 identify the untrained checkpoint as **Initial**, replacing **Frozen**. Their numerical entries are unchanged.
- Figure 27's embedded heading now states the supported reduction in average extra verified modes at every tested scale. It no longer implies that every domain exhibits the same accuracy–breadth tradeoff. Plotted data and geometry are unchanged.
- H.5 directly specifies the fixed-bank design: membership is fixed before step 384, no further modes are added, and replay continues through step 3,072. All 3,406 exemplars, 1,575 seed–prompt pairs, 29,041 visits, and 8–9 visits per exemplar were checked against the existing records.
- Token-score median change +.038, tenth percentile −.256, 2.61% below −.5, sequence-score median +.154, and associated intervals were checked. Mean-token and sequence-score units are distinguished, including exemplar lengths of 4–100 tokens. No measurements were changed.
- Figure 28 now describes the observed rising median and declining tail. It identifies equal-exemplar weighting, seed and pooled curves, zero/−.5 reference lines, and teacher-forced scores. These scores do not directly measure the probability of generating each canonical solution mode.
- Bootstrap wording matches the existing calculation: 10,000 replicates, prompt resampling within original seeds followed by five seeds with replacement, repeated-seed reuse of the within-replicate prompt sample, and linear percentile interpolation.
- The lack of a matched no-replay fixed-bank control and the observation of scores only at replay visits remain explicit. These trajectories do not establish the causal replay effect or validate the theorem's probability floor.

### Prompt-hint ablation

- The Python intervention jointly removes an executable nested-conditional example, divisor/dispatch guidance, and prompt length. The text no longer treats it as an isolated preference-cue intervention.
- Conditional diversity in Table 22 weights eligible checkpoint–problem groups equally. It is not an equal-seed mean: checkpoints with more eligible problems receive more weight. Only five of eighteen cells meet the threshold of thirty jointly eligible groups. The same 32 problems recur across checkpoints; group counts are not counts of unique problems.
- The two Python replay cells contain 66 and 57 eligible checkpoint–problem groups. At Level 3, seed 46 has one verified neutral output versus 256 original outputs, so it contributes no conditional-diversity groups. Accuracy and breadth still average all five checkpoints. No confidence intervals for these conditional-diversity point estimates are supplied by the existing analysis, and the prose does not imply otherwise.
- The Python failure paragraph now reports the categories directly. All 5,120 responses, the repeated extracted lambda, 2,320 verified original responses, and neutral-wording rejection counts were checked. Categories follow the first rejection under the extractor/parser/executor; they do not identify an internal training mechanism. Responses reaching the token limit without an extracted candidate remain counted.
- Observed MathIR collision concentration is distinguished from its uniform reference over five certified modes. Concentration in the sampled outputs does not imply that alternative valid modes are unreachable or that only one exists. The 4,179/4,382 correct-pair counts and 154/157 eligible-group counts are preserved.
- Figure 29 uses the main-body caption style, identifies the strict/normalized marks, paired effects, seed-specific uncertainty, and Pantry's two-checkpoint limitation. Its legend now says **Normalized**. Every plotted measurement is unchanged. Shorter table labels improve the table's printed size without altering its numerical columns.

### Decoding comparisons

- J now describes **Decoding Temperature, Nucleus, and Budget**, with the matching contents title.
- The eight-pass comparison comprises 300 checkpoint–setting evaluations: five domains, two methods, five seeds, and six temperatures. Four eight-response groups form each observed pool, with overlapping streams as described in H.1. The means are descriptive finite-pool estimates.
- Eligible prompts and seeds can differ across settings and arms. Python controls have no seed meeting the thirty-prompt reporting threshold; the missing statistic is not evidence that all alternative solutions have zero probability.
- All stated control maxima and default replay means were checked against the numerical records. The twelve paired A100/A5000 comparisons have mean absolute difference .0032 and maximum .0131; these observations do not bound hardware sensitivity generally.
- The separate twelve-pass comparison has 550 checkpoint–setting evaluations. Its larger-group settings increase group size fourfold but total responses per prompt only twofold, from 32 to 64. The old wording conflated these budgets.
- This cohort uses a different replay objective, combining separate mass and within-bank balancing terms with novelty and semantic entropy, as well as a longer training schedule. It is not an isolated comparison of the eight-pass Re:Dr objective.
- Figure 30 explicitly describes setting ranges rather than confidence intervals, eligible-seed means, Python's differing replay seed counts, and the minimum displayed range width of .004. Its embedded labels no longer narrate experiment history.
- Evaluation-time sweeps do not identify the effects of changing rollout temperature, optimizer settings, or training duration.

### Sampling budget and hosted interpretation

- K.1 directly defines the 64-response pools, finite-pool rarefaction, discovery gains, and correct-draw rarefaction. Incorrect, empty, refused, and truncated responses remain in the sampled pools.
- Holding the number of sampled correct outputs fixed does not equate model accuracy. Eligibility changes with the correct-draw budget, wording, method, and grading rule. Absolute estimates use each wording's own eligible population; paired differences use jointly eligible prompts.
- Separate eligibility counts are not the size of the joint population. Equal-checkpoint means and conditional bootstrap rules are described according to the existing implementation.
- Certified support sizes remain lower bounds. Their uniform-distribution references are not estimates of exhaustive model support or bounds on observed model breadth.
- All five numerical table environments, both displayed equations, and ten labels in the discovery fragment are byte-identical to the incoming fragment. The statistical and plotting functions are unchanged. The emitter reproduces the revised fragment from the existing report.
- Figure 31 identifies the axes, deployments, original/neutral line styles, strict/normalized encodings, sixteen problem pools per cell, and pointwise paired-problem bootstrap intervals. The three companion captions were revised consistently, including those beyond this ten-page boundary.
- The hosted interpretation directly states 36,864 responses, three deployments, six cells, two wordings, 64 draws, medium reasoning effort, and an 8,192-token output cap. These settings differ from the reasoning-disabled main-table evaluation.
- MathIR's fixed-correct-draw contrasts use 14/15 jointly eligible Level-2 problems for GPT-5.4/GPT-5.6 Sol and sixteen each at Level 3. The text preserves their intervals and explains the selected population. Pantry's late discoveries and Python's strict/normalized formatting effects retain their existing numerical values.

## Files and validation

The primary source is `paper/main.tex`, with edited fragments `modebench_prompt_ablation_20260911.tex`, `modebench_discovery_curves_20260911.tex`, and `discovery_hosted_interpretation_20260912.tex`. Corresponding prompt/discovery emitters and the Figure 27, 28, and 30 plotting sources were updated. Figure 29's embedded legend was regenerated through its existing emitter.

- Independent reviews checked prose, populations, calculations, and figure text against local records and implementation. No experiments were run, and no training code was changed by this pass.
- All ten final target pages were visually inspected. The final two pages were rechecked after tightening a sentence split around Figure 31.
- Main-length contract passes: nine scientific main pages, all twelve main figures, references starting on page 11. The complete final PDF has 122 pages.
- All 91 numbered appendix headings have contents entries. The new concurrent B.5 subsection was added to the manual contents.
- No unresolved references/citations, duplicate labels/destinations, overfull boxes, or wrap collisions. Two nonblocking main-body warnings state that stationary wrapfigures were forced to float (source lines 323 and 375); they are outside the edited appendix span.
- Concurrent changes outside this pass, including the new mode-keying subsection/macros, updated reference-KL figure, and Qwen level-trend figure/macros, were preserved. Consequently, the main PDF's text is not byte-identical to the incoming PDF: the changed figures/generated values and the B.5-to-B.6 cross-reference renumbering are included. This pass did not revise main-body prose.
- The full PDF and review copy are rebuilt from the validated source. The Overleaf archive is independently compiled and checked against the exact source snapshot, fragments, figures, and reference PDF it contains.
- The nine-page main-body export removes 146 dangling links to omitted references/appendix pages while retaining valid links. Text and metadata are preserved. Render comparison verifies that this export-only annotation repair leaves every page pixel-identical.

Other tasks continued updating main-body figures during final packaging. The delivered PDFs and archive therefore use a fixed source snapshot captured after this cleanup. The archive contains those exact sources and its matching reference PDF; live source/figure edits were preserved. `validation.json` records snapshot hashes and any live-source differences at release.

Machine-readable build, package, and export checks are recorded in `validation.json`.
