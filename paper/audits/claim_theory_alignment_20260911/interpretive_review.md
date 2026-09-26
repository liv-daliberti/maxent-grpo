# Current local manuscript interpretive audit

Scope: current local paper/main.tex, paper/mathai2026/main.tex, all non-theory workshop appendix sections (duplicated in the long paper), the two hosted appendix entry points and substantive nested protocol/sensitivity/temperature/provider-outcome includes. The proof agents own theorem corrections. No historical GitHub source was applied locally.

## Most material new finding: cross-level reference populations

The Level-2 r5 construction is not histogram-matched to all displayed native Level-1 terminal evaluation sets. The source var/results/modebench_level2_r5_frozen_repeat1/admission_fairness_report.json names native Level-1 training sets, but E117 development and confirmation reserves for Graph, Countdown, Python, and MathIR. Pantry uses native Level-1 development/evaluation. The displayed terminal Level-1 comparisons come from native E78/E118 evaluation logs (paper/results/modebench_level_comparison_snapshot.json, terminal_evaluations and terminal_sources).

Level-2 eval means computed from the admission manifest: Graph 6.0859375, Countdown 4.3671875, Python 251.6875, MathIR 5, Pantry 18.1875. Native Level-1 main table: Graph 6.265625, Countdown approximately 4.52, Python approximately 229.44, MathIR 5, Pantry 18.1875. The first three differ. The existing hosted appendix already explains precisely this distinction.

Figure modebench_level_admission panel A comes from frozen development admission, whereas panel B comes from separate native Level-1/Level-2 terminal test populations. Its Python frozen Level-1 P=.90625 is the development reserve, not a measured initial native E78 terminal-test value. The main figure caption should identify development admission.

Correct factual scope: Level 2 matches native training histograms and designated Level-1 construction reserves. Cross-level terminal changes also change test populations and therefore do not isolate a causal difficulty effect or exact test-support matching. Within-level paired replay effects remain valid and all numerical estimates should stay unchanged.

## Other current corrections communicated to parent

- In both non-theory appendices the direct-comparator figure calls B=D-P "correctness-adjusted breadth". Replace with "extra verified modes beyond the first success". This decomposition remains coupled to correctness.
- Baseline precheck caption "Both verifier-only objectives reduce verified solution breadth, at every scale" should specify the five-domain average of extra verified modes; raw D increases at 3B. Suggested title: "Verifier-only training reduces average extra verified modes at each tested scale".
- Workshop limitations sentence "Uniform correct mass requires full verified coverage" should refer to the full-support corollary assumption, not a universal requirement for a uniform policy. The same paragraph's "one decoding protocol" applies to training comparisons, since hosted experiments sweep temperature.
- Hosted pair-collision uncertainty is appropriately prompt-clustered, pair-weighted within domain, explicit about undefined cells and not a paired model-effect test. The temperature reports distinguish requested/effective settings, reasoning-none from historical medium reasoning, post hoc normalization, and empirical optimum from population optimum. No further major inference error found there.
- Historical impossible Falcon D<P values are absent from current core terminal tables. A recursive numerical audit checked 2,229 metric/count nodes over core terminal, GPT reference, hosted comparison, current campaign, and level comparison records; no violations of 0<=M<=P<=D<=8M or P<=1 were found.
- Current metric appendix contains correct per-prompt occupancy formulas and explicitly forbids applying the nonlinear formula to pooled accuracy; no repair needed to those identities. Existing finite-bank, length-normalization, observation-tail, and no-causal-fixed-bank-control caveats are substantially improved over the historical draft.

## Workshop edits owned by this agent

Only paper/mathai2026/main.tex was edited. Exact title: Mode Collapse in RLVR & ModeBench. Tightened binary-reward scope, sampled breadth wording, figure-1 factorial caveat, conditional categorical guarantee, aggregate/partial-cohort statements, admitted-exemplar scope, heterogeneous hosted behavior, and protocol differences. Cross-level statement and panel-A caption were corrected after the manifest audit. Current prose remains approximately the original length. Numerical results, figures, appendix, snapshots, submission receipts, and historical protocols were not edited by this agent.

Temporary build outputs: /tmp/mathai2026_claim_audit_build/. Baseline source backup: /tmp/mathai2026_main_before_claim_audit.tex. An isolated pdflatex build passed at four main pages after prose condensation; final build is checked again after the cross-level wording and parent theorem integration. The submission snapshot/receipt should be refreshed only by the parent after all shared-source changes finish.
