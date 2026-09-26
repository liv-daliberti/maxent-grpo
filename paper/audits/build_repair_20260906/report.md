# MATH-AI paper repair and results refresh — September 6, 2026

The workshop source had compiled, but two conclusion lines spilled onto a fifth main-content page. The submission checker rejected the page count, and Makefile's `.DELETE_ON_ERROR` removed `main.pdf`. The failure is recorded in `../revision_20260906/workshop-build-final.log`.

The repaired submission retains four content pages, all six main figures, seven supplementary figures, references beginning on page 5, the official anonymous template, and unchanged figure dimensions. Redundant prose was tightened. The user-requested conclusion now explains ModeBench, why different approaches may have different strengths when problems or constraints change, and the boundary between measured output diversity and untested general reasoning or later problem-solving utility. Both paper versions receive the clarified interpretation. The parent version also passes its caption and paragraph line-fill requirement.

Builds now compile and validate in a temporary directory before publishing the PDF and receipt. Compile errors, source mismatch and page-limit rejection preserve the last valid PDF. Failed TeX logs are retained as `main.failed.log`. Packaging verifies the exact archived bytes against the validated source and preserves the previous ZIP on failure. Eleven isolated build/regression cases cover success, rejection, recovery, missing receipts and clean archive compilation; evidence is in `build_regressions/`.

The dated endpoint census spans September 6, 16:56:42–17:09:35 UTC. E118 has 116/150 admitted endpoints, 57 matched MaxRL pairs and 11/15 complete blocks; E119 has 52/100 endpoints and one complete four-arm block; E120 has 33/45 endpoints and six complete blocks. The new Qwen3B Graph and MathIR pairs are single-seed descriptive updates. Original Figure 6 and primary E120 inputs retain their frozen identities. The complete census and its reproduction scripts are in `../submission_repair_20260906/results/`.

The newly complete registered Falcon Graph frequency-weighting block uses all five seeds 55–59. Uniform weighting adds .233 extra modes (95% paired bootstrap [.182, .320]); its pass@8 interval [−.0004, .0910] includes zero, while mean sampled correctness decreases by .0152. All paired seeds and the supporting negative finding are included in both appendices and the standalone bundle. Independent raw-record and exact-bootstrap verification is recorded in `e120_falcon_graph_independent_review.json`; 38 focused result-integrity tests also passed. No incomplete block is promoted to a five-seed efficacy estimate.

Validation: `make -C paper/mathai2026 bundle` passes source/hash/receipt, page, figure, anonymous-template and compiler-diagnostic checks. `make -C paper` passes the manuscript/data contracts and checks all 293 natural-prose blocks against its 50% final-line rule. The source archive is also compiled in a clean extracted directory. Logs are `workshop-build.log`, `parent-build.log`, and `standalone-bundle-build.log`. Source and asset changes are bound in the workshop snapshot history.

The separately authorized scheduler handoff moved E119 job 31075341 back to the queue from intact checkpoint 1248, without lost logged updates, and started E118 job 31100509 on node302 from checkpoint 1728. E119 depends on that E118 job finishing. The operational journal distinguishes allocation/checkpoint restoration from verified fresh optimizer steps; see `../../../var/artifacts/e118_node302_handoff_20260906/report.md`.


The final wording follows the author's requested softer scope statement:
“These controlled tasks provide a starting point for studying diversity in
broader mathematical reasoning.” The corresponding parent conclusion uses
the same framing; the causal and later-problem-solving limits remain explicit.


A final [workshop-fit revision](workshop_fit/report.md) moves the mathematical-agent
motivation into the abstract and first paragraph, with verified primary-source
attribution for candidate generation, search and revision. The first-page figure
and four-page main-text budget are preserved.
