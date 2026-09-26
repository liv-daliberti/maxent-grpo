# Active second ablation handoff

User requested64-response discovery curves and appendix integration with Yue et al.
The earlier prompt-hint ablation is complete locally and must remain intact.

- Experiment root: artifacts/modebench_discovery_curves_20260911.
- Root manifest SHA1d42de03313dc571329cbc14b4339d124c0994aae36eb6d2a8f4ea9d5b6b05e3.
- Six cells,16 old SHA-ranked problems each, both wordings,64 fresh draws.
- Local25 checkpoints,110592draws; Slurm31250797 tasks0–24%8 is running.
- Local plan SHA7973dcdcc881b8dedab242036ce48b5e32a41ce8f505fc0af6b77fc7c0a511fb.
- First completed4096-draw checkpoints: Python DrGRPO/Replay seed43. Others progressing.
- Source plan is frozen before collection; analyzer is being sealed before inspecting any outcomes.
- Eight n8 blocks preserve settings and use new911640000 seed namespace, stride128; child seeds disjoint.
- All support references deliberately conservative certified lower bounds.
- Hosted36,864requests prepared; zero collected, intended Azure credential absent.
- Hosted operational v2 SHA80eb49a0c35c1917c1370ad92ba8df5be2ceef567f293640cdb4e181c863c6aa.
- Immutable v2 at hosted_execution_revisions/v2/hosted_execution.json; source snapshot hosted_execution_code_v2.
- Runner ops/run_modebench_discovery_hosted.py supports status/preflight/full with private credential file.
- Twelve hosted tests passed; cross-arm provider identity validation and credential-free recovery included.
- Do not retrieve credentials from old chat logs. Prior automatic review rejected that source.

Agent ownership:
- local_prompt_ablation: local collector/array/integrity monitor until complete.
- prompt_ablation_analysis: analyzer, tests, sealed source, per-checkpoint regrading controller, report/figures/TeX.
- neutral_prompt_builder: publication checker/sync/build hooks/tests; independent stats review and exact-style synthetic layout check.
- Root: frozen design/hosted preparation+runner, final results interpretation, both manuscript edits, full builds/ZIP/audit.

New tools/files:
- ops/analyze_modebench_discovery_curves.py (in development until explicit seal).
- ops/check_paper_discovery_curves.py (new publication authenticator).
- ops/prepare_modebench_discovery_hosted.py; ops/run_modebench_discovery_hosted.py.
- tests/test_modebench_discovery_hosted.py (12pass).
- paper/audits/discovery_curves_20260911/source_before_integration.json and before/ preserve17publicationfiles.

Primary curves use exact without-replacement rarefaction from64; ordered draw-index
prefixes are secondary. Fixed-m correct-draw breadth uses joint eligibility in both
arms. Uniform collision1/L is an upper reference; uniform breadth atL is a lower
reference, not a model-breadth bound. Equal-seed mean collision differs from pooled
pair-count ratios; report labels both.20kpairedbootstrap; five training seeds in
Python/MathIR, two fixed/descriptive inPantry. No claims of full support or latent
reasoning capacity. Citation key yue2025rlvrlimit already exists and was verified
against official arXiv2504.13837v5. Preserve both different authors named YangYue.

Publication still to do after full local completion:
1. Seal/regrade/authenticate all25checkpoints and produce complete-local report with explicit partial_panels scope.
2. Visually inspect actual two figures and assess all-cell outcomes fairly.
3. Add new appendix section to both manuscripts, include generated results TeX,
   source-bound interpretation, and citation. Preserve concurrent unrelated edits.
4. Exact copies: paper/results/modebench_discovery_curves_20260911.{json,tex};
   figures modebench_discovery_curves_local and modebench_discovery_correct_budget_local each PDF/PNG/JSON.
5. Authenticate new report + prior ablation; run parent build, workshop sync in NEWauditdir,
   workshop bundle, page/layout checks, update main-body PDF based on actual main-page count.
6. Create focused standalone appendix PDF, final receipt/status and final user results.

Synthetic-only layout is in /tmp/modebench_discovery_synthetic_review; do not publish
these test figures as results. Revised six-page article passed line-fill8blocks,
no overfull or unresolvedrefs. Builder is checking exact ICLR style/counters before
analysis seal. If presentation must change after real outcomes are inspected,
preserve old sealed source and record numerical-equality editorial amendment.
