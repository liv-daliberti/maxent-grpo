# September 12 paper refresh and Python runtime recovery

Both manuscripts include the completed GPT/Grok discovery analysis and the full E118–E120 census. Historical datasets, registered primary estimates, exclusions and failed calibration outcomes remain preserved.

## Completed paper evidence

- E118: 150/150 endpoints, 75 MaxRL/ReplayMaxRL pairs and 15 five-seed domain/scale blocks.
- E119: 100/100 endpoints, all five four-method domain factorials complete on seeds 43–47. Pantry now appears in the main results, paired contrasts and training curves.
- E120-R1: 45/45 treatments, nine five-seed blocks. The original 25-cell Qwen-0.5B primary estimate remains unchanged.
- Training curves: 400 registered runs, 399 admitted after the original Falcon Countdown exclusion; 6,677 complete checkpoints. Missing/conflicted intermediate evaluations remain explicit.
- Conditional concentration: the explicit retrospective completion amendment adds ten Pantry checkpoints to the previous cache. The full report covers 475 runs, 902 available checkpoints and all 135 contrasts. Pantry's breadth improvement does not establish lower conditional concentration; sparse adverse collision estimates remain visible.
- Discovery analysis: 147,456 responses across 31 local/hosted cohorts, including 36,864 fresh GPT-5.4/GPT-5.6 Sol/Grok responses. Both grading conventions, 64-draw curves, matched-correct-budget analyses and full eligibility denominators are retained.

The main text includes GPT MathIR's matched-correct-budget wording gains, Grok Pantry's continued mode discovery, Pantry outage adaptation, mixed-model portfolios and coarse-key sensitivity. The appendices retain complete tables and protocol limits. Historical Level-3 inference problems are distinguished from the newly calibrated neutral dataset. Table 1's model icons were checked directly in the rendered PDF.

## Validated artifacts

- [Parent PDF](../../main.pdf): nine main pages, references on page 10, 99 total pages; all eight main figures and Table 1 present.
- [Workshop PDF](../../mathai2026/main.pdf): four content pages, references on page 5, eight main and 22 supplementary figures.
- [Workshop source bundle](../../mathai2026/mathai2026-source.zip): local submission package; no external submission was made.
- [Completed campaign census](../../results/current_campaign_results_20260912.md).
- [Full concentration report](../../results/conditional_concentration_20260912/report.md).
- [Hosted discovery interpretation](../../results/discovery_hosted_interpretation_20260912.tex).

Scientific checks authenticate the original proof blocks, endpoints, current figure sources and retained numerical records. The full discovery reconstruction passed with 41,472 CSV metric entries and 12 figure files (`discovery_validation_current_caption.log`). The targeted publication test suite passed after updating the stale Pantry cohort expectation; the twelve training-snapshot tests passed in `training_snapshot_tests_r3.log`. Build logs retain page, layout, anonymity, reference and asset checks. The caption-only discovery revision changes provenance and wording, with identical numerical fields and PDF/PNG bytes.

## Neutral Python Level 3

Fresh V5 confirmation passes the original numerical tolerances: **22.34% pass@1 and 70.51% pass@8**, with all 4,096 responses independently regraded. The historical 19.46% / 58.40% failure remains a separate result. The new pass is an observed approximate match to the fixed historical reference, not statistical equivalence.

A subsequent training startup exposed a separate CLI omission. Four jobs rejected the neutral template before creating scientific output; their retry loops were stopped. New immutable E122/E124 runtimes change only `src/oat_drgrpo/args.py`, admitting the already registered neutral template and enforcing its Python/domain/syntax contract. Four native CLI and inventory tests passed with the launcher's Python library path. No frozen parent runtime was edited.

The [repair commit](../../../var/artifacts/python_level3_cli_recovery_20260912/committed.json) replaces all 22 zero-step jobs. Old cancellations and new held states were independently read back. Current E122 IDs are 31259067–31259086; E124 IDs are 31259087–31259088. E124 retains its independent systems-qualification hold. Current launch defaults bind both the admitted data and tested runtime through [`ops/modebench_current_training.py`](../../../ops/modebench_current_training.py).

The four seed-43 Python jobs 31259067–31259070 have completed initial evaluation and produced positive optimizer-step records with finite loss/gradient metrics and zero restarts: drgrpo 32, replay_drgrpo 28, maxrl 11, replay_maxrl 32. The [final runtime status](../../../var/artifacts/python_level3_cli_recovery_20260912/FINAL_STATUS.json) binds retained metric prefixes and scheduler readbacks. The full training runs continue toward their registered endpoints.

## Capacity and storage recovery

The earlier finite bursts admitted nine additional E122 cells. After the CLI repair, a fresh GPU/CPU/memory census admitted seven more non-Python jobs, 31158704–31158710, with an 18-slot finite bound and more than 3 TiB of aggregate storage margin. All pending inference writers, including array 31258973, are reserved. These finite admissions preserve the ordinary cap-four continuation and all scientific recipes.

Node 302 is usable and runs a repaired Python job. Node 105 remains administratively down following an unexpected reboot; node 206 is draining and contributes no new capacity.

The full 5 GiB home quota prevented both sandbox startup and approval-session locks. User-authorized cleanup removed the VS Code extension download cache and one extension explicitly marked obsolete with no running-process reference. About 502 MiB of home quota was available afterward. Active editor installations, Codex sessions, credentials and experiment data were retained. See `home_cleanup_20260912.json`.

Final checks passed in `current_contract_final.log`, `parent_compile_final.log`, and `workshop_bundle_final.log`. Exact delivered artifact hashes are in `final_artifact_manifest.json`. CPU continuation31259141 is running on node917; its clean heartbeat confirms18 running,11 completed endpoints and71 held.
