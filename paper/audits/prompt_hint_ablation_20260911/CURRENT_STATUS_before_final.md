# Prompt-hint ablation: local panel complete; frontier credential pending

The user authorized the matched original-versus-neutral ablation on local
checkpoints and frontier models, followed by appendix integration.

## Frozen scientific inputs

- Root manifest: `manifest.json`, SHA256
  `0d0149905d29cf846c97cbfd2f8f23e5117cf837af78a5b2eb1c17f69d951f93`.
- Python factors, MathIR and Pantry at Levels 2 and 3; 32 fixed problems per
  cell, eight fresh responses per wording. User messages and verifiers remain
  unchanged. Exact system edits are in `transformations.json`.
- Hosted panel: GPT-5.6 Sol, GPT-5.4 and Grok 4.3; 9,216 responses across six
  fresh cohorts. Registry: `hosted_analysis_runs.json`.
- Local production panel: `local/plan_v2.json`, SHA256
  `f82f028dab3c2dfd2590db91297e0f6eac43101afde561f2ce328d0b493576f5`.
  Initial Qwen0.5B plus 24 complete archived Dr.GRPO/ReplayDr.GRPO checkpoints;
  27,648 responses. Trained checkpoints learned on Level 2; Level 3 is transfer.
- Intended total: 36,864 production responses. All failures count.
- Interpretation and uncertainty rules: `ANALYSIS_PLAN.md`.

## Execution

The hosted orchestrator is sealed at
`dbae7404a2047ff8d1a8b055ab81e38495d60b906218d0be2c2425d576d67ac6`.
`execution_amendment_v2.json` records the actual concurrent hosted collection
order and binds local production execution. Do not edit sealed code or source
manifests; record necessary amendments prospectively.

The first local production job is Slurm `31246892_0`. Additional disjoint ready
subsets are submitted by the local evaluator agent with dependencies and a
maximum of eight evaluation GPUs. Fresh scheduler capacity supported this
operational acceleration without additional responses or altered settings.
The prospective cap and dependency amendments are retained in
`local/execution_amendment_cap4.json`,
`local/execution_amendment_parallel_pairs.json`, and
`local/execution_amendment_cap8.json`; `local/execution_cap8_result.json`
records the verified eight-GPU bound and AND dependency barriers. Consult the current
`local/monitor_status_cap4.json` and durable submission records before any new
submission; do not duplicate queued jobs. The earlier two-GPU monitor is stale.

The earlier local job `31246812_0` was stopped before any saved generated
batches because vLLM's child-seed expansion required wider problem-seed spacing.
Its original plan and records remain preserved. It is not production evidence.
The corrected formula has disjoint child seeds across every candidate problem.
Checkpoint caches are restored in isolation from commit-pinned, SHA-verified
archives. Existing training jobs and original result files are unchanged.

Hosted collection has not started: no Azure credential is present in the
experiment environment. A secure credential source has been requested from the
user. Automatic approval review rejected retrieving credentials from a saved
chat log as an unintended credential source; do not repeat that search or use
an indirect workaround. Use only the environment or an explicitly supplied
private credential file. Never record the key in artifacts or logs.

## Commands

Use `var/seed_paper_eval/paper310/bin/python` for preparation, collection and
analysis; the system interpreter lacks required dependencies. The default
shell sandbox currently fails to create namespaces, so shell actions have
required reviewed escalation.

Hosted status:

```sh
var/seed_paper_eval/paper310/bin/python ops/run_hosted_prompt_ablation.py status
```

Once the user supplies an authorized credential source, run the retained
first-sample preflights, verify receipts, then the full six cohorts. Both stages
use the sealed orchestrator. Offline grading runs separately per cohort or
local checkpoint; the analyzer is `ops/analyze_modebench_prompt_ablation.py`.
It requires complete registered groups before inference and preserves raw and
reverified Python grades separately.

## Completed local results and publication status

All 25 registered local checkpoints are complete: 27,648 draws. Every
production job exited successfully; peak allocation was eight GPUs and total
allocation was 2.44 GPU-hours. `local/COMPLETE.json` and
`local/completion_integrity_audit.json` retain the full inventory. The offline
controller is complete. All grades were authenticated, with zero strict-grade
corrections and zero formatting-normalization rescues.

`LOCAL_FINDINGS.md` gives all-cell results and interpretation. The sealed
numerical report is `analysis_local_complete/analysis.json`. Publication uses
`analysis_local_complete_editorial_v2/`: only quotation marks and LaTeX
sizing/placement changed. `editorial_numerical_equality_v2.json` proves unchanged
scientific report fields, CSV bytes, and plotted values. Both report versions
are preserved.

Both paper sources include the ablation section and all-cell tables/figure,
explicitly marked local-only / `partial_panels`. The first full local appendix
builds passed: long paper 66 total pages with eight main pages; workshop four
main pages and a validated source ZIP. A later, explicitly post-hoc Python
failure audit is now in both TeX sources and copied as
`paper/results/modebench_prompt_ablation_python_failure_diagnostic_20260911.json`.
The final build passed the ablation and 5,120-draw diagnostic checks but paused
at an unrelated current-paper contract during a concurrent results refresh.
Another active process is updating the latest training census, Figures 5/6,
training curves, and prose. DO NOT roll back those changes or weaken checks.
Wait for that refresh to stabilize, then validate the combined papers, sync
workshop assets in a new audit directory, and build/package again.

The last validated outputs are preserved in
`paper/audits/prompt_hint_ablation_20260911/validated_before_concurrent_refresh/`.
Build logs and all prior paper/source snapshots are in the same audit root.
The ablation section, post-hoc paragraph, and exact report hash remain intact
in both manuscripts during the concurrent refresh.

The Python diagnostic is `LOCAL_PYTHON_FAILURE_DIAGNOSTIC.{json,md}` with
exhaustive per-draw records; it makes no model calls. All 2,560 original
DrGRPO Python outputs use the same example conditional body. All neutral draws
fail: 1,816 invalid expressions, 59 other language violations, 376 non-integer
returns, 276 divisor failures, and 33 missing extracted candidates at the token
limit. The post-hoc MathIR identity diagnostic finds 298 of 299 shared-solved
prompt–checkpoint groups retain the same mode; it is descriptive, not
independent-sample inference.

The hosted panel has collected zero responses because no authorized Azure
credential is available. The user has been asked for the intended secure
environment/file source. Once supplied, run the sealed preflight, inspect its
retained receipts, then run all six cohorts and the full analysis. Do not
retrieve a credential from saved chat logs; automatic approval review rejected
that source. Full publication must still include the registered frontier panel
before the overall experiment can be called complete.
