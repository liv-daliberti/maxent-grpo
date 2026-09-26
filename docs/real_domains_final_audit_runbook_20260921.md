# Final independent pilot audit

This runbook audits frozen CPU/GPU receipts; it neither submits jobs nor executes generated programs. A passing result establishes a reproducible pilot comparison, not Re:Max superiority. The completed `independent_hardened_final_v2` report passes for both domains, with all scheduler allocations terminal and 3.789444444 allocated GPU-hours accounted. Its summary SHA-256 is `7f69750aeb09c2e938e093ed8eda550566c6203369205060d98cafd7bdc04430`.

## Fixed evidence roles

All paths below are relative to `var/artifacts/real_domains_pilot_20260921/`.

| Domain | Base capability / endpoint | MaxRL training | Re:Max training | MaxRL endpoint | Re:Max endpoint |
| --- | --- | --- | --- | --- | --- |
| Reuters QA | `qa_endpoint_base_v2` | `qa_maxrl/training` | `qa_remax/training` | `qa_endpoint_maxrl_v2` | `qa_endpoint_remax_v2` |
| Hardened constructive code | `code_hardened_base` | `code_hardened_maxrl/training` | `code_hardened_remax/training` | `code_hardened_endpoint_maxrl` | `code_hardened_endpoint_remax` |

Each endpoint directory must contain `evaluation.json` with `status=complete`. Each training directory must contain `result.json` with `status=complete` and `completed_updates=32`, plus the sealed `checkpoint-32`. The auditor checks requested counts, original response tokens, raw/verdict identity bijections, checker/annotation agreement, prompts, actual checkpoint bytes and replay history. File existence alone is insufficient.

The corrected coding baseline and both checkpoint endpoints use all 21 admitted initial-development problems, 128 fresh responses per problem and seed 88421. The eight training problems are fixed in the training configurations; all 13 other development problems remain in their own stratum. **Those 13 untrained development problems are not the separately reserved 13 larger-study test problems.** Coding built-in final evaluation is only two development problems with eight responses each. It is an execution diagnostic, not the primary treatment comparison.

QA endpoints use all 32 dev and all 16 trained documents with 128 responses each. The 83 QA test documents and 13 larger-study coding test problems remain model-untouched in this pilot. The larger source cohort's independent closure receipt is separate from model evaluation readiness.

## Compatibility checks already prepared

`code_precompletion_compatibility_v1.json` records the matching initial LoRA hash, production trainer/adapter hashes, hardened dataset and prompt identities, frozen generation settings and actual checker/runtime executables. Its `compatibility_status=pass` deliberately coexists with `completed_result_readiness=unknown`.

The coding arms use separate build, runtime, scratch and launcher locations. The auditor discounts only those four location fields, after checking the launcher, per-task checker binaries, runtime image and critical interpreter files against admitted hashes. Dataset paths are normalized only through recorded mappings from the independently verified launch identities of all compared endpoints and training arms. Nested freezes are resolved to a fixed point; cycles or semantic drift fail. Model, limits, task order, seeds, objectives and all other semantic configuration fields remain compared. The frozen testlib relocation from `third_party/testlib/testlib.h` to `bundle/testlib/testlib.h` requires a unique original-to-snapshot byte binding in the launch identity.

The two training arms must have the same initial LoRA parameter hash and production source; both begin with empty banks. A full audit reconstructs each bank from the first accepted original policy sample per task/mode, checks round-robin scheduling and uniform mode replay, and requires zero applied replay gradient for MaxRL and positive replay gradient for eligible Re:Max updates. Binary fresh advantages are recomputed from each group's acceptance count. Resource diagnostics distinguish runner time from allocated scheduler GPU-hours.

Historical coding samples, canceled earlier pairs and CPU-regraded diagnostics do not enter the primary comparison. The revoked historical manifest remains explicitly rejected by the auditor.

## Run after all required jobs finish

Run from the repository root. Refresh the scheduler accounting once, then use that saved receipt in the independent audit. Accounting includes failed/canceled allocations and the separate QA resume proof; it queries Slurm only in the first command. The audit itself uses local receipts and cached tokenizers.

```bash
pilot_root=var/artifacts/real_domains_pilot_20260921
pilot_python=var/seed_paper_eval/paper310/bin/python

"$pilot_python" ops/account_real_domains_pilot_20260921.py --root "$pilot_root"

PYTHONPATH=src:ops "$pilot_python" ops/summarize_real_domains_pilot_20260921.py \
  --evaluation "qa=$pilot_root/qa_endpoint_base_v2" \
  --evaluation "code=$pilot_root/code_hardened_base" \
  --pair "qa=$pilot_root/qa_maxrl/training,$pilot_root/qa_remax/training" \
  --pair "code=$pilot_root/code_hardened_maxrl/training,$pilot_root/code_hardened_remax/training" \
  --endpoints "qa=$pilot_root/qa_endpoint_base_v2,$pilot_root/qa_endpoint_maxrl_v2,$pilot_root/qa_endpoint_remax_v2" \
  --endpoints "code=$pilot_root/code_hardened_base,$pilot_root/code_hardened_endpoint_maxrl,$pilot_root/code_hardened_endpoint_remax" \
  --qa-provenance var/artifacts/noncoding_multi_answer_sata_20260921/provenance_audit/news_source_comparison.json \
  --accounting "$pilot_root/budget.json" \
  --output "$pilot_root/independent_hardened_final_v2"

"$pilot_python" - <<'PY'
import json
from pathlib import Path
p = Path('var/artifacts/real_domains_pilot_20260921/independent_hardened_final_v2/summary.json')
s = json.loads(p.read_text())
assert s['readiness_gate'] == 'pass', s['readiness_gate']
for domain in ('code', 'qa'):
    d = s['domains'][domain]
    assert d['paired_training']['status'] == 'pass'
    assert d['endpoint_comparison']['status'] == 'pass'
assert s['scheduler_accounting']['pilot_accounting_complete']
assert s['scheduler_accounting']['allocated_gpu_hours'] <= 200
print('Complete independent pilot audit passed; treatment efficacy still requires interpretation.')
PY
```

Use a new numbered output directory for any later rerun; preserve earlier reports. Do not use `--skip-tokenizer` for the final report: it leaves token bindings and readiness unknown. A missing or incomplete endpoint must remain unknown or fail, never be filled in using an earlier verifier, earlier checkpoint, smaller sample set or different sampling configuration.

## Report checks

Read `report.md` together with `summary.json`. Keep whole-cohort acceptance and expected distinct @8/@32 visible. Report PCMD only with at least 30 accepted responses and paired differences only on the common eligible task intersection, always showing eligible/total counts. QA known support is its native annotated topic set; coding's total witness support is unknown. Separate all 32 QA dev from 16 trained QA documents, and eight trained coding-development from 13 untrained coding-development problems. Any subset selected because replay discovered several modes is exploratory and post-treatment.

The one-seed pilot cannot demonstrate seed-level reproducibility. Small or null changes remain reportable. Do not update the paper methods draft with coding treatment effects until the complete three-endpoint audit passes; the eight-response built-in training evaluation is not that endpoint comparison.

## Larger-study endpoint cohorts

The same auditor now accepts three explicit cohort types and writes `endpoint_cohort` into the endpoint comparison. `reserved_test_primary` contains exactly the full frozen reserved set: 83 QA test IDs or 13 admitted coding test IDs for the current larger dataset. QA requires both `adapter_config.allow_test=true` and a `test` split selection; coding requires `adapter_config.allow_heldout=true`. No reserved ID may be in `train_ids`. Dataset-bound split labels must agree with every endpoint receipt. Partial test sets and test/training mixtures fail this primary-cohort check.

`trained_support_diagnostic` contains exactly the training IDs, allowing a separate diagnostic report for each seed. `pilot_mixed_development` retains the complete existing pilot cohort. The reserved stratum is named `reserved_test`; the multi-bank exploratory subset is restricted to training tasks actually evaluated, so it is empty in test-only reports. Separate primary and trained reports must bind to the same audited training pair before seed-level aggregation.

For one larger-study seed, use the same `--evaluation`, `--pair`, `--endpoints`, `--accounting` and optional QA `--qa-provenance` arguments with that seed's reserved-only endpoint directories. Use another output directory and the trained-only endpoint directories for diagnostics. The unchanged CLI handles both; no held-out model run is needed to test these schemas.

A single-domain report can have global readiness unknown because the other domain is absent. A legitimate all-failure or single-mode result can also leave capability unestablished despite complete outcome evidence. Seed aggregation may retain such outcomes only when the requested domain's paired-training and endpoint audits, source/token bindings, required scheduler receipts and applicable provenance checks all pass; no failed integrity gate may be ignored. Capability is not a favorable-outcome filter.

The held-out-compatible auditor source SHA-256 at handoff is `13adb7df5bc19391f92ea1781c09a0477abda0c2a1ecfa4e98cf8d386a80ba71`; 58 focused integrity tests pass, including full test-only and trained-only endpoint comparisons, permission and coverage failures, train/test leakage rejection, nested freeze normalization and missing manifest split resolution through hash-bound task records. No larger-study GPU job was submitted by this work.

## Interrupted training: restart from the pinned initial state

The current full-history auditor supports an uninterrupted completed training directory. It intentionally rejects `identity.resume != null`; it does not stitch together original and continued candidate/metric files. The separate QA continuation check verifies functional state restoration, including exact initial adapter restoration, but does not establish an identical GPU trajectory: final fresh-count histories and one of 256 final evaluation outputs differ. It is engineering evidence, not a primary paired arm accepted by this auditor.

For a preempted larger-study arm, retain the partial run and its accounting, confirm that allocation is terminal, then **restart the full original update schedule from the same pinned initial model and training seed into a new directory** using the ordinary frozen launcher. Preserve the original request's semantic configuration, task order, model revision, source/data bytes and initial parameter identity. Before submission, verify that the launcher's current trainer, production learner and verifier dependencies still match the original frozen bundle; unrelated analysis-code changes need not alter experiment semantics. The normal launcher freezes the current repository, so merely reusing an old request filename is not sufficient evidence of source identity.

Use the successful complete restart as the arm supplied to `--pair`; freeze its final checkpoint into the corresponding endpoint. Keep the failed/partial allocations in `budget.json`. Do not concatenate logs, replace old checkpoint files, or present the truncated continuation directory as a full trajectory. This full-restart path preserves the requested comparison and is the supported reporting workaround until an explicit sealed continuation-chain auditor is implemented. The larger-study runbook owns the concrete retry preparation and budget commands.
