# Larger real-domain comparison: runnable handoff

The larger studies are **ready, protocol frozen, and unsubmitted**. The final independent corrected-pilot audit passed for both domains; readiness is based on integrity and method checks, not a favorable Re:Max effect. All predefined source/data pins, seeds, hyperparameters, task cohorts and analysis choices are preserved. The QA pipeline, production objective, GPU LoRA evaluation, and functional checkpoint continuation have been exercised; primary study evidence uses complete runs from an empty bank.

Final audit: `var/artifacts/real_domains_pilot_20260921/independent_hardened_final_v2/summary.json`, SHA256 `7f69750aeb09c2e938e093ed8eda550566c6203369205060d98cafd7bdc04430`. The combined CPU verification suite passed all **95 tests**. A reporting metadata correction preserves the original audit-summary input path and hash; both real-domain CPU analysis smokes pass with that correction. The previous freeze is preserved under `larger_study_protocol_freeze_history/v1`. The final frozen handoff receipt is `var/artifacts/real_domains_pilot_20260921/larger_study_protocol_freeze_v2_20260921.json`.

All commands below run from `/n/fs/similarity/maxent-grpo`.

## Fixed studies and pins

| | Reuters topic QA | Constructive coding |
|---|---|---|
| Request directory under `var/artifacts/real_domains_pilot_20260921/` | `qa_larger_study_unsubmitted` | `code_hardened_larger_study_unsubmitted` |
| Train / validation / heldout | 128 / 32 / 83 document groups | 29 / 2 / 13 problems |
| Paired seeds | 88301, 88302, 88303 | 88501, 88502, 88503 |
| Updates per arm | 512: four passes | 116: four passes |
| Response cap | 16 tokens | 1,024 tokens |
| Initial and final monitoring | 32 validation tasks × 32 draws; seed 99301 | 2 validation tasks × 32 draws; seed 99501 |
| Primary heldout endpoint | 83 tasks × 256 draws; seed 109301 | 13 tasks × 256 draws; seed 109501 |
| Optional trained diagnostic | — | 29 tasks × 128 draws; seed 109502 |
| Base checkpoint revision | Qwen2.5-7B-Instruct `a09a35458c702b33eeacc393d103063234e8bc28` | Qwen2.5-Coder-7B-Instruct `c03e6d358207e414f1eca0bb1891e29f1db0e242` |

Both arms use the existing production token-importance MaxRL objective, G=16 fresh rollouts/update, microbatch 1, rank 16 LoRA, learning rate 1e-5, and matched sampling budgets. Re:Max adds uniform verified policy exemplars with raw alpha 0.1 (effective coefficient 0.005859375); MaxRL traverses replay with zero applied replay derivative. Banks start empty. Checkpoints save every 16 updates and at the exact final update.

QA data: `var/artifacts/noncoding_multi_answer_sata_20260921`. Manifest SHA256 `49848ab116c9bcd43c734f48eecdc6fe1da1ed180dd10005bb629c320d1405d4`; records SHA256 `de530462b2f3e9769cef9332103ca3fcf578725d590676cb6a4e22594de6cb54`. SATA source revision `ba43a7ab537adfa3498e3a160a6d1eafbefc95c1`.

Coding data: `var/data/constructive_code_hardened_larger_20260921`. Manifest SHA256 `7cb281632b8e58be1bfa33d0400ac69c3470c0e998ee88c7d5c982cec0ee4986`; quality SHA256 `4509672ce27833cec74dd6a8b82cff945cf3f246546826a870371478db5747fc`. CodeContests+ revision `96c850540fade31d384a25766461e0da6b08f5fc`; CodeContests-O revision `1a765191567b429f633bbd1c6e67b5890dfaf267`. Exact split IDs and source-file hashes are in each request directory's `plan.json`. The old 32/2/16 quota was not met; the explicit strict 29/2/13 source gate passed.

QA's six final configurations passed CPU preflight under `qa_larger_study_unsubmitted/preflight_v2`. All six coding configurations passed complete CPU adapter/runtime/checker/tokenizer preflight under `code_hardened_larger_study_unsubmitted/preflight`. Their prompt, dataset and production-source hashes match; paired configurations differ only in isolated writable paths and the explicit arm. The largest prompt is 1,033 tokens (2,057 with the response cap), below the 8,192-token context limit. Do not use the obsolete pre-hardening requests.

## Verify pins and submit training

These checks compare the prepared source/data pins before freezing new jobs:

```bash
PILOT_PYTHON=var/seed_paper_eval/paper310/bin/python
PILOT_ROOT=var/artifacts/real_domains_pilot_20260921
"$PILOT_PYTHON" - <<'PY'
from pathlib import Path
import hashlib, json
root = Path('var/artifacts/real_domains_pilot_20260921')
for name in ('qa_larger_study_unsubmitted', 'code_hardened_larger_study_unsubmitted'):
    plan = json.loads((root / name / 'plan.json').read_text())
    for filename, expected in plan['source_pins'].items():
        assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == expected, filename
    assert hashlib.sha256(Path(plan['analysis_protocol']).read_bytes()).hexdigest() == plan['analysis_protocol_sha256']
    for filename, expected in plan['request_sha256'].items():
        assert hashlib.sha256((root / name / filename).read_bytes()).hexdigest() == expected, filename
qa = Path('var/artifacts/noncoding_multi_answer_sata_20260921')
code = Path('var/data/constructive_code_hardened_larger_20260921')
assert hashlib.sha256((qa / 'noncoding_multi_answer_sata_manifest.json').read_bytes()).hexdigest() == '49848ab116c9bcd43c734f48eecdc6fe1da1ed180dd10005bb629c320d1405d4'
assert hashlib.sha256((code / 'manifest.json').read_bytes()).hexdigest() == '7cb281632b8e58be1bfa33d0400ac69c3470c0e998ee88c7d5c982cec0ee4986'
assert hashlib.sha256((code / 'hardening_quality.json').read_bytes()).hexdigest() == '4509672ce27833cec74dd6a8b82cff945cf3f246546826a870371478db5747fc'
print('Prepared source and dataset pins match.')
PY
```

The integrity and protocol readiness conditions above are satisfied. When launching the larger studies, these commands freeze exact source/data and submit one GPU per arm. Existing directories are never overwritten; none of these larger submission commands has been executed.

```bash
for seed in 88301 88302 88303; do
  for arm in maxrl remax; do
    "$PILOT_PYTHON" ops/prepare_real_domains_run_20260921.py \
      --request "$PILOT_ROOT/qa_larger_study_unsubmitted/train_${arm}_seed${seed}_request.json" \
      --output "$PILOT_ROOT/qa_study_${arm}_seed${seed}" --submit
  done
done
for seed in 88501 88502 88503; do
  for arm in maxrl remax; do
    "$PILOT_PYTHON" ops/prepare_real_domains_run_20260921.py \
      --request "$PILOT_ROOT/code_hardened_larger_study_unsubmitted/train_${arm}_seed${seed}_request.json" \
      --output "$PILOT_ROOT/code_study_${arm}_seed${seed}" --submit
  done
done
```

## Materialize and evaluate sealed endpoints

Final-test templates are `final_test_base_request_template.json` and `final_test_{arm}_seed{seed}_request_template.json`. Coding also has `trained_diagnostic_*_request_template.json`. No heldout model responses have been generated.

Trained templates deliberately contain `__SEALED_CHECKPOINT_ROOT__` and `__EXACT_RESOLVED_TRAINING_CONFIG_SHA256__`. These mean the entire final checkpoint directory and the resolved training hash recorded by that checkpoint's completion seal. The latter is **not** the raw request-file hash. Primary endpoints must come from complete runs with an empty-bank start and a complete local rollout history. A full restart uses its new frozen training run as the endpoint parent.

The tested endpoint materializer verifies terminal results and seals, matches paired configurations and source, and freezes entire checkpoint roots. Its fourteen integrity tests include QA heldout authorization and resumed checkpoint parents. It supports 512 or 116 updates and never submits jobs. First freeze one baseline for each fixed endpoint cohort; a completed baseline evaluation is not required:

```bash
"$PILOT_PYTHON" ops/prepare_real_domains_run_20260921.py \
  --request "$PILOT_ROOT/qa_larger_study_unsubmitted/final_test_base_request_template.json" \
  --output "$PILOT_ROOT/qa_study_test_base"
"$PILOT_PYTHON" ops/prepare_real_domains_run_20260921.py \
  --request "$PILOT_ROOT/code_hardened_larger_study_unsubmitted/final_test_base_request_template.json" \
  --output "$PILOT_ROOT/code_study_test_base"
for seed in 88301 88302 88303; do
  "$PILOT_PYTHON" ops/prepare_real_domains_endpoints_20260921.py \
    --baseline-request "$PILOT_ROOT/qa_larger_study_unsubmitted/final_test_base_request_template.json" \
    --baseline-run "$PILOT_ROOT/qa_study_test_base" \
    --maxrl-run "$PILOT_ROOT/qa_study_maxrl_seed${seed}" \
    --remax-run "$PILOT_ROOT/qa_study_remax_seed${seed}" \
    --completed-updates 512 --output "$PILOT_ROOT/qa_study_test_seed${seed}_receipts" \
    --run-parent "$PILOT_ROOT" --run-prefix "qa_study_test_seed${seed}"
done
for seed in 88501 88502 88503; do
  "$PILOT_PYTHON" ops/prepare_real_domains_endpoints_20260921.py \
    --baseline-request "$PILOT_ROOT/code_hardened_larger_study_unsubmitted/final_test_base_request_template.json" \
    --baseline-run "$PILOT_ROOT/code_study_test_base" \
    --maxrl-run "$PILOT_ROOT/code_study_maxrl_seed${seed}" \
    --remax-run "$PILOT_ROOT/code_study_remax_seed${seed}" \
    --completed-updates 116 --output "$PILOT_ROOT/code_study_test_seed${seed}_receipts" \
    --run-parent "$PILOT_ROOT" --run-prefix "code_study_test_seed${seed}"
done
```

If an arm required a full restart, replace its `--maxrl-run` or `--remax-run` with the completed restart directory in all subsequent materialization and audit commands. Although the helper can freeze resumed checkpoints, the independent primary-evidence auditor does not support continuation-only histories. Optional coding trained diagnostics repeat the coding block with `trained_diagnostic_base_request_template.json` and fresh directory/prefix names replacing `test` with `trained`. They are separate from primary heldout inference.

Prepared runs are flat siblings under the standard artifact root, so existing accounting finds them. Submit their exact saved intents once; do not run the freeze launcher again into existing directories:

```bash
PYTHONPATH=ops:src "$PILOT_PYTHON" - <<'PYENDPOINTS'
import json, pathlib, subprocess
from prepare_real_domains_endpoints_20260921 import frozen_run
root = pathlib.Path('var/artifacts/real_domains_pilot_20260921')
names = ['qa_study_test_base', 'code_study_test_base']
for domain, seeds in [('qa', (88301, 88302, 88303)), ('code', (88501, 88502, 88503))]:
    names += [f'{domain}_study_test_seed{seed}_{arm}' for seed in seeds for arm in ('maxrl', 'remax')]
for name in names:
    run = root / name
    receipt_path = run / 'submission.json'
    if receipt_path.exists():
        raise FileExistsError(receipt_path)
    frozen_run(run, 'evaluate_real_domains_20260921.py')
    argv = json.loads((run / 'submission_intent.json').read_text())['argv']
    proc = subprocess.run(argv, text=True, capture_output=True)
    receipt = {'argv': argv, 'returncode': proc.returncode, 'stdout': proc.stdout, 'stderr': proc.stderr}
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    if proc.returncode:
        raise RuntimeError(proc.stderr)
    receipt['job_id'] = int(proc.stdout.strip().split(';')[0])
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
PYENDPOINTS
```


## Preemption and full restart

Each training job requests one 48 GB A6000 on `lowprio`, with a 240-minute cap and no automatic requeue. The four-hour placement test passed. `all` currently accepts 60-minute A6000 requests but rejected longer allocations; do not silently substitute it for larger jobs. Four-hour caps bound cost rather than guaranteeing completion. The completed corrected coding pilot averaged 63.45 seconds/update in both arms; 116 updates project 2.045 hours of learning, or roughly 2.3 hours including monitoring and overhead if the new task mix has similar cost. The pilot trained on 8 tasks, while the larger run uses 29; its peak allocated memory was 23.05 GiB, and the full larger task mix has not run on GPU.

The primary study's recovery path is a **full restart** from the same pinned request, seed, model and data, into a new normal frozen run. The independent auditor requires a complete local rollout history beginning with an empty bank. Preserve and account for every interrupted attempt; restart decisions depend on infrastructure interruption, not measured rewards. Allow at most two four-hour restart allocations per domain within its ten-hour reserve. Before each retry, refresh accounting and check spent time plus all live and remaining planned allocation caps stays below 200 GPUh.

The following example refreshes accounting, verifies the prepared pins and every original frozen source/data file, and checks prior attempts are terminal. It isolates the four writable coding work locations if present, freezes a normal replacement without submitting, compares both snapshots, and submits the saved intent. Only those work paths may differ scientifically. Adjust the original run, request, and new run names together for another arm/domain; keep the seed and update target unchanged. The remaining planned allocation check above must also pass:

```bash
PYTHONPATH=ops:src "$PILOT_PYTHON" - <<'PYRESTART'
from pathlib import Path
import copy, hashlib, json, subprocess
from account_real_domains_pilot_20260921 import account
from prepare_real_domains_run_20260921 import prepare
from prepare_real_domains_endpoints_20260921 import frozen_run
root = Path('var/artifacts/real_domains_pilot_20260921').resolve()
old = root / 'qa_study_remax_seed88301'
new = root / 'qa_study_remax_seed88301_restart1'
request = root / 'qa_larger_study_unsubmitted/train_remax_seed88301_request.json'
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
for name in ('qa_larger_study_unsubmitted', 'code_hardened_larger_study_unsubmitted'):
    plan = json.loads((root / name / 'plan.json').read_text())
    for path, expected in plan['source_pins'].items():
        assert sha(path) == expected, path
budget = account(root)
by_directory = {r.get('directory'): r for r in budget['jobs']}
assert by_directory[str(old)]['terminal'], 'Original allocation is still live.'
assert budget['available_for_new_allocations'] >= 4
for prior in root.glob('*/restart_provenance.json'):
    provenance = json.loads(prior.read_text())
    if provenance['original_run'] == str(old) and (prior.parent / 'submission.json').exists():
        assert by_directory[str(prior.parent)]['terminal'], 'A previous restart is still live.'
original = frozen_run(old, 'train_real_domains_pilot_20260921.py', 'remax')
identity = original['identity']
assert sha(request) == identity['request_sha256']
assert json.loads(request.read_text()) == identity['request']
for row in identity['files']:
    assert sha(row['source']) == row['sha256'], row['source']
retry = copy.deepcopy(identity['request'])
location_changes = {}
work = root / (new.name + '_work')
assert not work.exists(), 'Restart work directory must be new.'
for key in ('build_root', 'runtime_root', 'launcher', 'scratch_root'):
    config = retry['config']['adapter_config']
    if key in config:
        location_changes[key] = {'old': config[key], 'new': str(work / key)}
        config[key] = str(work / key)
retry_request = root / (new.name + '_request.json')
assert not retry_request.exists()
retry_request.write_text(json.dumps(retry, indent=2, sort_keys=True) + '\n')
prepare(retry_request, new, submit=False)
replacement = frozen_run(new, 'train_real_domains_pilot_20260921.py', 'remax')
prior_files = {r['source']: r['sha256'] for r in identity['files']}
new_files = {r['source']: r['sha256'] for r in replacement['identity']['files']}
assert prior_files == new_files, 'Source/data closure changed; do not submit.'
restored = copy.deepcopy(replacement['identity']['request'])
for key, change in location_changes.items():
    assert restored['config']['adapter_config'][key] == change['new']
    restored['config']['adapter_config'][key] = change['old']
assert restored == identity['request'], 'Scientific request changed.'
(new / 'restart_provenance.json').write_text(json.dumps({
    'original_run': str(old), 'original_identity_sha256': sha(old / 'identity.json'),
    'request_sha256': sha(request), 'retry_request_sha256': sha(retry_request),
    'allowed_writable_location_changes': location_changes, 'reason': 'infrastructure_interruption',
    'full_restart_from_empty_bank': True,
}, indent=2) + '\n')
argv = json.loads((new / 'submission_intent.json').read_text())['argv']
proc = subprocess.run(argv, text=True, capture_output=True)
receipt = {'argv': argv, 'returncode': proc.returncode, 'stdout': proc.stdout, 'stderr': proc.stderr}
(new / 'submission.json').write_text(json.dumps(receipt, indent=2) + '\n')
if proc.returncode:
    raise RuntimeError(proc.stderr)
receipt['job_id'] = int(proc.stdout.strip().split(';')[0])
(new / 'submission.json').write_text(json.dumps(receipt, indent=2) + '\n')
PYRESTART
```

Select the completed replacement directory for that arm in endpoint preparation and analysis. Do not combine a partial original history with a restarted run. The source comparison is deliberately strict: if a source or dataset changed, stop and resolve provenance before any retry.

A separate QA checkpoint 16→32 continuation proof completed in 0.0600 GPUh with exact initial adapter restoration and correct counters. It verified functional state restoration, but 6/256 subsequent samples and 1/256 final evaluation outputs differed; bit-identical GPU trajectories are not guaranteed. See `qa_resume_proof_run/report.md`. The reusable `ops/prepare_real_domains_resume_20260921.py` remains an engineering utility; resumed partial histories are not supported primary paired evidence in this protocol.

## Fixed reporting and uncertainty

Report each domain separately with task-macro means. Use the terminal checkpoint from every arm; do not select checkpoints or tasks by outcomes. Keep every heldout task in accuracy/pass@1, pass@8, and expected distinct verified modes at 8 and 32 draws (ED@8/ED@32), including zero-success tasks. Also report raw distinct@8 using fixed sample-index-consecutive eight-sample groups. QA additionally reports observed coverage of its known annotated topic support. Coding's trained-bank retention remains secondary; evaluation draws never enter a training bank.

PCMD requires at least 30 accepted draws for each task/policy. Report every arm's eligible numerator/denominator, each paired seed's common-eligible denominator, and sensitivity on the common-eligible intersection across all six trained endpoints. Include the base policy in that intersection for comparisons against base. Empty eligibility is undefined with 0/N, not zero. The corrected coding baseline currently has only 3/21 eligible tasks at 128 draws, so success-conditional conclusions may remain limited even with 256-draw endpoints; keep all-task ED and pass@k results visible.

Report each of the three paired-seed effects and their mean with SD or range. A prompt bootstrap is descriptive task-level uncertainty and cannot replace uncertainty across training seeds. Preserve the same task populations in reported PCMD contrasts rather than averaging changing eligible subsets. No effect-size or significance threshold determines whether to launch, stop, or report the study.

## Independently audit and aggregate completed studies

The auditor requires the full reserved test cohort for a primary report and the full training cohort for a separate diagnostic report. It verifies source, prompt, checkpoint, raw-token and native-verifier bindings before metrics. Do not pass `--skip-tokenizer`. All 58 auditor tests and all 23 three-seed reducer tests passed, including heldout/cohort, zero-success and nested frozen-data provenance cases. Replace any restarted arm's training directory below with its completed replacement directory.

```bash
export LD_LIBRARY_PATH=/n/fs/similarity/maxent-grpo/var/seed_paper_eval/paper310/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHONPATH=ops:src
"$PILOT_PYTHON" ops/account_real_domains_pilot_20260921.py --root "$PILOT_ROOT"
for seed in 88301 88302 88303; do
  "$PILOT_PYTHON" ops/summarize_real_domains_pilot_20260921.py \
    --evaluation "qa=$PILOT_ROOT/qa_study_test_base" \
    --pair "qa=$PILOT_ROOT/qa_study_maxrl_seed${seed},$PILOT_ROOT/qa_study_remax_seed${seed}" \
    --endpoints "qa=$PILOT_ROOT/qa_study_test_base,$PILOT_ROOT/qa_study_test_seed${seed}_maxrl,$PILOT_ROOT/qa_study_test_seed${seed}_remax" \
    --qa-provenance "$PILOT_ROOT/../noncoding_multi_answer_sata_20260921/provenance_audit/news_source_comparison.json" \
    --accounting "$PILOT_ROOT/budget.json" --output "$PILOT_ROOT/qa_study_report_seed${seed}"
done
for seed in 88501 88502 88503; do
  "$PILOT_PYTHON" ops/summarize_real_domains_pilot_20260921.py \
    --evaluation "code=$PILOT_ROOT/code_study_test_base" \
    --pair "code=$PILOT_ROOT/code_study_maxrl_seed${seed},$PILOT_ROOT/code_study_remax_seed${seed}" \
    --endpoints "code=$PILOT_ROOT/code_study_test_base,$PILOT_ROOT/code_study_test_seed${seed}_maxrl,$PILOT_ROOT/code_study_test_seed${seed}_remax" \
    --accounting "$PILOT_ROOT/budget.json" --output "$PILOT_ROOT/code_study_report_seed${seed}"
done
```

For optional coding trained diagnostics, repeat the coding loop with `code_study_trained_base`, `code_study_trained_seed${seed}_{arm}`, and output `code_study_trained_report_seed${seed}`; retain the same training runs. A single-domain report may mark its global cross-domain readiness unknown because the other domain is absent. Reduction requires that report's requested-domain training and endpoint audits to pass.

```bash
"$PILOT_PYTHON" ops/aggregate_real_domains_seeds_20260921.py --domain qa \
  --report "$PILOT_ROOT/qa_study_report_seed88301/summary.json" \
  --report "$PILOT_ROOT/qa_study_report_seed88302/summary.json" \
  --report "$PILOT_ROOT/qa_study_report_seed88303/summary.json" \
  --output "$PILOT_ROOT/qa_study_three_seed_report"
"$PILOT_PYTHON" ops/aggregate_real_domains_seeds_20260921.py --domain code \
  --report "$PILOT_ROOT/code_study_report_seed88501/summary.json" \
  --report "$PILOT_ROOT/code_study_report_seed88502/summary.json" \
  --report "$PILOT_ROOT/code_study_report_seed88503/summary.json" \
  --output "$PILOT_ROOT/code_study_three_seed_report"
```

If all optional trained diagnostic endpoints were run, add three `--trained-report` arguments pointing to their `summary.json` files to the coding aggregation command. The reducer requires exactly three distinct paired training seeds and one fixed heldout cohort; it reports seed effects and the common-eligible PCMD sensitivity without discarding zero-success tasks. The explicit single-seed diagnostic mode is for pilot descriptions only.

## Budget and interpretation

| Future allocation envelope | GPUh |
|---|---:|
| QA: six training caps | 24 |
| QA: base plus six heldout endpoints | 3.5 |
| QA reserve | 10 |
| Coding: six training caps | 24 |
| Coding: base plus six heldout endpoints | 28 |
| Coding: optional base plus six trained diagnostics | 28 |
| Coding reserve | 10 |
| **Combined future envelope** | **127.5** |

All 22 pilot allocations are terminal. Final allocated pilot cost is **3.7894 GPUh**, including the corrected coding endpoint jobs 31436911 and 31436912 (1,341 and 1,349 seconds). Adding the complete 127.5-GPUh future envelope gives **131.2894 GPUh**, leaving **68.7106 GPUh** below 200. No live pilot allocation remains. The dated receipt is `code_hardened_larger_study_unsubmitted/budget_envelope.json`, bound to the final accounting snapshot `pilot_accounting_snapshot_20260921_212447.json`. Refresh accounting before future submissions:

```bash
"$PILOT_PYTHON" ops/account_real_domains_pilot_20260921.py --root "$PILOT_ROOT"
```

QA's smoke had no mixed-reward groups: MaxRL remained at its initial weights and Re:Max's coverage effects were small and mixed. Treat this as a ceiling-sensitive annotated-topic experiment, not demonstrated improvement or diverse reasoning. Hierarchical Reuters labels may overlap semantically.

Coding's heldout set has only 13 problems: 7 assignment, 4 ordered sequence, 2 unordered set; no heldout partition problem remains. All admitted source panels pass 12/12 positives and 12/12 negatives, but those panels also helped repair the verifier and do not establish an independent false-positive rate. Five source-derived supplemental inputs were appended to unchanged released suites/checkers. Results establish finite-suite correctness and observable witness coverage, not universal program correctness or algorithm diversity. Use the heldout endpoint as primary and report task-level uncertainty; do not choose tasks or a stopping point based on a favorable treatment effect.
