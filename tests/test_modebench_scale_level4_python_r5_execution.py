"""Synthetic saved-proof tests; no actual scheduler, model, grader or audit."""
from copy import deepcopy
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('python_execution_under_test', ROOT/'artifacts/verify_modebench_scale_level4_python_r5_execution_20260913.py')
a = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(a)
actual_reviewed = a.reviewed
JOB = 910  # Synthetic only, never a production job identity.


def put(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists(): path.chmod(0o600)  # Synthetic fixture mutation only.
    path.write_text(json.dumps(value, sort_keys=True, indent=2)+'\n')
    return path


def stamp(path, value='2026-09-12T19:54:00+00:00'):
    ns = int(datetime.fromisoformat(value).timestamp()*1_000_000_000)
    os.utime(path, ns=(ns, ns))


@pytest.fixture
def saved(tmp_path, monkeypatch):
    """Fully synthetic execution/proof fixture; job numbers only model routing."""
    state = tmp_path/'state'; root = state/'development'
    root.mkdir(parents=True)
    monkeypatch.setattr(a, 'STATE', state); monkeypatch.setattr(a, 'RECOVERY', root)
    review=put(tmp_path/'transport-review.json', {'synthetic':'reviewed'})
    monkeypatch.setattr(a, 'TRANSPORT_REVIEW', review)
    activation=put(tmp_path/'activation.json',{'transport_review_sha256':a.file_sha(review)})
    monkeypatch.setattr(a,'REVIEW',activation);monkeypatch.setattr(a,'reviewed',lambda *args,**kwargs:{})
    view = put(tmp_path/'view.json', {'scratch': True})
    source = tmp_path/'revision'; put(source/'protocol.json', {'scratch': True})
    pool = source/'level4/pools/python_factors'
    certificate = put(pool/'identity.json', {'scratch': True})
    tasks = []
    for tier in range(4):
        output = source/f'level4/results/development/python_factors/difficulty_{tier}.json'
        tasks.append({'level': 'level4', 'domain': 'python_factors', 'split': 'dev', 'output': str(output),
                      'seeds': [7204000,7204001,7204002,7204003], 'batch_size': 8, 'row_offset': 0, 'row_limit': 0})
    taskpath = put(root/'tasks/cell.json', tasks)
    cell = {'id': 'level4_python_factors_r5_dev', 'level': 'level4', 'domain': 'python_factors',
            'phase': 'dev', 'model_label': '7b', 'source_kind': 'domain_revision_v1',
            'source_root': str(source), 'tasks': str(taskpath), 'command': ['/scratch/python', '/scratch/evaluate', '--resume']}
    environment = {'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '1', 'VLLM_USE_V1': '0',
                   'VLLM_ATTENTION_BACKEND': 'XFORMERS', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                   'PYTHONDONTWRITEBYTECODE': '1'}
    ready=put(state/'certificate.json', {'preserved_python_failure_sha256':'old_failed_recipe'})
    put(state/'amendment.json', {'actual_r4_predecessor':{'revision':3,'scientific_gates_passed':False}})
    shared=put(tmp_path/'shared-analysis.json',{'synthetic':'same diagnostic'})
    contract={'schema':'modebench_scale_python_harder_portable_evidence_v1', 'shared_source':str(shared),
        'historical_destination':'/tmp/synthetic-only/analysis.json','sha256':a.file_sha(shared),
        'overwrite_permitted':False,'science_or_live_fence_changed':False}
    stager=put(tmp_path/'stager.py',{'scratch':'read-only contract helper'})
    stager_tests=put(tmp_path/'stager_tests.py',{'scratch':'stager tests'})
    preservation=put(tmp_path/'stager_preservation.json',{'scratch':'preservation'})
    stager_review=put(tmp_path/'stager_review.json',{'scratch':'reviewed stager'})
    txschema='modebench_scale_level4_python_r5_development_transport_v1'
    plan = {'schema':txschema,'readiness_path':str(ready),'provider':'synthetic_provider.py','created_at': '2026-09-12T19:50:00+00:00', 'cells': [cell], 'provider_sha256': 'provider',
            'readiness_sha256': 'ready', 'view_manifest': str(view), 'python': '/scratch/python',
            'inputs_sha256': {str(p):a.file_sha(p) for p in (taskpath,stager,stager_tests,shared,preservation,stager_review)}, 'review_sha256': a.file_sha(review)}
    put(root/'plan.json', plan); put(root/'plan.sha256.json', {'sha256': a.file_sha(root/'plan.json')})
    intent = {'at_utc': '2026-09-12T19:51:15.100000+00:00'}
    put(root/'submission_intent.json', intent)
    submitted = {'array_job_id': JOB, 'at_utc': '2026-09-12T19:51:15.600000+00:00'}
    put(root/'submission_result.json', submitted)
    env = {**environment, 'SLURMD_NODENAME': 'node202', 'SLURM_JOB_ACCOUNT': 'allcs', 'SLURM_JOB_PARTITION': 'cs',
           'SLURM_ARRAY_JOB_ID': str(JOB), 'SLURM_ARRAY_TASK_ID': '0', 'SLURM_JOB_ID': str(JOB),
           'SLURM_CPUS_PER_TASK': '6', 'SLURM_MEM_PER_NODE': '61440',
           'PYTHONPYCACHEPREFIX': '/tmp/NONPRODUCTION-scale-frozen-pycache-scratch/pycache'}
    runtime = {'schema': txschema, 'status': 'validated_before_unchanged_evaluator_subprocess',
        'array_job_id': JOB, 'array_index': 0, 'cell': cell['id'], 'level': 'level4',
        'plan_sha256': a.file_sha(root/'plan.json'), 'submission_result_sha256': a.file_sha(root/'submission_result.json'),
        'readiness_sha256': 'ready', 'provider_sha256': 'provider', 'view_manifest': str(view),
        'view_manifest_sha256': a.file_sha(view), 'evaluator_command': cell['command'], 'environment': env,
        'hostname': 'node202.ionic.cs.princeton.edu', 'vllm_version': '0.8.4',
        'portable_evidence_staging':{**contract,'observer_host':'node202.ionic.cs.princeton.edu','observer_pid':12345,
            'observer_uid':1000,'created_destination':True,'status':'created_exact_diagnostic_copy',
            'at_utc':'2026-09-12T19:51:18+00:00','destination_stat':{'device':50,'inode':123,'uid':1000,
            'size_bytes':shared.stat().st_size,'mtime_ns':123,'mode':'0o444'}},
        'visible_gpu_names': ['NVIDIA RTX A5000']*2, 'at_utc': '2026-09-12T19:51:40.546756+00:00',
        'outputs_before_exec': {t['output']: {'receipt_exists': False, 'batch_directory_exists': False} for t in tasks},
        'gpu_metadata_probe': {'command': [plan['python'], '-B', '-c',
            'import json,torch;print(json.dumps([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]))'],
            'timeout_seconds': 60, 'returncode': 0, 'stdout': '["NVIDIA RTX A5000", "NVIDIA RTX A5000"]\n',
            'stderr': '', 'started_at_utc': '2026-09-12T19:51:35+00:00', 'finished_at_utc': '2026-09-12T19:51:39+00:00'}}
    put(root/'runtime/0.json', runtime)
    outcome = {'schema': txschema, 'status': 'actual_evaluator_process_return_observed', 'array_job_id': JOB,
        'array_index': 0, 'cell': cell['id'], 'command': cell['command'], 'runtime_sha256': a.file_sha(root/'runtime/0.json'),
        'observer_host': runtime['hostname'], 'observer_pid': 12345, 'returncode': 0, 'scheduler_success_claimed': False,
        'started_at_utc': '2026-09-12T19:51:40.561303+00:00', 'finished_at_utc': '2026-09-12T19:57:19.651109+00:00'}
    put(root/'runtime/0.evaluator_exit.json', outcome)
    summaries, allrows, allpaths = [], [], []
    for tier, task in enumerate(tasks):
        receipt = put(task['output'], {'generated_at': '2026-09-12T19:54:00+00:00', 'metrics': {'pass@1': .25, 'pass@8': .5}})
        paths = [receipt, put(str(receipt)+'.batches/run.json', {'scratch': True})]
        for seed in task['seeds']:
            for start in range(0,193,8):
                paths.append(put(str(receipt)+f'.batches/seed-{seed}__rows-{start:06d}-{min(start+8,193):06d}.json', {'scratch': True}))
        for path in paths: stamp(path)
        summary = {'tier': tier, 'receipt': str(receipt), 'receipt_sha256': a.file_sha(receipt),
                   'rows': 193, 'batches': 100, 'attempts': 6176, 'metrics': {'pass@1': .25, 'pass@8': .5},
                   'rendered_prompts_sha256': f'prompt{tier}', 'source_certificate': str(certificate)}
        summaries.append(summary); allrows.append([{'scratch_row': i} for i in range(193)]); allpaths.append(paths)
    out = root/'logs'/f'{JOB}_0.out'; out.parent.mkdir()
    out.write_text('\n'.join(json.dumps({'event':'task_complete','output':s['receipt'],'metrics':s['metrics']}) for s in summaries)+'\n')
    err = root/'logs'/f'{JOB}_0.err'; err.write_text('scratch terminal warning\n')
    rows = []
    for suffix in ('', '.batch', '.extern'):
        state_value, code = ('COMPLETED','0:0')
        rows.append([str(JOB)+suffix, f'{JOB}_0'+suffix, state_value, code,
                     '2026-09-12T19:51:17','2026-09-12T19:57:19','node202','6','60G' if not suffix else '',
                     'cpu=6,gres/gpu:a5000=2,gres/gpu=2,mem=60G,node=1',
                     '2026-09-12T19:51:15' if not suffix else '2026-09-12T19:51:17','allcs','cs' if not suffix else ''])
    terminal = {'schema': a.TERMINAL_SCHEMA, 'array_job_id': JOB, 'command': a.terminal_command(JOB),
        'environment': {'TZ':'UTC'}, 'returncode': 0, 'stderr': '', 'stdout': '\n'.join('|'.join(r) for r in rows)+'\n',
        'observer_host': a.contract.CAPTURE_HOST, 'observer_uid': a.contract.UID, 'observer_pid': 5678, 'captured_at_utc': '2026-09-12T19:59:19.813404+00:00',
        'logs_sha256': {str(p): a.file_sha(p) for p in (out,err)}}
    put(root/'terminal_accounting.json', terminal)
    protocol = {'files_sha256': {str(source/'protocol.json'):a.file_sha(source/'protocol.json')}}
    modules = SimpleNamespace(revision=SimpleNamespace(authenticate=lambda source_root:protocol))
    transport = SimpleNamespace(SOURCE=a.TRANSPORT,STAGER=stager,STAGER_SHA=a.file_sha(stager),STAGER_TESTS=stager_tests,
        SHARED_EVIDENCE=shared,EVIDENCE_SHA=a.file_sha(shared),PRESERVATION=preservation,PRESERVATION_SHA=a.file_sha(preservation),
        STAGER_REVIEW=stager_review,STAGER_REVIEW_SHA=a.file_sha(stager_review),
        portable_helper=lambda:SimpleNamespace(contract=lambda:deepcopy(contract)),SCHEMA=txschema, NODES=['node202','node203','node204'], ENVIRONMENT=environment,
        verified=lambda actual_state:(plan, None, None, None, {}), submission_identity=lambda actual_state:submitted,
        DEPENDENCIES=[31254520])
    monkeypatch.setattr(a, 'load', lambda *args:transport)
    monkeypatch.setattr(a.cell_helper, 'scientific_modules', lambda:modules)
    completed_calls = []
    def completed(actual_modules, actual_plan, actual_cell, task, tier, actual_protocol, end):
        assert actual_modules is modules and actual_plan is plan and actual_cell is cell and actual_protocol is protocol
        assert end == {'worker_started_at_utc':outcome['started_at_utc'], 'end_utc':outcome['finished_at_utc']}
        completed_calls.append(tier)
        assert all(path.exists() for path in allpaths[tier])
        return summaries[tier], allrows[tier], allpaths[tier]
    monkeypatch.setattr(a.cell_helper, 'completed_task', completed)
    activation_pins = a.required_scoring_pins(root, a.file_sha(root/'terminal_accounting.json'))
    monkeypatch.setattr(a,'reviewed',lambda *args,**kwargs:dict(activation_pins))
    completed_calls.clear()
    return SimpleNamespace(root=root, state=state, plan=plan, cell=cell, submitted=submitted, runtime=runtime, outcome=outcome,
        terminal=terminal, terminal_rows=rows, transport=transport, summaries=summaries, rows=allrows, paths=allpaths,
        modules=modules, completed_calls=completed_calls, activation_pins=activation_pins)


def rewrite_runtime(saved, monkeypatch):
    put(saved.root/'runtime/0.json', saved.runtime)
    saved.outcome['runtime_sha256'] = a.file_sha(saved.root/'runtime/0.json')
    put(saved.root/'runtime/0.evaluator_exit.json', saved.outcome)


def test_composed_inspect_preserves_successful_actual_scheduler_sidecar_and_all_four_native_calls(saved):
    context = a.inspect(saved.root, a.file_sha(saved.root/'terminal_accounting.json'))
    assert context.execution['state'] == 'COMPLETED' and context.execution['exit_code'] == '0:0'
    assert context.execution['scheduler_success'] is True and context.execution['evaluator_returncode'] == 0
    assert context.preparation['preserved_python_r4_failure_sha256']=='old_failed_recipe'
    assert context.execution['portable_evidence_staging']['destination_stat']['device']==50
    assert context.execution['evaluator_finished_at_utc'].startswith('2026-09-12T19:57:19.651109')
    assert saved.completed_calls == [0,1,2,3]
    assert len(context.output_file_times) == 408 and sum(s['attempts'] for s in context.summaries) == 24704


@pytest.mark.parametrize('kind', ['main_failed', 'main_exit', 'extern_failed', 'missing_step', 'duplicate', 'raw_id',
    'account', 'partition', 'cpu', 'gpu', 'start', 'end', 'submit', 'timezone', 'command', 'observer', 'logs'])
def test_only_exact_observed_terminal_semantics_pass(saved, kind):
    value = deepcopy(saved.terminal); rows = deepcopy(saved.terminal_rows)
    if kind == 'main_failed': rows[0][2:4] = ['FAILED','1:0']
    elif kind == 'main_exit': rows[0][3] = '143:0'
    elif kind == 'extern_failed': rows[2][2:4] = ['FAILED','1:0']
    elif kind == 'missing_step': rows.pop()
    elif kind == 'duplicate': rows[2] = rows[1]
    elif kind == 'raw_id': rows[0][0] = '999'
    elif kind == 'account': rows[0][11] = 'mltheory'
    elif kind == 'partition': rows[0][12] = 'all'
    elif kind == 'cpu': rows[0][7] = '4'
    elif kind == 'gpu': rows[0][9] = rows[0][9].replace('gpu=2','gpu=1')
    elif kind == 'start': rows[0][4] = '2026-09-12T19:51:18'
    elif kind == 'end': rows[0][5] = '2026-09-12T19:57:18'
    elif kind == 'submit': rows[0][10] = '2026-09-12T19:51:16'
    elif kind == 'timezone': value['environment'] = {'TZ':'America/New_York'}
    elif kind == 'command': value['command'] = ['sacct','-X']
    elif kind == 'observer': value['observer_host'] = 'wash.cs.princeton.edu'
    elif kind == 'logs': value['logs_sha256'] = {}
    value['stdout'] = '\n'.join('|'.join(r) for r in rows)+'\n'
    with pytest.raises(ValueError): a.terminal_execution(saved.root, saved.plan, saved.submitted, saved.runtime, saved.outcome, value)


@pytest.mark.parametrize('kind', ['return1', 'return_minus15', 'boolean_return', 'outcome_command', 'outcome_host',
    'scheduler_claim', 'finish_next_second', 'wrong_runtime_hash', 'gpu_name', 'probe_command', 'probe_timeout',
    'probe_after_entry', 'runtime_plan', 'ready', 'environment', 'preexisting_output', 'wrong_index'])
def test_runtime_and_evaluator_return_are_independent_strict_evidence(saved, monkeypatch, kind):
    if kind == 'return1': saved.outcome['returncode'] = 1
    elif kind == 'return_minus15': saved.outcome['returncode'] = -15
    elif kind == 'boolean_return': saved.outcome['returncode'] = False
    elif kind == 'outcome_command': saved.outcome['command'] = ['different']
    elif kind == 'outcome_host': saved.outcome['observer_host'] = 'node203.ionic.cs.princeton.edu'
    elif kind == 'scheduler_claim': saved.outcome['scheduler_success_claimed'] = True
    elif kind == 'finish_next_second': saved.outcome['finished_at_utc'] = '2026-09-12T19:57:20+00:00'
    elif kind == 'wrong_runtime_hash': saved.runtime['cell'] = 'different'
    elif kind == 'gpu_name': saved.runtime['visible_gpu_names'] = ['NVIDIA A100']*2
    elif kind == 'probe_command': saved.runtime['gpu_metadata_probe']['command'] = ['changed']
    elif kind == 'probe_timeout': saved.runtime['gpu_metadata_probe']['timeout_seconds'] = 0
    elif kind == 'probe_after_entry': saved.runtime['gpu_metadata_probe']['finished_at_utc'] = '2026-09-12T19:52:00+00:00'
    elif kind == 'runtime_plan': saved.runtime['plan_sha256'] = 'wrong'
    elif kind == 'ready': saved.runtime['readiness_sha256'] = 'wrong'
    elif kind == 'environment': saved.runtime['environment']['OMP_NUM_THREADS'] = '1'
    elif kind == 'preexisting_output': saved.runtime['outputs_before_exec'][saved.summaries[0]['receipt']]['receipt_exists'] = True
    elif kind == 'wrong_index': saved.runtime['array_index'] = 1
    rewrite_runtime(saved, monkeypatch)
    with pytest.raises(ValueError):
        runtime, outcome = a.runtime_linkage(saved.root, saved.transport, saved.plan, saved.submitted)
        a.terminal_execution(saved.root, saved.plan, saved.submitted, runtime, outcome, saved.terminal)


def test_explicit_capture_and_runtime_sidecar_linkage_are_required(saved):
    changed={**saved.outcome,'runtime_sha256':'wrong'}
    put(saved.root/'runtime/0.evaluator_exit.json',changed)
    with pytest.raises(ValueError):a.runtime_linkage(saved.root,saved.transport,saved.plan,saved.submitted)
    with pytest.raises(ValueError,match='SHA'):a.inspect(saved.root,'0'*64)


@pytest.mark.parametrize('when', ['2026-09-12T19:51:40.550000+00:00','2026-09-12T19:57:19.700000+00:00'])
def test_output_mtime_must_lie_inside_exact_evaluator_interval(saved, when):
    stamp(saved.paths[0][0], when)
    with pytest.raises(ValueError, match='saved output'): a.output_times(saved.paths[0], saved.outcome)


def test_last_scheduler_second_is_accepted_without_rewriting_timestamps(saved):
    stamp(saved.paths[0][0], '2026-09-12T19:57:19.600000+00:00')
    assert a.output_times(saved.paths[0], saved.outcome)
    result = a.terminal_execution(saved.root, saved.plan, saved.submitted, saved.runtime, saved.outcome, saved.terminal)
    assert result['end_utc'] == '2026-09-12T19:57:19'
    assert result['evaluator_finished_at_utc'] == '2026-09-12T19:57:19.651109+00:00'


@pytest.mark.parametrize('kind', ['missing_receipt', 'missing_batch', 'wrong_totals', 'prior_runtime', 'extra_completion_event'])
def test_registration_waits_complete_outputs_and_clean_runtime(saved, kind):
    if kind == 'missing_receipt': saved.paths[3][0].unlink()
    elif kind == 'missing_batch': saved.paths[3][-1].unlink()
    elif kind == 'wrong_totals': saved.summaries[3]['batches'] = 99
    elif kind == 'prior_runtime': put(saved.root/'runtime/extra.json', {'unexpected':True})
    else:
        path = saved.root/'logs'/f'{JOB}_0.out'
        path.write_text(path.read_text()+json.dumps({'event':'task_complete','output':'extra','metrics':{}})+'\n')
    with pytest.raises((AssertionError, ValueError, FileNotFoundError)):
        a.register(saved.root, a.file_sha(saved.root/'terminal_accounting.json'))
    assert not (saved.root/'execution_audit/registration.json').exists()
    assert not (saved.root/'execution_audit/claim.json').exists()


def grader_fixture(saved, monkeypatch, *, fail_tier=None):
    calls = []
    monkeypatch.setattr(a.cell_helper, 'tokenizer_for', lambda context:object())
    monkeypatch.setattr(a.cell_helper, 'native_prompts', lambda context,tier,tokenizer:
                        {'rendered_prompts_sha256':context.summaries[tier]['rendered_prompts_sha256']})
    def grader(output, rows, protocol, phase, *, regrade):
        tier = next(i for i,s in enumerate(saved.summaries) if s['receipt'] == output)
        assert (saved.root/'execution_audit/claim.json').exists()
        assert phase == 'dev' and regrade is True and len(rows) == 193
        calls.append(tier)
        if tier == fail_tier: raise RuntimeError('synthetic native disagreement')
        return {'metrics':saved.summaries[tier]['metrics']}, {str(i):{} for i in range(193)}
    saved.modules.revision.receipt_scores = grader
    return calls


def test_one_durable_grader_pass_and_readonly_verify_never_regrades(saved, monkeypatch):
    calls = grader_fixture(saved, monkeypatch)
    result = a.audit(saved.root, a.file_sha(saved.root/'terminal_accounting.json'))
    assert calls == [0,1,2,3]
    assert result['new_grader_invocations'] == 24704 and result['completed_batches'] == 400
    assert result['scheduler_success'] is True and result['status']==a.STATUS
    assert result['evaluator_returncode'] == 0 and result['terminal_policy']==a.POLICY
    assert result == a.verify_existing(saved.root)
    assert calls == [0,1,2,3]
    with pytest.raises(ValueError, match='already claimed'): a.audit(saved.root, a.file_sha(saved.root/'terminal_accounting.json'))
    assert calls == [0,1,2,3]


def test_grader_exception_saves_failure_and_cannot_retry(saved, monkeypatch):
    calls = grader_fixture(saved, monkeypatch, fail_tier=1)
    with pytest.raises(RuntimeError, match='synthetic'): a.audit(saved.root, a.file_sha(saved.root/'terminal_accounting.json'))
    assert calls == [0,1] and (saved.root/'execution_audit/failure.json').exists()
    assert not (saved.root/'execution_reconciliation.json').exists()
    with pytest.raises(ValueError, match='already claimed'): a.audit(saved.root, a.file_sha(saved.root/'terminal_accounting.json'))
    assert calls == [0,1]


def test_claim_survives_interruption_and_partial_report_is_not_completion(saved, monkeypatch):
    calls = grader_fixture(saved, monkeypatch)
    def interrupted(*args, **kwargs): raise KeyboardInterrupt()
    monkeypatch.setattr(a.cell_helper, 'tokenizer_for', interrupted)
    with pytest.raises(KeyboardInterrupt): a.audit(saved.root, a.file_sha(saved.root/'terminal_accounting.json'))
    assert (saved.root/'execution_audit/claim.json').exists() and calls == []
    with pytest.raises(ValueError, match='already claimed'): a.audit(saved.root, a.file_sha(saved.root/'terminal_accounting.json'))


@pytest.mark.parametrize('kind', ['claim', 'report', 'certificate', 'output', 'mtime'])
def test_readonly_verification_rejects_mutated_proof_without_regrading(saved, monkeypatch, kind):
    calls = grader_fixture(saved, monkeypatch); a.audit(saved.root, a.file_sha(saved.root/'terminal_accounting.json'))
    if kind == 'claim':
        path = saved.root/'execution_audit/claim.json'; value = a.read(path); value['script_sha256'] = 'wrong'; put(path,value)
    elif kind == 'report':
        path = saved.root/'execution_audit/tiers/0.json'; value = a.read(path); value['new_grader_invocations'] = 1; put(path,value)
    elif kind == 'certificate':
        path = saved.root/'execution_reconciliation.json'; value = a.read(path); value['scheduler_success'] = False; put(path,value)
    elif kind == 'output': put(saved.paths[0][-1], {'changed':True})
    else: stamp(saved.paths[0][-1], '2026-09-12T19:55:00+00:00')
    with pytest.raises(ValueError): a.verify_existing(saved.root)
    assert calls == [0,1,2,3]


@pytest.mark.parametrize('corrupt_batch', [False, True])
def test_real_sealed_completed_task_interface_and_batch_draw_checks(saved, corrupt_batch):
    # Load the sealed helper anew so the composed fixture's delegation stub is
    # bypassed. Only external evaluator/model metadata functions are scratch.
    spec = importlib.util.spec_from_file_location('real_saved_batch_validator', a.CELL)
    sealed = importlib.util.module_from_spec(spec); spec.loader.exec_module(sealed)
    task = a.read(saved.cell['tasks'])[0]
    rows = [{'problem': 'SCRATCH_NATIVE_INTERFACE_'+str(i), 'answer': '{}'} for i in range(193)]
    source_path = Path(saved.cell['source_root'])/'level4/pools/python_factors/difficulty_0.jsonl'
    source_path.write_text('\n'.join(json.dumps(row) for row in rows)+'\n')
    source = {'path': str(source_path), 'file_sha256': a.file_sha(source_path)}
    labels = task['seeds']; schedule = {'scratch_exact_rows': a.sha(rows), 'seeds': labels}
    runtime_settings = {'max_model_len':2048,'tensor_parallel_size':2,'gpu_memory_utilization':.82,
                        'swap_space':4.0,'enable_prefix_caching':True}
    interface = {'scratch_native_interface':True}
    model = {'path':'/scratch/model','model_label':'7b'}
    identity = {'source':source,'seeds':labels,'runtime':runtime_settings,'interface':interface,
                'model':{**model,'vllm_version':'0.8.4'},'seed_schedule_sha256':a.sha(schedule),
                'rendered_prompts_sha256':'scratch_rendered_prompts'}
    results = [{'draws':[{'attempts':[{'text':f'SCRATCH_{i}_{seed}_{j}','canonical_key':None} for j in range(8)]}
                         for seed in labels]} for i in range(193)]
    receipt = {'level':'level4','domain':'python_factors','split':'dev','model_label':'7b',
               'identity':identity,'identity_sha256':a.sha(identity),'generated_at':'2026-09-12T19:54:00+00:00',
               'prompt_results':results,'metrics':{'pass@1':.25,'pass@8':.5}}
    put(task['output'], receipt)
    directory = Path(task['output']+'.batches')
    put(directory/'run.json', {'identity':identity,'identity_sha256':receipt['identity_sha256']})
    for index, seed in enumerate(labels):
        for start in range(0,193,8):
            end = min(start+8,193); draws = [r['draws'][index] for r in results[start:end]]
            put(directory/f'seed-{seed}__rows-{start:06d}-{end:06d}.json',
                {'seed':seed,'start':start,'end':end,'identity_sha256':receipt['identity_sha256'],
                 'draws':draws,'draws_sha256':a.sha(draws)})
    if corrupt_batch:
        path = directory/f'seed-{labels[0]}__rows-000000-000008.json'
        batch = a.read(path); batch['draws'][0]['attempts'][0]['text'] = 'changed'
        batch['draws_sha256'] = a.sha(batch['draws']); put(path,batch)
    certificate = Path(saved.cell['source_root'])/'level4/pools/python_factors/identity.json'
    put(certificate, {'protocol_sha256':a.file_sha(Path(saved.cell['source_root'])/'protocol.json'),
                      'tiers':{'0':{'rows':193,'rows_sha256':a.sha(rows)}}})
    def validate_task(actual, confirmation):
        assert actual is task and confirmation is False
    def validate_receipt(actual, actual_rows):
        assert actual_rows is rows and len(actual['prompt_results']) == 193
    def runtime(**kwargs):
        assert kwargs == runtime_settings
        return runtime_settings
    evaluator = SimpleNamespace(validate_task=validate_task, load_rows=lambda t:(rows,source),
        validate_seed_receipt=validate_receipt, runtime_settings=runtime,
        frozen_interface=lambda domain:interface, model_identity=lambda path,label:model,
        frozen=SimpleNamespace(schedule_record=lambda domain,actual_rows,seeds:schedule))
    plan = {**saved.plan,'models':{'7b':{'path':'/scratch/model'}},
            'rng_admission':{'task_seed_schedule_sha256':{task['output']:a.sha(schedule)}}}
    args = (SimpleNamespace(evaluator=evaluator),plan,saved.cell,task,0,{'draw_labels':{'dev':labels}},
            {'worker_started_at_utc':saved.outcome['started_at_utc'],'end_utc':saved.outcome['finished_at_utc']})
    if corrupt_batch:
        with pytest.raises(ValueError, match='saved batch draws'): sealed.completed_task(*args)
    else:
        summary, actual_rows, paths = sealed.completed_task(*args)
        assert summary['rows'] == 193 and summary['batches'] == 100 and summary['attempts'] == 6176
        assert actual_rows is rows and len(paths) == 102


@pytest.mark.parametrize('kind',['destination','overwrite','host','pid','uid','status','created_mode','public_mode','size','after_probe'])
def test_exact_staging_record_is_audited_without_replaying_foreign_device(saved,monkeypatch,kind):
    staging=saved.runtime['portable_evidence_staging']
    if kind=='destination':staging['historical_destination']='/tmp/other'
    elif kind=='overwrite':staging['overwrite_permitted']=True
    elif kind=='host':staging['observer_host']='spin.cs.princeton.edu'
    elif kind=='pid':staging['observer_pid']=99999
    elif kind=='uid':staging['destination_stat']['uid']=999
    elif kind=='status':staging['status']='invented'
    elif kind=='created_mode':staging['destination_stat']['mode']='0o644'
    elif kind=='public_mode':staging['destination_stat']['mode']='0o666'
    elif kind=='size':staging['destination_stat']['size_bytes']+=1
    else:staging['at_utc']='2026-09-12T19:51:36+00:00'
    rewrite_runtime(saved,monkeypatch)
    with pytest.raises(ValueError,match='staging evidence'):
        a.runtime_linkage(saved.root,saved.transport,saved.plan,saved.submitted)


def test_foreign_staging_inode_device_are_truthful_history_only(saved,monkeypatch):
    staging=saved.runtime['portable_evidence_staging']
    staging['destination_stat']['device']=987654
    staging['destination_stat']['inode']=123456789
    staging['created_destination']=False
    staging['status']='verified_existing_exact_diagnostic'
    staging['destination_stat']['mode']='0o644'
    rewrite_runtime(saved,monkeypatch)
    runtime,_=a.runtime_linkage(saved.root,saved.transport,saved.plan,saved.submitted)
    assert runtime['portable_evidence_staging']==staging
    assert not Path(staging['historical_destination']).exists()




@pytest.mark.parametrize('status',['draft','pending',None])
def test_final_review_must_exist_and_be_reviewed_before_native_claim(tmp_path,monkeypatch,status):
    review=put(tmp_path/'review.json',{'status':status,'files_sha256':{}})
    monkeypatch.setattr(a,'REVIEW',review)
    with pytest.raises(ValueError):actual_reviewed(a.file_sha(review))
    assert not (tmp_path/'execution_audit').exists()

def test_native_audit_and_runtime_functions_are_identical_sealed_predecessor_objects():
    # r5 wraps the true original, not the predecessor wrapper
    assert a._original_runtime_linkage is a.engine._original_runtime_linkage
    assert a.output_times is a.engine.output_times
    assert a.engine.audit.__code__.co_filename == str(a.PRIOR)
    assert a.cell_helper.completed_task.__code__.co_filename == str(a.CELL)


@pytest.fixture
def activation(tmp_path,monkeypatch):
    for key in ('SOURCE','TESTS','RECORDER','RECORDER_TESTS','PRIOR','PRIOR_TESTS','CONTRACT','CONTRACT_TESTS',
                'TRANSPORT','TRANSPORT_TESTS','PREPARATION','PREPARATION_TESTS','CELL'):
        monkeypatch.setattr(a,key,put(tmp_path/(key+'.json'),{'scratch':key}))
    state=tmp_path/'state';root=state/'development'
    monkeypatch.setattr(a,'STATE',state);monkeypatch.setattr(a,'RECOVERY',root)
    monkeypatch.setattr(a,'REVIEW',tmp_path/'final_review.json')
    monkeypatch.setattr(a,'TRANSPORT_REVIEW',tmp_path/'transport_review.json')
    monkeypatch.setattr(a,'PREPARATION_REVIEW',tmp_path/'preparation_review.json')
    monkeypatch.setattr(a,'TRANSPORT_SHA',a.file_sha(a.TRANSPORT));monkeypatch.setattr(a,'PREPARATION_SHA',a.file_sha(a.PREPARATION))
    dependency=put(tmp_path/'dependency.json',{'scratch':'full transitive dependency'})
    history=put(tmp_path/'negative_history.json',{'scratch':'completed negative r3'})
    monkeypatch.setattr(a.contract,'negative_predecessor_pins',lambda **kwargs:{str(history):a.file_sha(history)})
    for p,schema in ((a.TRANSPORT_REVIEW,'modebench_scale_level4_python_r5_transport_independent_review_v1'),
                     (a.PREPARATION_REVIEW,'modebench_scale_level4_python_r5_registration_independent_review_v1')):
        put(p,{'schema':schema,'status':'reviewed','files_sha256':{str(dependency):a.file_sha(dependency)}})
    paths=[state/'certificate.json',state/'amendment.json',*[root/name for name in ('plan.json','plan.sha256.json',
        'submission_intent.json','submission_result.json','terminal_accounting.json','runtime/0.json','runtime/0.evaluator_exit.json')]]
    for path in paths:put(path,{'scratch':True})
    put(root/'submission_result.json',{'array_job_id':70000001})
    pins={str(p):a.file_sha(p) for p in tmp_path.rglob('*') if p.is_file()}
    pins.update(a.preserved_history_pins(guest=True))
    value={'schema':a.REVIEW_SCHEMA,'status':'reviewed','array_job_id':70000001,'terminal_policy':a.POLICY,
        'transport_review_sha256':a.file_sha(a.TRANSPORT_REVIEW),'terminal_sha256':a.file_sha(root/'terminal_accounting.json'),
        'runtime_sha256':a.file_sha(root/'runtime/0.json'),'evaluator_exit_sha256':a.file_sha(root/'runtime/0.evaluator_exit.json'),
        'files_sha256':pins}
    put(a.REVIEW,value)
    return SimpleNamespace(value=value,root=root,history=history,dependency=dependency)

@pytest.fixture
def observed_activation(activation,monkeypatch):
    """activation uses a fresh job id, so the OBSERVED_JOB branch of reviewed() is never
    entered. That gap is why an r4 exit_cause literal survived into r5's gate. This
    fixture drives the real authorized branch."""
    root=activation.root
    put(root/'submission_result.json',{'array_job_id':a.contract.OBSERVED_JOB})
    value=dict(activation.value)
    value['array_job_id']=a.contract.OBSERVED_JOB
    value['observed_terminal_decision_sha256']=a.contract.OBSERVED_DECISION_SHA
    value['user_authorization_sha256']=a.contract.AUTHORIZATION_SHA
    value['scheduler_success']=False
    value['exit_cause']=a.contract.execution_classification(a.contract.OBSERVED_JOB)['exit_cause']
    pins=dict(value['files_sha256'])
    pins[str(root/'submission_result.json')]=a.file_sha(root/'submission_result.json')
    pins.update(a.contract.observed_terminal_pins(guest=True))
    value['files_sha256']=pins
    put(a.REVIEW,value)
    return SimpleNamespace(value=value,root=root)


def test_authorized_branch_binds_the_real_classification_not_a_predecessor_literal(observed_activation):
    """The exit cause in the review must be the one this revision's contract derives."""
    expected=a.contract.execution_classification(a.contract.OBSERVED_JOB)['exit_cause']
    assert expected=='post_completion_teardown'
    assert observed_activation.value['exit_cause']==expected
    pins=actual_reviewed(a.file_sha(a.REVIEW))
    assert pins[str(a.REVIEW)]==a.file_sha(a.REVIEW)


@pytest.mark.parametrize('bad',['unknown','successful_execution','',None])
def test_authorized_branch_refuses_any_other_exit_cause(observed_activation,bad):
    """'unknown' is revision 4's vocabulary; writing it here would erase the very fact
    the user's authorization rests on."""
    value=dict(observed_activation.value);value['exit_cause']=bad;put(a.REVIEW,value)
    with pytest.raises(ValueError,match='explicit user authorization'):actual_reviewed(a.file_sha(a.REVIEW))


@pytest.mark.parametrize('field',['observed_terminal_decision_sha256','user_authorization_sha256'])
def test_authorized_branch_refuses_a_substituted_permission_digest(observed_activation,field):
    value=dict(observed_activation.value);value[field]='0'*64;put(a.REVIEW,value)
    with pytest.raises(ValueError,match='explicit user authorization'):actual_reviewed(a.file_sha(a.REVIEW))


def test_actual_review_contract_accepts_complete_scratch_evidence_closure(activation):
    pins=actual_reviewed(a.file_sha(a.REVIEW))
    assert pins[str(a.REVIEW)]==a.file_sha(a.REVIEW)
    assert str(activation.history) in pins

@pytest.mark.parametrize('kind',['source','recorder','dependency','history','future_audit','terminal','old_job','transport_review'])
def test_actual_auditor_review_rejects_missing_or_rehashed_wrong_observations(activation,kind):
    value=activation.value
    if kind=='source':value['files_sha256'].pop(str(a.SOURCE))
    elif kind=='recorder':value['files_sha256'].pop(str(a.RECORDER))
    elif kind=='dependency':value['files_sha256'].pop(str(activation.dependency))
    elif kind=='history':value['files_sha256'].pop(str(activation.history))
    elif kind=='future_audit':value['files_sha256'][str(activation.root/'execution_audit/claim.json')]='0'*64
    elif kind=='terminal':value['terminal_sha256']='0'*64
    elif kind=='old_job':value['array_job_id']=31260226
    else:value['transport_review_sha256']='0'*64
    put(a.REVIEW,value)
    with pytest.raises(ValueError):actual_reviewed(a.file_sha(a.REVIEW))
    assert not (activation.root/'execution_audit').exists()


def test_required_scoring_pins_collects_complete_readonly_closure_without_activation(saved,monkeypatch):
    def no_review(*args,**kwargs):
        raise AssertionError('read-only collection must not depend on activation')
    monkeypatch.setattr(a,'reviewed',no_review)
    pins=a.required_scoring_pins(saved.root,a.file_sha(saved.root/'terminal_accounting.json'))
    assert pins==saved.activation_pins and saved.completed_calls==[0,1,2,3]
    assert all(pins[str(path)]==a.file_sha(path) for paths in saved.paths for path in paths)
    assert not (saved.root/'execution_audit').exists()

@pytest.mark.parametrize('kind',['source','receipt','batch'])
@pytest.mark.parametrize('mutation',['omitted','rehashed'])
def test_every_consumed_source_receipt_and_batch_is_reviewed_before_any_claim(saved,monkeypatch,kind,mutation):
    calls=grader_fixture(saved,monkeypatch)
    path = (Path(saved.summaries[0]['source_certificate']) if kind=='source'
            else saved.paths[0][0 if kind=='receipt' else 2])
    if mutation=='omitted':
        saved.activation_pins.pop(str(path))
    else:
        value=a.read(path);value['scratch_changed_after_review']=True;put(path,value)
        if kind!='source':stamp(path)
        if kind=='receipt':saved.summaries[0]['receipt_sha256']=a.file_sha(path)
        # Current structural evidence is internally consistent; the review still
        # identifies the older bytes and must reject them before native grading.
        assert saved.activation_pins[str(path)]!=a.file_sha(path)
    with pytest.raises(ValueError,match='pre-pin every consumed'):
        a.audit(saved.root,a.file_sha(saved.root/'terminal_accounting.json'))
    assert not calls and not (saved.root/'execution_audit/claim.json').exists()
    assert not (saved.root/'execution_audit/registration.json').exists()


@pytest.fixture
def authorized_saved(saved,monkeypatch):
    from test_modebench_scale_level4_python_r5_postscoring_contract import bind_authorized_terminal
    evidence=bind_authorized_terminal(a.contract,monkeypatch,saved.root,saved.plan,saved.submitted,
                                      saved.runtime,saved.outcome,saved.terminal)
    saved.activation_pins.clear();saved.activation_pins.update(a.required_scoring_pins(saved.root,a.file_sha(saved.root/'terminal_accounting.json')))
    saved.completed_calls.clear()
    return saved,evidence

def test_authorized_failed_job_runs_original_four_native_stubs_once_then_readonly(authorized_saved,monkeypatch):
    saved,evidence=authorized_saved;calls=grader_fixture(saved,monkeypatch)
    proof=a.audit(saved.root,a.file_sha(saved.root/'terminal_accounting.json'))
    assert proof['scheduler_success'] is False and proof['exit_cause']=='post_completion_teardown'
    assert proof['execution']['state']=='FAILED' and proof['execution']['exit_code']=='143:0'
    assert proof['user_authorization_sha256']==a.file_sha(evidence.authorization)
    assert proof['new_grader_invocations']==24704 and len(calls)==4
    assert a.verify_existing(saved.root)==proof and len(calls)==4
    with pytest.raises(ValueError,match='already claimed'):a.audit(saved.root,proof['terminal_sha256'])

@pytest.mark.parametrize('kind',['authorization','decision','receipt','batch'])
def test_authorized_failed_job_cannot_bypass_pre_audit_review_closure(authorized_saved,monkeypatch,kind):
    saved,e=authorized_saved;calls=grader_fixture(saved,monkeypatch)
    path={'authorization':e.authorization,'decision':e.decision,
          'receipt':saved.paths[0][0],'batch':saved.paths[0][2]}[kind]
    saved.activation_pins.pop(str(path))
    with pytest.raises(ValueError,match='pre-pin every consumed'):a.audit(saved.root,a.file_sha(saved.root/'terminal_accounting.json'))
    assert not calls and not (saved.root/'execution_audit/claim.json').exists()


def test_actual_r5_plan_layout_uses_only_ephemeral_contract_and_original_runtime_checks(saved,monkeypatch):
    assert 'portable_evidence' not in saved.plan
    before=deepcopy(saved.plan);disk=(saved.root/'plan.json').read_bytes();calls=[]
    original=a._original_runtime_linkage
    def checked(root,transport,view,submitted):
        assert view is not saved.plan and {k:v for k,v in view.items() if k!='portable_evidence'}==saved.plan
        assert view['portable_evidence']==saved.transport.portable_helper().contract()
        calls.append('unchanged original runtime checker')
        return original(root,transport,view,submitted)
    monkeypatch.setattr(a,'_original_runtime_linkage',checked)
    runtime,outcome=a.runtime_linkage(saved.root,saved.transport,saved.plan,saved.submitted)
    assert runtime==saved.runtime and outcome==saved.outcome and len(calls)==1
    assert saved.plan==before and (saved.root/'plan.json').read_bytes()==disk
    assert runtime['plan_sha256']==a.file_sha(saved.root/'plan.json')

def test_fixture_layout_matches_literal_registered_r5_transport_plan_source():
    import ast
    source=ast.parse(a.TRANSPORT.read_text())
    prepare=next(node for node in source.body if isinstance(node,ast.FunctionDef) and node.name=='prepare')
    plans=[node.value for node in ast.walk(prepare) if isinstance(node,ast.Assign)
           and any(isinstance(target,ast.Name) and target.id=='plan' for target in node.targets)]
    assert len(plans)==1 and isinstance(plans[0],ast.Dict)
    keys={key.value for key in plans[0].keys if isinstance(key,ast.Constant)}
    assert 'portable_evidence' not in keys and {'schema','inputs_sha256'} <= keys

@pytest.mark.parametrize('kind',['embedded_contract','wrong_schema','missing_stager','missing_shared','missing_preservation','missing_review','changed_stager','changed_shared','changed_preservation','wrong_transport'])
def test_metadata_projection_rejects_conflicting_or_unpinned_layout_inputs(saved,monkeypatch,kind):
    plan=deepcopy(saved.plan);transport=saved.transport
    if kind=='embedded_contract':plan['portable_evidence']=transport.portable_helper().contract()
    elif kind=='wrong_schema':plan['schema']='old_transport'
    elif kind.startswith('missing_'):
        attr={'missing_stager':'STAGER','missing_shared':'SHARED_EVIDENCE','missing_preservation':'PRESERVATION','missing_review':'STAGER_REVIEW'}[kind]
        plan['inputs_sha256'].pop(str(getattr(transport,attr)))
    elif kind.startswith('changed_'):
        attr={'changed_stager':'STAGER','changed_shared':'SHARED_EVIDENCE','changed_preservation':'PRESERVATION'}[kind]
        path=getattr(transport,attr);path.write_text(path.read_text()+' ')
        plan['inputs_sha256'][str(path)]=a.file_sha(path)
    else:monkeypatch.setattr(transport,'SOURCE',saved.root/'other_transport.py')
    with pytest.raises(ValueError):a.runtime_linkage(saved.root,transport,plan,saved.submitted)
    assert not (saved.root/'execution_audit/claim.json').exists()

def test_derived_contract_still_must_match_every_recorded_staging_field(saved,monkeypatch):
    contract=deepcopy(saved.transport.portable_helper().contract());contract['historical_destination']='/tmp/unrecorded-destination.json'
    monkeypatch.setattr(saved.transport,'portable_helper',lambda:SimpleNamespace(contract=lambda:contract))
    with pytest.raises(ValueError,match='staging evidence'):
        a.runtime_linkage(saved.root,saved.transport,saved.plan,saved.submitted)

@pytest.mark.parametrize('kind',['preserved_generator','burned_pilot_row'])
def test_final_review_requires_the_preserved_generator_and_every_burned_pilot_row(activation,kind):
    """Dropping either the sealed generator bytes or one model-exposed pilot row must
    be refused: a pilot row missing from the closure is a re-emission exposure."""
    pins=a.preserved_history_pins(guest=True)
    path=next(p for p in pins if p.endswith('.py')) if kind=='preserved_generator' \
        else next(p for p in pins if p.endswith('.jsonl'))
    value=activation.value;value['files_sha256'].pop(path);put(a.REVIEW,value)
    with pytest.raises(ValueError,match='preserved generator bytes or a burned'):actual_reviewed(a.file_sha(a.REVIEW))


# --- Real-record binding for the predecessor labels ---------------------------
# The revision token advanced r3->r4 in the predecessor's auditor but not r4->r5
# here, producing a KeyError on every audit and, on the adjacent line, a silent
# false provenance claim that would have been published into the certificate.
# Both survived because the fixture stamps the amendment from the same token.

REAL_STATE = ROOT/'var/artifacts/modebench_scale_level4_python_r5_20260913'


def test_auditor_reads_the_predecessor_key_the_sealed_amendment_actually_has():
    amendment=a.read(REAL_STATE/'amendment.json')
    assert 'actual_r4_predecessor' in amendment
    assert 'actual_r3_predecessor' not in amendment
    src=Path(a.SOURCE).read_text()
    assert "amendment['actual_r4_predecessor']" in src
    assert "amendment['actual_r3_predecessor']" not in src


def test_preserved_failure_digest_is_the_revision_four_recipe_not_revision_three():
    """The label must name the revision the digest actually came from."""
    amendment=a.read(REAL_STATE/'amendment.json')
    r4=ROOT/'var/data/modebench_scale_domain_revisions_v1/level4_python_factors_r4/level4/recipes/python_factors.json'
    r3=ROOT/'var/data/modebench_scale_domain_revisions_v1/level4_python_factors_r3/level4/recipes/python_factors.json'
    assert amendment['failed_recipe_sha256']==a.file_sha(r4)
    assert amendment['failed_recipe_sha256']!=a.file_sha(r3)
    src=Path(a.SOURCE).read_text()
    assert 'preserved_python_r4_failure_sha256' in src
    assert 'preserved_python_r3_failure_sha256' not in src
