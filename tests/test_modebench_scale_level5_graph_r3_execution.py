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
SPEC = importlib.util.spec_from_file_location('graph_execution_under_test', ROOT/'artifacts/verify_modebench_scale_level5_graph_r3_execution_20260912.py')
a = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(a)


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
    state = tmp_path/'state'; root = state/'development_recovery_1'
    root.mkdir(parents=True); (state/'development/runtime').mkdir(parents=True)
    monkeypatch.setattr(a, 'STATE', state); monkeypatch.setattr(a, 'RECOVERY', root)
    view = put(tmp_path/'view.json', {'scratch': True})
    source = tmp_path/'revision'; put(source/'protocol.json', {'scratch': True})
    pool = source/'level5/pools/graph_coloring'
    certificate = put(pool/'identity.json', {'scratch': True})
    tasks = []
    for tier in range(4):
        output = source/f'level5/results/development/graph_coloring/difficulty_{tier}.json'
        tasks.append({'level': 'level5', 'domain': 'graph_coloring', 'split': 'dev', 'output': str(output),
                      'seeds': [8103000,8103001,8103002,8103003], 'batch_size': 8, 'row_offset': 0, 'row_limit': 0})
    taskpath = put(root/'tasks/cell.json', tasks)
    cell = {'id': 'level5_graph_coloring_r3_dev', 'level': 'level5', 'domain': 'graph_coloring',
            'phase': 'dev', 'model_label': '14b', 'source_kind': 'domain_revision_v1',
            'source_root': str(source), 'tasks': str(taskpath), 'command': ['/scratch/python', '/scratch/evaluate', '--resume']}
    environment = {'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '1', 'VLLM_USE_V1': '0',
                   'VLLM_ATTENTION_BACKEND': 'XFORMERS', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                   'PYTHONDONTWRITEBYTECODE': '1'}
    plan = {'created_at': '2026-09-12T19:50:00+00:00', 'cells': [cell], 'provider_sha256': 'provider',
            'readiness_sha256': 'ready', 'view_manifest': str(view), 'python': '/scratch/python',
            'inputs_sha256': {str(taskpath): a.file_sha(taskpath)}, 'review_sha256': a.TRANSPORT_REVIEW_SHA}
    put(root/'plan.json', plan); put(root/'plan.sha256.json', {'sha256': a.file_sha(root/'plan.json')})
    intent = {'at_utc': '2026-09-12T19:51:15.100000+00:00'}
    put(root/'submission_intent.json', intent)
    submitted = {'array_job_id': a.JOB, 'at_utc': '2026-09-12T19:51:15.600000+00:00'}
    put(root/'submission_result.json', submitted)
    env = {**environment, 'SLURMD_NODENAME': 'node202', 'SLURM_JOB_ACCOUNT': 'allcs', 'SLURM_JOB_PARTITION': 'cs',
           'SLURM_ARRAY_JOB_ID': str(a.JOB), 'SLURM_ARRAY_TASK_ID': '0', 'SLURM_JOB_ID': str(a.JOB),
           'SLURM_CPUS_PER_TASK': '6', 'SLURM_MEM_PER_NODE': '61440',
           'PYTHONPYCACHEPREFIX': '/tmp/NONPRODUCTION-scale-frozen-pycache-scratch/pycache'}
    txschema = 'modebench_scale_level5_graph_r3_development_transport_v2'
    runtime = {'schema': txschema, 'status': 'validated_before_unchanged_evaluator_subprocess',
        'array_job_id': a.JOB, 'array_index': 0, 'cell': cell['id'], 'level': 'level5',
        'plan_sha256': a.file_sha(root/'plan.json'), 'submission_result_sha256': a.file_sha(root/'submission_result.json'),
        'readiness_sha256': 'ready', 'provider_sha256': 'provider', 'view_manifest': str(view),
        'view_manifest_sha256': a.file_sha(view), 'evaluator_command': cell['command'], 'environment': env,
        'hostname': 'node202.ionic.cs.princeton.edu', 'vllm_version': '0.8.4',
        'visible_gpu_names': ['NVIDIA RTX A5000']*2, 'at_utc': '2026-09-12T19:51:40.546756+00:00',
        'outputs_before_exec': {t['output']: {'receipt_exists': False, 'batch_directory_exists': False} for t in tasks},
        'gpu_metadata_probe': {'command': [plan['python'], '-B', '-c',
            'import json,torch;print(json.dumps([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]))'],
            'timeout_seconds': 60, 'returncode': 0, 'stdout': '["NVIDIA RTX A5000", "NVIDIA RTX A5000"]\n',
            'stderr': '', 'started_at_utc': '2026-09-12T19:51:35+00:00', 'finished_at_utc': '2026-09-12T19:51:39+00:00'}}
    put(root/'runtime/0.json', runtime)
    outcome = {'schema': txschema, 'status': 'actual_evaluator_process_return_observed', 'array_job_id': a.JOB,
        'array_index': 0, 'cell': cell['id'], 'command': cell['command'], 'runtime_sha256': a.file_sha(root/'runtime/0.json'),
        'observer_host': runtime['hostname'], 'observer_pid': 12345, 'returncode': 0, 'scheduler_success_claimed': False,
        'started_at_utc': '2026-09-12T19:51:40.561303+00:00', 'finished_at_utc': '2026-09-12T19:57:19.651109+00:00'}
    put(root/'runtime/0.evaluator_exit.json', outcome)
    summaries, allrows, allpaths = [], [], []
    for tier, task in enumerate(tasks):
        receipt = put(task['output'], {'generated_at': '2026-09-12T19:54:00+00:00', 'metrics': {'pass@1': .25, 'pass@8': .5}})
        paths = [receipt, put(str(receipt)+'.batches/run.json', {'scratch': True})]
        for seed in task['seeds']:
            for start in range(0,146,8):
                paths.append(put(str(receipt)+f'.batches/seed-{seed}__rows-{start:06d}-{min(start+8,146):06d}.json', {'scratch': True}))
        for path in paths: stamp(path)
        summary = {'tier': tier, 'receipt': str(receipt), 'receipt_sha256': a.file_sha(receipt),
                   'rows': 146, 'batches': 76, 'attempts': 4672, 'metrics': {'pass@1': .25, 'pass@8': .5},
                   'rendered_prompts_sha256': f'prompt{tier}', 'source_certificate': str(certificate)}
        summaries.append(summary); allrows.append([{'scratch_row': i} for i in range(146)]); allpaths.append(paths)
    out = root/'logs'/f'{a.JOB}_0.out'; out.parent.mkdir()
    out.write_text('\n'.join(json.dumps({'event':'task_complete','output':s['receipt'],'metrics':s['metrics']}) for s in summaries)+'\n')
    err = root/'logs'/f'{a.JOB}_0.err'; err.write_text('scratch terminal warning\n')
    rows = []
    for suffix in ('', '.batch', '.extern'):
        state_value, code = ('COMPLETED','0:0') if suffix == '.extern' else ('FAILED','1:0')
        rows.append([str(a.JOB)+suffix, f'{a.JOB}_0'+suffix, state_value, code,
                     '2026-09-12T19:51:17','2026-09-12T19:57:19','node202','6','60G' if not suffix else '',
                     'cpu=6,gres/gpu:a5000=2,gres/gpu=2,mem=60G,node=1',
                     '2026-09-12T19:51:15' if not suffix else '2026-09-12T19:51:17','allcs','cs' if not suffix else ''])
    terminal = {'schema': a.TERMINAL_SCHEMA, 'array_job_id': a.JOB, 'command': a.terminal_command(),
        'environment': {'TZ':'UTC'}, 'returncode': 0, 'stderr': '', 'stdout': '\n'.join('|'.join(r) for r in rows)+'\n',
        'observer_host': 'spin.cs.princeton.edu', 'captured_at_utc': '2026-09-12T19:59:19.813404+00:00',
        'logs_sha256': {str(p): a.file_sha(p) for p in (out,err)}}
    put(root/'terminal_accounting.json', terminal)
    monkeypatch.setattr(a, 'TERMINAL_SHA', a.file_sha(root/'terminal_accounting.json'))
    monkeypatch.setattr(a, 'RUNTIME_SHA', a.file_sha(root/'runtime/0.json'))
    monkeypatch.setattr(a, 'EVALUATOR_EXIT_SHA', a.file_sha(root/'runtime/0.evaluator_exit.json'))
    predecessor = {'array_job_id':31259795, 'stdout':'31259795|31259795_0|FAILED|1:0\n'}
    protocol = {'files_sha256': {str(source/'protocol.json'):a.file_sha(source/'protocol.json')}}
    modules = SimpleNamespace(revision=SimpleNamespace(authenticate=lambda source_root:protocol))
    transport = SimpleNamespace(SCHEMA=txschema, NODES=['node202','node203','node204'], ENVIRONMENT=environment,
        verified=lambda actual_state:(plan, None, None, None, {}), submission_identity=lambda actual_state:submitted,
        failed_predecessor=lambda base:predecessor, FAILURE=state/'development/pre_evaluator_failure_observation.json',
        FAILURE_SHA='prior', STATIC=tmp_path/'static.py', STATIC_SHA='static')
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
    return SimpleNamespace(root=root, state=state, plan=plan, cell=cell, submitted=submitted, runtime=runtime, outcome=outcome,
        terminal=terminal, terminal_rows=rows, transport=transport, summaries=summaries, rows=allrows, paths=allpaths,
        modules=modules, completed_calls=completed_calls)


def rewrite_runtime(saved, monkeypatch):
    put(saved.root/'runtime/0.json', saved.runtime)
    saved.outcome['runtime_sha256'] = a.file_sha(saved.root/'runtime/0.json')
    put(saved.root/'runtime/0.evaluator_exit.json', saved.outcome)
    monkeypatch.setattr(a, 'RUNTIME_SHA', a.file_sha(saved.root/'runtime/0.json'))
    monkeypatch.setattr(a, 'EVALUATOR_EXIT_SHA', a.file_sha(saved.root/'runtime/0.evaluator_exit.json'))


def test_composed_inspect_retains_mixed_states_sidecar_and_all_four_native_calls(saved):
    context = a.inspect(saved.root, a.TERMINAL_SHA)
    assert context.execution['state'] == 'FAILED' and context.execution['exit_code'] == '1:0'
    assert context.execution['scheduler_success'] is False and context.execution['evaluator_returncode'] == 0
    assert context.predecessor['array_job_id'] == 31259795 and context.predecessor['evaluator_not_started']
    assert context.execution['evaluator_finished_at_utc'].startswith('2026-09-12T19:57:19.651109')
    assert saved.completed_calls == [0,1,2,3]
    assert len(context.output_file_times) == 312 and sum(s['attempts'] for s in context.summaries) == 18688


@pytest.mark.parametrize('kind', ['main_success', 'main_exit', 'extern_failed', 'missing_step', 'duplicate', 'raw_id',
    'account', 'partition', 'cpu', 'gpu', 'start', 'end', 'submit', 'timezone', 'command', 'observer', 'logs'])
def test_only_exact_observed_terminal_semantics_pass(saved, kind):
    value = deepcopy(saved.terminal); rows = deepcopy(saved.terminal_rows)
    if kind == 'main_success': rows[0][2:4] = ['COMPLETED','0:0']
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
    elif kind == 'end': rows[0][5] = '2026-09-12T19:57:20'
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


def test_exact_capture_and_sidecar_hashes_cannot_be_replaced(saved):
    put(saved.root/'runtime/0.evaluator_exit.json', {**saved.outcome, 'added':True})
    with pytest.raises(ValueError, match='bytes'): a.runtime_linkage(saved.root, saved.transport, saved.plan, saved.submitted)
    with pytest.raises(ValueError, match='SHA'): a.inspect(saved.root, '0'*64)


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
def test_registration_waits_complete_outputs_and_preserves_failed_predecessor(saved, kind):
    if kind == 'missing_receipt': saved.paths[3][0].unlink()
    elif kind == 'missing_batch': saved.paths[3][-1].unlink()
    elif kind == 'wrong_totals': saved.summaries[3]['batches'] = 75
    elif kind == 'prior_runtime': put(saved.state/'development/runtime/0.json', {'unexpected':True})
    else:
        path = saved.root/'logs'/f'{a.JOB}_0.out'
        path.write_text(path.read_text()+json.dumps({'event':'task_complete','output':'extra','metrics':{}})+'\n')
    with pytest.raises((AssertionError, ValueError, FileNotFoundError)):
        a.register(saved.root, a.TERMINAL_SHA)
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
        assert phase == 'dev' and regrade is True and len(rows) == 146
        calls.append(tier)
        if tier == fail_tier: raise RuntimeError('synthetic native disagreement')
        return {'metrics':saved.summaries[tier]['metrics']}, {str(i):{} for i in range(146)}
    saved.modules.revision.receipt_scores = grader
    return calls


def test_one_durable_grader_pass_and_readonly_verify_never_regrades(saved, monkeypatch):
    calls = grader_fixture(saved, monkeypatch)
    result = a.audit(saved.root, a.TERMINAL_SHA)
    assert calls == [0,1,2,3]
    assert result['new_grader_invocations'] == 18688 and result['completed_batches'] == 304
    assert result['scheduler_success'] is False and result['recovery_scheduler_success'] is False
    assert result['evaluator_returncode'] == 0 and result['exit_cause'] == 'unknown'
    assert result == a.verify_existing(saved.root)
    assert calls == [0,1,2,3]
    with pytest.raises(ValueError, match='already claimed'): a.audit(saved.root, a.TERMINAL_SHA)
    assert calls == [0,1,2,3]


def test_grader_exception_saves_failure_and_cannot_retry(saved, monkeypatch):
    calls = grader_fixture(saved, monkeypatch, fail_tier=1)
    with pytest.raises(RuntimeError, match='synthetic'): a.audit(saved.root, a.TERMINAL_SHA)
    assert calls == [0,1] and (saved.root/'execution_audit/failure.json').exists()
    assert not (saved.root/'execution_reconciliation.json').exists()
    with pytest.raises(ValueError, match='already claimed'): a.audit(saved.root, a.TERMINAL_SHA)
    assert calls == [0,1]


def test_claim_survives_interruption_and_partial_report_is_not_completion(saved, monkeypatch):
    calls = grader_fixture(saved, monkeypatch)
    def interrupted(*args, **kwargs): raise KeyboardInterrupt()
    monkeypatch.setattr(a.cell_helper, 'tokenizer_for', interrupted)
    with pytest.raises(KeyboardInterrupt): a.audit(saved.root, a.TERMINAL_SHA)
    assert (saved.root/'execution_audit/claim.json').exists() and calls == []
    with pytest.raises(ValueError, match='already claimed'): a.audit(saved.root, a.TERMINAL_SHA)


@pytest.mark.parametrize('kind', ['claim', 'report', 'certificate', 'output', 'mtime'])
def test_readonly_verification_rejects_mutated_proof_without_regrading(saved, monkeypatch, kind):
    calls = grader_fixture(saved, monkeypatch); a.audit(saved.root, a.TERMINAL_SHA)
    if kind == 'claim':
        path = saved.root/'execution_audit/claim.json'; value = a.read(path); value['script_sha256'] = 'wrong'; put(path,value)
    elif kind == 'report':
        path = saved.root/'execution_audit/tiers/0.json'; value = a.read(path); value['new_grader_invocations'] = 1; put(path,value)
    elif kind == 'certificate':
        path = saved.root/'execution_reconciliation.json'; value = a.read(path); value['scheduler_success'] = True; put(path,value)
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
    rows = [{'problem': 'SCRATCH_NATIVE_INTERFACE_'+str(i), 'answer': '{}'} for i in range(146)]
    source_path = Path(saved.cell['source_root'])/'level5/pools/graph_coloring/difficulty_0.jsonl'
    source_path.write_text('\n'.join(json.dumps(row) for row in rows)+'\n')
    source = {'path': str(source_path), 'file_sha256': a.file_sha(source_path)}
    labels = task['seeds']; schedule = {'scratch_exact_rows': a.sha(rows), 'seeds': labels}
    runtime_settings = {'max_model_len':1024,'tensor_parallel_size':2,'gpu_memory_utilization':.82,
                        'swap_space':4.0,'enable_prefix_caching':True}
    interface = {'scratch_native_interface':True}
    model = {'path':'/scratch/model','model_label':'14b'}
    identity = {'source':source,'seeds':labels,'runtime':runtime_settings,'interface':interface,
                'model':{**model,'vllm_version':'0.8.4'},'seed_schedule_sha256':a.sha(schedule),
                'rendered_prompts_sha256':'scratch_rendered_prompts'}
    results = [{'draws':[{'attempts':[{'text':f'SCRATCH_{i}_{seed}_{j}','canonical_key':None} for j in range(8)]}
                         for seed in labels]} for i in range(146)]
    receipt = {'level':'level5','domain':'graph_coloring','split':'dev','model_label':'14b',
               'identity':identity,'identity_sha256':a.sha(identity),'generated_at':'2026-09-12T19:54:00+00:00',
               'prompt_results':results,'metrics':{'pass@1':.25,'pass@8':.5}}
    put(task['output'], receipt)
    directory = Path(task['output']+'.batches')
    put(directory/'run.json', {'identity':identity,'identity_sha256':receipt['identity_sha256']})
    for index, seed in enumerate(labels):
        for start in range(0,146,8):
            end = min(start+8,146); draws = [r['draws'][index] for r in results[start:end]]
            put(directory/f'seed-{seed}__rows-{start:06d}-{end:06d}.json',
                {'seed':seed,'start':start,'end':end,'identity_sha256':receipt['identity_sha256'],
                 'draws':draws,'draws_sha256':a.sha(draws)})
    if corrupt_batch:
        path = directory/f'seed-{labels[0]}__rows-000000-000008.json'
        batch = a.read(path); batch['draws'][0]['attempts'][0]['text'] = 'changed'
        batch['draws_sha256'] = a.sha(batch['draws']); put(path,batch)
    certificate = Path(saved.cell['source_root'])/'level5/pools/graph_coloring/identity.json'
    put(certificate, {'protocol_sha256':a.file_sha(Path(saved.cell['source_root'])/'protocol.json'),
                      'tiers':{'0':{'rows':146,'rows_sha256':a.sha(rows)}}})
    def validate_task(actual, confirmation):
        assert actual is task and confirmation is False
    def validate_receipt(actual, actual_rows):
        assert actual_rows is rows and len(actual['prompt_results']) == 146
    def runtime(**kwargs):
        assert kwargs == runtime_settings
        return runtime_settings
    evaluator = SimpleNamespace(validate_task=validate_task, load_rows=lambda t:(rows,source),
        validate_seed_receipt=validate_receipt, runtime_settings=runtime,
        frozen_interface=lambda domain:interface, model_identity=lambda path,label:model,
        frozen=SimpleNamespace(schedule_record=lambda domain,actual_rows,seeds:schedule))
    plan = {**saved.plan,'models':{'14b':{'path':'/scratch/model'}},
            'rng_admission':{'task_seed_schedule_sha256':{task['output']:a.sha(schedule)}}}
    args = (SimpleNamespace(evaluator=evaluator),plan,saved.cell,task,0,{'draw_labels':{'dev':labels}},
            {'worker_started_at_utc':saved.outcome['started_at_utc'],'end_utc':saved.outcome['finished_at_utc']})
    if corrupt_batch:
        with pytest.raises(ValueError, match='saved batch draws'): sealed.completed_task(*args)
    else:
        summary, actual_rows, paths = sealed.completed_task(*args)
        assert summary['rows'] == 146 and summary['batches'] == 76 and summary['attempts'] == 4672
        assert actual_rows is rows and len(paths) == 78
