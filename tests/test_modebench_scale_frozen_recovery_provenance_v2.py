"""Scratch-only recovery provenance guards; no scientific stage or scheduler calls."""
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[1] / 'artifacts/verify_modebench_scale_frozen_recovery_provenance_v2_20260912.py'
spec = importlib.util.spec_from_file_location('frozen_recovery_provenance_tested', SOURCE)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


def mutate(path, function):
    value = audit.read(path)
    function(value)
    write(path, value)


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    workspace = tmp_path / 'workspace'; root = workspace / 'recovery'; root.mkdir(parents=True)
    helper = workspace / 'helper.py'; helper.write_text('# sealed helper fixture\n')
    legacy = workspace / 'legacy.py'; legacy.write_text('# legacy fixture\n')
    source = workspace / 'auditor.py'; source.write_text('# auditor fixture\n')
    canonical = workspace / 'src/templates.py'; canonical.parent.mkdir(); canonical.write_text('frozen\n')
    target_python = workspace / 'bin/python3.10'; target_python.parent.mkdir(); target_python.write_text('# interpreter\n')
    venv = workspace / 'venv/python'; venv.parent.mkdir(); venv.symlink_to(target_python)
    view = workspace / 'view'; view.mkdir()
    frozen_copy = view / 'templates.scale_pinned.py'; frozen_copy.write_text('frozen\n')
    manifest = {'mappings': [{'source': str(frozen_copy), 'destination': str(canonical), 'sha256': audit.file_sha(frozen_copy)}],
                'proot': {'path': '/proot', 'sha256': 'a' * 64},
                'runner': {'path': '/runner', 'sha256': 'b' * 64},
                'host_source': {'path': str(canonical), 'sha256': 'neutral-host-expectation'}}
    write(view / 'manifest.json', manifest)
    (view / 'manifest.sha256').write_text(audit.file_sha(view / 'manifest.json') + '  manifest.json\n')
    for path in view.iterdir(): path.chmod(0o444)
    view.chmod(0o555)
    configuration = {name: '1' * 64 for name in audit.MODEL_CONFIGS}
    model = {'label': '14b', 'path': str(workspace / 'model'), 'configuration_sha256': configuration,
             'weights': [{'name': 'model.safetensors', 'bytes': 100, 'mtime_ns': 7}]}
    settings = {'dtype': 'float16', 'enable_prefix_caching': True, 'gpu_memory_utilization': .82,
                'swap_space': 4.0, 'tensor_parallel_size': 2}
    command = [str(venv), str(workspace / 'evaluator.py'), '--resume']
    environment = {'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '1', 'VLLM_USE_V1': '0',
                   'VLLM_ATTENTION_BACKEND': 'XFORMERS', 'HF_HUB_OFFLINE': '1',
                   'TRANSFORMERS_OFFLINE': '1', 'PYTHONDONTWRITEBYTECODE': '1'}
    tasks = []; preserved = {}; all_inputs = {str(helper): audit.file_sha(helper), str(venv): audit.file_sha(venv),
        str(canonical): audit.file_sha(canonical)}
    for tier in range(4):
        source_rows = workspace / f'pool_{tier}.jsonl'; source_rows.write_text('{"problem":"fixture"}\n')
        output = workspace / 'outputs' / f'difficulty_{tier}.json'
        task = {'output': str(output), 'rows_jsonl': str(source_rows), 'domain': 'pantry',
                'level': 'level5', 'batch_size': 8, 'seeds': [6558000, 6558001, 6558002, 6558003]}
        tasks.append(task); all_inputs[str(source_rows)] = audit.file_sha(source_rows)
        identity = {'model': {'label': '14b', 'path': model['path'], 'vllm_version': '0.8.4',
                             'weight_file_manifest': model['weights'], 'configuration_sha256': configuration},
                    'runtime': {**settings, 'max_model_len': 2048}, 'interface': {'fixture': 'original'},
                    'code_sha256': {'src/templates.py': audit.file_sha(canonical)},
                    'sampling_engine': {'engine': 'V0', 'parallel_sample_seed_policy': 'request_seed_plus_sample_index', 'vllm_version': '0.8.4'},
                    'batch_size': 8, 'seeds': task['seeds'],
                    'source': {'path': str(source_rows), 'file_sha256': audit.file_sha(source_rows), 'selected_rows': 232}}
        receipt = {'status': 'complete', 'level': 'level5', 'domain': 'pantry', 'model_label': '14b', 'split': 'dev',
                   'metrics': {'rows': 232, 'pass1': .1}, 'prompt_results': [{'draws': [{'fixture': i, 'seed': seed} for seed in task['seeds']]} for i in range(232)], 'identity': identity, 'identity_sha256': audit.sha(identity),
                   'generated_at': '2026-09-12T01:10:00+00:00' if tier >= 2 else '2026-09-11T23:00:00+00:00'}
        write(output, receipt)
        batch_dir = Path(str(output) + '.batches')
        write(batch_dir / 'run.json', {'identity': identity, 'identity_sha256': audit.sha(identity)})
        if tier < 2: preserved[str(output)] = audit.file_sha(output)
        if tier < 3: preserved[str(batch_dir / 'run.json')] = audit.file_sha(batch_dir / 'run.json')
        for seed in task['seeds']:
            for start in range(0, 232, 8):
                path = batch_dir / f'seed-{seed}__rows-{start:06d}-{start + 8:06d}.json'
                draws = [{'fixture': start + i, 'seed': seed} for i in range(8)]
                write(path, {'identity_sha256': audit.sha(identity), 'seed': seed, 'start': start, 'end': start + 8,
                             'draws': draws, 'draws_sha256': audit.sha(draws)})
                if tier < 3: preserved[str(path)] = audit.file_sha(path)
    assert len(preserved) == 353
    tasks_path = workspace / 'tasks.json'; write(tasks_path, tasks)
    cell = {'id': 'level5_pantry', 'domain': 'pantry', 'model_label': '14b', 'command': command, 'tasks': str(tasks_path)}
    original = {'python': str(venv), 'runtime_profile': settings, 'models': {'14b': model}, 'cells': [{}] * 9 + [cell]}
    original_plan = workspace / 'original_plan.json'; write(original_plan, original)
    original_runtime = workspace / 'original_runtime.json'; write(original_runtime, {'command': command})
    terminal_original = {'original_row': audit.OLD_JOB, 'at_utc': '2026-09-12T01:00:00+00:00',
        'sacct_command': ['sacct', '-j', '31243495', '-n', '-P', '--format=JobIDRaw,JobID,State,ExitCode,End,NodeList'],
        'sacct_stdout': '|'.join(audit.OLD_JOB) + '\n', 'squeue_stdout': 'otherjob|RUNNING\n'}
    write(root / 'original_terminal.json', terminal_original)
    old_stderr = workspace / 'original.err'; old_stderr.write_text('validate_seed_receipt: receipt evaluator/helper code hashes missing or changed\n')
    write(root / 'original_source_failure.json', {'source_path': str(old_stderr), 'source_sha256': audit.file_sha(old_stderr), 'stderr': old_stderr.read_text()})
    for path in [view / 'manifest.json', view / 'manifest.sha256', frozen_copy, tasks_path, original_plan, original_runtime,
                 root / 'original_terminal.json', root / 'original_source_failure.json', old_stderr]:
        all_inputs[str(path)] = audit.file_sha(path)
    all_inputs.update(preserved)
    plan = {'original_plan': str(original_plan), 'original_runtime': str(original_runtime), 'command': command,
            'view_manifest': str(view / 'manifest.json'), 'environment': environment,
            'prepared_at_utc': '2026-09-12T01:00:44+00:00', 'submit_command': ['sbatch', 'fixed-worker'],
            'inputs_sha256': all_inputs, 'preserved_outputs_sha256': preserved}
    write(root / 'plan.json', plan); plan_sha = audit.file_sha(root / 'plan.json')
    write(root / 'plan.sha256.json', {'sha256': plan_sha})
    intent = {'status': 'submission_attempt_started', 'plan_sha256': plan_sha, 'command': plan['submit_command'],
              'at_utc': '2026-09-12T01:01:34.600000+00:00', 'original_terminal': terminal_original}
    write(root / 'submission_intent.json', intent)
    write(root / 'submission_result.json', {'status': 'submitted', 'job_id': audit.JOB, 'returncode': 0,
        'stdout': str(audit.JOB) + '\n', 'plan_sha256': plan_sha, 'intent_sha256': audit.file_sha(root / 'submission_intent.json'),
        'at_utc': '2026-09-12T01:01:34.700000+00:00'})
    runtime = {'status': 'validated_before_exact_evaluator_exec', 'job_id': audit.JOB, 'plan_sha256': plan_sha,
        'original_array_job_id': 31243495, 'original_array_index': 9, 'original_runtime': str(original_runtime),
        'original_runtime_sha256': audit.file_sha(original_runtime), 'view_manifest': str(view / 'manifest.json'),
        'view_manifest_sha256': audit.file_sha(view / 'manifest.json'), 'command': command,
        'at_utc': '2026-09-12T01:02:04+00:00', 'hostname': 'node105.ionic.cs.princeton.edu',
        'vllm_version': '0.8.4', 'visible_gpu_names': ['NVIDIA RTX A5000'] * 2,
        'environment': {**environment, 'SLURM_JOB_ID': str(audit.JOB), 'SLURMD_NODENAME': 'node105',
            'SLURM_CPUS_PER_TASK': '6', 'SLURM_MEM_PER_NODE': '60416',
            'PYTHONPYCACHEPREFIX': '/tmp/NONPRODUCTION-scale-frozen-pycache-fixture/pycache'}}
    write(root / 'runtime.json', runtime)
    auxiliary_path = workspace / 'auxiliary.json'
    scratch = workspace / 'var/tmp/NONPRODUCTION-wash-node105-lock-fixture'
    probe_code = '# authenticated diagnostic code fixture'
    auxiliary = {'schema': 'modebench_scale_scratch_cross_host_lock_qualification_v1', 'status': 'passed',
        'allocation_job_id': audit.JOB, 'host': 'wash.cs.princeton.edu',
        'at_utc': '2026-09-12T01:12:04.500000+00:00', 'scratch': str(scratch), 'probe_code': probe_code,
        'neutral_source_before': manifest['host_source']['sha256'],
        'neutral_source_after': manifest['host_source']['sha256'],
        'command_prefix': ['srun', '--jobid=' + str(audit.JOB), '--overlap', '--nodes=1', '--ntasks=1',
            '--cpus-per-task=1', '--gres=none', '--cpu-bind=none', '--mem=16M', '--time=00:01:00',
            str(venv), '-B', '-c', probe_code, str(scratch / 'probe.lock')],
        'phases': [{'phase': phase, 'expected': expected, 'returncode': 0, 'stderr': '',
            'stdout': json.dumps({'host': 'node105.ionic.cs.princeton.edu', 'job': str(audit.JOB),
                'step': str(index), 'pid': 1000 + index, 'observed': expected, 'inode': 99,
                'contents_sha256': '1' * 64})}
            for index, (phase, expected) in enumerate([('while_wash_holds', 'blocked'), ('after_wash_releases', 'acquired')])]}
    write(auxiliary_path, auxiliary)
    step_rows = [[str(audit.JOB) + '.' + str(index)] * 2 + ['COMPLETED', '0:0',
        '2026-09-12T01:12:0' + str(index * 2), '2026-09-12T01:12:0' + str(index * 2 + 1),
        'node105', '1', '', 'cpu=1,mem=16M,node=1', '2026-09-12T01:12:0' + str(index * 2)]
        for index in range(2)]
    row = [str(audit.JOB), str(audit.JOB), 'FAILED', '1:0', '2026-09-12T01:01:35', audit.FAILED_END,
           'node105', '6', '59G', 'billing=37,cpu=6,gres/gpu:a5000=2,gres/gpu=2,mem=59G,node=1', '2026-09-12T01:01:34']
    write(root / 'terminal_accounting.json', {'schema': audit.TERMINAL_SCHEMA, 'job_id': audit.JOB,
        'captured_at_utc': '2026-09-12T02:10:00+00:00', 'observer_host': 'wash.cs.princeton.edu',
        'command': ['sacct', '-j', str(audit.JOB), '-n', '-P', '--format=' + audit.TERMINAL_FIELDS],
        'environment': {'TZ': 'UTC'}, 'returncode': 0,
        'stdout': ''.join('|'.join(item) + '\n' for item in [row, *step_rows,
            [str(audit.JOB) + '.batch'] * 2 + ['FAILED', '1:0', row[4], audit.FAILED_END, 'node105', '6', '', row[9], row[4]],
            [str(audit.JOB) + '.extern'] * 2 + ['COMPLETED', '0:0', row[4], audit.FAILED_END, 'node105', '6', '', row[9], row[4]]]), 'stderr': ''})
    for key, value in {'ROOT': workspace, 'SOURCE': source, 'RECOVERY': root, 'HELPER': helper, 'LEGACY': legacy,
                       'LEGACY_SHA': audit.file_sha(legacy), 'PLAN_SHA': plan_sha,
                       'AUXILIARY': auxiliary_path, 'AUXILIARY_SHA': audit.file_sha(auxiliary_path)}.items():
        monkeypatch.setattr(audit, key, value)
    v1 = workspace / 'v1_verifier.py'; v1.write_text('# preserved v1 fixture')
    monkeypatch.setattr(audit, 'V1_SOURCE', v1)
    monkeypatch.setattr(audit, 'V1_SHA', audit.file_sha(v1))
    monkeypatch.setattr(audit, 'TERMINAL_SHA', audit.file_sha(root / 'terminal_accounting.json'))
    logs = root / 'logs'; logs.mkdir()
    log_events = [{'event': 'task_complete', 'output': task['output'], 'metrics': audit.read(task['output'])['metrics']}
                  for task in tasks[2:]]
    (logs / '31253118.out').write_text(''.join(json.dumps(value) + '\n' for value in log_events)
        + 'Terminating local vLLM worker processes\nWorker exiting\n')
    (logs / '31253118.err').write_text('scratch shutdown warning\n')
    monkeypatch.setattr(audit, 'LOG_PINS', {p.name: audit.file_sha(p) for p in logs.iterdir()})
    graders = {}
    for tier, task in enumerate(tasks):
        path = workspace / f'grader_audit_{tier}.json'
        receipt = audit.read(task['output'])
        record = {'schema': 'modebench_scale_original_completed_tier_audit_v1', 'status': 'passed',
            'tier': tier, 'domain': 'pantry', 'level': 'level5', 'receipt': task['output'],
            'receipt_sha256': audit.file_sha(task['output']), 'source': task['rows_jsonl'],
            'source_sha256': audit.file_sha(task['rows_jsonl']), 'metrics': receipt['metrics'],
            'runtime': receipt['identity']['runtime'], 'interface': receipt['identity']['interface'],
            'attempts_validated': 7424, 'new_grader_invocations': 7424, 'prompts_validated': 232,
            'validation_entrypoint': 'receipt_scores(regrade=True)',
            'started_utc': '2026-09-12T02:08:00+00:00', 'created_utc': '2026-09-12T02:09:00+00:00',
            'files_sha256': {str(p): audit.file_sha(p) for p in [task['output'], task['rows_jsonl']]}}
        record.update({field: True for field in ['original_grader_agrees', 'original_grader_replayed_by_this_audit',
            'all_saved_metrics_recomputed_without_rounding', 'source_pins_unchanged',
            'exact_frozen_task_source_model_interface_rng_runtime_validated']})
        record.update({field: False for field in ['model_sampling_performed', 'fit_invoked',
                                                'heldout_data_loaded', 'scheduler_or_controller_actions']})
        write(path, record); graders[path] = audit.file_sha(path)
    monkeypatch.setattr(audit, 'GRADER_AUDITS', graders)
    calls = []
    def sealed_stage_stub(path, *, require_original_inventory):
        calls.append((path, require_original_inventory))
        assert require_original_inventory is False
        current = audit.read(path / 'plan.json')
        audit.checked_pins(current['inputs_sha256'])
        return current
    monkeypatch.setattr(audit, 'load_helper', lambda path: SimpleNamespace(verify=sealed_stage_stub))
    yield SimpleNamespace(root=root, workspace=workspace, plan=plan, runtime=runtime, row=row,
                          tasks=tasks, canonical=canonical, view=view, venv=venv, target_python=target_python, calls=calls,
                          auxiliary_path=auxiliary_path, step_rows=step_rows)
    view.chmod(0o755)


def test_valid_recovery_is_scoped_and_retains_literal_venv(fixture):
    result = audit.audit(fixture.root)
    assert result['schema'] == audit.SCHEMA and result['status'] == 'verified_scientific_outputs_with_failed_execution'
    assert result['final_supplemental_provenance'] is False
    assert result['original_execution']['state'] == 'FAILED'
    assert result['original_execution']['exit_code'] == '1:0'
    assert result['recovery_execution']['state'] == 'FAILED'
    assert result['scheduler_success'] is False and result['scientific_outputs_complete'] is True
    assert result['exit_cause'] == 'unknown' and result['completed_output_files'] == 472
    assert len(result['original_grader_audits']) == 4
    auxiliary = result['known_auxiliary_activity']
    assert auxiliary['step_ids'] == [str(audit.JOB) + '.0', str(audit.JOB) + '.1']
    assert auxiliary['soak_mount_or_process_observed'] is False
    assert auxiliary['new_job_submitted'] is auxiliary['model_or_grader_called'] is False
    assert auxiliary['requested_per_step'] == {'cpus': 1, 'gpus': 0, 'memory': '16M', 'time': '00:01:00'}
    assert result['files_sha256'][str(fixture.auxiliary_path)] == audit.AUXILIARY_SHA
    assert len(result['recovery_execution']['auxiliary_terminal_steps']) == 2
    assert result['preserved_output_files'] == 353
    assert sum(item['batches'] for item in result['completed_development_tiers']) == 464
    assert result['literal_evaluator_command'][0] == str(fixture.venv)
    assert str(fixture.venv) not in result['files_sha256']
    assert result['files_sha256'][str(fixture.target_python)] == audit.file_sha(fixture.venv)
    assert 'controller authority transition' in result['not_covered']
    assert fixture.calls == [(fixture.root, False)]
    assert not (fixture.root / 'execution_audit.json').exists()


@pytest.mark.parametrize('name', ['terminal_accounting.json', 'runtime.json', 'submission_result.json', 'original_terminal.json'])
def test_missing_evidence_rejected(fixture, name):
    (fixture.root / name).unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        audit.audit(fixture.root)


@pytest.mark.parametrize('state,code', [('RUNNING', '0:0'), ('COMPLETED', '0:0'), ('CANCELLED', '0:15'), ('FAILED', '2:0')])
def test_terminal_failure_or_live_job_cannot_pass(fixture, state, code):
    row = fixture.row.copy(); row[2:4] = [state, code]
    mutate(fixture.root / 'terminal_accounting.json', lambda record: record.update(stdout='|'.join(row) + '\n'))
    with pytest.raises(ValueError, match='terminal FAILED'):
        audit.audit(fixture.root)


@pytest.mark.parametrize('index,value', [(0, 'other'), (6, 'node007'), (7, '8'), (8, '60G'),
    (9, 'cpu=6,gres/gpu:a5000=4,gres/gpu=4,mem=59G,node=1'),
    (9, 'cpu=6,cpu=6,gres/gpu:a5000=2,gres/gpu=2,mem=59G,node=1'),
    (4, '2026-09-12T01:03:00'), (5, '2026-09-12T01:01:36'), (10, '2026-09-11T23:00:00')])
def test_terminal_allocation_identity_and_chronology_guards(fixture, index, value):
    row = fixture.row.copy(); row[index] = value
    mutate(fixture.root / 'terminal_accounting.json', lambda record: record.update(stdout='|'.join(row) + '\n'))
    with pytest.raises(ValueError):
        audit.audit(fixture.root)


@pytest.mark.parametrize('change', [lambda r: r.update(environment={'TZ': 'America/New_York'}),
    lambda r: r.update(returncode=1), lambda r: r.update(job_id=12),
    lambda r: r.update(captured_at_utc='2026-09-12T01:00:00+00:00'),
    lambda r: r.update(stdout=r['stdout'] + r['stdout'])])
def test_terminal_observation_envelope_guards(fixture, change):
    mutate(fixture.root / 'terminal_accounting.json', change)
    with pytest.raises(ValueError):
        audit.audit(fixture.root)


@pytest.mark.parametrize('change', [lambda r: r.update(job_id=12), lambda r: r.update(plan_sha256='0' * 64),
    lambda r: r.update(command=['/different/python']), lambda r: r.update(hostname='soak.cs.princeton.edu'),
    lambda r: r.update(visible_gpu_names=['NVIDIA RTX A5000']), lambda r: r.update(vllm_version='different'),
    lambda r: r['environment'].update(SLURM_MEM_PER_NODE='61440'),
    lambda r: r['environment'].update(VLLM_ATTENTION_BACKEND='different'),
    lambda r: r['environment'].update(PYTHONPYCACHEPREFIX='/stale'),
    lambda r: r.update(at_utc='2026-09-12T01:00:00+00:00')])
def test_runtime_linkage_host_settings_and_cache_guards(fixture, change):
    mutate(fixture.root / 'runtime.json', change)
    with pytest.raises(ValueError):
        audit.audit(fixture.root)


def test_original_failure_is_not_relabelled_success(fixture):
    mutate(fixture.root / 'original_terminal.json', lambda r: r['original_row'].__setitem__(2, 'COMPLETED'))
    with pytest.raises(ValueError, match='pinned evidence changed'):
        audit.audit(fixture.root)


def test_neutral_host_view_cannot_satisfy_old_pins(fixture):
    fixture.canonical.write_text('neutral\n')
    with pytest.raises(ValueError, match='pinned evidence changed'):
        audit.audit(fixture.root)


def test_retained_batch_mutation_cannot_pass(fixture):
    path = next(Path(fixture.tasks[1]['output'] + '.batches').glob('seed-*.json'))
    mutate(path, lambda record: record.update(seed=1))
    with pytest.raises(ValueError, match='pinned evidence changed'):
        audit.audit(fixture.root)


def test_new_batch_missing_or_wrong_identity_rejected(fixture):
    path = next(Path(fixture.tasks[3]['output'] + '.batches').glob('seed-*.json'))
    mutate(path, lambda record: record.update(identity_sha256='bad'))
    with pytest.raises(ValueError, match='batch linkage'):
        audit.audit(fixture.root)
    path.unlink()
    with pytest.raises(ValueError, match='batch inventory'):
        audit.audit(fixture.root)


def test_completed_receipt_model_change_rejected_after_rehash(fixture):
    path = Path(fixture.tasks[3]['output']); receipt = audit.read(path)
    receipt['identity']['runtime']['dtype'] = 'float32'; receipt['identity_sha256'] = audit.sha(receipt['identity'])
    write(path, receipt)
    write(Path(str(path) + '.batches') / 'run.json', {'identity': receipt['identity'], 'identity_sha256': receipt['identity_sha256']})
    with pytest.raises(ValueError, match='model/settings'):
        audit.audit(fixture.root)


def test_view_mode_and_extra_file_guards(fixture):
    fixture.view.chmod(0o755)
    with pytest.raises(ValueError, match='directory layout/mode'):
        audit.audit(fixture.root)
    (fixture.view / 'extra').write_text('extra'); fixture.view.chmod(0o555)
    with pytest.raises(ValueError, match='directory layout/mode'):
        audit.audit(fixture.root)


def test_plan_sidecar_and_legacy_validator_drift_rejected(fixture):
    mutate(fixture.root / 'plan.sha256.json', lambda record: record.update(sha256='bad'))
    with pytest.raises(ValueError, match='recovery plan changed'):
        audit.audit(fixture.root)
    write(fixture.root / 'plan.sha256.json', {'sha256': audit.PLAN_SHA})
    audit.LEGACY.write_text('changed')
    with pytest.raises(ValueError, match='legacy provenance validator changed'):
        audit.audit(fixture.root)


def test_audit_publication_is_only_scoped_and_never_overwrites(fixture):
    value = audit.audit(fixture.root); destination = fixture.root / 'execution_reconciliation.json'
    with pytest.raises(ValueError, match='only the recovery-reconciliation'):
        audit.publish_reconciliation(value, fixture.workspace / 'execution_provenance.json')
    with pytest.raises(ValueError, match='recovery-only'):
        audit.publish_reconciliation({**value, 'schema': audit.FINAL_SCHEMA, 'status': 'verified'}, destination)
    audit.publish_reconciliation(value, destination)
    original = destination.read_bytes()
    with pytest.raises(FileExistsError): audit.publish_reconciliation(value, destination)
    assert destination.read_bytes() == original
    assert audit.read(destination) == audit.audit(fixture.root)


@pytest.mark.parametrize('timestamp', ['2026-09-12T01:01:00+00:00', '2026-09-12T02:04:22+00:00'])
def test_new_receipts_must_fall_within_recovery_lifetime(fixture, timestamp):
    mutate(Path(fixture.tasks[3]['output']), lambda receipt: receipt.update(generated_at=timestamp))
    with pytest.raises(ValueError, match='outside recovery'):
        audit.audit(fixture.root)


def test_retained_output_must_appear_in_completed_output_inventory(fixture):
    plan = dict(fixture.plan); plan['preserved_outputs_sha256'] = dict(plan['preserved_outputs_sha256'])
    unrelated = fixture.workspace / 'unrelated.json'; unrelated.write_text('{}')
    plan['preserved_outputs_sha256'].pop(fixture.tasks[0]['output'])
    plan['preserved_outputs_sha256'][str(unrelated)] = audit.file_sha(unrelated)
    original = audit.read(plan['original_plan'])
    with pytest.raises(ValueError, match='retained output pin linkage'):
        audit.validate_outputs(plan, original, original['cells'][9], fixture.runtime, {'end_utc': fixture.row[5]})


def test_retained_byte_drift_between_checks_is_detected(fixture, monkeypatch):
    original_check = audit.checked_pins
    target = next(Path(fixture.tasks[0]['output'] + '.batches').glob('seed-*.json'))
    def race(pins):
        result = original_check(pins)
        if pins is fixture.plan['preserved_outputs_sha256']:
            target.write_text(target.read_text() + '\n')  # Valid same JSON, different preserved bytes.
        return result
    monkeypatch.setattr(audit, 'checked_pins', race)
    original = audit.read(fixture.plan['original_plan'])
    with pytest.raises(ValueError, match='retained output pin linkage'):
        audit.validate_outputs(fixture.plan, original, original['cells'][9], fixture.runtime, {'end_utc': fixture.row[5]})


def test_missing_or_changed_auxiliary_report_rejected(fixture):
    fixture.auxiliary_path.write_text(fixture.auxiliary_path.read_text() + '\n')
    with pytest.raises(ValueError, match='auxiliary lock probe changed'):
        audit.audit(fixture.root)
    fixture.auxiliary_path.unlink()
    with pytest.raises(FileNotFoundError):
        audit.audit(fixture.root)


@pytest.mark.parametrize('change', [lambda r: r.update(allocation_job_id=12),
    lambda r: r.update(host='soak.cs.princeton.edu'), lambda r: r.update(neutral_source_after='changed'),
    lambda r: r['command_prefix'].__setitem__(6, '--gres=gpu:2'),
    lambda r: r.update(at_utc='2026-09-12T01:01:00+00:00'),
    lambda r: r['phases'].pop(), lambda r: r['phases'][0].update(returncode=1),
    lambda r: r['phases'][0].update(stdout=json.dumps({'job': str(audit.JOB), 'step': '2'})),
    lambda r: r.update(scratch='/tmp/controller.lock')])
def test_auxiliary_semantics_remain_checked_after_repin(fixture, monkeypatch, change):
    mutate(fixture.auxiliary_path, change)
    monkeypatch.setattr(audit, 'AUXILIARY_SHA', audit.file_sha(fixture.auxiliary_path))
    with pytest.raises(ValueError):
        audit.audit(fixture.root)


def rewrite_step(fixture, change, *, index=0):
    def transform(record):
        rows = [line.split('|') for line in record['stdout'].splitlines()]
        change(rows[index + 1])
        record['stdout'] = ''.join('|'.join(row) + '\n' for row in rows)
    mutate(fixture.root / 'terminal_accounting.json', transform)


@pytest.mark.parametrize('column,value', [(2, 'FAILED'), (2, 'RUNNING'), (3, '1:0'), (6, 'node007'),
    (4, '2026-09-12T01:01:40'), (5, '2026-09-12T01:12:05'),
    (5, '2026-09-12T02:05:00')])
def test_auxiliary_terminal_state_node_and_chronology(fixture, column, value):
    rewrite_step(fixture, lambda row: row.__setitem__(column, value))
    with pytest.raises(ValueError, match='auxiliary'):
        audit.audit(fixture.root)


@pytest.mark.parametrize('kind', ['missing', 'duplicate', 'unreviewed', 'wrong_raw_id'])
def test_auxiliary_terminal_inventory_is_exact(fixture, kind):
    def change(record):
        rows = record['stdout'].splitlines()
        if kind == 'missing': rows.pop()
        if kind == 'duplicate': rows.append(rows[-1])
        if kind == 'unreviewed': rows.append(rows[-1].replace(str(audit.JOB) + '.1', str(audit.JOB) + '.2'))
        if kind == 'wrong_raw_id': rows[-1] = 'other|' + rows[-1].split('|', 1)[1]
        record['stdout'] = '\n'.join(rows) + '\n'
    mutate(fixture.root / 'terminal_accounting.json', change)
    with pytest.raises(ValueError):
        audit.audit(fixture.root)


def test_auxiliary_record_must_precede_main_terminal_end(fixture, monkeypatch):
    mutate(fixture.auxiliary_path, lambda r: r.update(at_utc='2026-09-12T02:04:22+00:00'))
    monkeypatch.setattr(audit, 'AUXILIARY_SHA', audit.file_sha(fixture.auxiliary_path))
    with pytest.raises(ValueError, match='auxiliary observation falls outside'):
        audit.audit(fixture.root)


def test_auxiliary_steps_follow_recorded_sequential_phases(fixture):
    rewrite_step(fixture, lambda row: row.__setitem__(4, '2026-09-12T01:12:00'), index=1)
    with pytest.raises(ValueError, match='auxiliary terminal step chronology'):
        audit.audit(fixture.root)


def test_auxiliary_report_byte_drift_during_audit_rejected(fixture, monkeypatch):
    validate = audit.validate_auxiliary
    def race(*args):
        result = validate(*args)
        fixture.auxiliary_path.write_text(fixture.auxiliary_path.read_text() + '\n')
        return result
    monkeypatch.setattr(audit, 'validate_auxiliary', race)
    with pytest.raises(ValueError, match='pinned evidence changed'):
        audit.audit(fixture.root)


@pytest.mark.parametrize('tier', range(4))
def test_every_original_grader_audit_is_required(fixture, tier):
    path = list(audit.GRADER_AUDITS)[tier]
    path.unlink()
    with pytest.raises(FileNotFoundError):
        audit.audit(fixture.root)


@pytest.mark.parametrize('field,value', [('original_grader_agrees', False), ('heldout_data_loaded', True),
    ('attempts_validated', 7423), ('tier', 1), ('metrics', {'rows': 232, 'pass1': .2}),
    ('source', '/unrelated'), ('receipt_sha256', '0' * 64),
    ('validation_entrypoint', 'receipt_scores(regrade=False)')])
def test_grader_semantics_checked_even_after_repin(fixture, monkeypatch, field, value):
    path = list(audit.GRADER_AUDITS)[0]
    mutate(path, lambda record: record.update({field: value}))
    pins = dict(audit.GRADER_AUDITS); pins[path] = audit.file_sha(path)
    monkeypatch.setattr(audit, 'GRADER_AUDITS', pins)
    with pytest.raises(ValueError, match='grader'):
        audit.audit(fixture.root)


def test_new_batch_rehashed_but_inconsistent_with_receipt_rejected(fixture):
    path = next(Path(fixture.tasks[3]['output'] + '.batches').glob('seed-*.json'))
    def change(record):
        record['draws'][0]['fixture'] = -1
        record['draws_sha256'] = audit.sha(record['draws'])
    mutate(path, change)
    with pytest.raises(ValueError, match='batches differ from audited receipt'):
        audit.audit(fixture.root)


def test_unexpected_batch_directory_file_rejected(fixture):
    directory = Path(fixture.tasks[3]['output'] + '.batches')
    (directory / 'unexpected.json').write_text('{}')
    with pytest.raises(ValueError, match='batch inventory'):
        audit.audit(fixture.root)


@pytest.mark.parametrize('name', ['31253118.out', '31253118.err'])
def test_exact_terminal_log_bytes_preserved(fixture, name):
    path = fixture.root / 'logs' / name
    path.write_text(path.read_text() + 'changed')
    with pytest.raises(ValueError, match='pinned evidence changed'):
        audit.audit(fixture.root)


@pytest.mark.parametrize('kind', ['missing_event', 'metrics', 'order'])
def test_log_completion_linkage_checked_even_after_repin(fixture, monkeypatch, kind):
    path = fixture.root / 'logs/31253118.out'
    lines = path.read_text().splitlines()
    if kind == 'missing_event': lines.pop(0)
    if kind == 'metrics':
        value = json.loads(lines[0]); value['metrics']['pass1'] = .8; lines[0] = json.dumps(value)
    if kind == 'order': lines = lines[2:] + lines[:2]
    path.write_text('\n'.join(lines) + '\n')
    monkeypatch.setattr(audit, 'LOG_PINS', {**audit.LOG_PINS, path.name: audit.file_sha(path)})
    with pytest.raises(ValueError, match='log linkage|shutdown sequence'):
        audit.audit(fixture.root)


@pytest.mark.parametrize('state,code,end', [('COMPLETED', '0:0', audit.FAILED_END),
    ('FAILED', '2:0', audit.FAILED_END), ('FAILED', '1:0', '2026-09-12T02:04:20')])
def test_batch_step_exact_failure_required(fixture, state, code, end):
    def change(record):
        rows = [line.split('|') for line in record['stdout'].splitlines()]
        batch = next(row for row in rows if row[1].endswith('.batch'))
        batch[2:4] = [state, code]; batch[5] = end
        record['stdout'] = ''.join('|'.join(row) + '\n' for row in rows)
    mutate(fixture.root / 'terminal_accounting.json', change)
    with pytest.raises(ValueError, match='batch/extern'):
        audit.audit(fixture.root)


def test_successful_v1_destination_stays_unused(fixture):
    (fixture.root / 'execution_audit.json').write_text('{}')
    with pytest.raises(ValueError, match='v1 execution audit must remain unused'):
        audit.audit(fixture.root)


def test_reconciliation_verifies_saved_equality_and_never_relabels_failure(fixture):
    value = audit.audit(fixture.root)
    destination = fixture.root / 'execution_reconciliation.json'
    audit.publish_reconciliation(value, destination)
    assert audit.verify_reconciliation(fixture.root) == value
    assert value['recovery_execution']['state'] == 'FAILED'
    assert value['scheduler_success'] is False and value['exit_cause'] == 'unknown'
    assert not (fixture.root / 'execution_audit.json').exists()
    destination.chmod(0o644)
    mutate(destination, lambda record: record.update(scheduler_success=True))
    with pytest.raises(ValueError, match='differs from current evidence'):
        audit.verify_reconciliation(fixture.root)


def test_publication_rejects_fabricated_current_evidence(fixture):
    value = audit.audit(fixture.root)
    value['completed_development_tiers'][0]['batches'] = 999
    with pytest.raises(ValueError, match='differs from authenticated current evidence'):
        audit.publish_reconciliation(value, fixture.root / 'execution_reconciliation.json')
    assert not (fixture.root / 'execution_reconciliation.json').exists()
