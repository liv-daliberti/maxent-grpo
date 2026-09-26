"""Supplemental provenance is a read-only proof, never a scheduling authority."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ops/exp_scaling'))
import modebench_scale_execution_provenance as provenance


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + '\n')
    return path


def refresh(path):
    value = provenance.read(path)
    value['files_sha256'] = provenance.pins_for(value['files_sha256'])
    write(path, value)


def memory_receipt(*, peak=40 * 1024**3, ceiling='max'):
    scope = '/sys/fs/cgroup/slurm/job_73'
    snapshots = {}
    for stage in ('before_load', 'after_load', 'after_generation'):
        snapshots[stage] = {'job_id': '73', 'job_scope': scope, 'shared_ancestors_excluded': True,
                            'scopes_leaf_to_job': [{'path': scope, 'is_job_scope': True,
                                'counters': {'memory.current': peak, 'memory.peak': peak,
                                             'memory.max': ceiling, 'memory.events': {'oom': 0, 'oom_kill': 0}}}]}
    return {'job_id': '73', 'memory_snapshots': snapshots, 'memory_qualification': {
        'requested_host_limit_bytes': 59 * 1024**3, 'observed_job_peak_bytes': peak,
        'visible_process_ancestor_limit_bytes': None if ceiling == 'max' else ceiling,
        'hard_ceiling_at_or_below_59_gib_visible': ceiling != 'max' and ceiling <= 59 * 1024**3,
        'gates': {'job_peak_below_59_gib': True, 'job_oom_events_zero': True}, 'passed': True,
        'cache_state': 'uncontrolled_existing_node_cache', 'cold_load_claim': False}}


def test_observed_memory_does_not_claim_cold_loading_or_an_unobserved_ceiling():
    result = provenance.qualify_memory(memory_receipt())
    assert result['observed_job_peak_bytes'] == 40 * 1024**3
    assert result['cold_load_claim'] is False
    assert result['hard_ceiling_at_or_below_59_gib_visible'] is False
    assert result['visible_process_ancestor_limit_bytes'] is None


@pytest.mark.parametrize('mutation', ['peak_at_limit', 'oom', 'shared_ancestor', 'wrong_job', 'scope_changed',
                                     'missing_stage', 'false_cold_claim', 'false_ceiling_claim', 'fabricated_peak'])
def test_memory_claim_rejects_wrong_scope_oom_or_overstated_evidence(mutation):
    receipt = memory_receipt()
    last = receipt['memory_snapshots']['after_generation']
    counters = last['scopes_leaf_to_job'][-1]['counters']
    if mutation == 'peak_at_limit': counters['memory.peak'] = 59 * 1024**3
    elif mutation == 'oom': counters['memory.events']['oom_kill'] = 1
    elif mutation == 'shared_ancestor': last['scopes_leaf_to_job'].insert(0, {'path': '/sys/fs/cgroup/slurm', 'is_job_scope': False, 'counters': {}})
    elif mutation == 'wrong_job': last['job_id'] = '74'
    elif mutation == 'scope_changed': last['job_scope'] = '/sys/fs/cgroup/other/job_73'
    elif mutation == 'missing_stage': receipt['memory_snapshots'].pop('before_load')
    elif mutation == 'false_cold_claim': receipt['memory_qualification']['cold_load_claim'] = True
    elif mutation == 'false_ceiling_claim': receipt['memory_qualification']['hard_ceiling_at_or_below_59_gib_visible'] = True
    elif mutation == 'fabricated_peak': receipt['memory_qualification']['observed_job_peak_bytes'] = 1
    with pytest.raises(ValueError): provenance.qualify_memory(receipt)


@pytest.fixture
def mutation(tmp_path):
    root = tmp_path / 'schedule_v2'
    expected = {key: 'fixture' for key in provenance.RESOURCE_FIELDS}
    expected.update({'ArrayJobId': str(provenance.ARRAY_JOB_ID), 'Command': '/fixture/worker.slurm',
                     'ReqNodeList': 'node105', 'NumCPUs': '6', 'CPUs/Task': '6', 'NumTasks': '1',
                     'TresPerNode': 'gres/gpu:a5000:2', 'TimeLimit': '08:00:00'})

    def snapshot(path, task, memory, throttle):
        value = {**expected, 'ArrayTaskId': str(task), 'MinMemoryNode': memory,
                 'ArrayTaskThrottle': str(throttle), 'JobState': 'RUNNING' if task == 4 else 'PENDING',
                 'SubmitLine': 'sbatch --array=0-9%1 --mem=60G /fixture/worker.slurm'}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(' '.join(key + '=' + value for key, value in value.items()) + '\n')
        return path

    def node(path):
        path.write_text('NodeName=node105 CfgTRES=gres/gpu:a5000=10 AllocTRES=gres/gpu:a5000=5 CPUTot=96 CPUAlloc=38 RealMemory=515096 AllocMem=454656\n')
        return path

    before = [snapshot(root / f'before_task_{index}.txt', index, '60G', 1) for index in range(4, 10)]
    receipt = write(root / 'qualification.json', {'job_id': '73'})
    amendment = {'created_at': '2026-09-11T18:00:00+00:00', 'unchanged_per_cell': expected,
                 'qualification': {'receipt': str(receipt)}, 'files_sha256': provenance.pins_for(before)}
    amendment_path = write(root / 'amendment.json', amendment)
    sidecar = write(root / 'amendment.sha256.json', {'sha256': provenance.file_sha(amendment_path)})
    operator = write(root / 'apply_schedule.py', {'fixture': True})
    accounting = root / 'pre_apply_qualification_sacct.txt'; accounting.write_text('73|COMPLETED|0:0\n')
    initial = [operator, accounting, node(root / 'pre_apply_node105.txt'),
               *[snapshot(root / f'pre_apply_task_{index}.txt', index, '60G', 1) for index in range(4, 10)]]
    commands = provenance.expected_commands()
    common = {'array_job_id': provenance.ARRAY_JOB_ID, 'commands': commands, 'before_throttle': 1, 'after_throttle': 2}
    intent = write(root / 'mutation_intent.json', {**common, 'schema': 'modebench_scale_schedule_mutation_intent_v2',
        'created_at': '2026-09-11T18:00:01+00:00', 'files_sha256': provenance.pins_for([amendment_path, sidecar, *initial])})
    dispatches, action_paths = [], []
    for index, command in enumerate(commands):
        action_before = [snapshot(root / f'action_{index}_before_task_{task}.txt', task,
                                 '59G' if 4 < task <= index + 4 else '60G', 1) for task in range(4, 10)]
        linked = [intent, *action_before]
        if index: linked.append(root / f'action_{index - 1}_result.json')
        if index == 5: linked.append(node(root / 'before_throttle_node105.txt'))
        action_intent = write(root / f'action_{index}_intent.json', {'schema': 'modebench_scale_schedule_action_intent_v2',
            'index': index, 'command': command, 'created_at': f'2026-09-11T18:00:{2 + index * 3:02}+00:00',
            'files_sha256': provenance.pins_for(linked)})
        dispatch = {'index': index, 'command': command, 'returncode': 0, 'stdout': '', 'stderr': '', 'timed_out': False}
        dispatch_path = write(root / f'action_{index}_dispatch.json', {**dispatch,
            'created_at': f'2026-09-11T18:00:{3 + index * 3:02}+00:00'})
        raw = [snapshot(root / f'action_{index}_after_task_{task}.txt', task,
                        '59G' if 4 < task <= index + 5 else '60G', 2 if index == 5 else 1) for task in range(4, 10)]
        action_result = write(root / f'action_{index}_result.json', {**dispatch,
            'schema': 'modebench_scale_schedule_action_result_v2', 'status': 'applied_observed',
            'created_at': f'2026-09-11T18:00:{4 + index * 3:02}+00:00',
            'tasks': {str(task): provenance.scheduler_fields(path.read_text()) for task, path in zip(range(4, 10), raw)},
            'files_sha256': provenance.pins_for([action_intent, dispatch_path, *raw])})
        dispatches.append(dispatch); action_paths.extend([action_intent, dispatch_path, action_result])
    after = [snapshot(root / f'after_task_{index}.txt', index, '60G' if index == 4 else '59G', 2)
             for index in range(4, 10)]
    accounting_after = root / 'after_sacct.txt'; accounting_after.write_text('parent remains live\n')
    after_resources = node(root / 'after_node105.txt')
    result_path = write(root / 'mutation_result.json', {**common, 'schema': 'modebench_scale_schedule_mutation_result_v2',
        'created_at': '2026-09-11T18:00:25+00:00', 'status': 'applied_observed', 'per_cell_memory_verified': True,
        'unchanged_per_cell_verified': True, 'dispatches': dispatches,
        'tasks': {str(task): provenance.scheduler_fields(path.read_text()) for task, path in zip(range(4, 10), after)},
        'files_sha256': provenance.pins_for([amendment_path, sidecar, intent, *initial, *action_paths, *after, accounting_after, after_resources])})

    def reseal():
        result = provenance.read(result_path)
        for index in range(6):
            action_intent = root / f'action_{index}_intent.json'; refresh(action_intent)
            action_result = root / f'action_{index}_result.json'; refresh(action_result)
            value = provenance.read(action_result)
            result['dispatches'][index] = {key: value[key] for key in
                                           ('index', 'command', 'returncode', 'stdout', 'stderr', 'timed_out')}
        result['files_sha256'] = provenance.pins_for(result['files_sha256'])
        write(result_path, result)
    return SimpleNamespace(root=root, amendment=amendment, result=result_path, reseal=reseal)


def test_all_six_observed_actions_are_required_in_the_registered_order(mutation):
    pins = provenance.validate_mutation(mutation.root, mutation.amendment)
    assert str(mutation.result) in pins
    assert all(str(mutation.root / f'action_{index}_result.json') in pins for index in range(6))
    assert all(str(mutation.root / f'after_task_{index}.txt') in pins for index in range(4, 10))


@pytest.mark.parametrize('mutation_name', ['failed_dispatch', 'timed_out', 'wrong_command', 'wrong_order',
                                         'early_throttle', 'wrong_memory', 'changed_7b_memory', 'missing_intermediate',
                                         'partial_result', 'missing_ledger', 'unqualified_resources'])
def test_partial_ambiguous_or_wrong_scheduler_changes_never_validate(mutation, mutation_name):
    s = mutation; action_path = s.root / 'action_0_result.json'; value = provenance.read(action_path)
    if mutation_name == 'failed_dispatch': value['returncode'] = 1
    elif mutation_name == 'timed_out': value['timed_out'] = True
    elif mutation_name == 'wrong_command': value['command'] = provenance.COMMAND
    elif mutation_name == 'wrong_order': value['index'] = 4
    elif mutation_name == 'early_throttle':
        path = s.root / 'action_0_after_task_5.txt'; path.write_text(path.read_text().replace('ArrayTaskThrottle=1', 'ArrayTaskThrottle=2'))
    elif mutation_name == 'wrong_memory':
        path = s.root / 'after_task_8.txt'; path.write_text(path.read_text().replace('MinMemoryNode=59G', 'MinMemoryNode=58G'))
    elif mutation_name == 'changed_7b_memory':
        path = s.root / 'after_task_4.txt'; path.write_text(path.read_text().replace('MinMemoryNode=60G', 'MinMemoryNode=59G'))
    elif mutation_name == 'missing_intermediate': value['files_sha256'].pop(str(s.root / 'action_0_after_task_5.txt'))
    elif mutation_name == 'unqualified_resources':
        path = s.root / 'after_task_7.txt'; path.write_text(path.read_text().replace('CPUs/Task=6', 'CPUs/Task=4'))
    write(action_path, value)
    s.reseal()
    if mutation_name == 'partial_result':
        result = provenance.read(s.result); result['dispatches'].pop(); write(s.result, result)
    elif mutation_name == 'missing_ledger': (s.root / 'action_5_result.json').unlink()
    with pytest.raises(ValueError): provenance.validate_mutation(s.root, s.amendment)


@pytest.fixture
def release(tmp_path):
    root, source_root = tmp_path / 'release', tmp_path / 'source'
    protocol = write(source_root / 'protocol.json', {'scientific_fixture': True})
    admissions = []
    for level, model in provenance.LEVELS.items():
        domains, sources, pins = {}, {}, {}
        for domain in provenance.DOMAINS:
            source = {'source_kind': 'campaign_v1', 'source_root': str(source_root),
                      'protocol_sha256': provenance.file_sha(protocol), 'parent_recipe_sha256': 'fixture'}
            sources[domain] = source.copy()
            base = source_root / level / 'dataset' / domain
            recipe = write(source_root / level / 'recipes' / (domain + '.json'), {'development_fit_pass': True})
            records = {}
            for split, count in provenance.SPLITS.items():
                rows = [{'problem': f'{level} {domain} {split} {index}', 'answer': 'fixture'} for index in range(count)]
                jsonl = base / (split + '.jsonl'); jsonl.parent.mkdir(parents=True, exist_ok=True)
                jsonl.write_text(''.join(json.dumps(row) + '\n' for row in rows))
                arrow = write(base / split / 'data.arrow', {'fixture_deserialized_elsewhere': True})
                pins.update(provenance.pins_for([jsonl, arrow]))
                records[split] = {'rows': count, 'rows_sha256': provenance.sha(rows)}
            identity = write(base / 'identity.json', {'level': level, 'domain': domain,
                'protocol_sha256': provenance.file_sha(protocol), 'recipe_sha256': provenance.file_sha(recipe), 'splits': records})
            receipt = write(source_root / level / 'results/confirmation' / (domain + '.json'), {'recorded_model_outcomes': 'fixture'})
            audit = write(source_root / level / 'confirmation' / (domain + '.json'), {'level': level, 'domain': domain,
                'difficulty_matched': True, 'original_grader_replayed_attempts': 4096,
                'receipt_sha256': provenance.file_sha(receipt), 'recipe_sha256': provenance.file_sha(recipe),
                'dataset_identity_sha256': provenance.file_sha(identity)})
            pins.update(provenance.pins_for([protocol, recipe, identity, receipt, audit]))
            domains[domain] = {**source, 'splits': {split: {'path': str(base / split), **record} for split, record in records.items()}}
            link = root / level / 'dataset' / domain; link.parent.mkdir(parents=True, exist_ok=True); link.symlink_to(base)
        manifest = write(root / level / 'source_manifest.json', {'level': level, 'model_label': model, 'sources': sources})
        pins.update(provenance.pins_for([manifest]))
        admissions.append(write(root / level / 'admission.json', {'schema': 'modebench_scale_composite_level_admission_v1',
            'level': level, 'model_label': model, 'difficulty_matched': True, 'test_split': 'eval',
            'treatment_training_started': False, 'domains': domains, 'files_sha256': pins}))
    campaign = write(root / 'admission.json', {'schema': 'modebench_scale_composite_campaign_admission_v1',
        'difficulty_matched': True, 'levels': {level: {'status': 'admitted', 'admission': str(root / level / 'admission.json')}
                                            for level in provenance.LEVELS}, 'files_sha256': provenance.pins_for(admissions)})
    return SimpleNamespace(root=root, source=source_root, campaign=campaign)


def test_release_requires_both_levels_all_ten_domains_and_actual_exact_row_counts(release):
    levels, pins = provenance.validate_release(release.root)
    assert set(levels) == {'level4', 'level5'}
    assert all(level['rows_per_domain'] == {'train': 384, 'dev': 128, 'eval': 128} for level in levels.values())
    assert str(release.campaign) in pins


@pytest.mark.parametrize('changed', ['missing_level', 'missing_domain', 'wrong_model', 'wrong_level_route',
                                     'wrong_count', 'missing_replay', 'failed_heldout', 'changed_data',
                                     'wrong_split_path', 'missing_pin', 'missing_link'])
def test_incomplete_or_mutated_release_cannot_receive_execution_provenance(release, changed):
    s = release; level_path = s.root / 'level5/admission.json'; level = provenance.read(level_path)
    if changed == 'missing_domain': level['domains'].pop('pantry')
    elif changed == 'wrong_model': level['model_label'] = '7b'
    elif changed == 'wrong_count': level['domains']['pantry']['splits']['train']['rows'] = 383
    elif changed == 'wrong_split_path': level['domains']['pantry']['splits']['eval']['path'] = str(s.root / 'other')
    elif changed == 'missing_pin': level['files_sha256'].pop(str(s.source / 'level5/dataset/pantry/eval.jsonl'))
    elif changed in ('missing_replay', 'failed_heldout'):
        path = s.source / 'level5/confirmation/pantry.json'; audit = provenance.read(path)
        audit['original_grader_replayed_attempts' if changed == 'missing_replay' else 'difficulty_matched'] = 4095 if changed == 'missing_replay' else False
        write(path, audit); level['files_sha256'][str(path)] = provenance.file_sha(path)
    elif changed == 'changed_data':
        path = s.source / 'level5/dataset/pantry/eval.jsonl'; path.write_text(path.read_text() + '{}\n')
        level['files_sha256'][str(path)] = provenance.file_sha(path)
    elif changed == 'missing_link': (s.root / 'level5/dataset/pantry').unlink()
    write(level_path, level); refresh(s.campaign)
    if changed in ('missing_level', 'wrong_level_route'):
        campaign = provenance.read(s.campaign)
        if changed == 'missing_level': campaign['levels'].pop('level5')
        else: campaign['levels']['level5']['admission'] = str(s.root / 'level4/admission.json')
        write(s.campaign, campaign)
    with pytest.raises(ValueError): provenance.validate_release(s.root)


def test_publisher_is_read_only_by_default_immutable_and_pins_its_own_source(release, monkeypatch, tmp_path):
    evidence = write(tmp_path / 'checked_schedule.json', {'fixture': True})
    amendment = {'initial_submit_array_spec': '0-9%1', 'per_cell_memory_overrides': {'5': {'before': '60G', 'after': '59G'}}}
    monkeypatch.setattr(provenance, 'validate_schedule', lambda *args, **kwargs:
                        (amendment, provenance.pins_for([evidence]), {'cold_load_claim': False}))
    destination = release.root / 'execution_provenance.json'
    value = provenance.verify(release=release.root)
    assert not destination.exists()
    assert str(Path(provenance.__file__).resolve()) in value['files_sha256']
    provenance.verify(release=release.root, publish=True)
    first = destination.read_bytes()
    provenance.verify(release=release.root, publish=True)
    assert destination.read_bytes() == first
    write(destination, {'status': 'fabricated'})
    with pytest.raises(ValueError, match='immutable supplemental'):
        provenance.verify(release=release.root, publish=True)


def test_publication_never_creates_admissions_or_skips_failed_validation(release, monkeypatch):
    calls = []

    def missing_mutation(*args, **kwargs):
        calls.append('schedule')
        raise ValueError('missing mutation result')

    monkeypatch.setattr(provenance, 'validate_schedule', missing_mutation)
    with pytest.raises(ValueError, match='missing mutation result'):
        provenance.verify(release=release.root, publish=True)
    assert calls == ['schedule'] and not (release.root / 'execution_provenance.json').exists()

@pytest.fixture
def qualification(tmp_path):
    root, model_root = tmp_path / 'qualification', tmp_path / 'model'
    config = write(model_root / 'config.json', {'fixture_model': True})
    weight = write(model_root / 'shard.safetensors', {'fixture_weight': True})
    weights = [{'name': weight.name, 'resolved_path': str(weight), 'bytes': weight.stat().st_size,
                'mtime_ns': weight.stat().st_mtime_ns}]
    baseline_model = {'label': '14b', 'path': str(model_root), 'weights': weights,
                      'configuration_sha256': {'config.json': provenance.file_sha(config)}}
    qualified_model = {key: value for key, value in baseline_model.items() if key != 'weights'}
    qualified_model['weight_file_manifest'] = weights
    baseline_profile = {'dtype': 'float16', 'tensor_parallel_size': 2, 'gpu_memory_utilization': .82,
                        'swap_space': 4.0, 'enable_prefix_caching': True, 'batch_size': 8,
                        'memory': '60G', 'partition': 'mltheory', 'time': '08:00:00'}
    baseline = {'runtime_profile': baseline_profile, 'models': {'14b': baseline_model}}
    profile = {**baseline_profile, 'memory': '59G', 'partition': 'all', 'time': '00:15:00'}
    settings = {key: baseline_profile[key] for key in
                ('dtype', 'tensor_parallel_size', 'gpu_memory_utilization', 'swap_space', 'enable_prefix_caching')}
    settings['max_model_len'] = 2048
    source = write(root / 'smoke.py', {'fixture_synthetic_source': True})
    worker = write(root / 'worker.slurm', {'fixture_worker': True})
    manifest = write(root / 'manifest.json', {'schema': 'modebench_scale_14b_host_memory_smoke_manifest_v1',
        'files_sha256': provenance.pins_for([config]), 'models': {'14b': qualified_model},
        'numerical_settings': settings, 'runtime_profile': profile, 'smoke_source_sha256': provenance.file_sha(source),
        'synthetic_sampling': {'prompts': 8, 'n': 8, 'temperature': 1., 'top_p': 1., 'max_tokens': 192,
                               'min_tokens': 192, 'ignore_eos': True, 'seeds': list(range(1000, 9000, 1000))}})
    command = ['sbatch', '--mem=59G', str(worker)]
    plan = write(root / 'plan.json', {'schema': 'modebench_scale_14b_host_memory_smoke_plan_v1',
        'files_sha256': provenance.pins_for([manifest, source, worker]), 'runtime_profile': profile,
        'model_identity': qualified_model, 'command': command})
    intent = write(root / 'submission_intent.json', {'plan_sha256': provenance.file_sha(plan), 'command': command,
        'files_sha256': provenance.pins_for([plan])})
    write(root / 'submission_result.json', {'status': 'submitted', 'job_id': 73, 'command': command,
        'plan_sha256': provenance.file_sha(plan), 'intent_sha256': provenance.file_sha(intent)})
    receipt = {**memory_receipt(), 'schema': 'modebench_scale_14b_host_memory_smoke_v1', 'status': 'pass',
        'model_label': '14b', 'actual_llm_settings': settings, 'runtime_profile': profile, 'model': qualified_model,
        'manifest_sha256': provenance.file_sha(manifest), 'smoke_source_sha256': provenance.file_sha(source),
        'benchmark_candidates_loaded': False, 'calibration_outcomes_produced': False,
        'output_token_counts': [[192] * 8 for _ in range(8)], 'prompt_tokens': [1572] * 8,
        'allocation_environment': {'SLURM_MEM_PER_NODE': '60416', 'SLURM_CPUS_PER_TASK': '6', 'SLURM_JOB_PARTITION': 'all'},
        'host': 'node105', 'devices': [{'name': 'A5000'}, {'name': 'A5000'}]}
    receipt_path = write(root / '14b.json', receipt)
    accounting = root / 'terminal_sacct.txt'; accounting.write_text('73|COMPLETED|0:0|00:03:51\n')
    audit = write(root / 'qualification_audit.json', {'schema': 'modebench_scale_14b_59g_qualification_audit_v1',
        'status': 'passed', 'job_id': 73, 'model_label': '14b', 'outputs': 64, 'synthetic_prompt_count': 8,
        'tokens_per_output': 192, 'terminal_completed_exit0': True, 'unchanged_numerical_settings': True,
        'memory_qualification': receipt['memory_qualification'], 'actual_allocation': receipt['allocation_environment'],
        'files_sha256': provenance.pins_for([receipt_path, accounting])})
    reference = {'root': str(root), 'receipt': str(receipt_path), 'plan': str(plan), 'terminal_accounting': str(accounting),
                 'audit': str(audit), 'job_id': 73, 'memory_qualification': receipt['memory_qualification']}
    return SimpleNamespace(root=root, baseline=baseline, reference=reference, receipt=receipt_path,
                           validate=lambda: provenance.validate_qualification(reference, baseline,
                               provenance.pins_for(path for path in root.iterdir() if path.is_file())))


def test_qualification_proves_the_full_workload_and_preserves_memory_limits(qualification):
    result, pins = qualification.validate()
    assert result['workload'] == '8 prompts x 8 samples x 192 output tokens'
    assert result['job_id'] == '73' and result['cold_load_claim'] is False
    assert str(qualification.receipt) in pins


@pytest.mark.parametrize('changed', ['missing_prompt', 'short_output', 'changed_numerics', 'changed_checkpoint',
                                     'wrong_terminal_job', 'nonzero_exit', 'running_job', 'wrong_submission_intent',
                                     'false_audit'])
def test_synthetic_success_alone_cannot_replace_full_qualification_evidence(qualification, changed):
    s = qualification; receipt = provenance.read(s.receipt)
    if changed == 'missing_prompt': receipt['output_token_counts'].pop()
    elif changed == 'short_output': receipt['output_token_counts'][0][0] = 191
    elif changed == 'changed_numerics': receipt['actual_llm_settings']['swap_space'] = 2.0
    elif changed == 'changed_checkpoint': receipt['model']['path'] = '/unregistered/checkpoint'
    elif changed in ('wrong_terminal_job', 'nonzero_exit', 'running_job'):
        text = {'wrong_terminal_job': '74|COMPLETED|0:0\n', 'nonzero_exit': '73|COMPLETED|1:0\n',
                'running_job': '73|RUNNING|0:0\n'}[changed]
        (s.root / 'terminal_sacct.txt').write_text(text)
    elif changed == 'wrong_submission_intent':
        path = s.root / 'submission_result.json'; value = provenance.read(path); value['intent_sha256'] = 'wrong'; write(path, value)
    write(s.receipt, receipt)
    refresh(s.root / 'qualification_audit.json')
    if changed == 'false_audit':
        path = s.root / 'qualification_audit.json'; value = provenance.read(path); value['terminal_completed_exit0'] = False; write(path, value)
    with pytest.raises(ValueError): s.validate()


def test_registered_amendment_digest_cannot_be_replaced_by_resealing_a_sidecar(tmp_path):
    path = write(tmp_path / 'amendment.json', {'schema': 'modebench_scale_parent_array_schedule_amendment_v2'})
    write(tmp_path / 'amendment.sha256.json', {'sha256': provenance.file_sha(path)})
    with pytest.raises(ValueError, match='registered amendment or sidecar changed'):
        provenance.validate_schedule(tmp_path)


def test_cli_help_requires_neither_scheduler_nor_final_admissions(capsys):
    with pytest.raises(SystemExit) as result:
        provenance.main(['--help'])
    assert result.value.code == 0
    text = capsys.readouterr().out
    assert '--publish' in text and 'read-only' in text
