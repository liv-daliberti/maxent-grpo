"""Scientific/execution tamper guards for the adaptive v3 completed audit."""
import copy
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ops'))
import audit_modebench_level3_v3 as audit
common = audit.common


@pytest.mark.parametrize('value', ['0', '01', '123;bad;cluster', '123;', '-1', '123\n456', '12.0', ''])
def test_malformed_scheduler_ids_refuse(value):
    with pytest.raises(ValueError):
        audit.parse_jobid(value)


def test_actual_slurm_cluster_suffix_is_allowed():
    assert audit.parse_jobid('123;cluster-1\n') == '123'


@pytest.mark.parametrize('value', ['gres/gpu:rtx_6000=1', 'cpu=6,gres/gpu=1,gres/gpu:rtx_6000=1', 'gpu:rtx_6000:1'])
def test_exact_single_gpu(value):
    assert audit.single_rtx6000_tres(value)


@pytest.mark.parametrize('value', ['gres/gpu:rtx_6000=10', 'gpu:rtx_6000:10', 'gres/gpu=2,gres/gpu:rtx_6000=1',
    'gres/gpu=1', 'gres/gpu:rtx_6000=1,gres/gpu:a100=1', 'gres/gpu:rtx_6000=1,gres/gpu:rtx_6000=1',
    'gres/gpu:rtx_6000=01', 'gres/gpu:rtx_6000=1junk', ''])
def test_extra_missing_or_prefix_gpu_count_refuses(value):
    assert not audit.single_rtx6000_tres(value)


def account():
    return {'JobIDRaw': '123', 'State': 'COMPLETED', 'ExitCode': '0:0', 'Partition': 'all',
            'AllocCPUS': '6', 'ReqMem': '48G', 'Timelimit': '01:00:00', 'NodeList': 'node1',
            'Start': '2026-09-09T01:00:00', 'End': '2026-09-09T01:10:00',
            'ReqTRES': 'gres/gpu:rtx_6000=1', 'AllocTRES': 'cpu=6,mem=48G,gres/gpu=1,gres/gpu:rtx_6000=1'}


@pytest.mark.parametrize('field,value', [('State', 'RUNNING'), ('ExitCode', '1:0'), ('Partition', 'preempt'),
    ('AllocCPUS', '8'), ('ReqMem', '96G'), ('Timelimit', '02:00:00'), ('NodeList', 'Unknown'),
    ('AllocTRES', 'gres/gpu:rtx_6000=10')])
def test_incomplete_or_wrong_actual_allocation_refuses(field, value):
    record = account(); audit.verify_account(record, '123'); record[field] = value
    with pytest.raises(ValueError):
        audit.verify_account(record, '123')


def plan(stage='development'):
    jobs = []
    for domain, revision in zip(common.REVISED, ('graph_v8', 'python_v7')):
        for tier in range(4) if stage == 'development' else (None,):
            rows = common.contract()['calibration_rows_per_tier'][domain] if tier is not None else 128
            jobs.append({'name': f'{revision}_d{tier}' if tier is not None else '3b_' + domain,
                'domain': domain, 'tier': tier, 'model_label': '3b', 'rows': rows, 'batch_count': 4 * ((rows + 7) // 8),
                'output': str(common.RESULTS / (f'calibration_3b_{revision}_d{tier}.json' if tier is not None
                              else f'confirmation/confirmation_3b_{domain}.json'))})
    return {'jobs': jobs}


def test_full_new_job_shapes_include_partial_final_batches():
    jobs = audit.verify_plan('development', plan())
    assert [job['batch_count'] for job in jobs] == [72] * 4 + [84] * 4
    assert sum(job['rows'] for job in jobs) == 1232
    assert len(audit.verify_plan('confirmation', plan('confirmation'))) == 2


@pytest.mark.parametrize('field,value', [('rows', 128), ('batch_count', 64), ('name', 'graph_v7_d0'),
    ('tier', 1), ('domain', 'python_factors'), ('model_label', '05b'), ('output', '/old/receipt.json')])
def test_actual_plan_cannot_substitute_old_or_partial_cells(field, value):
    record = plan(); record['jobs'][0][field] = value
    with pytest.raises(ValueError):
        audit.verify_plan('development', record)


def cache_fixture(tmp_path):
    folder = tmp_path / 'receipt.json.batches'; folder.mkdir()
    task = {'output': str(tmp_path / 'receipt.json'), 'seeds': list(common.DEV_LABELS), 'batch_size': 8}
    identity = {'source': 'synthetic'}
    rows = [{} for _ in range(14)]
    receipt = {'identity': identity, 'identity_sha256': audit.sha(identity), 'prompt_results': [
        {'draws': [{'text': f'row{i}/draw{label}'} for label in common.DEV_LABELS]} for i in range(14)]}
    common.atomic_new(folder / 'run.json', {'identity': identity, 'identity_sha256': receipt['identity_sha256']})
    for label_index, label in enumerate(common.DEV_LABELS):
        for start in (0, 8):
            end = min(14, start + 8)
            draws = [row['draws'][label_index] for row in receipt['prompt_results'][start:end]]
            common.atomic_new(folder / f'seed-{label}__rows-{start:06d}-{end:06d}.json', {
                'identity_sha256': receipt['identity_sha256'], 'seed': label, 'start': start, 'end': end,
                'draws': draws, 'draws_sha256': audit.sha(draws)})
    return folder, task, rows, receipt


def test_exact_partial_batch_cache_is_covered(tmp_path):
    folder, task, rows, receipt = cache_fixture(tmp_path)
    pins = {}; value = audit.verify_completed_cache(task, receipt, rows, pins)
    assert value['batches'] == 8 and value['draws'] == 56 and value['attempts'] == 448
    assert len(pins) == 9


@pytest.mark.parametrize('tamper', ['missing_tail', 'extra_batch', 'changed_draw', 'run_identity'])
def test_cache_tampering_or_short_coverage_refuses(tmp_path, tamper):
    folder, task, rows, receipt = cache_fixture(tmp_path)
    if tamper == 'missing_tail':
        (folder / f'seed-{common.DEV_LABELS[0]}__rows-000008-000014.json').unlink()
    elif tamper == 'extra_batch':
        (folder / 'extra.json').write_text('{}')
    elif tamper == 'changed_draw':
        receipt['prompt_results'][13]['draws'][3]['text'] = 'different sampled answer'
    else:
        (folder / 'run.json').write_text('{}')
    with pytest.raises((ValueError, FileNotFoundError)):
        audit.verify_completed_cache(task, receipt, rows, {})


def runtime_fixture(tmp_path):
    task = tmp_path / 'tasks.json'; task.write_text('[]')
    launcher = SimpleNamespace(JOB_PREFIX='mb-l3-v3-dev-', WORKER=tmp_path / 'worker.slurm')
    job = {'name': 'graph_v8_d0', 'tasks': str(task), 'output': '/new/output.json'}
    identity = {'job_id': '123', 'cell': job['name'], 'seal_sha256': 'a' * 64,
        'gpu_names': ['Quadro RTX 6000'], 'partition': 'all', 'preemption_mode': 'OFF', 'cpus': 6, 'effective_qos': 'normal'}
    claim = {'identity': {'job_id': '123', 'cell': job['name'], 'seal_sha256': 'a' * 64,
                         'task_sha256': common.digest(task), 'output': job['output']}}
    scheduler = {'job_id': '123', 'partition': 'all', 'preemption_mode': 'OFF', 'cpus': 6, 'effective_qos': 'normal',
        'scheduler_job': f'JobId=123 JobState=RUNNING JobName=mb-l3-v3-dev-graph_v8_d0 Command={launcher.WORKER} '
            f'UserId=test({os.getuid()}) Partition=all NumCPUs=6 MinMemoryNode=48G QOS=normal TimeLimit=01:00:00 NodeList=node1 '
            'AllocTRES=cpu=6,mem=48G,gres/gpu=1,gres/gpu:rtx_6000=1',
        'scheduler_partition': 'PartitionName=all PreemptMode=OFF'}
    runtime = {'identity': identity, 'scheduler': scheduler}
    claim['runtime'] = copy.deepcopy(runtime); claim['gpu_names'] = list(identity['gpu_names'])
    return job, launcher, claim, runtime


def test_runtime_proves_exact_actual_worker(tmp_path, monkeypatch):
    job, launcher, claim, runtime = runtime_fixture(tmp_path)
    monkeypatch.setattr(audit, 'verify_engine_log', lambda text, model: None)
    text = json.dumps({'event': 'worker_authenticated', **runtime['identity']})
    audit.verify_runtime(job, '123', 'a' * 64, claim, runtime, account(), text, launcher, '/model')
    runtime['scheduler']['scheduler_job'] = runtime['scheduler']['scheduler_job'].replace('graph_v8_d0', 'python_v7_d0')
    with pytest.raises(ValueError):
        audit.verify_runtime(job, '123', 'a' * 64, claim, runtime, account(), text, launcher, '/model')


def gate_recipe():
    import fit_modebench_level3_fixed_reference as fitter
    differences = {'pass1': .04, 'pass8': -.08}
    return {'schema': fitter.SCHEMA, 'development_fit_pass': True, 'development': {
        'gates': {stage: {'pass1': True, 'pass8': True} for stage in ('expected_dev', 'expected_eval', 'selected')},
        'selected_development_sets_scored': 1, 'selected_evaluation_sets_scored': 0,
        'weight_combinations_considered': 1771, 'tolerances': dict(common.TOLERANCES),
        'expected_dev_differences': dict(differences), 'expected_eval_differences': dict(differences),
        'differences': dict(differences)}}


def test_all_six_gates_use_unchanged_inclusive_threshold():
    recipe = gate_recipe(); audit.validate_recipe_gates(recipe)
    recipe['development']['expected_eval_differences']['pass1'] = .04000001
    with pytest.raises(ValueError):
        audit.validate_recipe_gates(recipe)
    recipe['development']['gates']['expected_eval']['pass1'] = False
    recipe['development_fit_pass'] = False
    audit.validate_recipe_gates(recipe)


@pytest.mark.parametrize('field,value', [('selected_development_sets_scored', 2), ('selected_evaluation_sets_scored', 1),
    ('weight_combinations_considered', 1770), ('tolerances', {'pass1': .05, 'pass8': .1})])
def test_new_recipe_policy_cannot_relax(field, value):
    recipe = gate_recipe(); recipe['development'][field] = value
    with pytest.raises(ValueError):
        audit.validate_recipe_gates(recipe)


def report_fixture():
    targets = {domain: {'metrics': {'pass1': .2, 'pass8': .5}, 'provenance': {'role': 'frozen_benchmark_reference'}}
               for domain in common.DOMAINS}
    old = {'domains': {domain: {'observed_approximate_match': True, 'within_tolerance': {'pass1': True, 'pass8': True},
        'means': {'baseline': {'pass1': .2, 'pass8': .5}, 'candidate': {'pass1': .21, 'pass8': .52}},
        'receipts': {'candidate': {'path': '/old/candidate', 'sha256': 'a' * 64}}, 'source_datasets': {'candidate': '/old/data'}}
        for domain in common.DOMAINS}}
    cells = [{'domain': domain, 'metrics': {'pass1': .21, 'pass8': .52}, 'receipt_path': '/fresh/' + domain,
              'receipt_sha256': 'b' * 64, 'source': {'path': '/fresh/data'}, 'job_id': str(123 + i)}
             for i, domain in enumerate(common.REVISED)]
    return old, targets, cells


def test_aggregate_discloses_retained_and_fresh_sources():
    old, targets, cells = report_fixture()
    records = audit.fixed_reference_report_domains(old, targets, cells)
    assert sum(record['fresh_candidate_confirmation'] for record in records.values()) == 2
    for domain in common.RETAINED:
        assert records[domain]['candidate_confirmation_round'] == 1
        assert records[domain]['retained_original_report_domain'] == old['domains'][domain]
    for domain in common.REVISED:
        assert records[domain]['candidate_confirmation_round'] == 2
        assert records[domain]['draw_labels'] == list(common.CONF_LABELS)


def test_actual_fresh_failure_remains_failure():
    old, targets, cells = report_fixture(); cells[0]['metrics']['pass1'] = .1
    records = audit.fixed_reference_report_domains(old, targets, cells)
    assert records['graph_coloring']['within_tolerance']['pass1'] is False
    assert records['graph_coloring']['observed_approximate_match'] is False


@pytest.mark.parametrize('tamper', ['missing_fresh', 'failed_retained', 'changed_target'])
def test_aggregate_cannot_promote_missing_or_changed_evidence(tamper):
    old, targets, cells = report_fixture()
    if tamper == 'missing_fresh': cells.pop()
    elif tamper == 'failed_retained': old['domains']['pantry']['observed_approximate_match'] = False
    else: targets['countdown']['metrics']['pass1'] = .3
    with pytest.raises(ValueError):
        audit.fixed_reference_report_domains(old, targets, cells)


@pytest.mark.parametrize('function', [audit.audit_completed_development, audit.audit_completed_confirmation])
def test_no_publication_without_actual_original_grading(function):
    with pytest.raises(ValueError):
        function(seal_sha256='a' * 64, registration_sha256='b' * 64, publish=True, regrade=False)
    with pytest.raises(ValueError):
        function(seal_sha256='a' * 64, registration_sha256='b' * 64, publish=True, saved_accounts={})


def test_original_grader_replay_checks_failures_and_full_keys(monkeypatch):
    import oat_drgrpo.math_grader as grader
    row = {'problem': 'fixed prompt', 'answer': 'fixed specification'}
    attempts = [{'text': str(index), 'verified': index % 2 == 0,
                 'canonical_key': ['mode', index] if index % 2 == 0 else None} for index in range(8)]
    receipt = {'prompt_results': [{'row_sha256': audit.sha(row),
                 'draws': [{'attempts': copy.deepcopy(attempts)} for _ in range(4)]}]}
    observed = []
    def original(text, specification):
        assert specification == row['answer']
        observed.append(text)
        index = int(text)
        return ['mode', index] if index % 2 == 0 else None
    monkeypatch.setattr(grader, 'validated_modebench_outcome_key', original)
    assert audit.grade_completed_receipt(receipt, [row]) == 32
    assert len(observed) == 32 and observed.count('1') == 4
    receipt['prompt_results'][0]['draws'][0]['attempts'][0]['canonical_key'] = ['mode', 'different']
    with pytest.raises(ValueError):
        audit.grade_completed_receipt(receipt, [row])


@pytest.mark.parametrize('mutation', ['allocated_cpu', 'duplicate_cpu', 'partition_name', 'claim_runtime', 'claim_gpu'])
def test_runtime_rejects_inconsistent_allocation_or_duplicate_evidence(tmp_path, monkeypatch, mutation):
    job, launcher, claim, runtime = runtime_fixture(tmp_path)
    monkeypatch.setattr(audit, 'verify_engine_log', lambda *args: None)
    if mutation == 'allocated_cpu':
        runtime['scheduler']['scheduler_job'] = runtime['scheduler']['scheduler_job'].replace('AllocTRES=cpu=6', 'AllocTRES=cpu=4')
        claim['runtime'] = copy.deepcopy(runtime)
    elif mutation == 'duplicate_cpu':
        runtime['scheduler']['scheduler_job'] = runtime['scheduler']['scheduler_job'].replace('AllocTRES=cpu=6', 'AllocTRES=cpu=6,cpu=6')
        claim['runtime'] = copy.deepcopy(runtime)
    elif mutation == 'partition_name':
        runtime['scheduler']['scheduler_partition'] = 'PartitionName=other PreemptMode=OFF'
        claim['runtime'] = copy.deepcopy(runtime)
    elif mutation == 'claim_runtime':
        claim['runtime']['scheduler']['cpus'] = 4
    else:
        claim['gpu_names'] = ['NVIDIA A100']
    text = json.dumps({'event': 'worker_authenticated', **runtime['identity']})
    with pytest.raises(ValueError):
        audit.verify_runtime(job, '123', 'a' * 64, claim, runtime, account(), text, launcher, '/model')


def test_original_grader_timeout_propagates_as_base_exception_and_restores_timer(monkeypatch):
    import signal
    import time
    previous = signal.getsignal(signal.SIGALRM)
    monkeypatch.setattr(audit, 'grade_completed_receipt', lambda *args: time.sleep(.2))
    with pytest.raises(audit.OriginalGraderTimeout):
        audit.replay_original({}, [], seconds=.01)
    assert signal.getitimer(signal.ITIMER_REAL)[0] == 0
    assert signal.getsignal(signal.SIGALRM) == previous


def test_timed_out_execution_cannot_publish_completed_proof(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, 'CANONICAL', tmp_path / 'proof.json')
    launcher = SimpleNamespace(SEAL=tmp_path/'seal.json', PLAN=tmp_path/'plan.json')
    monkeypatch.setattr(audit, 'stage_launcher', lambda *args: (launcher, {'files_sha256':{}, 'directory_files':{}}, {}))
    monkeypatch.setattr(audit, 'add_pin', lambda *args: None)
    monkeypatch.setattr(audit, 'read', lambda *args: {})
    def timed_out(*args, **kwargs):
        raise audit.OriginalGraderTimeout('original-grader timeout')
    monkeypatch.setattr(audit, 'audit_cells', timed_out)
    with pytest.raises(audit.OriginalGraderTimeout):
        audit.audit_completed_development(seal_sha256='a'*64, registration_sha256='b'*64, publish=True)
    assert not audit.CANONICAL.exists()
