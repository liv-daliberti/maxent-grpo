"""Isolated launch/recovery safety tests; never contact Slurm or read outcomes."""
from copy import deepcopy
import importlib.util
import itertools
import json
import os
from pathlib import Path
import pwd
import re
import shlex
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'ops/exp_scaling/launch_e124_qwen7b_three_level.py'


@pytest.fixture
def ctl(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('e124_launcher_under_test', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name, path in {
        'ROOT': tmp_path, 'ART': tmp_path / 'var/artifacts/e124_qwen7b_three_level',
        'PLAN': tmp_path / 'var/artifacts/e124_qwen7b_three_level/plan.json',
        'TX': tmp_path / 'var/artifacts/e124_qwen7b_three_level/transaction.json',
        'LEDGER': tmp_path / 'var/artifacts/e124_qwen7b_three_level_jobs.json',
        'PROTOCOL': tmp_path / 'paper/preregistration/e124.md',
    }.items(): monkeypatch.setattr(module, name, path)
    def forbidden(*args, **kwargs): raise AssertionError('unmocked scheduler command')
    monkeypatch.setattr(module, 'command', forbidden)
    return module


@pytest.fixture
def sample(ctl):
    row = {'cell_id': 'l1-graph_coloring-maxrl-s70', 'level': 1, 'domain': 'graph_coloring',
           'arm': 'maxrl', 'seed': 70, 'model_size': '7b', 'run_stamp': 'e124_l1_graph_maxrl_s70',
           'run_dir': str(ctl.ROOT / 'var/data/e124/l1/graph/maxrl'),
           'environment': {'RUN_STAMP': 'e124_l1_graph_maxrl_s70', 'SAVE_PATH': str(ctl.ROOT / 'var/data/e124/l1/graph/maxrl'),
                           'OAT_ZERO_NUM_SAMPLES': '16', 'OAT_ZERO_PRETRAIN': '/frozen/qwen7b', 'OAT_ZERO_RESUME_STEPS': '96'}}
    plan = {'plan_sha256': 'a' * 64, 'snapshot': {'root': str(ctl.ROOT / 'snapshot')}, 'cells': [row]}
    tx = {'rows': {}, 'events': [], 'status': 'prepared', 'plan_sha256': plan['plan_sha256']}
    return plan, tx, row


def scheduler_record(ctl, row, item, changes=None, command_override=None):
    argv = item['command'] if command_override is None else command_override
    name = next(x.split('=', 1)[1] for x in item['command'] if x.startswith('--job-name='))
    comment = next(x.split('=', 1)[1] for x in item['command'] if x.startswith('--comment='))
    fields = {'JobId': str(item['job_id']), 'JobName': name,
              'UserId': f'{pwd.getpwuid(os.getuid()).pw_name}({os.getuid()})',
              'Account': 'mltheory', 'Partition': 'lowprio', 'MinMemoryNode': '256G',
              'NumCPUs': '8', 'CPUs/Task': '8', 'TresPerNode': 'gres/gpu:a6000:1',
              'TimeLimit': '3-00:00:00', 'Nice': '100', 'Requeue': '1', 'Dependency': '(null)',
              'Comment': comment, 'ReqNodeList': 'node[205-208]', 'ExcNodeList': ctl.PVL,
              'JobState': 'PENDING', 'Reason': 'JobHeldUser', 'Priority': '0'}
    fields.update(changes or {})
    # scontrol renders this reason with underscores; squeue uses spaces.
    if fields['Reason'] == 'job requeued in held state':
        fields['Reason'] = 'job_requeued_in_held_state'
    return ' '.join(f'{k}={v}' for k, v in fields.items()) + ' SubmitLine=' + shlex.join(argv) + ' WorkDir=' + str(ctl.ROOT) + ' StdOut=/logs/output StdErr=/logs/error'


def mock_held_record(ctl, monkeypatch, record):
    monkeypatch.setattr(ctl, 'show', lambda _: record)
    monkeypatch.setattr(ctl, 'command', lambda argv, **_: '\n'.join(ctl.POOL.split(',')) if argv[:3] == ['scontrol', 'show', 'hostnames'] else pytest.fail('unexpected scheduler command'))


def test_valid_real_shape_held_record_passes_full_audit(ctl, sample, monkeypatch):
    plan, _, row = sample
    item = {'job_id': 700, 'command': ctl.job_command(plan, row)}
    record = scheduler_record(ctl, row, item)
    mock_held_record(ctl, monkeypatch, record)
    result = ctl.audit_job(plan, row, item, held=True)
    assert result['CPUs/Task'] == '8' and result['JobId'] == '700'


@pytest.mark.parametrize('field,bad', [('JobId', '701'), ('JobName', 'other-campaign'), ('UserId', 'other(99999)'),
    ('MinMemoryNode', '128G'), ('TresPerNode', 'gres/gpu:a5000:1'), ('TimeLimit', '01:00:00'),
    ('Priority', '1'), ('Reason', 'Resources'), ('CPUs/Task', '4'), ('Dependency', 'afterok:1')])
def test_scheduler_identity_and_resource_drift_rejected(ctl, sample, monkeypatch, field, bad):
    plan, _, row = sample
    item = {'job_id': 700, 'command': ctl.job_command(plan, row)}
    mock_held_record(ctl, monkeypatch, scheduler_record(ctl, row, item, {field: bad}))
    with pytest.raises(RuntimeError): ctl.audit_job(plan, row, item, held=True)


def test_full_environment_drift_rejected_without_subset_audit(ctl, sample, monkeypatch):
    plan, _, row = sample
    item = {'job_id': 700, 'command': ctl.job_command(plan, row)}
    changed = [x.replace('OAT_ZERO_NUM_SAMPLES=16', 'OAT_ZERO_NUM_SAMPLES=8') for x in item['command']]
    mock_held_record(ctl, monkeypatch, scheduler_record(ctl, row, item, command_override=changed))
    with pytest.raises(RuntimeError, match='exports'): ctl.audit_job(plan, row, item, held=True)


@pytest.mark.parametrize('response', ['', 'Submitted batch job 700', '700\n701'])
def test_ambiguous_sbatch_response_preserves_intent_and_never_resubmits(ctl, sample, monkeypatch, response):
    plan, tx, row = sample
    calls = []
    def submit(argv, **kwargs):
        persisted = ctl.read(ctl.TX)
        assert persisted['rows'][row['cell_id']]['status'] == 'submit_intent'
        calls.append(argv)
        return response
    monkeypatch.setattr(ctl, 'command', submit)
    with pytest.raises(RuntimeError, match='ambiguous'): ctl.stage_one(plan, tx, row)
    restarted = ctl.read(ctl.TX)
    with pytest.raises(RuntimeError, match='unresolved submission intent'): ctl.stage_one(plan, restarted, row)
    assert len(calls) == 1


def test_sbatch_transport_failure_does_not_infer_non_submission(ctl, sample, monkeypatch):
    plan, tx, row = sample
    calls = []
    def uncertain(argv, **kwargs):
        calls.append(argv)
        assert ctl.read(ctl.TX)['rows'][row['cell_id']]['status'] == 'submit_intent'
        raise TimeoutError('response lost after scheduler accepted request')
    monkeypatch.setattr(ctl, 'command', uncertain)
    with pytest.raises(TimeoutError): ctl.stage_one(plan, tx, row)
    with pytest.raises(RuntimeError, match='unresolved'): ctl.stage_one(plan, ctl.read(ctl.TX), row)
    assert len(calls) == 1


def test_failed_post_submit_audit_reconciles_existing_id_without_resubmission(ctl, sample, monkeypatch):
    plan, tx, row = sample
    submitted = []
    monkeypatch.setattr(ctl, 'command', lambda argv, **kw: submitted.append(argv) or '700')
    def unavailable(*args, **kwargs): raise RuntimeError('temporary scontrol outage')
    monkeypatch.setattr(ctl, 'audit_job', unavailable)
    with pytest.raises(RuntimeError, match='outage'): ctl.stage_one(plan, tx, row)
    persisted = ctl.read(ctl.TX)
    assert persisted['rows'][row['cell_id']]['job_id'] == 700
    assert persisted['rows'][row['cell_id']]['status'] == 'held_unverified'
    monkeypatch.setattr(ctl, 'audit_job', lambda *a, **k: {'JobId': '700', 'JobState': 'PENDING'})
    ctl.stage_one(plan, persisted, row)
    assert len(submitted) == 1
    assert ctl.read(ctl.TX)['rows'][row['cell_id']]['status'] == 'held'


def test_export_parser_rejects_duplicate_keys(ctl):
    with pytest.raises(RuntimeError, match='duplicate'):
        ctl.exports(['sbatch', '--export=ALL,A=1,A=2'])


def test_runtime_pantry_wrapper_admits_only_level2_and3_maxrl_seed70(ctl):
    original = (ctl.HEALTHY_OPS / 'run_experiment.sh').read_bytes()
    transformed = ctl.runtime_bytes('ops/run_experiment.sh', original).decode()
    pattern = re.search(r'\^e124_l\[23\]_pantry_.*?\$', transformed).group()
    for level, arm, seed in itertools.product((1, 2, 3), ('maxrl', 'replay_maxrl', 'drgrpo', 'replay_drgrpo'), (43, 70, 71)):
        actual = re.fullmatch(pattern, f'e124_l{level}_pantry_{arm}_s{seed}') is not None
        assert actual == (level in (2, 3) and arm in ('maxrl', 'replay_maxrl') and seed == 70)
    assert 'e124_l[123]_*' in transformed
    assert ctl.runtime_bytes('src/oat_drgrpo/learner/run.py', original) == original


def fake_prepare_dependencies(ctl, monkeypatch, duplicate=False):
    domains = ('graph_coloring', 'countdown', 'python_factors', 'mathir', 'pantry_plan')
    rows = [{'level': level, 'domain': domain, 'arm': arm, 'seed': 70,
             'run_stamp': f'e124_l{level}_{domain}_{arm}_s70',
             'run_dir': str(ctl.ROOT / 'var/data/e124' / f'l{level}' / domain / arm),
             'environment': {'RUN_STAMP': f'e124_l{level}_{domain}_{arm}_s70'}}
            for level, domain, arm in itertools.product((1, 2, 3), domains, ('maxrl', 'replay_maxrl'))]
    if duplicate: rows[-1] = deepcopy(rows[0])
    monkeypatch.setitem(sys.modules, 'e124_qwen7b_recipes', SimpleNamespace(dataset_admission=lambda: {'admitted': True}, build_cells=lambda *a, **k: deepcopy(rows)))
    def benchmark_prepare(plan_path, output):
        output.mkdir(parents=True)
        (output / 'plan.json').write_text('{}')
    monkeypatch.setitem(sys.modules, 'benchmark_e124_qwen7b', SimpleNamespace(prepare=benchmark_prepare))
    monkeypatch.setattr(ctl, 'model_identity', lambda: {'files': {}, 'model': 'Qwen/Qwen2.5-7B-Instruct'})
    monkeypatch.setattr(ctl, 'runtime_description', lambda: {'root': str(ctl.ROOT / 'snapshot'), 'inventory': {}})
    monkeypatch.setattr(ctl, 'publish_snapshot', lambda snapshot: Path(snapshot['root']).mkdir())
    compiler = ctl.ROOT / 'var/cuda124_toolkit/bin/nvcc'; compiler.parent.mkdir(parents=True); compiler.write_text('fixture')
    ctl.PROTOCOL.parent.mkdir(parents=True); ctl.PROTOCOL.write_text('prospective fixture')
    return rows


def test_prepare_preserves_all_thirty_level_identities(ctl, monkeypatch):
    fake_prepare_dependencies(ctl, monkeypatch)
    result = ctl.prepare()
    plan = ctl.read(ctl.PLAN)
    assert result['cells'] == len(plan['cells']) == 30
    assert len({x['cell_id'] for x in plan['cells']}) == 30
    assert len({x['run_dir'] for x in plan['cells']}) == 30
    assert {x['level'] for x in plan['cells']} == {1, 2, 3}
    assert {x['arm'] for x in plan['cells']} == {'maxrl', 'replay_maxrl'}
    assert {x['seed'] for x in plan['cells']} == {70}
    assert ctl.read(ctl.LEDGER)['runs'] == []


def test_prepare_rejects_duplicate_cross_level_output_before_publication(ctl, monkeypatch):
    fake_prepare_dependencies(ctl, monkeypatch, duplicate=True)
    with pytest.raises(RuntimeError, match='duplicate'): ctl.prepare()
    assert not ctl.PLAN.exists() and not ctl.TX.exists() and not ctl.LEDGER.exists()


def test_prepare_refuses_existing_namespace(ctl, monkeypatch):
    ctl.PLAN.parent.mkdir(parents=True); ctl.PLAN.write_text('{}')
    with pytest.raises(RuntimeError, match='already prepared'): ctl.prepare()


def test_atomic_and_new_writes_fsync_parent_directory(ctl, monkeypatch):
    synced = []
    monkeypatch.setattr(ctl, 'fsync_directory', lambda path: synced.append(Path(path)))
    target = ctl.ROOT / 'state/item.json'
    ctl.write(target, {'intent': 1}, new=True)
    ctl.write(target, {'result': 2})
    assert synced == [target.parent, target.parent]
    assert ctl.read(target) == {'result': 2}


def test_stage_submits_exact_thirty_science_plus_one_systems_all_held(ctl, monkeypatch):
    rows = fake_prepare_dependencies(ctl, monkeypatch)
    for row in rows:
        row.update(cell_id=f"l{row['level']}-{row['domain']}-{row['arm']}-s70", model_size='7b')
    plan = {'cells': rows, 'plan_sha256': 'a' * 64, 'snapshot': {'root': str(ctl.ROOT / 'snapshot')}}
    tx = {'rows': {}, 'events': [], 'status': 'prepared'}
    monkeypatch.setattr(ctl, 'verify', lambda *a, **k: None)
    submitted = []
    def scheduler(argv, **kw):
        if argv[0] == 'squeue': return ''
        assert argv[0] == 'sbatch' and '--hold' in argv
        submitted.append(argv)
        return str(1000 + len(submitted))
    monkeypatch.setattr(ctl, 'command', scheduler)
    def held_audit(plan, row, item, *, held=None):
        assert held is True
        return {'JobId': str(item['job_id']), 'JobState': 'PENDING'}
    monkeypatch.setattr(ctl, 'audit_job', held_audit)
    result = ctl.stage(plan, tx)
    ledger = ctl.read(ctl.LEDGER)
    assert len(submitted) == 31
    assert len(set(result['science_job_ids'])) == len(ledger['runs']) == 30
    assert result['systems_job_id'] not in result['science_job_ids']
    assert not ledger['released']
    assert {row['controller_status'] for row in ledger['runs']} == {'held'}


def minimal_verified_plan(ctl):
    ctl.PROTOCOL.parent.mkdir(parents=True); ctl.PROTOCOL.write_text('fixed protocol')
    dataset = ctl.ROOT / 'data/train'; dataset.parent.mkdir(); dataset.write_text('fixed data')
    frozen = ctl.ROOT / 'snapshot/control/runtime.py'; frozen.parent.mkdir(parents=True); frozen.write_text('fixed runtime')
    plan = {'protocol_sha256': ctl.digest(ctl.PROTOCOL),
            'admission': {'files_sha256': {str(dataset): ctl.digest(dataset)}, 'directory_files': {str(dataset.parent): [str(dataset)]}},
            'model': {'files': {}}, 'snapshot': {'root': str(ctl.ROOT / 'snapshot'), 'inventory': {'control/runtime.py': {'sha256': ctl.digest(frozen)}}}}
    plan['plan_sha256'] = ctl.seal(plan)
    return plan, {'plan_sha256': plan['plan_sha256'], 'auxiliary_pins': {}}, dataset, frozen


@pytest.mark.parametrize('changed', ['dataset', 'runtime', 'protocol', 'inventory'])
def test_verify_rejects_data_or_frozen_runtime_drift(ctl, changed):
    plan, tx, dataset, frozen = minimal_verified_plan(ctl)
    ctl.verify(plan, tx)
    if changed == 'dataset': dataset.write_text('modified data')
    elif changed == 'runtime': frozen.write_text('modified runtime')
    elif changed == 'protocol': ctl.PROTOCOL.write_text('modified protocol')
    else: (dataset.parent / 'extra').write_text('unregistered row shard')
    with pytest.raises(RuntimeError, match='changed'): ctl.verify(plan, tx)


def write_recovery_checkpoint(row, step, *, optimizer_steps=None, model_step=None, bank=True, protocol=4):
    import pickle
    import zipfile
    cp = Path(row['run_dir']) / 'debug_job700/checkpoints' / f'step_{step:05d}'
    cp.mkdir(parents=True)
    state_step = step if model_step is None else model_step
    model = dict(global_steps=state_step, global_step=state_step, policy_sgd_step=state_step, prompt_batches_consumed_total=state_step)
    if bank: model['online_canonical_bank_state'] = {'restored': True}
    optimizer = {'state': {i: {'step': value} for i, value in enumerate(optimizer_steps or [step, step])}}
    for name, payload in [('mp_rank_00_model_states.pt', model), ('zero_pp_rank_0_mp_rank_00_optim_states.pt', optimizer)]:
        with zipfile.ZipFile(cp / name, 'w') as archive:
            archive.writestr('archive/data.pkl', pickle.dumps(payload, protocol=protocol))
    return cp


@pytest.mark.parametrize('protocol', [2, 4, 5])
def test_valid_checkpoint_all_counters_and_memoized_optimizer_entries(ctl, sample, protocol):
    _, _, row = sample
    cp = write_recovery_checkpoint(row, 192, protocol=protocol)
    detail = ctl.checkpoint(row)
    assert detail['step'] == 192 and detail['path'] == str(cp)
    assert len(detail['files']) == 2


@pytest.mark.parametrize('bad', ['counter', 'memoized_optimizer', 'missing_bank', 'missing_optimizer', 'off_grid', 'past_endpoint', 'corrupt'])
def test_checkpoint_rejects_partial_stale_memoized_or_wrong_grid_metadata(ctl, sample, bad):
    _, _, row = sample
    step = 48 if bad == 'off_grid' else 3264 if bad == 'past_endpoint' else 192
    cp = write_recovery_checkpoint(row, step,
        optimizer_steps=[step, step - 1] if bad == 'memoized_optimizer' else None,
        model_step=step - 1 if bad == 'counter' else None, bank=bad != 'missing_bank')
    if bad == 'missing_optimizer': (cp / 'zero_pp_rank_0_mp_rank_00_optim_states.pt').unlink()
    if bad == 'corrupt': (cp / 'mp_rank_00_model_states.pt').write_bytes(b'partial ZIP write')
    with pytest.raises(RuntimeError, match='no validated checkpoint'): ctl.checkpoint(row)


def test_registered_pantry96_checkpoint_grid_is_accepted(ctl, sample):
    _, _, row = sample
    row['environment']['OAT_ZERO_RESUME_STEPS'] = '96'
    write_recovery_checkpoint(row, 96)
    assert ctl.checkpoint(row)['step'] == 96


def test_incomplete_latest_checkpoint_falls_back_with_rejection_evidence(ctl, sample):
    _, _, row = sample
    write_recovery_checkpoint(row, 192)
    write_recovery_checkpoint(row, 384, optimizer_steps=[384, 383])
    result = ctl.checkpoint(row)
    assert result['step'] == 192
    assert result['rejected'] and 'optimizer step differs' in result['rejected'][0]['error']


def timeout_fixture(ctl, sample, monkeypatch, step=192, floor=0):
    plan, tx, row = sample
    plan.update(max_retries_per_cell=64, deadline_utc='2099-01-01T00:00:00+00:00')
    item = {'job_id': 700, 'status': 'released', 'retries': 0, 'last_resume_step': floor,
            'command': ctl.job_command(plan, row)}
    tx['rows'][row['cell_id']] = item
    detail = {'path': str(Path(row['run_dir']) / f'debug_job700/checkpoints/step_{step:05d}'), 'step': step, 'files': [{'name': 'verified.pt', 'bytes': 100, 'mtime_ns': 1}]}
    monkeypatch.setattr(ctl, 'checkpoint', lambda _: deepcopy(detail))
    monkeypatch.setattr(ctl, 'completed', lambda _: None)
    monkeypatch.setattr(ctl, 'other_writers', lambda *a: None)
    monkeypatch.setattr(ctl, 'audit_job', lambda *a, **k: {'Restarts': '3'})
    return plan, tx, row, item, detail


@pytest.mark.parametrize('state,inactive', [('FAILED', True), ('OUT_OF_MEMORY', True), ('RUNNING', False), ('PENDING', False)])
def test_timeout_recovery_rejects_active_or_noneligible_failure(ctl, sample, monkeypatch, state, inactive):
    plan, tx, row, item, _ = timeout_fixture(ctl, sample, monkeypatch)
    with pytest.raises(RuntimeError, match='only accounted inactive'):
        ctl.recover_timeout(plan, tx, row, item, {'state': state, 'inactive': inactive})
    assert item['status'] == 'released'


@pytest.mark.parametrize('step,floor', [(192, 192), (192, 384)])
def test_no_durable_progress_or_regression_cannot_requeue(ctl, sample, monkeypatch, step, floor):
    plan, tx, row, item, _ = timeout_fixture(ctl, sample, monkeypatch, step, floor)
    with pytest.raises(RuntimeError, match='no durable progress'):
        ctl.recover_timeout(plan, tx, row, item, {'state': 'TIMEOUT', 'inactive': True})
    assert 'requeue_intent' not in item


def test_requeue_cap_is_enforced_before_scheduler_mutation(ctl, sample, monkeypatch):
    plan, tx, row, item, _ = timeout_fixture(ctl, sample, monkeypatch)
    item['retries'] = plan['max_retries_per_cell']
    with pytest.raises(RuntimeError, match='retry cap'):
        ctl.recover_timeout(plan, tx, row, item, {'state': 'TIMEOUT', 'inactive': True})


def test_lost_requeue_ack_reconciles_own_hold_without_second_requeue(ctl, sample, monkeypatch):
    plan, tx, row, item, detail = timeout_fixture(ctl, sample, monkeypatch)
    calls = []
    def lost_ack(argv, **kwargs):
        calls.append(argv)
        stored = ctl.read(ctl.TX)['rows'][row['cell_id']]
        assert stored['status'] == 'requeue_intent'
        assert stored['requeue_intent']['checkpoint'] == detail
        raise TimeoutError('requeue accepted but response lost')
    monkeypatch.setattr(ctl, 'command', lost_ack)
    with pytest.raises(TimeoutError):
        ctl.recover_timeout(plan, tx, row, item, {'state': 'TIMEOUT', 'inactive': True})
    tx = ctl.read(ctl.TX); item = tx['rows'][row['cell_id']]
    monkeypatch.setattr(ctl, 'audit_job', lambda *a, **k: {'Restarts': '4', 'JobState': 'PENDING', 'Reason': 'job_requeued_in_held_state', 'Priority': '0'})
    monkeypatch.setattr(ctl, 'command', lambda argv, **kwargs: calls.append(argv) or '')
    released = []
    monkeypatch.setattr(ctl, 'release_owned', lambda *a, **k: released.append(k))
    ctl.recover_timeout(plan, tx, row, item, {'state': 'PENDING', 'inactive': False})
    assert sum(argv[1] == 'requeuehold' for argv in calls) == 1
    assert item['retries'] == 1 and item['last_resume_step'] == 192
    assert released == [{'retry': True}]


@pytest.mark.parametrize('bad', ['restart_count', 'checkpoint_change', 'unowned_hold'])
def test_uncertain_requeue_never_reconciles_unproven_transition(ctl, sample, monkeypatch, bad):
    plan, tx, row, item, detail = timeout_fixture(ctl, sample, monkeypatch)
    item.update(status='requeue_intent', requeue_intent={'restarts_before': 3, 'retry_number': 1, 'checkpoint': deepcopy(detail)})
    record = {'Restarts': '4', 'JobState': 'PENDING', 'Reason': 'job_requeued_in_held_state', 'Priority': '0'}
    if bad == 'restart_count': record['Restarts'] = '5'
    elif bad == 'unowned_hold': record['Reason'] = 'JobHeldAdmin'
    else: detail['files'][0]['mtime_ns'] = 2
    monkeypatch.setattr(ctl, 'audit_job', lambda *a, **k: record)
    with pytest.raises(RuntimeError): ctl.recover_timeout(plan, tx, row, item, {'state': 'PENDING', 'inactive': False})
    assert item['retries'] == 0


def test_lost_release_ack_observed_running_does_not_issue_second_release(ctl, sample, monkeypatch):
    plan, tx, row = sample
    item = {'job_id': 700, 'status': 'release_intent', 'command': ctl.job_command(plan, row)}
    tx['rows'][row['cell_id']] = item
    monkeypatch.setattr(ctl, 'scheduler_state', lambda _: {'state': 'RUNNING', 'inactive': False})
    monkeypatch.setattr(ctl, 'audit_job', lambda *a, **k: {'JobState': 'RUNNING'})
    ctl.release_owned(plan, tx, row, item)
    assert item['status'] == 'released'


@pytest.mark.parametrize('state,reason', [('RUNNING', 'None'), ('PENDING', 'Resources')])
def test_other_writer_guard_rejects_active_and_released_pending_collision(ctl, sample, monkeypatch, state, reason):
    _, _, row = sample
    monkeypatch.setattr(ctl, 'command', lambda *a, **k: f'701|{state}|{reason}')
    monkeypatch.setattr(ctl, 'show', lambda _: 'SubmitLine=sbatch --export=ALL,SAVE_PATH=' + row['run_dir'] + ' /entrypoint WorkDir=' + str(ctl.ROOT))
    with pytest.raises(RuntimeError, match='other active or released-pending writer'):
        ctl.other_writers(row, 700)


def test_other_writer_guard_ignores_owned_id_and_dormant_user_hold(ctl, sample, monkeypatch):
    _, _, row = sample
    monkeypatch.setattr(ctl, 'command', lambda *a, **k: '700|RUNNING|None\n701|PENDING|JobHeldUser')
    monkeypatch.setattr(ctl, 'show', lambda _: pytest.fail('held/own job must not be probed as another writer'))
    ctl.other_writers(row, 700)


def test_untracked_e124_scheduler_job_blocks_mutation(ctl, monkeypatch):
    monkeypatch.setattr(ctl, 'command', lambda *a, **k: '700|e124-l1-graph-m-s70\n701|e124-controller')
    tx = {'rows': {'cell': {'job_id': 700}}}
    with pytest.raises(RuntimeError, match='untracked E124'): ctl.reject_untracked_jobs(tx)
    tx['controller'] = {'job_id': 701}
    ctl.reject_untracked_jobs(tx)


def test_fresh_storage_denial_after_held_audit_preserves_hold(ctl, sample, monkeypatch):
    plan, tx, row = sample
    item = {'job_id': 700, 'status': 'held', 'command': ctl.job_command(plan, row)}
    tx['rows'][row['cell_id']] = item
    sequence = []
    monkeypatch.setattr(ctl, 'audit_job', lambda *a, **k: sequence.append('audit') or {})
    monkeypatch.setattr(ctl, 'other_writers', lambda *a, **k: sequence.append('writers'))
    def blocked(*args, **kwargs):
        assert sequence == ['audit', 'writers']
        sequence.append('storage')
        return {'allowed': False, 'blocked_reason': 'waiting_disk'}
    monkeypatch.setattr(ctl, 'storage_report', blocked)
    try:
        ctl.release_owned(plan, tx, row, item)
    except RuntimeError as exc:
        assert 'storage' in str(exc).lower() or 'disk' in str(exc).lower()
    assert sequence == ['audit', 'writers', 'storage']
    assert item['status'] == 'held'
    assert not any(event['event'] == 'release_intent' for event in tx['events'])


def terminal_fixture(row, *, missing_shard=False):
    root = Path(row['run_dir'])
    export = root / 'debug_job700/saved_models/step_03073'; export.mkdir(parents=True)
    names = [f'model-0000{i}-of-00004.safetensors' for i in range(1, 5)]
    for name in names[:3] if missing_shard else names:
        with (export / name).open('wb') as stream:
            stream.truncate(15231233024 // 4)  # Sparse metadata fixture, no tensor allocation.
    (export / 'model.safetensors.index.json').write_text(json.dumps({'metadata': {'total_size': 15231233024}, 'weight_map': {f'tensor{i}': name for i, name in enumerate(names)}}))
    (root / 'TRAINING_COMPLETE.json').write_text(json.dumps({'schema': 'oat_zero_training_complete_v1', 'terminal_step': 3073, 'terminal_export': str(export)}))
    return export


def test_terminal_receipt_requires_all_indexed_7b_shards(ctl, sample):
    _, _, row = sample
    export = terminal_fixture(row, missing_shard=True)
    with pytest.raises(RuntimeError): ctl.completed(row)
    with (export / 'model-00004-of-00004.safetensors').open('wb') as stream:
        stream.truncate(15231233024 // 4)
    assert ctl.completed(row)['step'] == 3073


def test_terminal_index_cannot_reference_outside_export(ctl, sample):
    _, _, row = sample
    export = terminal_fixture(row)
    (export / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'weight': '../outside.safetensors'}}))
    with pytest.raises(RuntimeError): ctl.completed(row)


def test_observer_absolute_deadline_blocks_new_mutation(ctl, sample, monkeypatch):
    plan, tx, _ = sample
    plan['deadline_utc'] = '2000-01-01T00:00:00+00:00'
    monkeypatch.setattr(ctl, 'verify', lambda *a: None)
    monkeypatch.setattr(ctl, 'reject_untracked_jobs', lambda *a: None)
    with pytest.raises(RuntimeError, match='deadline'): ctl.observe_pass(plan, tx)


def test_cpu_watcher_requeues_only_itself_after24h_preserving_science(ctl, monkeypatch):
    ctl.ART.mkdir(parents=True)
    plan = {'plan_sha256': 'p'}
    tx = {'controller': {'job_id': 900}, 'rows': {'science': {'job_id': 700, 'status': 'held'}}, 'events': [], 'status': 'waiting_storage'}
    ctl.write(ctl.PLAN, plan); ctl.write(ctl.TX, tx)
    monkeypatch.setenv('SLURM_JOB_ID', '900')
    monkeypatch.setattr(ctl, 'observe_pass', lambda *a: False)
    times = iter([0, 86401])
    monkeypatch.setattr(ctl.time, 'monotonic', lambda: next(times))
    calls = []
    monkeypatch.setattr(ctl, 'command', lambda argv, **kwargs: calls.append(argv) or '')
    assert ctl.watch() == 0
    stored = ctl.read(ctl.TX)
    assert calls == [['scontrol', 'requeue', '900']]
    assert stored['controller']['self_requeues'] == 1
    assert stored['rows'] == tx['rows']


def test_cpu_transition_errors_are_bounded_even_after_successful_observation(ctl, monkeypatch):
    ctl.ART.mkdir(parents=True)
    tx = {'controller': {'job_id': 900}, 'rows': {}, 'events': [], 'status': 'waiting_storage'}
    ctl.write(ctl.PLAN, {'plan_sha256': 'p'}); ctl.write(ctl.TX, tx)
    monkeypatch.setenv('SLURM_JOB_ID', '900')
    monkeypatch.setattr(ctl, 'observe_pass', lambda *a: False)
    tick = iter([0] + [86401] * 12)
    monkeypatch.setattr(ctl.time, 'monotonic', lambda: next(tick))
    calls = []
    def fail(argv, **kwargs):
        calls.append(argv)
        raise RuntimeError('CPU scheduler unavailable')
    monkeypatch.setattr(ctl, 'command', fail)
    sleeps = []
    class UnboundedLoop(BaseException): pass
    def sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) > 8: raise UnboundedLoop('CPU transition errors are reset before being counted')
    monkeypatch.setattr(ctl.time, 'sleep', sleep)
    assert ctl.watch() == 2
    assert len(calls) <= 8
    assert ctl.read(ctl.TX)['status'] == 'needs_review'


def test_scientific_checkpoint_cadence_drift192_is_rejected(ctl, sample):
    _, _, row = sample
    row['environment']['OAT_ZERO_RESUME_STEPS'] = '192'
    write_recovery_checkpoint(row, 192)
    with pytest.raises(RuntimeError, match='cadence changed'):
        ctl.checkpoint(row)


def test_observer_storage_denial_keeps_ambiguous_owned_hold_under_observation(ctl, sample, monkeypatch):
    plan, tx, row = sample
    plan['deadline_utc'] = '2099-01-01T00:00:00+00:00'
    tx['rows'] = {'systems': {'job_id': 699, 'status': 'qualified'}, row['cell_id']: {'job_id': 700, 'status': 'release_intent'}}
    qualification = ctl.ROOT / 'profile.json'; qualification.write_text('{}')
    tx['qualification'] = {'path': str(qualification), 'sha256': ctl.digest(qualification)}
    monkeypatch.setattr(ctl, 'verify', lambda *a: None)
    monkeypatch.setattr(ctl, 'reject_untracked_jobs', lambda *a: None)
    monkeypatch.setattr(ctl, 'release_owned', lambda *a, **k: False)
    monkeypatch.setattr(ctl, 'scheduler_state', lambda _: {'state': 'PENDING', 'reason': 'JobHeldUser', 'inactive': False})
    monkeypatch.setattr(ctl, 'audit_job', lambda *a, **k: pytest.fail('a deliberate storage hold must not be audited as released'))
    monkeypatch.setattr(ctl, 'storage_report', lambda *a, **k: {'allowed': False, 'blocked_reason': 'waiting_disk'})
    assert ctl.observe_pass(plan, tx) is False
    assert ctl.read(ctl.TX)['rows'][row['cell_id']]['status'] == 'release_intent'


def test_fresh_science_release_refuses_preexisting_artifacts(ctl, sample, monkeypatch):
    plan, tx, row = sample
    item = {'job_id': 700, 'status': 'held', 'command': ctl.job_command(plan, row)}
    tx['rows'][row['cell_id']] = item
    Path(row['run_dir']).mkdir(parents=True)
    monkeypatch.setattr(ctl, 'audit_job', lambda *a, **k: {})
    monkeypatch.setattr(ctl, 'other_writers', lambda *a, **k: None)
    monkeypatch.setattr(ctl, 'storage_report', lambda *a, **k: {'allowed': True})
    with pytest.raises(RuntimeError, match='fresh scientific namespace'): ctl.release_owned(plan, tx, row, item)
    assert item['status'] == 'held'


def test_sbatch_scrubs_unregistered_inherited_science_and_scheduler_options(ctl, monkeypatch):
    for key in ('OAT_ZERO_UNREGISTERED_OVERRIDE', 'SBATCH_PARTITION', 'SLURM_JOB_ID', 'RUN_STAMP', 'SAVE_PATH', 'PYTHONPATH', 'OMP_NUM_THREADS'):
        monkeypatch.setenv(key, 'caller-value')
    received = []
    def run(argv, **kwargs):
        received.append(kwargs.get('env', dict(os.environ)))
        return SimpleNamespace(returncode=0, stdout='777', stderr='')
    monkeypatch.setattr(ctl.subprocess, 'run', run)
    # Restore this one pure wrapper, since the fixture forbids unmocked calls.
    raw_spec = importlib.util.spec_from_file_location('e124_launcher_command_test', SOURCE)
    raw = importlib.util.module_from_spec(raw_spec); raw_spec.loader.exec_module(raw)
    raw.command(['sbatch', '--parsable', '--export=ALL,OAT_ZERO_NUM_SAMPLES=16', '/entrypoint'])
    submitted_env = received[0]
    assert not any(k.startswith(('OAT_ZERO_', 'SBATCH_', 'SLURM_')) for k in submitted_env)
    assert not {'RUN_STAMP', 'SAVE_PATH', 'PYTHONPATH', 'OMP_NUM_THREADS'} & submitted_env.keys()


@pytest.mark.parametrize('reason', ['job_requeued_in_held_state', 'job requeued in held state'])
def test_owned_requeue_hold_audit_requires_exact_prior_restart_proof(ctl, sample, monkeypatch, reason):
    plan, _, row = sample
    item = {'job_id': 700, 'status': 'held_retry', 'retries': 1, 'last_resume_step': 192,
            'command': ctl.job_command(plan, row),
            'requeue_intent': {'restarts_before': 3, 'retry_number': 1, 'checkpoint': {'step': 192}}}
    record = scheduler_record(ctl, row, item, {'Reason': reason, 'Restarts': '4'})
    mock_held_record(ctl, monkeypatch, record)
    assert ctl.audit_job(plan, row, item, held=True)['Restarts'] == '4'
    with pytest.raises(RuntimeError, match='unexpectedly held'):
        ctl.audit_job(plan, row, item, held=False)
    item['requeue_intent']['restarts_before'] = 2
    with pytest.raises(RuntimeError, match='provenance'):
        ctl.audit_job(plan, row, item, held=True)


@pytest.mark.parametrize('reason', ['job_requeued_in_held_state', 'job requeued in held state'])
@pytest.mark.parametrize('bad', ['missing_intent', 'fresh_status', 'admin_hold'])
def test_requeue_hold_cannot_borrow_fresh_job_hold_authority(ctl, sample, monkeypatch, reason, bad):
    plan, _, row = sample
    item = {'job_id': 700, 'status': 'held_retry', 'command': ctl.job_command(plan, row), 'requeue_intent': {'restarts_before': 3}}
    if bad == 'missing_intent': item.pop('requeue_intent')
    elif bad == 'fresh_status': item['status'] = 'held'
    else: reason = 'JobHeldAdmin'
    mock_held_record(ctl, monkeypatch, scheduler_record(ctl, row, item, {'Reason': reason, 'Restarts': '4'}))
    with pytest.raises(RuntimeError): ctl.audit_job(plan, row, item, held=True)


@pytest.mark.parametrize('reason', ['job_requeued_in_held_state', 'job requeued in held state'])
def test_pending_requeue_release_intent_is_not_mistaken_for_released_job(ctl, sample, monkeypatch, reason):
    plan, tx, row = sample
    item = {'job_id': 700, 'status': 'release_intent', 'retries': 1, 'last_resume_step': 192,
            'command': ctl.job_command(plan, row),
            'requeue_intent': {'restarts_before': 3, 'retry_number': 1, 'checkpoint': {'step': 192}}}
    tx['rows'][row['cell_id']] = item
    released = []
    monkeypatch.setattr(ctl, 'scheduler_state', lambda _: {'state': 'PENDING', 'reason': reason, 'inactive': False})
    def show(_):
        return scheduler_record(ctl, row, item, {'Reason': 'Resources' if released else reason,
            'Priority': '123' if released else '0', 'Restarts': '4'})
    monkeypatch.setattr(ctl, 'show', show)
    def command(argv, **kwargs):
        if argv[:3] == ['scontrol', 'show', 'hostnames']: return '\n'.join(ctl.POOL.split(','))
        assert argv == ['scontrol', 'release', '700']
        released.append(argv)
        return ''
    monkeypatch.setattr(ctl, 'command', command)
    monkeypatch.setattr(ctl, 'other_writers', lambda *a: None)
    monkeypatch.setattr(ctl, 'storage_report', lambda *a, **k: {'allowed': True})
    assert ctl.release_owned(plan, tx, row, item) is True
    assert released == [['scontrol', 'release', '700']]
    assert item['status'] == 'released'


@pytest.mark.parametrize('reason', ['job_requeued_in_held_state', 'job requeued in held state'])
def test_requeue_hold_storage_denial_keeps_proven_retry_reserved(ctl, sample, monkeypatch, reason):
    plan, tx, row = sample
    item = {'job_id': 700, 'status': 'release_intent', 'command': ctl.job_command(plan, row), 'requeue_intent': {'restarts_before': 3}}
    tx['rows'][row['cell_id']] = item
    mock_held_record(ctl, monkeypatch, scheduler_record(ctl, row, item, {'Reason': reason, 'Restarts': '4'}))
    monkeypatch.setattr(ctl, 'scheduler_state', lambda _: {'state': 'PENDING', 'reason': reason, 'inactive': False})
    monkeypatch.setattr(ctl, 'other_writers', lambda *a: None)
    monkeypatch.setattr(ctl, 'storage_report', lambda *a, **k: {'allowed': False, 'blocked_reason': 'waiting_disk'})
    assert ctl.release_owned(plan, tx, row, item) is False
    assert item['status'] == 'release_intent'
    assert ctl.read(ctl.TX)['status'] == 'waiting_storage'


def test_cpu_controller_uses_2g_without_gpu_and_retains_all31_holds(ctl, monkeypatch):
    plan = {'plan_sha256': 'a' * 64}
    tx = {'rows': {f'cell{i}': {'job_id': 700 + i, 'status': 'held'} for i in range(31)}, 'events': []}
    monkeypatch.setattr(ctl, 'verify', lambda *a, **k: None)
    record = f'JobId=900 JobName=e124-controller UserId=test({os.getuid()}) JobState=PENDING Reason=JobHeldUser Priority=0 Account=mltheory Partition=lowprio ReqNodeList=node[916-917] ReqTRES=cpu=1,mem=2G,node=1 MinMemoryNode=2G TimeLimit=1-01:10:00'
    monkeypatch.setattr(ctl, 'show', lambda _: record)
    calls = []
    def command(argv, **kwargs):
        calls.append(argv)
        if argv[0] == 'sbatch':
            assert '--mem=2G' in argv and '--gres=none' in argv and '--hold' in argv
            return '900'
        if argv[:3] == ['scontrol', 'show', 'hostnames']: return 'node916\nnode917'
        assert argv == ['scontrol', 'release', '900']
        return ''
    monkeypatch.setattr(ctl, 'command', command)
    assert ctl.start_controller(plan, tx)['job_id'] == 900
    assert all(row['status'] == 'held' for row in tx['rows'].values())
    assert [argv for argv in calls if argv[:2] == ['scontrol', 'release']] == [['scontrol', 'release', '900']]
