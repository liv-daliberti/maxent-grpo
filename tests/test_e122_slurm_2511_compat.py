"""Additive compatibility accepts one exact Slurm spelling and never retries index0."""
from __future__ import annotations
import importlib.util
import json
import os
from pathlib import Path
import pwd
import sys
from types import SimpleNamespace
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ops/exp_scaling'))
import launch_e122_level3_factorial as frozen

@pytest.fixture(scope='module')
def compat():
    spec = importlib.util.spec_from_file_location('e122_2511_test', ROOT / 'ops/exp_scaling/e122_slurm_2511_compat.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

@pytest.fixture
def plan(compat):
    return compat.read(compat.PLAN)

def raw_record(cell, value='1-1'):
    resource = cell['resources']
    fields = {'JobId': '31158645', 'JobName': cell['run_stamp'], 'JobState': 'PENDING',
        'Reason': 'JobHeldUser', 'Account': resource['account'], 'Partition': resource['partition'],
        'QOS': resource['expected_qos'], 'NumCPUs': str(resource['cpus']), 'NumNodes': value, 'NumTasks': '1',
        'Requeue': '1', 'Nice': '0', 'UserId': f'{pwd.getpwuid(os.getuid()).pw_name}({os.getuid()})',
        'WorkDir': str(ROOT), 'Command': cell['command'][-1], 'RunTime': '00:00:00', 'Restarts': '0',
        'MinMemoryNode': str(resource['memory_gib']) + 'G', 'TimeLimit': resource['walltime'],
        'ExcNodeList': resource['exclude'] or '(null)', 'ReqNodeList': 'node[205-207,302]',
        'ReqTRES': f'cpu={resource["cpus"]},node=1,mem={resource["memory_gib"]}G,gres/gpu=1'}
    return ' '.join(key + '=' + value for key, value in fields.items()) + ' ' + ','.join(
        key + '=' + value for key, value in cell['environment'].items())

@pytest.mark.parametrize('value', ['1', '1-1'])
def test_equivalent_single_node_preserves_raw_and_rechecks_pins(compat, plan, value):
    calls = []; audit = compat.make_auditor(frozen.audit_held_record, lambda: calls.append('verify'))
    raw = raw_record(plan['cells'][0], value)
    assert audit(raw, 31158645, plan['cells'][0]) == raw
    assert audit(raw, 31158645, plan['cells'][0]) == raw
    assert calls == ['verify', 'verify']
    assert compat.normalize_single_node(raw) == raw.replace('NumNodes=' + value, 'NumNodes=1', 1)

@pytest.mark.parametrize('value', ['0', '2', '1-2', '0-1', '01', '1-01', '1--1', '1,1', '1-1-1'])
def test_malformed_ranges_and_multiple_nodes_rejected(compat, plan, value):
    with pytest.raises(ValueError, match='only exact single-node'):
        compat.make_auditor(frozen.audit_held_record, lambda: None)(raw_record(plan['cells'][0], value), 31158645, plan['cells'][0])

def test_duplicate_numnodes_field_rejected(compat, plan):
    with pytest.raises(ValueError, match='only exact single-node'):
        compat.normalize_single_node(raw_record(plan['cells'][0]) + ' NumNodes=1')

@pytest.mark.parametrize('before,after', [('NumTasks=1', 'NumTasks=2'), ('QOS=medium', 'QOS=none'),
    ('JobHeldUser', 'Priority'), ('Requeue=1', 'Requeue=0'), ('OAT_ZERO_SEED=43', 'OAT_ZERO_SEED=44'),
    ('OAT_ZERO_LEARNING_RATE=2e-07', 'OAT_ZERO_LEARNING_RATE=1e-07')])
def test_all_other_frozen_audit_checks_remain_strict(compat, plan, before, after):
    with pytest.raises(ValueError):
        compat.make_auditor(frozen.audit_held_record, lambda: None)(
            raw_record(plan['cells'][0]).replace(before, after), 31158645, plan['cells'][0])

def test_actual_frozen_inputs_match_and_compatibility_files_are_hash_bound(compat, tmp_path, monkeypatch):
    compat.verify_bindings(compat.digest(compat.SOURCE), compat.digest(compat.TEST))
    source = tmp_path / 'source.py'; source.write_text('source')
    test = tmp_path / 'test.py'; test.write_text('test')
    monkeypatch.setattr(compat, 'SOURCE', source); monkeypatch.setattr(compat, 'TEST', test)
    source_sha, test_sha = compat.digest(source), compat.digest(test)
    compat.verify_bindings(source_sha, test_sha)
    source.write_text('changed')
    with pytest.raises(ValueError, match='input changed'):
        compat.verify_bindings(source_sha, test_sha)

@pytest.fixture
def initial_state(compat, plan, tmp_path, monkeypatch):
    here = tmp_path / 'e122'; here.mkdir()
    for index, cell in enumerate(plan['cells']):
        cell['run_dir'] = str(tmp_path / 'runs' / str(index))
    cell_fields = ('domain', 'dataset_domain', 'arm', 'seed', 'run_stamp', 'run_dir', 'target_steps')
    prospective = {'schema': frozen.LEDGER_SCHEMA, 'released': False, 'runs': [],
        'planned_runs': [{key: cell[key] for key in cell_fields} for cell in plan['cells']],
        'domains': list(dict.fromkeys(cell['domain'] for cell in plan['cells'])),
        'arms': list(dict.fromkeys(cell['arm'] for cell in plan['cells'])),
        'seeds': sorted({cell['seed'] for cell in plan['cells']}),
        'train_rows': 384, 'target_steps': 3072, 'passes': 8}
    ledger_path = tmp_path / 'ledger.json'; ledger_path.write_text(json.dumps(prospective))
    initial_sha = compat.digest(ledger_path)
    claim = {'schema': 'e122_once_only_submission_claim_v1', 'jobs': 100,
             'plan_sha256': compat.PLAN_SHA256, 'initial_ledger_sha256': initial_sha}
    intent = {'cell': plan['cells'][0], 'command_sha256': frozen.sha(plan['cells'][0]['command'])}
    result = {'returncode': 0, 'stdout': '31158645\n', 'stderr': ''}
    values = {'submission_claim.json': claim, 'submission_000_intent.json': intent, 'submission_000_result.json': result}
    for name, value in values.items():
        (here / name).write_text(json.dumps(value))
    monkeypatch.setattr(compat, 'HERE', here)
    monkeypatch.setattr(compat, 'RESUME_CLAIM', here / 'compat/resume_claim.json')
    monkeypatch.setattr(compat, 'INITIAL_LEDGER_SHA256', initial_sha)
    monkeypatch.setattr(compat, 'FIRST_PINS', {name: compat.digest(here / name) for name in values})
    launcher = SimpleNamespace(LEDGER=ledger_path, sha=frozen.sha, verify_initial_ledger=frozen.verify_initial_ledger)
    jobs = [{'job_id': '31158645', 'job_name': plan['cells'][0]['run_stamp'], 'state': 'PENDING'}]
    return SimpleNamespace(here=here, launcher=launcher, jobs=jobs, initial_sha=initial_sha)

def test_initial_known_first_evidence_is_adoptable(compat, plan, initial_state):
    assert compat.validate_initial_evidence(initial_state.launcher, plan, initial_state.jobs)['runs'] == []

@pytest.mark.parametrize('change', ['first_result', 'extra_intent', 'extra_result', 'held_audit', 'duplicate_claim', 'extra_job'])
def test_resume_rejects_changed_evidence_extra_indices_and_existing_claim(compat, plan, initial_state, change):
    state = initial_state
    if change == 'first_result':
        (state.here / 'submission_000_result.json').write_text('{"stdout":"31158646"}')
    elif change in ('extra_intent', 'extra_result'):
        (state.here / ('submission_001_' + change.removeprefix('extra_') + '.json')).write_text('{}')
    elif change == 'held_audit':
        (state.here / 'held_audit_000.json').write_text('{}')
    elif change == 'duplicate_claim':
        compat.RESUME_CLAIM.parent.mkdir(); compat.RESUME_CLAIM.write_text('{}')
    else:
        state.jobs.append({'job_id': '31158646', 'job_name': plan['cells'][1]['run_stamp'], 'state': 'PENDING'})
    with pytest.raises(ValueError):
        compat.validate_initial_evidence(state.launcher, plan, state.jobs)

def test_wrong_or_unpinned_controller_compatibility_binding_rejected(compat, tmp_path):
    source_sha, test_sha = compat.digest(compat.SOURCE), compat.digest(compat.TEST)
    path = tmp_path / 'ledger.json'
    value = {'slurm_readback_compatibility': compat.compatibility_record(source_sha, test_sha)}
    path.write_text(json.dumps(value)); expected = compat.digest(path)
    compat.verify_controller_ledger_binding(path, expected, source_sha, test_sha)
    with pytest.raises(ValueError, match='compatibility implementation binding'):
        compat.verify_controller_ledger_binding(path, expected, '0' * 64, test_sha)
    path.write_text('{}')
    with pytest.raises(ValueError, match='ledger SHA256'):
        compat.verify_controller_ledger_binding(path, expected, source_sha, test_sha)


def test_resume_adopts_zero_and_submits_only_remaining_indices_once(compat, plan, initial_state, monkeypatch):
    state = initial_state; submitted = []; audits = []
    old_evidence = {name: (state.here / name).read_bytes() for name in compat.FIRST_PINS}
    monkeypatch.setattr(compat, 'install_compat', lambda *args: None)
    monkeypatch.setattr(compat, 'verify_bindings', lambda *args: None)
    monkeypatch.setattr(compat, 'queue_e122_jobs', lambda: list(state.jobs))
    def submit(cell, index):
        assert index != 0; submitted.append(index)
        frozen.atomic_new(state.here / f'submission_{index:03d}_intent.json', {'index': index})
        frozen.atomic_new(state.here / f'submission_{index:03d}_result.json', {'job_id': 31159000 + index})
        return 31159000 + index
    def audit(job_id, cell):
        audits.append(job_id); return 'ORIGINAL NumNodes=1-1 JobId=' + str(job_id)
    launcher = state.launcher
    launcher.verify_plan = lambda **kwargs: plan
    launcher.audit_held = audit; launcher.submit_one_held = submit
    launcher.atomic_new = frozen.atomic_new; launcher.now = frozen.now; launcher.e119 = frozen.e119
    result = compat.resume_known_first(launcher, 'a' * 64, 'b' * 64)
    assert submitted == list(range(1, 100))
    assert result['adopted_first_job'] == 31158645 and result['submitted_new_jobs'] == 99 and result['released'] == 0
    ledger = compat.read(launcher.LEDGER)
    assert len(ledger['runs']) == 100 and ledger['runs'][0]['job_id'] == 31158645
    assert all('NumNodes=1-1' in row['held_scheduler_record'] for row in ledger['runs'])
    assert len(audits) == 200
    assert all((state.here / name).read_bytes() == content for name, content in old_evidence.items())
    with pytest.raises(ValueError, match='resume claim already exists'):
        compat.resume_known_first(launcher, 'a' * 64, 'b' * 64)
    assert submitted == list(range(1, 100))
