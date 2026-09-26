from __future__ import annotations
import copy
import importlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ops/exp_scaling'))
x = importlib.import_module('amend_e119_generic_pool_20260909')


@pytest.fixture
def item(tmp_path):
    run = str(tmp_path / 'run')
    command = ['sbatch', '--parsable', '--hold', '--account=mltheory', '--partition=lowprio',
               '--nodelist=node205,node207', '--mem=128G', '--time=3-00:00:00', '--gres=gpu:a6000:1',
               '--nice=200', '--requeue', f'--export=ALL,SAVE_PATH={run},RUN_STAMP=stamp,OAT_ZERO_RESUME_STEPS=192', '/snapshot/train.slurm']
    fields = {key: '(null)' for key in x.FIELDS}
    fields.update(JobId='31163339', UserId='user(123)', JobName='science', Account='mltheory', Partition='lowprio',
                  MinMemoryNode='128G', TimeLimit='3-00:00:00', Nice='200', NumCPUs='8', NumTasks='1', Requeue='1',
                  Restarts='0', ReqNodeList='node205,node207', TresPerNode='gres/gpu:a6000:1',
                  ReqTRES='cpu=8,mem=128G,node=1,billing=3,gres/gpu=1,gres/gpu:a6000=1', NumNodes='1-1',
                  JobState='PENDING', Priority='100', Reason='Resources', NodeList='', RunTime='00:00:00', StartTime='2026-09-11T12:00:00',
                  WorkDir='/workspace', Command='/snapshot/train.slurm')
    row = {'original_job_id': 1, 'continuation_job_id': 31163339, 'domain': 'pantry_plan', 'arm': 'maxrl',
           'seed': 45, 'run_dir': run, 'run_stamp': 'stamp', 'previous_continuation_job_ids': [99, 31124279],
           'repair_audit': 'old/audit', 'actual_requested_nodes': ['node205', 'node207']}
    value = {**row, 'old_job_id': 31124279, 'new_job_id': 31163339, 'command': command,
             'original_command': command, 'row_before_pool': copy.deepcopy(row), '_fields': fields}
    value['pool_before_record'] = record(value)
    return value


def record(item, *, generic=False, held=False, **updates):
    fields = copy.deepcopy(item['_fields'])
    if generic:
        fields.update(ReqNodeList='node205,node207,node302', TresPerNode='gres/gpu:1',
                      ReqTRES='cpu=8,mem=128G,node=1,billing=3,gres/gpu=1')
    if held:
        fields.update(Priority='0', Reason='JobHeldUser', StartTime='Unknown')
    fields.update(updates)
    workdir = fields.pop('WorkDir')
    return ' '.join(f'{k}={v}' for k, v in fields.items()) + ' SubmitLine=' + shlex.join(item['command']) + ' WorkDir=' + workdir


@pytest.fixture(autouse=True)
def simple_hosts(monkeypatch):
    monkeypatch.setattr(x, 'hosts', lambda value: set(value.split(',')))


def test_generic_profile_preserves_128g_72h_and_science(item):
    item['pool_hold_requested'] = True
    assert x.profile(item, record(item, generic=True, held=True), generic=True, held=True)
    for generic in (False, True):
        cmd = x.forecast_command(item, generic)
        assert '--mem=128G' in cmd and '--time=3-00:00:00' in cmd and '--nice=200' in cmd
        assert '--hold' not in cmd and '--test-only' in cmd
        assert x.b.exports(cmd) == x.b.exports(item['original_command'])
    assert '--gres=gpu:1' in x.forecast_command(item, True)
    assert '--nodelist=node205,node207,node302' in x.forecast_command(item, True)


@pytest.mark.parametrize('key,value', [('MinMemoryNode', '116G'), ('TimeLimit', '1-12:00:00'), ('Nice', '0'),
    ('NumCPUs', '4'), ('Dependency', 'afterok:123'), ('Restarts', '1'), ('JobId', '77'),
    ('ReqNodeList', 'node302'), ('TresPerNode', 'gres/gpu:a100:1'),
    ('ReqTRES', 'cpu=8,mem=128G,node=1,billing=3,gres/gpu=2')])
def test_profile_drift_rejected(item, key, value):
    with pytest.raises(RuntimeError):
        x.profile(item, record(item, generic=True, **{key: value}), generic=True)


def test_submission_drift_rejected(item):
    changed = record(item, generic=True).replace('OAT_ZERO_RESUME_STEPS=192', 'OAT_ZERO_RESUME_STEPS=96')
    with pytest.raises(RuntimeError, match='SubmitLine'):
        x.profile(item, changed, generic=True)


@pytest.mark.parametrize('node,kind,state', [('node302', 'a6000', 'MIXED'), ('node205', 'a100', 'MIXED'),
                                          ('node207', 'a6000', 'MIXED+DRAIN')])
def test_node_hardware_and_health_fail_closed(monkeypatch, node, kind, state):
    def command(args):
        current = args[-1]
        gpu = kind if current == node else x.POOL_TYPES[current]
        st = state if current == node else 'MIXED'
        return subprocess.CompletedProcess(args, 0, f'NodeName={current} State={st} Gres=gpu:{gpu}:4 Partitions=all,lowprio', '')
    monkeypatch.setattr(x.b, 'command', command)
    with pytest.raises(RuntimeError):
        x.healthy_nodes()


def test_non_improving_forecast_still_preserves_broader_pool(item, monkeypatch):
    monkeypatch.setattr(x.launch, 'submit', lambda cmd: subprocess.CompletedProcess(cmd, 0, '', 'Job to start at 2026-09-12T10:00:00'))
    assert x.forecast(item, False)['start'] == x.forecast(item, True)['start']


def no_science_mutations(monkeypatch):
    monkeypatch.setattr(x.launch, 'old_guard', lambda *args: None)
    monkeypatch.setattr(x.launch, 'safe_run', lambda *args: None)
    monkeypatch.setattr(x.launch, 'authoritative', lambda *args: None)
    monkeypatch.setattr(x, 'event', lambda *args: None)


def test_lost_hold_ack_reconciles_without_repeat(item, monkeypatch):
    no_science_mutations(monkeypatch)
    item['pool_hold_requested'] = True
    monkeypatch.setattr(x.b, 'show', lambda job: record(item, held=True))
    monkeypatch.setattr(x.b, 'command', lambda *a, **kw: pytest.fail('must not repeat hold'))
    assert x.hold({}, item) and item['pool_held']


def test_uncertain_hold_not_applied_fails_without_repeat(item, monkeypatch):
    no_science_mutations(monkeypatch)
    item['pool_hold_requested'] = True
    monkeypatch.setattr(x.b, 'show', lambda job: record(item))
    monkeypatch.setattr(x.b, 'command', lambda *a, **kw: pytest.fail('must not repeat uncertain hold'))
    with pytest.raises(RuntimeError, match='owned never-allocated hold'):
        x.hold({}, item)


def test_preexisting_hold_is_not_adopted(item, monkeypatch):
    no_science_mutations(monkeypatch)
    monkeypatch.setattr(x.b, 'show', lambda job: record(item, held=True))
    monkeypatch.setattr(x.b, 'command', lambda *a, **kw: pytest.fail('must not touch preexisting hold'))
    with pytest.raises(RuntimeError, match='preexisting hold'):
        x.hold({}, item)


def test_allocation_wins_race_before_hold(item, monkeypatch):
    no_science_mutations(monkeypatch)
    answers = iter([record(item), record(item, JobState='RUNNING', NodeList='node205')])
    monkeypatch.setattr(x.b, 'show', lambda job: next(answers))
    monkeypatch.setattr(x.b, 'command', lambda *a, **kw: pytest.fail('running allocation must not be changed'))
    assert not x.hold({}, item)
    assert item['pool_allocation_preserved'] and not item.get('pool_hold_requested')


def test_allocation_wins_after_hold_call_never_amended_or_cancelled(item, monkeypatch):
    no_science_mutations(monkeypatch)
    answers = iter([record(item), record(item), record(item, JobState='RUNNING', NodeList='node205')])
    calls = []
    monkeypatch.setattr(x.b, 'show', lambda job: next(answers))
    monkeypatch.setattr(x.b, 'command', lambda args, **kw: (calls.append(args) or subprocess.CompletedProcess(args, 1, '', 'already running')))
    assert not x.hold({}, item)
    assert calls == [['scontrol', 'hold', str(item['new_job_id'])]]
    assert item['pool_allocation_preserved']


def ledger_fixture(item, tmp_path, monkeypatch, others=74):
    rows = [{'continuation_job_id': 1000 + i, 'run_dir': f'/other/{i}', 'preserved': i} for i in range(others)]
    ledger = tmp_path / 'ledger.json'
    ledger.write_text(json.dumps({'continuations': [copy.deepcopy(item['row_before_pool']), *rows], 'repair_history': [{'audit': 'earlier'}]}))
    monkeypatch.setattr(x.launch, 'LEDGER', ledger)
    monkeypatch.setattr(x, 'ART', tmp_path)
    monkeypatch.setattr(x, 'TX', tmp_path / 'tx.json')
    item['pool_held_after'] = record(item, generic=True, held=True)
    return ledger, rows


def test_75_row_ledger_preserves_science_history_and_lost_ack(item, tmp_path, monkeypatch):
    no_science_mutations(monkeypatch)
    ledger, others = ledger_fixture(item, tmp_path, monkeypatch)
    x.update_ledger({}, item)
    after = json.loads(ledger.read_text())
    assert after['continuations'][1:] == others
    row = after['continuations'][0]
    assert row['continuation_job_id'] == 31163339
    assert row['previous_continuation_job_ids'] == [99, 31124279]
    assert row['repair_audit'] == 'old/audit'
    assert row['actual_requested_nodes'] == ['node205', 'node207', 'node302']
    assert row['actual_requested_gres'] == 'gpu:1'
    assert row['actual_scheduler_profile']['MinMemoryNode'] == '128G'
    assert after['repair_history'][0] == {'audit': 'earlier'}
    raw = ledger.read_bytes()
    item.pop('pool_ledger_updated')
    x.update_ledger({}, item)
    assert ledger.read_bytes() == raw and item['pool_ledger_updated']


def test_wrong_ledger_denominator_not_written(item, tmp_path, monkeypatch):
    no_science_mutations(monkeypatch)
    ledger, _ = ledger_fixture(item, tmp_path, monkeypatch, others=73)
    before = ledger.read_bytes()
    with pytest.raises(RuntimeError, match='denominator'):
        x.update_ledger({}, item)
    assert ledger.read_bytes() == before


def test_lost_release_ack_no_repeated_mutation(item, tmp_path, monkeypatch):
    no_science_mutations(monkeypatch)
    ledger, _ = ledger_fixture(item, tmp_path, monkeypatch)
    x.update_ledger({}, item)
    item['pool_release_requested'] = True
    monkeypatch.setattr(x, 'load', lambda: {'items': [item]})
    monkeypatch.setattr(x.b, 'show', lambda job: record(item, generic=True))
    monkeypatch.setattr(x.b, 'command', lambda *a, **kw: pytest.fail('must not repeat release'))
    x.apply(item['old_job_id'])
    assert item['pool_released']


def test_uncertain_release_still_held_never_repeated(item, monkeypatch):
    no_science_mutations(monkeypatch)
    item.update(pool_hold_requested=True, pool_release_requested=True)
    monkeypatch.setattr(x, 'load', lambda: {'items': [item]})
    monkeypatch.setattr(x.b, 'show', lambda job: record(item, generic=True, held=True))
    monkeypatch.setattr(x.b, 'command', lambda *a, **kw: pytest.fail('must not repeat uncertain release'))
    with pytest.raises(RuntimeError, match='manual reconciliation'):
        x.apply(item['old_job_id'])


def test_lost_update_ack_reconciles_then_releases_once(item, monkeypatch):
    no_science_mutations(monkeypatch)
    item.update(pool_hold_requested=True, pool_update_requested=True)
    monkeypatch.setattr(x, 'load', lambda: {'items': [item]})
    monkeypatch.setattr(x, 'healthy_nodes', lambda: {})
    monkeypatch.setattr(x, 'forecast', lambda *a: {'start': 'later'})
    monkeypatch.setattr(x, 'update_ledger', lambda *a: None)
    state = {'held': True}; calls = []
    monkeypatch.setattr(x.b, 'show', lambda job: record(item, generic=True, held=state['held']))
    def command(args, **kwargs):
        calls.append(args)
        assert args == ['scontrol', 'release', str(item['new_job_id'])]
        state['held'] = False
        return subprocess.CompletedProcess(args, 0, '', '')
    monkeypatch.setattr(x.b, 'command', command)
    x.apply(item['old_job_id'])
    assert len(calls) == 1 and item['pool_released']


def test_uncertain_update_not_applied_never_repeated(item, monkeypatch):
    no_science_mutations(monkeypatch)
    item.update(pool_hold_requested=True, pool_update_requested=True)
    monkeypatch.setattr(x, 'load', lambda: {'items': [item]})
    monkeypatch.setattr(x.b, 'show', lambda job: record(item, held=True))
    monkeypatch.setattr(x.b, 'command', lambda *a, **kw: pytest.fail('must not repeat uncertain update'))
    with pytest.raises(RuntimeError, match='node pool changed'):
        x.apply(item['old_job_id'])


def test_allocation_win_with_confirmed_owned_hold_clears_only_that_hold(item, monkeypatch):
    no_science_mutations(monkeypatch)
    answers = iter([record(item), record(item),
                    record(item, JobState='RUNNING', NodeList='node205', Priority='0', Reason='None'),
                    record(item, JobState='RUNNING', NodeList='node205', Priority='0', Reason='None'),
                    record(item, JobState='RUNNING', NodeList='node205', Priority='100', Reason='None')])
    calls = []
    monkeypatch.setattr(x.b, 'show', lambda job: next(answers))
    monkeypatch.setattr(x.b, 'command', lambda args, **kw: (calls.append(args) or subprocess.CompletedProcess(args, 0, '', '')))
    assert not x.hold({}, item)
    assert calls == [['scontrol', 'hold', str(item['new_job_id'])], ['scontrol', 'release', str(item['new_job_id'])]]
    assert item['pool_allocation_preserved'] and item['pool_active_hold_release_requested']
    assert not item.get('pool_update_requested')


def test_active_zero_priority_without_confirmed_own_hold_is_not_released(item, monkeypatch):
    no_science_mutations(monkeypatch)
    monkeypatch.setattr(x.b, 'show', lambda job: record(item, JobState='RUNNING', NodeList='node205', Priority='0'))
    monkeypatch.setattr(x.b, 'command', lambda *a, **kw: pytest.fail('must not release an unowned active hold'))
    with pytest.raises(RuntimeError, match='not our confirmed mutation'):
        x.hold({}, item)


def test_lost_active_hold_release_ack_reconciles_without_repeat(item, monkeypatch):
    no_science_mutations(monkeypatch)
    item.update(pool_hold_requested=True, pool_active_hold_release_requested=True,
                pool_hold_result={'returncode': 0}, pool_hold_before=record(item))
    monkeypatch.setattr(x.b, 'show', lambda job: record(item, JobState='RUNNING', NodeList='node205', Priority='100'))
    monkeypatch.setattr(x.b, 'command', lambda *a, **kw: pytest.fail('must not repeat active hold cleanup'))
    assert not x.hold({}, item)
    assert item['pool_allocation_preserved']
