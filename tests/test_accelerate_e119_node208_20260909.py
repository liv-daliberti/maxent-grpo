from __future__ import annotations
import copy
import importlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ops/exp_scaling'))
x = importlib.import_module('accelerate_e119_node208_20260909')


@pytest.fixture
def item():
    command = ['sbatch', '--parsable', '--hold', '--job-name=e119-pantry-m-s44',
               '--account=allcs', '--partition=cs', '--nodelist=node205,node206,node207',
               '--gres=gpu:1', '--mem=40G', '--cpus-per-task=8', '--time=1-12:00:00',
               '--nice=0', '--requeue', '--chdir=/repo', '--dependency=afterany:5',
               '--export=ALL,SAVE_PATH=/run,RUN_STAMP=stamp,OAT_ZERO_AUTO_RESUME=1,OAT_ZERO_VLLM_GPU_RATIO=0.25,OAT_ZERO_EVAL_PROMPT_INTERVAL=192',
               '/snapshot/train.slurm']
    return {'old_job_id': 31048181, 'new_job_id': None, 'memory_gib': 116,
            'time_limit': '1-12:00:00', 'before': 'Nice=0', 'original_command': command,
            'comment': 'e119-node208-20260909-old31048181', 'domain': 'pantry_plan',
            'arm': 'maxrl', 'seed': 44, 'run_dir': '/run', 'run_stamp': 'stamp',
            'checkpoint': {'path': None, 'step': 0, 'files': {}}, 'status': 'prepared'}


def test_clone_preserves_every_explicit_science_export_and_script(item):
    result = x.build_command(item)
    assert x.b.exports(result) == x.b.exports(item['original_command'])
    assert result[-1] == item['original_command'][-1]
    assert '--gres=gpu:a6000:1' in result
    assert '--mem=116G' in result and '--time=1-12:00:00' in result
    assert '--account=mltheory' in result and '--partition=lowprio' in result
    assert '--nodelist=node208' in result and '--requeue' in result
    assert '--dependency=afterany:5' not in result
    assert result.count('--hold') == 1


def test_max45_keeps_72_hour_128g_request(item):
    item.update(memory_gib=128, time_limit='3-00:00:00')
    result = x.build_command(item)
    assert '--time=3-00:00:00' in result and '--mem=128G' in result


def test_no_checkpoint_preserves_fresh_auto_resume(item, monkeypatch):
    value = item['checkpoint']
    monkeypatch.setattr(x.prior, 'checkpoint', lambda path: value)
    monkeypatch.setattr(x.checkpoints, 'checkpoint', lambda item: pytest.fail('no checkpoint exists'))
    assert x.checkpoint(item) == value


def test_newer_partial_checkpoint_blocks(item, monkeypatch):
    monkeypatch.setattr(x.prior, 'checkpoint', lambda path: {'path': '/run/step_00096', 'step': 96})
    monkeypatch.setattr(x.checkpoints, 'checkpoint', lambda item: {'step': 96, 'rejected': {'/run/step_00192': ['partial']}})
    with pytest.raises(RuntimeError, match='newer incomplete'):
        x.checkpoint(item)


def test_duplicate_writer_blocks(item, monkeypatch):
    monkeypatch.setattr(x.recovery, 'complete', lambda path: False)
    monkeypatch.setattr(x.recovery, 'active_writers', lambda: {'/run': {item['old_job_id'], 123}})
    with pytest.raises(RuntimeError, match='unexpected same-cell writer'):
        x.safe_run(item, [item['old_job_id']])


def test_completed_cell_cannot_be_released(item, monkeypatch):
    monkeypatch.setattr(x.recovery, 'complete', lambda path: True)
    with pytest.raises(RuntimeError, match='already complete'):
        x.safe_run(item, [])


def test_sbatch_ambient_experiment_options_removed(monkeypatch):
    for name in ['OAT_ZERO_EVAL_PROMPT_INTERVAL', 'SBATCH_ACCOUNT', 'SLURM_JOB_ID',
                 'SAVE_PATH', 'RUN_STAMP', 'PYTHONPATH', 'OMP_NUM_THREADS']:
        monkeypatch.setenv(name, 'unwanted')
    monkeypatch.setenv('PATH', '/usr/bin')
    result = x.clean_submit_environment()
    assert result['PATH'] == '/usr/bin'
    assert not any(k.startswith(('OAT_ZERO_', 'SBATCH_', 'SLURM_')) for k in result)
    assert all(k not in result for k in ['SAVE_PATH', 'RUN_STAMP', 'PYTHONPATH', 'OMP_NUM_THREADS'])


@pytest.fixture
def staged_mock(item, monkeypatch):
    item.update(command=x.build_command(item), old_held=True)
    tx = {'items': [item]}
    calls = []
    monkeypatch.setattr(x, 'load_transaction', lambda: tx)
    monkeypatch.setattr(x, 'old_guard', lambda *args: 'JobState=PENDING Reason=JobHeldUser Priority=0')
    monkeypatch.setattr(x, 'safe_run', lambda *args, **kwargs: None)
    monkeypatch.setattr(x, 'authoritative', lambda *args: None)
    monkeypatch.setattr(x, 'event', lambda tx, message: calls.append(message))
    monkeypatch.setattr(x, 'new_guard', lambda *args, **kwargs: 'JobState=PENDING Reason=JobHeldUser Priority=0')
    monkeypatch.setattr(x, 'promote', lambda tx, row: row.update(ledger_committed=True))
    return tx, item, calls


def test_submission_failure_persists_intent_and_prevents_retry(staged_mock, monkeypatch):
    tx, item, calls = staged_mock
    count = []
    def failed(command):
        assert item['submission_uncertain']
        count.append(command)
        raise subprocess.TimeoutExpired(command, 30)
    monkeypatch.setattr(x, 'submit', failed)
    with pytest.raises(subprocess.TimeoutExpired):
        x.stage(item['old_job_id'])
    with pytest.raises(RuntimeError, match='uncertain held submission'):
        x.stage(item['old_job_id'])
    assert len(count) == 1


def test_acknowledged_successor_is_never_submitted_twice(staged_mock, monkeypatch):
    tx, item, calls = staged_mock
    count = []
    def submitted(command):
        count.append(command)
        return subprocess.CompletedProcess(command, 0, '99999;cluster\n', '')
    monkeypatch.setattr(x, 'submit', submitted)
    x.stage(item['old_job_id']); x.stage(item['old_job_id'])
    assert item['new_job_id'] == 99999 and item['status'] == 'staged'
    assert len(count) == 1


def test_hold_race_preserves_live_writer(staged_mock, monkeypatch):
    tx, item, calls = staged_mock
    item.pop('old_held')
    commands = []
    monkeypatch.setattr(x.b, 'show', lambda job: 'JobState=RUNNING Reason=None Priority=5')
    monkeypatch.setattr(x.b, 'command', lambda args, **kwargs: (commands.append(args) or subprocess.CompletedProcess(args, 0, '', '')))
    monkeypatch.setattr(x, 'submit', lambda command: pytest.fail('must not clone allocated cell'))
    x.stage(item['old_job_id'])
    assert item['status'] == 'skipped_started'
    assert commands == [['scontrol', 'hold', str(item['old_job_id'])], ['scontrol', 'release', str(item['old_job_id'])]]


def test_release_lost_ack_does_not_release_again(staged_mock, monkeypatch):
    tx, item, calls = staged_mock
    item.update(new_job_id=99999, ledger_committed=True, release_requested=True)
    monkeypatch.setattr(x, 'new_guard', lambda *args, **kwargs: 'Priority=10 JobState=RUNNING')
    monkeypatch.setattr(x.b, 'command', lambda *args, **kwargs: pytest.fail('no repeated scheduler mutation'))
    x.release(item['old_job_id'])
    assert item['released'] and item['status'] == 'released'


def test_adopt_requires_exact_successor_audit(staged_mock, monkeypatch):
    tx, item, calls = staged_mock
    item['submission_uncertain'] = True
    def reject(*args, **kwargs):
        raise RuntimeError('replacement full SubmitLine changed')
    monkeypatch.setattr(x, 'new_guard', reject)
    with pytest.raises(RuntimeError, match='SubmitLine changed'):
        x.adopt(item['old_job_id'], 999)
    assert item['new_job_id'] is None and item['submission_uncertain']


def test_promotion_preserves_other_74_rows_and_reconciles_ack(item, tmp_path, monkeypatch):
    item.update(new_job_id=99999, new_held_record='held')
    old = {k: item[k] for k in x.IDENTITY}
    old.update(original_job_id=1, continuation_job_id=item['old_job_id'])
    item['row_before'] = copy.deepcopy(old)
    other = [{'original_job_id': k + 2, 'continuation_job_id': k + 100,
              'run_dir': f'/other/{k}', 'keep': {'value': k}} for k in range(74)]
    ledger = tmp_path / 'ledger.json'; ledger.write_text(json.dumps({'continuations': [old, *other]}))
    monkeypatch.setattr(x, 'LEDGER', ledger); monkeypatch.setattr(x, 'ART', tmp_path)
    monkeypatch.setattr(x, 'TX', tmp_path / 'tx.json')
    monkeypatch.setattr(x, 'authoritative', lambda *args: None)
    monkeypatch.setattr(x, 'event', lambda *args: None)
    tx = {'items': [item]}
    x.promote(tx, item)
    after = json.loads(ledger.read_text())
    assert after['continuations'][1:] == other and len(after['continuations']) == 75
    assert after['continuations'][0]['continuation_job_id'] == 99999
    assert after['continuations'][0]['dormant_fallback_job_id'] == item['old_job_id']
    item.pop('ledger_committed')
    before = ledger.read_bytes(); x.promote(tx, item)
    assert ledger.read_bytes() == before and item['ledger_committed']


def test_data_fingerprint_detects_changed_dataset(item, tmp_path, monkeypatch):
    directory = tmp_path / 'dataset'; directory.mkdir(); (directory / 'data.json').write_text('one')
    monkeypatch.setattr(x.b, 'exports', lambda command: {'OAT_ZERO_PROMPT_DATA': str(directory), 'OAT_ZERO_EVAL_DATA': str(directory)})
    before = x.data_fingerprints([item]); (directory / 'data.json').write_text('two')
    assert x.data_fingerprints([item]) != before


def test_prepare_scope_excludes_guard_owned_cells():
    assert 31048178 not in x.TARGETS and 31048182 not in x.TARGETS
    assert 31037827 not in x.TARGETS and len(x.TARGETS) == 10

@pytest.fixture
def audited_new_record(item, monkeypatch):
    import shlex
    item['new_job_id'] = 99999
    preserved = {'UserId': 'od2961(363432)', 'JobName': 'e119-pantry-m-s44',
                 'Nice': '0', 'Requeue': '1', 'NumCPUs': '8', 'NumTasks': '1',
                 'CPUs/Task': '8', 'ExcNodeList': x.b.PVL, 'WorkDir': '/repo',
                 'Command': '/snapshot/train.slurm', 'Features': '(null)'}
    item['before'] = ' '.join(f'{k}={v}' for k, v in preserved.items())
    item['command'] = x.build_command(item)
    fields = {**preserved, 'JobId': '99999', 'Account': 'mltheory', 'Partition': 'lowprio',
              'QOS': 'none', 'ReqNodeList': 'node208', 'TimeLimit': '1-12:00:00',
              'MinMemoryNode': '116G', 'TresPerNode': 'gres/gpu:a6000:1', 'Dependency': '(null)',
              'Comment': item['comment'], 'NumNodes': '1', 'JobState': 'PENDING',
              'Reason': 'JobHeldUser', 'Priority': '0',
              'ReqTRES': 'cpu=8,mem=116G,node=1,gres/gpu=1,gres/gpu:a6000=1',
              'StdOut': str(x.ROOT / 'var/artifacts/logs/e119-pantry-m-s44-99999.out'),
              'StdErr': str(x.ROOT / 'var/artifacts/logs/e119-pantry-m-s44-99999.err')}
    def render():
        return ' '.join(f'{k}={v}' for k, v in fields.items()) + ' SubmitLine=' + shlex.join(item['command']) + ' WorkDir=/repo'
    monkeypatch.setattr(x.b, 'show', lambda job: render())
    return item, fields


def test_exact_held_successor_profile_passes(audited_new_record):
    item, fields = audited_new_record
    x.new_guard(item, held=True)


@pytest.mark.parametrize(('key', 'wrong'), [
    ('UserId', 'someoneelse(1)'), ('MinMemoryNode', '96G'), ('JobId', '99998'),
    ('ReqNodeList', 'node205'), ('TimeLimit', '01:00:00'), ('Comment', 'unowned'),
    ('TresPerNode', 'gres/gpu:a5000:1'), ('Reason', 'JobHeldAdmin'),
    ('StdOut', '/tmp/wrong.out'), ('ReqTRES', 'cpu=8,mem=116G,node=1,gres/gpu=2,gres/gpu:a6000=2'),
])
def test_successor_resource_or_ownership_drift_fails(audited_new_record, key, wrong):
    item, fields = audited_new_record
    fields[key] = wrong
    with pytest.raises(RuntimeError):
        x.new_guard(item, held=True)
