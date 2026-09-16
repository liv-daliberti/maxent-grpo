from __future__ import annotations
import copy
import importlib
import json
from pathlib import Path
import subprocess
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ops/exp_scaling'))
x = importlib.import_module('amend_e119_healthy_route_20260909')


@pytest.fixture
def item():
    original = {'original_job_id': 1, 'continuation_job_id': 31163339, 'domain': 'pantry_plan',
                'arm': 'maxrl', 'seed': 45, 'run_dir': '/run', 'run_stamp': 'stamp'}
    fields = 'Account=mltheory Partition=lowprio ReqNodeList=node208 MinMemoryNode=128G TimeLimit=3-00:00:00 TresPerNode=gres/gpu:a6000:1 Nice=200'
    command = ['sbatch', '--parsable', '--hold', '--account=mltheory', '--partition=lowprio',
               '--nodelist=node208', '--mem=128G', '--time=3-00:00:00', '--gres=gpu:a6000:1',
               '--nice=200', '--export=ALL,SAVE_PATH=/run,RUN_STAMP=stamp,OAT_ZERO_RESUME_STEPS=192', '/snapshot/train.slurm']
    return {**original, 'old_job_id': 31124279, 'new_job_id': 31163339,
            'row_before_amendment': copy.deepcopy(original), 'before': fields.replace('Partition=lowprio', 'Partition=mltheory').replace('node208', 'node302'),
            'held_before_amendment': fields, 'command': command, 'original_command': command}


def test_forecast_uses_exact_resources_and_preserves_science(item):
    command = x.route_command(item)
    assert '--nodelist=node205,node207' in command
    assert '--mem=128G' in command and '--time=3-00:00:00' in command
    assert '--test-only' in command and '--hold' not in command
    assert x.b.exports(command) == x.b.exports(item['original_command'])
    original = x.route_command(item, original=True)
    assert '--nodelist=node302' in original and '--partition=mltheory' in original


@pytest.mark.parametrize('healthy', ['2026-09-12T10:00:00', '2026-09-13T10:00:00'])
def test_nonimproving_forecast_blocks(item, monkeypatch, healthy):
    answers = iter([healthy, '2026-09-12T10:00:00'])
    monkeypatch.setattr(x.launch, 'submit', lambda command: subprocess.CompletedProcess(command, 0, '', 'sbatch: Job to start at ' + next(answers)))
    with pytest.raises(RuntimeError, match='not forecast earlier'):
        x.forecast(item)


def test_earlier_forecast_passes(item, monkeypatch):
    answers = iter(['2026-09-10T22:30:00', '2026-09-12T10:00:00'])
    monkeypatch.setattr(x.launch, 'submit', lambda command: subprocess.CompletedProcess(command, 0, '', 'sbatch: Job to start at ' + next(answers)))
    assert x.forecast(item)['healthy']['start'] == '2026-09-10T22:30:00'


def test_any_drained_node_blocks_release(monkeypatch):
    monkeypatch.setattr(x.b, 'command', lambda command: subprocess.CompletedProcess(command, 0, 'State=MIXED+DRAIN Gres=gpu:a6000:10 Partitions=lowprio', ''))
    with pytest.raises(RuntimeError, match='no longer healthy'):
        x.healthy_nodes()


def test_ledger_amendment_preserves_ids_others_and_history(item, tmp_path, monkeypatch):
    others = [{'continuation_job_id': k + 100, 'run_dir': f'/other/{k}', 'keep': k} for k in range(74)]
    ledger = tmp_path / 'ledger.json'; ledger.write_text(json.dumps({'continuations': [copy.deepcopy(item['row_before_amendment']), *others], 'repair_history': [{'audit': 'old'}]}))
    monkeypatch.setattr(x.launch, 'LEDGER', ledger); monkeypatch.setattr(x, 'ART', tmp_path)
    monkeypatch.setattr(x, 'TX', tmp_path / 'tx.json')
    monkeypatch.setattr(x.launch, 'authoritative', lambda *args: None)
    monkeypatch.setattr(x, 'event', lambda *args: None)
    item['held_after_amendment'] = 'ReqNodeList=node205,node207'
    x.update_ledger({'items': [item]}, item)
    after = json.loads(ledger.read_text())
    assert after['continuations'][1:] == others
    assert after['continuations'][0]['continuation_job_id'] == 31163339
    assert after['continuations'][0]['actual_requested_nodes'] == ['node205', 'node207']
    assert after['repair_history'][0] == {'audit': 'old'}
    raw = ledger.read_bytes(); item.pop('ledger_updated'); x.update_ledger({'items': [item]}, item)
    assert ledger.read_bytes() == raw and item['ledger_updated']


def test_lost_release_ack_does_not_repeat_mutation(item, monkeypatch):
    item['release_requested'] = True
    monkeypatch.setattr(x, 'load', lambda: {'items': [item]})
    monkeypatch.setattr(x.launch, 'old_guard', lambda *args: None)
    monkeypatch.setattr(x.launch, 'authoritative', lambda *args: None)
    monkeypatch.setattr(x.b, 'show', lambda job: 'Priority=100')
    monkeypatch.setattr(x, 'profile', lambda *args, **kwargs: None)
    monkeypatch.setattr(x, 'event', lambda *args: None)
    monkeypatch.setattr(x.b, 'command', lambda *args, **kwargs: pytest.fail('no repeated mutation'))
    x.apply(item['old_job_id'])
    assert item['released']
