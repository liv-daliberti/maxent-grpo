from __future__ import annotations
import copy
import importlib
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ops/exp_scaling'))
x = importlib.import_module('rollback_e119_node208_20260909')


@pytest.fixture
def item(tmp_path):
    original = {'original_job_id': 1, 'continuation_job_id': 31124279, 'domain': 'pantry_plan',
                'arm': 'maxrl', 'seed': 45, 'run_dir': str(tmp_path / 'run'), 'run_stamp': 'stamp'}
    promoted = {**original, 'continuation_job_id': 31163339, 'repair_audit': '/route/tx.json'}
    return {**original, 'old_job_id': 31124279, 'new_job_id': 31163339,
            'row_before': original, 'promoted_row_before': promoted, 'before': 'JobName=test'}


@pytest.mark.parametrize('bad', ['StartTime=2026-09-09T19:00:00', 'Restarts=1',
                                  'NodeList=node208', 'RunTime=00:00:01'])
def test_never_started_rejects_any_allocation_evidence(item, monkeypatch, bad):
    base = 'StartTime=Unknown Restarts=0 NodeList= RunTime=00:00:00'
    field = bad.split('=')[0]
    record = ' '.join(p for p in base.split(' ') if p.split('=')[0] != field) + ' ' + bad
    monkeypatch.setattr(x.launch, 'new_guard', lambda *args, **kwargs: record)
    monkeypatch.setattr(x, 'no_gpu_writes', lambda item: None)
    with pytest.raises(RuntimeError, match='was allocated'):
        x.never_started(item)


def test_written_run_directory_blocks_cancellation(item):
    path = Path(item['run_dir']) / f"debug_job{item['new_job_id']}"; path.mkdir(parents=True)
    with pytest.raises(RuntimeError, match='run directory'):
        x.no_gpu_writes(item)


def test_restore_exact_original_row_preserves_other_74_and_history(item, tmp_path, monkeypatch):
    others = [{'continuation_job_id': k + 100, 'run_dir': f'/other/{k}', 'keep': k} for k in range(74)]
    data = {'continuations': [copy.deepcopy(item['promoted_row_before']), *others],
            'repair_history': [{'audit': 'existing history'}]}
    ledger = tmp_path / 'ledger.json'; ledger.write_text(json.dumps(data))
    monkeypatch.setattr(x.launch, 'LEDGER', ledger); monkeypatch.setattr(x, 'ART', tmp_path)
    monkeypatch.setattr(x, 'TX', tmp_path / 'rollback.json')
    monkeypatch.setattr(x.launch, 'authoritative', lambda *args: None)
    monkeypatch.setattr(x, 'event', lambda *args: None)
    tx = {'items': [item]}; x.restore_row(tx, item)
    after = json.loads(ledger.read_text())
    assert after['continuations'][0] == item['row_before']
    assert after['continuations'][1:] == others
    assert after['repair_history'][0] == {'audit': 'existing history'}
    assert after['repair_history'][1]['canceled_new'] == item['new_job_id']
    image = ledger.read_bytes(); item.pop('mapping_restored'); x.restore_row(tx, item)
    assert ledger.read_bytes() == image and item['mapping_restored']


def test_unrelated_restoration_cannot_be_claimed(item, tmp_path, monkeypatch):
    ledger = tmp_path / 'ledger.json'; ledger.write_text(json.dumps({'continuations': [item['row_before']]}))
    monkeypatch.setattr(x.launch, 'LEDGER', ledger)
    with pytest.raises(RuntimeError, match='without this rollback intent'):
        x.restore_row({'items': [item]}, item)


def test_cancellation_requires_inactive_accounting_confirmation(item, monkeypatch):
    item['cancel_requested'] = True
    monkeypatch.setattr(x.b, 'queue', lambda: {item['new_job_id']: 'RUNNING'})
    with pytest.raises(RuntimeError, match='still appears in queue'):
        x.canceled_successor(item)


def test_scope_contains_only_two_verified_staged_pairs():
    assert x.PAIRS == {31124279: 31163339, 31048181: 31163361}
