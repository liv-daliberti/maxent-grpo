"""Adversarial checks for the final cross-run identity and completion audit."""
import fcntl
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


spec = importlib.util.spec_from_file_location(
    'audit_all_levels512', Path(__file__).resolve().parents[1] / 'ops/audit_gpt56_all_levels512_collection.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


@pytest.fixture(scope='module')
def inventory():
    rows = [{'level': level, 'domain': domain, 'row_index': index}
            for level in (1, 2, 3)
            for domain in ('graph_coloring', 'countdown', 'python_factors', 'mathir', 'pantry_plan')
            for index in range(16)]
    records = [dict(row, sample_index=draw,
                    provider_sample_identity=[f"{row['level']}/{row['domain']}/{row['row_index']}/{draw}", 0])
               for row in rows for draw in range(512)]
    return rows, records


def test_complete_all_level_inventory(inventory):
    rows, records = inventory
    result = audit.check_inventory(records, rows)
    assert result['total_authenticated_responses'] == result['unique_provider_samples'] == 122880
    assert result['unique_prompts'] == 240
    assert len(result['cells']) == 15


@pytest.mark.parametrize('corruption', ['provider_reuse', 'duplicate_slot', 'missing_terminal_draw', 'unexpected_problem'])
def test_inventory_rejects_cross_run_identity_corruption(inventory, corruption):
    rows, records = inventory
    last = dict(records[-1])
    if corruption == 'provider_reuse':
        last['provider_sample_identity'] = records[0]['provider_sample_identity']
    elif corruption == 'duplicate_slot':
        last['sample_index'] = 510
    elif corruption == 'missing_terminal_draw':
        last['sample_index'] = 512
    else:
        last['row_index'] = 99
    with pytest.raises(ValueError):
        audit.check_inventory(records[:-1] + [last], rows)


def test_active_scheduler_prevents_completion_attestation(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, 'load_collector', lambda: SimpleNamespace(load_helper=lambda: SimpleNamespace()))
    with (tmp_path / '.production_scheduler.lock').open('w') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            audit.audit(tmp_path)
    assert not (tmp_path / 'collection_completion_audit.json').exists()
