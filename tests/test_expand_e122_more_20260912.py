import importlib.util
from pathlib import Path
import pytest

SOURCE = Path(__file__).resolve().parents[1] / 'ops/exp_scaling/expand_e122_more_20260912.py'
spec = importlib.util.spec_from_file_location('more_e122', SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def node(name, memory=131072, used=0, state=None):
    return {'name': name, 'state': state or ['MIXED'], 'partitions': ['lowprio'],
            'gres': 'gpu:a6000:10', 'gres_used': f'gpu:a6000:{used}',
            'real_memory': memory, 'alloc_memory': 0, 'cpus': 16, 'alloc_cpus': 0}


def test_capacity_uses_other_qualified_nodes_without_node208_room():
    rows = module.capacity_rows([node('node208', memory=56000), node('node207', memory=73000),
                                 node('node302', memory=187428), node('node206', state=['MIXED', 'DRAIN'])])
    assert {r['node']: r['free_64g_slots'] for r in rows} == {
        'node208': 0, 'node207': 1, 'node302': 2, 'node206': 0}


def test_capacity_never_counts_unqualified_or_full_nodes():
    with pytest.raises(ValueError, match='No schedulable'):
        module.capacity_rows([node('node209'), node('node207', used=10), node('node302', memory=64000)])


def test_both_replacement_array_tasks_keep_full_storage_reservations(monkeypatch):
    budget = module.m.budget_helper
    monkeypatch.setattr(budget, 'CAP', 21)
    monkeypatch.setitem(budget.ARRAYS, module.INFERENCE_ID, (2, module.WORKER, module.INFERENCE_PLAN))
    monkeypatch.setattr(budget.base, 'canonical_registry', lambda: {})
    monkeypatch.setattr(budget, 'run', lambda _: f'UserId=user({module.os.getuid()}) WorkDir={module.ROOT} {module.WORKER}')
    monkeypatch.setattr(budget, 'digest', lambda _: 'x')
    value = budget.storage_budget({'jobs': []}, queue=(
        '31259131_0|RUNNING|None|gres/gpu:a5000:2\n'
        '31259131_1|PENDING|JobArrayTaskLimit|gres/gpu:a5000:2'))
    assert sum(r['bytes'] for r in value['reservations']) == 64 * 1024**3
    with pytest.raises(ValueError, match='unregistered'):
        budget.storage_budget({'jobs': []}, queue='31259131_2|PENDING|Resources|gres/gpu:a5000:2')
