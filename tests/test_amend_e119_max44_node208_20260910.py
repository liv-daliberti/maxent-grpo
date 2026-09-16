from datetime import datetime, timedelta, timezone
import importlib
import json
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ops/exp_scaling'))
x = importlib.import_module('amend_e119_max44_node208_20260910')


@pytest.fixture
def target(monkeypatch, tmp_path):
    item = {'identity': {'run_dir': str(tmp_path)}, 'resources': {'ReqNodeList': 'node205,node207,node302'}}
    monkeypatch.setattr(x.guard, 'stable', lambda *a: None)
    monkeypatch.setattr(x.guard, 'dormant', lambda *a: None)
    monkeypatch.setattr(x.guard, 'writer_check', lambda *a: None)
    monkeypatch.setattr(x.guard, 'show', lambda job: f'JobId={x.TARGET} JobState=PENDING Restarts=1 RunTime=00:00:00 Reason=JobHeldUser Priority=0')
    return item


def test_running_target_is_never_held_or_rerouted(target, monkeypatch):
    monkeypatch.setattr(x.guard, 'show', lambda job: f'JobId={x.TARGET} JobState=RUNNING')
    monkeypatch.setattr(x.b, 'command', lambda *a, **k: pytest.fail('no mutation'))
    with pytest.raises(RuntimeError, match='no longer pending'):
        x.target_record(target, amended=False, held=False)


def test_owned_target_hold_requires_zero_priority(target, monkeypatch):
    monkeypatch.setattr(x.guard, 'show', lambda job: f'JobId={x.TARGET} JobState=PENDING Restarts=1 RunTime=00:00:00 Reason=JobHeldUser Priority=10')
    with pytest.raises(RuntimeError, match='owned target hold'):
        x.target_record(target, amended=True, held=True)


def test_runtime_output_prevents_never_started_route(target):
    debug = Path(target['identity']['run_dir']) / f'debug_job{x.TARGET}'
    debug.mkdir(); (debug / 'train_metrics.jsonl').write_text('{}')
    with pytest.raises(RuntimeError, match='runtime output'):
        x.target_record(target, amended=False, held=True)


def test_amended_stability_check_changes_only_node_pool(target, monkeypatch):
    target['resources']['MinMemoryNode'] = '116G'
    seen = []
    monkeypatch.setattr(x.guard, 'stable', lambda item, record: seen.append(item))
    x.target_record(target, amended=True, held=True)
    assert seen[0]['resources'] == {'ReqNodeList': x.NODES, 'MinMemoryNode': '116G'}
    assert target['resources']['ReqNodeList'] == 'node205,node207,node302'


def test_unresolved_old_retry_prevents_cpu_handoff(monkeypatch, tmp_path):
    path = tmp_path / 'tx.json'
    path.write_text(json.dumps({'plan_sha256': 'old', 'jobs': {'1': {'attempts': [{'released': False}]}}}))
    monkeypatch.setattr(x, 'OLD_TX', path)
    with pytest.raises(RuntimeError, match='unresolved retry'):
        x.final_old_state({'old_plan_sha256': 'old'})


def test_only_target_ledger_route_fields_change():
    before = {'continuation_job_id': x.TARGET, 'run_dir': '/same', 'seed': 44,
              'actual_requested_nodes': ['node205','node207','node302'],
              'actual_scheduler_profile': {'ReqNodeList': 'node205,node207,node302', 'MinMemoryNode': '116G'},
              'nested_history': [{'old': 31048181}]}
    after = x.amended_row({'ledger_row_before': before})
    assert after['continuation_job_id'] == before['continuation_job_id'] and after['run_dir'] == '/same'
    assert after['actual_scheduler_profile']['MinMemoryNode'] == '116G'
    assert after['actual_requested_nodes'] == x.NODES.split(',')
    assert after['nested_history'] == before['nested_history']
    assert before['actual_requested_nodes'] == ['node205','node207','node302']


def test_old_cpu_resource_drift_blocks_cancellation(monkeypatch):
    monkeypatch.setattr(x.guard, 'show', lambda job: 'Account=someoneelse Restarts=0 JobState=RUNNING')
    monkeypatch.setattr(x.b, 'submit_tokens', lambda record: ['same'])
    plan = {'old_cpu_submit_tokens': ['same'], 'old_cpu_resources': {'Account': 'mltheory'}, 'old_cpu_restarts': '0'}
    with pytest.raises(RuntimeError, match='identity/resources changed'):
        x.old_cpu(plan, running=True)


def test_different_successor_cpu_cannot_duplicate_handoff(monkeypatch):
    monkeypatch.setattr(x, 'load', lambda: ({}, {'new_cpu_job_id': 100}))
    monkeypatch.setattr(x, 'new_cpu', lambda *a, **k: pytest.fail('different CPU must be rejected first'))
    with pytest.raises(RuntimeError, match='duplicate the guard'):
        x.route(101)


def test_new_cpu_explicit_two_thread_requirement(monkeypatch):
    plan = {'cpu_submit_tokens': ['same'], 'old_cpu_resources': {'UserId': 'user(1)'}}
    monkeypatch.setattr(x.b, 'submit_tokens', lambda record: ['same'])
    monkeypatch.setattr(x.guard, 'show', lambda job: 'UserId=user(1) ReqTRES=cpu=1,mem=2G MinMemoryNode=2G NumCPUs=1 TimeLimit=1-01:10:00')
    with pytest.raises(RuntimeError, match='actual resources differ'):
        x.new_cpu(plan, 100, held=True)
    assert '--cpus-per-task=2' in x.cpu_command()


def test_original_cpu_must_exit_before_successor_ready(monkeypatch):
    monkeypatch.setattr(x.b, 'queue', lambda: {x.OLD_CPU: 'RUNNING'})
    with pytest.raises(RuntimeError, match='Old CPU became active'):
        x.ready({}, {})


def test_stale_successor_heartbeat_prevents_target_release(monkeypatch, tmp_path):
    monkeypatch.setattr(x, 'NEW', tmp_path)
    monkeypatch.setattr(x.b, 'queue', lambda: {})
    monkeypatch.setattr(x, 'new_cpu', lambda *a, **k: 'JobState=RUNNING')
    (tmp_path / 'ready.json').write_text(json.dumps({'job_id': 100, 'plan_sha256': 'new',
        'at': (datetime.now(timezone.utc) - timedelta(minutes=4)).isoformat()}))
    with pytest.raises(RuntimeError, match='heartbeat is stale'):
        x.ready({'new_guard_plan_sha256': 'new'}, {'new_cpu_job_id': 100})


def test_new_drain_blocks_otherwise_successful_physical_probe(monkeypatch, tmp_path):
    proof = {'at_utc': datetime.now(timezone.utc).isoformat(), 'node': 'node208',
             'gpu': {'name': 'NVIDIA RTX A6000', 'memory_total_MiB': 49140, 'temperature_C': 26},
             'host': {'MemAvailable_kB': 400 * 1024**2}}
    path = tmp_path / 'probe.json'; path.write_text(json.dumps(proof))
    monkeypatch.setattr(x, 'HEALTH', path)
    class Result:
        stdout = 'State=MIXED+DRAIN Gres=gpu:a6000:10 Partitions=lowprio'
    monkeypatch.setattr(x.b, 'command', lambda *a, **k: Result())
    with pytest.raises(RuntimeError, match='became unhealthy'):
        x.health()


def test_empty_preempted_startup_with_frozen_restart_is_allowed(target):
    (Path(target['identity']['run_dir']) / f'debug_job{x.TARGET}').mkdir()
    assert x.target_record(target, amended=False, held=True)
