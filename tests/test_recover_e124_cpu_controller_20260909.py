from contextlib import contextmanager
import copy
from datetime import timedelta
import importlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ops/exp_scaling'))
x = importlib.import_module('recover_e124_cpu_controller_20260909')


@pytest.fixture
def harness(tmp_path, monkeypatch):
    plan = {'sha256': 'recovery', 'science_plan_sha256': 'science', 'pins': {str(x.V2): 'v2'},
            'command': ['sbatch', '--hold', '--gres=none', '/waiter.slurm']}
    tx = {'plan_sha256': 'recovery', 'status': 'released', 'job_id': 900, 'events': []}
    monkeypatch.setattr(x, 'GATE', tmp_path / 'gate.json')
    monkeypatch.setattr(x, 'READY', tmp_path / 'ready.json')
    monkeypatch.setattr(x, 'CANONICAL_TX', tmp_path / 'canonical.json')
    monkeypatch.setattr(x, 'PLAN', tmp_path / 'plan.json')
    monkeypatch.setattr(x, 'TX', tmp_path / 'tx.json')
    x.CANONICAL_TX.write_text(json.dumps({'plan_sha256': 'science', 'controller': {'job_id': x.OLD, 'self_requeues': 7}, 'rows': {'science': {'job_id': 42, 'status': 'held'}}}))
    monkeypatch.setattr(x, 'load', lambda: (plan, tx))
    monkeypatch.setattr(x, 'pins', lambda p: None)
    monkeypatch.setattr(x, 'event', lambda t, label, **kw: t['events'].append(label))
    monkeypatch.setattr(x, 'all_gpu_holds', lambda p: None)
    monkeypatch.setattr(x, 'old_identity', lambda p: None)
    monkeypatch.setattr(x, 'audit_cpu', lambda *a, **kw: {'Priority': '0'})
    monkeypatch.setattr(x, 'ready_waiter', lambda *a: {'ready': True})
    @contextmanager
    def unlocked():
        yield
    monkeypatch.setattr(x, 'original_locks', unlocked)
    return plan, tx


def test_cpu_pool_excludes_unresponsive_private_and_pvl_nodes():
    assert 'node916' not in x.POOL
    assert x.POOL == {'node009', 'node010', 'node012', 'node013', 'node014', 'node015', 'node016', 'node915', 'node917'}


def test_submission_ambiguity_never_blindly_retries(harness, monkeypatch):
    plan, tx = harness; tx.pop('job_id'); tx['status'] = 'prepared'
    calls = []
    def timeout(args):
        assert tx['submit_intent']; calls.append(args)
        raise subprocess.TimeoutExpired(args, 120)
    monkeypatch.setattr(x, 'command', timeout)
    with pytest.raises(subprocess.TimeoutExpired):x.stage()
    with pytest.raises(RuntimeError, match='uncertain'):x.stage()
    assert len(calls) == 1


def test_adoption_requires_exact_owned_held_cpu(harness, monkeypatch):
    plan, tx = harness; tx.pop('job_id'); tx['submit_intent'] = True
    monkeypatch.setattr(x, 'audit_cpu', lambda *a, **kw: (_ for _ in ()).throw(RuntimeError('wrong CPU')))
    with pytest.raises(RuntimeError, match='wrong CPU'):x.adopt(999)
    assert 'job_id' not in tx


def test_staging_ack_reconciles_without_new_submission(harness, monkeypatch):
    plan, tx = harness; tx['status'] = 'held_unverified'
    monkeypatch.setattr(x, 'command', lambda *a: pytest.fail('no scheduler mutation'))
    assert x.stage() == {'status': 'held', 'job_id': 900}


def test_no_old_cancellation_until_allocated_waiter_ready(harness, monkeypatch):
    monkeypatch.setattr(x, 'ready_waiter', lambda *a: (_ for _ in ()).throw(RuntimeError('not ready')))
    monkeypatch.setattr(x, 'command', lambda *a: pytest.fail('must preserve old CPU'))
    with pytest.raises(RuntimeError, match='not ready'):x.handoff()


def test_cancels_only_old_cpu_once_after_ready(harness, monkeypatch):
    plan, tx = harness; calls = []
    monkeypatch.setattr(x, 'inactive_old', lambda: None)
    monkeypatch.setattr(x, 'command', lambda args: calls.append(args))
    assert x.handoff()['status'] == 'waiting_old_CPU_inactive'
    assert x.handoff()['status'] == 'waiting_old_CPU_inactive'
    assert calls == [['scancel', str(x.OLD)]]
    assert not x.GATE.exists()


def test_live_old_cpu_never_opens_gate(harness, monkeypatch):
    plan, tx = harness; tx['cancel_intent'] = True
    monkeypatch.setattr(x, 'inactive_old', lambda: None)
    monkeypatch.setattr(x, 'promote_controller', lambda *a: pytest.fail('old still active'))
    assert x.handoff()['status'] == 'waiting_old_CPU_inactive'
    assert not x.GATE.exists()


def test_held_original_lock_blocks_promotion_and_gate(harness, monkeypatch):
    monkeypatch.setattr(x, 'inactive_old', lambda: {'state': 'CANCELLED'})
    @contextmanager
    def blocked():
        raise BlockingIOError('old lock still owned')
        yield
    monkeypatch.setattr(x, 'original_locks', blocked)
    assert x.handoff()['status'] == 'waiting_old_locks'
    assert not x.GATE.exists()


def test_cpu_promotion_preserves_gpu_rows_and_counters(harness):
    plan, tx = harness; before = x.read(x.CANONICAL_TX)
    x.promote_controller(plan, tx); after = x.read(x.CANONICAL_TX)
    assert after['rows'] == before['rows']
    assert after['controller']['self_requeues'] == 7
    assert after['controller']['job_id'] == 900
    assert after['controller_history'] == [before['controller']]
    original = x.CANONICAL_TX.read_bytes(); x.promote_controller(plan, tx)
    assert x.CANONICAL_TX.read_bytes() == original


def test_promotion_precedes_gate_and_reconciles_existing_gate(harness, monkeypatch):
    plan, tx = harness
    monkeypatch.setattr(x, 'inactive_old', lambda: {'state': 'CANCELLED'})
    assert x.handoff()['status'] == 'gate_open'
    assert x.read(x.CANONICAL_TX)['controller']['job_id'] == 900
    monkeypatch.setattr(x, 'ready_waiter', lambda *a: pytest.fail('gate already committed'))
    assert x.handoff()['status'] == 'gate_open'


def test_auto_disabled_cannot_cancel_old(harness, monkeypatch):
    monkeypatch.setattr(x, 'command', lambda *a: pytest.fail('unapproved automatic action'))
    with pytest.raises(RuntimeError, match='not enabled'):x.handoff(automatic=True)


def test_recovered_old_controller_is_preserved(harness, monkeypatch):
    plan, tx = harness; tx['auto_handoff'] = True
    monkeypatch.setattr(x, 'old_monitoring_state', lambda: 'recovered')
    monkeypatch.setattr(x, 'command', lambda *a: pytest.fail('healthy old CPU must survive'))
    assert x.handoff(automatic=True)['status'] == 'old_controller_recovered'
    assert not tx.get('cancel_intent') and not x.GATE.exists()


def test_auto_freshness_checked_again_immediately_before_cancel(harness, monkeypatch):
    plan, tx = harness; tx['auto_handoff'] = True; states = iter(['stale', 'recovered'])
    monkeypatch.setattr(x, 'old_monitoring_state', lambda: next(states))
    monkeypatch.setattr(x, 'inactive_old', lambda: None)
    monkeypatch.setattr(x, 'command', lambda *a: pytest.fail('old recovered during audit'))
    assert x.handoff(automatic=True)['status'] == 'old_controller_recovered'
    assert not tx.get('cancel_intent')


@pytest.mark.parametrize('age,expected', [(60, 'recovered'), (600, 'waiting'), (1000, 'stale')])
def test_old_monitoring_freshness_thresholds(harness, monkeypatch, tmp_path, age, expected):
    monkeypatch.setattr(x, 'BASE', tmp_path)
    (tmp_path / 'controller_ready.json').write_text(json.dumps({'job_id': x.OLD, 'at': (x.now() - timedelta(seconds=age)).isoformat()}))
    assert x.old_monitoring_state() == expected


def test_gpu_released_during_recovery_blocks_cancel(harness, monkeypatch):
    monkeypatch.setattr(x, 'all_gpu_holds', lambda p: (_ for _ in ()).throw(RuntimeError('GPU no longer held')))
    monkeypatch.setattr(x, 'command', lambda *a: pytest.fail('must stop before cancellation'))
    with pytest.raises(RuntimeError, match='GPU no longer held'):x.handoff()


def test_gate_cannot_start_v2_before_canonical_promotion(harness, monkeypatch):
    plan, tx = harness
    gate = {'job_id': 900, 'old_job_id': x.OLD, 'plan_sha256': 'recovery', 'source_sha256': 'v2'}
    monkeypatch.setattr(x, 'inactive_old', lambda: {'state': 'CANCELLED'})
    with pytest.raises(RuntimeError, match='canonical CPU promotion missing'):
        x.gate_valid(plan, tx, gate)


@pytest.fixture
def waiter_clock(harness, monkeypatch):
    plan, tx = harness; plan['max_wait_seconds'] = 2
    x.TX.write_text(json.dumps(tx)); monkeypatch.setenv('SLURM_JOB_ID', '900')
    clock = [0.0]
    monkeypatch.setattr(x.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(x.time, 'sleep', lambda value: clock.__setitem__(0, clock[0] + value))
    timers = []
    monkeypatch.setattr(x.signal, 'signal', lambda *a: None)
    monkeypatch.setattr(x.signal, 'setitimer', lambda kind, duration: timers.append(duration))
    return plan, tx, timers


def test_waiter_without_gate_expires_without_mutating_jobs(waiter_clock, monkeypatch):
    plan, tx, timers = waiter_clock
    monkeypatch.setattr(x, 'command', lambda *a: pytest.fail('waiting must not mutate a scheduler job'))
    monkeypatch.setattr(x.os, 'execv', lambda *a: pytest.fail('gate not opened'))
    assert x.waiter()['status'] == 'waiter_expired_without_gate'
    assert x.read(x.READY)['phase'] == 'expired_without_gate'
    assert timers[-1] == 0


def test_waiter_retries_busy_original_lock_within_its_bound(waiter_clock, monkeypatch):
    plan, tx, timers = waiter_clock; tx['auto_handoff'] = True; x.TX.write_text(json.dumps(tx))
    @contextmanager
    def unlocked():yield
    monkeypatch.setattr(x, 'recovery_lock', unlocked)
    monkeypatch.setattr(x, 'handoff', lambda **kw: (_ for _ in ()).throw(BlockingIOError('old lock retained')))
    assert x.waiter()['status'] == 'waiter_expired_without_gate'
    assert timers[-1] == 0


def test_open_gate_executes_exact_unchanged_v2_and_clears_alarm(waiter_clock, monkeypatch):
    plan, tx, timers = waiter_clock; x.GATE.write_text('{}')
    checked = []; monkeypatch.setattr(x, 'gate_valid', lambda *a: checked.append(True))
    class Executed(BaseException):pass
    def execute(path, args):
        assert path == x.PYTHON and args == [x.PYTHON, str(x.V2), 'watch']
        assert timers[-1] == 0
        raise Executed()
    monkeypatch.setattr(x.os, 'execv', execute)
    with pytest.raises(Executed):x.waiter()
    assert checked == [True]
    assert x.read(x.READY)['phase'] == 'starting_unchanged_v2'



def test_old_scheduler_record_purge_uses_user_scan_and_accounting(harness, monkeypatch):
    calls = []
    def observe(args):
        calls.append(args)
        if args[0] == 'squeue':
            assert '-j' not in args and '-u' in args
            return subprocess.CompletedProcess(args, 0, '900|RUNNING\n', '')
        assert args[0] == 'sacct'
        return subprocess.CompletedProcess(args, 0, f'{x.OLD}|CANCELLED by 363432|0:0\n', '')
    monkeypatch.setattr(x, 'command', observe)
    assert x.inactive_old()['state'] == 'CANCELLED'
    assert len(calls) == 2


def test_old_in_user_queue_still_blocks_gate_without_accounting(harness, monkeypatch):
    def observe(args):
        assert args[0] == 'squeue'
        return subprocess.CompletedProcess(args, 0, f'{x.OLD}|COMPLETING\n', '')
    monkeypatch.setattr(x, 'command', observe)
    assert x.inactive_old() is None


def test_waiter_retries_recovery_lock_contention_within_bound(waiter_clock, monkeypatch):
    plan, tx, timers = waiter_clock; tx['auto_handoff'] = True; x.TX.write_text(json.dumps(tx))
    @contextmanager
    def busy():
        raise BlockingIOError('root is auditing the same recovery')
        yield
    monkeypatch.setattr(x, 'recovery_lock', busy)
    monkeypatch.setattr(x, 'handoff', lambda **kw: pytest.fail('root owns mutation lock'))
    assert x.waiter()['status'] == 'waiter_expired_without_gate'
    assert timers[-1] == 0
