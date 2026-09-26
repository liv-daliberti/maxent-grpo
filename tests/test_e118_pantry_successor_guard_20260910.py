from __future__ import annotations

from datetime import datetime, timedelta, timezone
import importlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ops/exp_scaling'))
x = importlib.import_module('guard_e118_pantry_successor_20260910')


@pytest.fixture
def scope(monkeypatch, tmp_path):
    now = datetime(2026, 9, 11, 3, tzinfo=timezone.utc)
    monkeypatch.setattr(x, 'utcnow', lambda: now)
    plan = {'not_before_utc': (now - timedelta(hours=1)).isoformat(),
            'deadline_utc': (now + timedelta(days=7)).isoformat(),
            'old_cpu_job_id': 88, 'old_plan_sha256': 'frozen',
            'rows': [{'job_id': 31048143, 'initial_resume_step': 192,
                      'identity': {'run_dir': '/run'}, 'max_requeues': 2}],
            'runtime_fingerprints': {'source': 'sha'}}
    old = {'plan_sha256': 'frozen', 'deadline_utc': plan['not_before_utc'],
           'jobs': {'31048143': {'status': 'monitoring', 'last_resume_step': 288,
                                'attempts': [{'released': True}]}}}
    path = tmp_path / 'old.json'; path.write_text(json.dumps(old))
    monkeypatch.setattr(x, 'OLD_TX', path)
    monkeypatch.setattr(x.base, 'queue', lambda: {})
    monkeypatch.setattr(x, 'save', lambda *a: None)
    return plan, {'jobs': {}, 'events': []}, old


def test_takeover_inherits_final_floor_and_separate_retry_budget(scope):
    plan, tx, _ = scope
    x.takeover(plan, tx)
    state = tx['jobs']['31048143']
    assert state['last_resume_step'] == 288
    assert state['attempts'] == [] and state['predecessor_retries'] == 1
    assert tx['old_final_tx_sha256'] == x.recovery.digest(x.OLD_TX)


def test_no_takeover_before_original_absolute_deadline(scope, monkeypatch):
    plan, tx, _ = scope
    monkeypatch.setattr(x, 'utcnow', lambda: datetime.fromisoformat(plan['not_before_utc']) - timedelta(microseconds=1))
    with pytest.raises(RuntimeError, match='deadline not reached'):
        x.takeover(plan, tx)
    assert tx['jobs'] == {}


def test_active_predecessor_blocks_takeover_even_after_deadline(scope, monkeypatch):
    plan, tx, _ = scope
    monkeypatch.setattr(x.base, 'queue', lambda: {88: 'RUNNING'})
    with pytest.raises(RuntimeError, match='Predecessor CPU remains active'):
        x.takeover(plan, tx)


def test_unresolved_predecessor_hold_blocks_takeover(scope):
    plan, tx, old = scope
    old['jobs']['31048143']['attempts'][0]['released'] = False
    x.OLD_TX.write_text(json.dumps(old))
    with pytest.raises(RuntimeError, match='unresolved mutation'):
        x.takeover(plan, tx)


def test_changed_predecessor_after_takeover_stops_successor(scope):
    plan, tx, old = scope
    x.takeover(plan, tx)
    old['changed'] = True; x.OLD_TX.write_text(json.dumps(old))
    with pytest.raises(RuntimeError, match='changed after takeover'):
        x.takeover(plan, tx)


def test_both_duplicate_guard_boundaries_use_real_flock(tmp_path):
    # Distinct opens match separate CPU processes' open file descriptions.
    for name in ['successor.lock', 'predecessor.lock']:
        with (tmp_path / name).open('a+') as first, (tmp_path / name).open('a+') as second:
            assert x.acquire(first)
            assert not x.acquire(second)
        with (tmp_path / name).open('a+') as next_owner:
            assert x.acquire(next_owner)


@pytest.fixture
def observing(scope, monkeypatch):
    plan, tx, _ = scope
    item = plan['rows'][0]
    x.takeover(plan, tx)
    monkeypatch.setattr(x.prior, 'mapping', lambda: {item['job_id']: {'identity': item['identity']}})
    monkeypatch.setattr(x.prior, 'stable', lambda *a: None)
    monkeypatch.setattr(x.prior, 'show', lambda job: 'JobState=TIMEOUT Reason=TimeLimit Restarts=1 RunTime=12:00:00')
    monkeypatch.setattr(x.prior, 'checkpoint_and_writer', lambda item: {'step': 384})
    monkeypatch.setattr(x.prior, 'save', lambda *a: None)
    monkeypatch.setattr(x.recovery, 'complete', lambda path: False)
    monkeypatch.setattr(x.recovery, 'state', lambda job: 'TIMEOUT')
    monkeypatch.setattr(x.base, 'command', lambda *a, **k: pytest.fail('unexpected scheduler mutation'))
    return plan, tx, item


def test_terminal_completion_suppresses_retry_without_controller_record(observing, monkeypatch):
    plan, tx, item = observing
    monkeypatch.setattr(x.recovery, 'complete', lambda path: True)
    monkeypatch.setattr(x.prior, 'show', lambda job: pytest.fail('completed cell needs no controller'))
    assert x.prior.observe_one(tx, item, apply=True)['status'] == 'completed'


def test_no_progress_is_rejected(observing, monkeypatch):
    plan, tx, item = observing
    monkeypatch.setattr(x.prior, 'checkpoint_and_writer', lambda item: {'step': 288})
    with pytest.raises(RuntimeError, match='No newer valid checkpoint'):
        x.prior.observe_one(tx, item, apply=True)


def test_advancing_timeout_dry_run_has_no_mutation(observing):
    plan, tx, item = observing
    result = x.prior.observe_one(tx, item, apply=False)
    assert result['status'] == 'would_requeue_same_id' and result['retry_number'] == 1
    assert tx['jobs'][str(item['job_id'])]['attempts'] == []


def test_two_additional_retry_cap(observing):
    plan, tx, item = observing
    tx['jobs'][str(item['job_id'])]['attempts'] = [{'released': True}] * 2
    with pytest.raises(RuntimeError, match='allowance exhausted'):
        x.prior.observe_one(tx, item, apply=True)


def test_purged_controller_is_visible_manual_stop(observing, monkeypatch):
    plan, tx, item = observing
    def purged(job):
        raise RuntimeError('Job no longer has a controller record; manual continuation required')
    monkeypatch.setattr(x.prior, 'show', purged)
    result = x.observe(plan, tx, apply=True)
    assert result[0]['status'] == 'manual_stop'
    assert 'controller record' in tx['jobs'][str(item['job_id'])]['error']


def test_uncertain_requeue_is_visible_stop_and_never_reissued(observing, monkeypatch):
    plan, tx, item = observing
    calls = []
    def fail(parts, **kwargs):
        calls.append(parts)
        assert tx['jobs'][str(item['job_id'])]['attempts'][0]['hold_intent']
        raise subprocess.TimeoutExpired(parts, 45)
    monkeypatch.setattr(x.base, 'command', fail)
    assert x.observe(plan, tx, apply=True)[0]['status'] == 'manual_stop'
    assert x.observe(plan, tx, apply=True)[0]['status'] == 'manual_stop'
    assert calls == [['scontrol', 'requeuehold', str(item['job_id'])]]
    assert not tx['jobs'][str(item['job_id'])]['attempts'][0]['released']


@pytest.mark.parametrize('verb', ['requeuehold', 'release'])
def test_expiry_checked_at_each_mutation_boundary(scope, monkeypatch, verb):
    plan, tx, _ = scope
    x.takeover(plan, tx)
    # Restore the process-local overrides when this test ends.
    for module, name in [(x.prior, 'mapping'), (x.prior, 'checkpoint_and_writer'),
                         (x.prior, 'save'), (x.base, 'command')]:
        monkeypatch.setattr(module, name, getattr(module, name))
    x.install_primitives(plan, tx, apply=True)
    monkeypatch.setattr(x, 'utcnow', lambda: datetime.fromisoformat(plan['deadline_utc']))
    monkeypatch.setattr(x, 'command', lambda *a, **k: pytest.fail('expired mutation must never execute'))
    with pytest.raises(RuntimeError, match='outside owned successor window'):
        x.base.command(['scontrol', verb, '31048143'])


def test_deadline_rechecked_after_slow_fingerprint_reads(scope, monkeypatch):
    plan, tx, _ = scope
    x.takeover(plan, tx)
    for module, name in [(x.prior, 'mapping'), (x.prior, 'checkpoint_and_writer'),
                         (x.prior, 'save'), (x.base, 'command')]:
        monkeypatch.setattr(module, name, getattr(module, name))
    x.install_primitives(plan, tx, apply=True)
    def expensive(items):
        monkeypatch.setattr(x, 'utcnow', lambda: datetime.fromisoformat(plan['deadline_utc']))
        return plan['runtime_fingerprints']
    monkeypatch.setattr(x.runtime, 'runtime_fingerprints', expensive)
    monkeypatch.setattr(x, 'command', lambda *a, **k: pytest.fail('deadline passed during metadata reads'))
    with pytest.raises(RuntimeError, match='deadline passed during validation'):
        x.base.command(['scontrol', 'release', '31048143'])


def test_newer_partial_checkpoint_blocks_retry(scope, monkeypatch):
    plan, tx, _ = scope
    monkeypatch.setattr(x.recovery, 'complete', lambda path: False)
    monkeypatch.setattr(x.recovery, 'select_latest_checkpoint', lambda path: (Path('/run/step_00384'), {'/run/step_00576': ['partial']}))
    monkeypatch.setattr(x.checkpoints, 'checked_checkpoint', lambda path: {'step': 384})
    with pytest.raises(RuntimeError, match='Newer incomplete checkpoint'):
        x.checkpoint_and_writer(plan['rows'][0])


def test_prepared_cpu_requests_two_threads():
    assert '--cpus-per-task=2' in x.cpu_command()
    assert '--gres=none' in x.cpu_command()


@pytest.mark.parametrize('state', ['RUNNING', 'PENDING'])
def test_acknowledged_release_reconciles_without_second_release(observing, monkeypatch, state):
    plan, tx, item = observing
    action = {'before_restarts': 1, 'release_requested': True, 'released': False,
              'checkpoint_before_release': {'step': 384}}
    job_state = tx['jobs'][str(item['job_id'])]
    job_state['attempts'] = [action]
    monkeypatch.setattr(x.prior, 'show', lambda job: f'JobState={state} Reason=None Restarts=2 RunTime=00:00:00')
    x.prior.reconcile_action(tx, item, action, job_state)
    assert action['released'] and job_state['status'] == 'monitoring'
    assert job_state['last_resume_step'] == 384
