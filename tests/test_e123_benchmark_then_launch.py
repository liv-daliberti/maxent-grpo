"""Failure boundaries for the authorized deferred E123 launch."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('e123_auto', ROOT / 'ops/exp_scaling/run_e123_benchmark_then_launch.py')
auto = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auto)


def test_projection_allows_only_registered_physical_environment_changes():
    preview = {'snapshot': {'id': 'frozen'}, 'model_identity': {'revision': 'fixed'},
               'admission_proof': {'status': 'passed'}, 'files_sha256': {},
               'cells': [{'domain': 'mathir', 'seed': 70, 'arm': 'maxrl',
                          'environment': {'OAT_ZERO_LEARNING_RATE': '1e-7',
                                          'OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE': '1'},
                          'resources': {'memory_gib': 128}, 'command': ['baseline']}]}
    measured = copy.deepcopy(preview)
    measured['cells'][0]['environment']['OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE'] = '8'
    measured['cells'][0]['resources']['memory_gib'] = 56
    measured['cells'][0]['command'] = ['measured']
    assert auto.science_projection(measured) == auto.science_projection(preview)
    measured['cells'][0]['environment']['OAT_ZERO_LEARNING_RATE'] = '2e-7'
    assert auto.science_projection(measured) != auto.science_projection(preview)


def test_ambiguous_submission_retains_intent_and_cannot_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(auto, 'HERE', tmp_path)
    (tmp_path / 'orchestration.json').write_text(json.dumps({'benchmark_command': ['sbatch']}))
    monkeypatch.setattr(auto, 'check', lambda plan: {})
    calls = []
    def run(*args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout='unexpected response', stderr='')
    monkeypatch.setattr(auto.subprocess, 'run', run)
    with pytest.raises(ValueError, match='ambiguous'):
        auto.submit()
    assert (tmp_path / 'benchmark_submission_intent.json').exists()
    with pytest.raises((ValueError, FileExistsError)):
        auto.submit()
    assert len(calls) == 1


def test_accounting_requires_exact_allocation_not_step_success(monkeypatch):
    monkeypatch.setattr(auto.subprocess, 'run', lambda *a, **kw: SimpleNamespace(stdout='123.batch|COMPLETED|0:0|\n'))
    with pytest.raises(ValueError, match='missing or ambiguous'):
        auto.benchmark_status(123)


def test_failed_benchmark_never_launches_science(tmp_path, monkeypatch):
    monkeypatch.setattr(auto, 'HERE', tmp_path)
    for name, value in [('orchestration.json', {}), ('benchmark_held_audit.json', {'job_id': 123}),
                        ('benchmark_release_result.json', {'returncode': 0})]:
        (tmp_path / name).write_text(json.dumps(value))
    monkeypatch.setattr(auto, 'check', lambda plan: {})
    monkeypatch.setattr(auto, 'benchmark_status', lambda job_id: ('OUT_OF_MEMORY', '0:125'))
    monkeypatch.setattr(auto, 'launch_measured', lambda *a: pytest.fail('failed benchmark launched science'))
    with pytest.raises(ValueError, match='OUT_OF_MEMORY'):
        auto.watch()


def test_success_requires_zero_allocation_exit(tmp_path, monkeypatch):
    monkeypatch.setattr(auto, 'HERE', tmp_path)
    for name, value in [('orchestration.json', {}), ('benchmark_held_audit.json', {'job_id': 123}),
                        ('benchmark_release_result.json', {'returncode': 0})]:
        (tmp_path / name).write_text(json.dumps(value))
    monkeypatch.setattr(auto, 'check', lambda plan: {})
    monkeypatch.setattr(auto, 'benchmark_status', lambda job_id: ('COMPLETED', '1:0'))
    monkeypatch.setattr(auto, 'launch_measured', lambda *a: pytest.fail('nonzero exit launched science'))
    with pytest.raises(ValueError, match='exit successfully'):
        auto.watch()


def test_queue_identity_can_bridge_brief_accounting_delay(monkeypatch):
    responses = iter([SimpleNamespace(stdout=''), SimpleNamespace(stdout='123|PENDING\n')])
    monkeypatch.setattr(auto.subprocess, 'run', lambda *a, **kw: next(responses))
    assert auto.benchmark_status(123) == ('PENDING', 'accounting_pending')


def test_profile_cannot_change_source_snapshot():
    base = {'snapshot': {'sha': 'benchmarked'}, 'model_identity': {}, 'admission_proof': {},
            'files_sha256': {}, 'cells': []}
    changed = dict(base, snapshot={'sha': 'different_runtime'})
    assert auto.science_projection(base) != auto.science_projection(changed)
