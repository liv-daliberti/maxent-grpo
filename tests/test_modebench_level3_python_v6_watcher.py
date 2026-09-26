"""Single-owner v6 fitting preserves failure, source pins and immutable phases."""
import fcntl
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ops/exp_scaling'))
import watch_modebench_level3_python_v6 as watcher


def json_file(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    for name, filename in [('SEAL', 'implementation_seal.json'), ('LAUNCHER', 'launch.py'),
                           ('RECIPE', 'recipe.json'), ('INTENT', 'development_fit_intent.json'),
                           ('RESULT', 'development_fit_report.json'), ('FAILURE', 'development_fit_failure.json'),
                           ('EVENTS', 'development_fit_events.jsonl'), ('LOCK', '.python_v6_fit.lock')]:
        monkeypatch.setattr(watcher, name, tmp_path / filename)
    baseline = json_file(tmp_path / 'baseline.json', {'status': 'complete'})
    scores = [json_file(tmp_path / f'd{i}.json', {'status': 'complete'}) for i in range(4)]
    monkeypatch.setattr(watcher, 'BASELINE', baseline)
    monkeypatch.setattr(watcher, 'SCORES', scores)
    monkeypatch.setattr(watcher, 'RECEIPTS', [baseline, *scores])
    monkeypatch.setattr(watcher, 'verify', lambda expected: {'seal': expected})
    return tmp_path


def recipe(passed):
    return {'development_fit_pass': passed, 'decision': 'fixed_decision',
            'weights': [0.1, 0.2, 0.3, 0.4], 'development': {'fixed_forecast': True}}


def test_only_four_registered_v6_receipts_are_fit_once(campaign, monkeypatch):
    calls = []
    def fit(*args):
        calls.append(args)
        assert watcher.INTENT.is_file()
        assert not watcher.RECIPE.exists()
        # An independent file description cannot acquire the domain lock.
        with watcher.LOCK.open('a') as lock:
            with pytest.raises(BlockingIOError):
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return recipe(True)
    monkeypatch.setattr(watcher.adapter, 'fit_recipe', fit)
    result = watcher.run('seal')
    assert calls == [(watcher.BASELINE, watcher.SCORES, 'python_factors')]
    assert result['status'] == 'development_fit_pass'
    assert json.loads(watcher.RECIPE.read_text()) == recipe(True)
    assert result['recipe_sha256'] == watcher.digest(watcher.RECIPE)
    assert result['intent_sha256'] == watcher.digest(watcher.INTENT)
    assert not watcher.FAILURE.exists()
    with pytest.raises(FileExistsError):
        watcher.run('seal')
    assert len(calls) == 1


def test_failed_development_gate_is_published_without_retry_or_changes(campaign, monkeypatch):
    monkeypatch.setattr(watcher.adapter, 'fit_recipe', lambda *args: recipe(False))
    result = watcher.run('seal')
    assert result['status'] == 'needs_calibration_revision'
    assert result['development_fit_pass'] is False
    assert json.loads(watcher.RECIPE.read_text()) == recipe(False)
    assert not watcher.FAILURE.exists()


@pytest.mark.parametrize('name', ['INTENT', 'RESULT', 'RECIPE', 'FAILURE'])
def test_existing_phase_evidence_stops_before_fit(campaign, monkeypatch, name):
    json_file(getattr(watcher, name), {'already': 'owned'})
    monkeypatch.setattr(watcher.adapter, 'fit_recipe', lambda *args: pytest.fail('must not refit'))
    with pytest.raises(FileExistsError):
        watcher.run('seal')


def test_missing_receipt_waits_without_claim_or_fit(campaign, monkeypatch):
    watcher.SCORES[2].unlink()
    monkeypatch.setattr(watcher.adapter, 'fit_recipe', lambda *args: pytest.fail('must wait for all receipts'))
    result = watcher.run('seal')
    assert result['missing_receipts'] == [str(watcher.SCORES[2])]
    assert result['status'] == 'waiting_for_python_v6_receipts'
    assert not watcher.INTENT.exists()
    assert not watcher.RECIPE.exists()


def test_noncomplete_receipt_fails_before_claim(campaign, monkeypatch):
    json_file(watcher.SCORES[3], {'status': 'partial'})
    monkeypatch.setattr(watcher.adapter, 'fit_recipe', lambda *args: pytest.fail('must not fit partial receipt'))
    with pytest.raises(ValueError, match='must be complete'):
        watcher.run('seal')
    assert not watcher.INTENT.exists()


def test_fitter_exception_leaves_intent_failure_and_forbids_retry(campaign, monkeypatch):
    def fit(*args):
        raise ValueError('unauthenticated receipt')
    monkeypatch.setattr(watcher.adapter, 'fit_recipe', fit)
    with pytest.raises(ValueError, match='unauthenticated receipt'):
        watcher.run('seal')
    assert watcher.INTENT.exists() and watcher.FAILURE.exists()
    assert not watcher.RECIPE.exists() and not watcher.RESULT.exists()
    with pytest.raises(FileExistsError):
        watcher.run('seal')


def test_receipt_mutation_during_fit_cannot_publish(campaign, monkeypatch):
    def fit(*args):
        json_file(watcher.SCORES[1], {'status': 'complete', 'changed': True})
        return recipe(True)
    monkeypatch.setattr(watcher.adapter, 'fit_recipe', fit)
    with pytest.raises(ValueError, match='receipt changed during fitting'):
        watcher.run('seal')
    assert watcher.FAILURE.exists() and not watcher.RECIPE.exists()


def test_source_mutation_after_fitting_cannot_publish(campaign, monkeypatch):
    calls = []
    def verify(expected):
        calls.append(expected)
        if len(calls) == 3:
            raise ValueError('sealed source changed')
    monkeypatch.setattr(watcher, 'verify', verify)
    monkeypatch.setattr(watcher.adapter, 'fit_recipe', lambda *args: recipe(True))
    with pytest.raises(ValueError, match='sealed source changed'):
        watcher.run('seal')
    assert watcher.FAILURE.exists() and not watcher.RECIPE.exists()


def test_partial_publication_preserves_recipe_and_forbids_retry(campaign, monkeypatch):
    original = watcher.atomic_new
    def publish(path, payload):
        if path == watcher.RESULT:
            raise OSError('publication interrupted')
        return original(path, payload)
    monkeypatch.setattr(watcher, 'atomic_new', publish)
    monkeypatch.setattr(watcher.adapter, 'fit_recipe', lambda *args: recipe(True))
    with pytest.raises(OSError, match='publication interrupted'):
        watcher.run('seal')
    assert watcher.RECIPE.exists() and watcher.FAILURE.exists()
    assert not watcher.RESULT.exists()
    before = watcher.digest(watcher.RECIPE)
    with pytest.raises(FileExistsError):
        watcher.run('seal')
    assert watcher.digest(watcher.RECIPE) == before


def test_concurrent_owner_cannot_write_fit_evidence(campaign, monkeypatch):
    monkeypatch.setattr(watcher.adapter, 'fit_recipe', lambda *args: pytest.fail('already owned'))
    with watcher.LOCK.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            watcher.run('seal')
    assert not watcher.INTENT.exists()


def test_verifier_binds_watcher_and_launcher_before_loading(tmp_path, monkeypatch):
    seal_path = tmp_path / 'seal.json'
    launcher = tmp_path / 'launch.py'
    launcher.write_text('def authenticate_saved_seal(expected):\n    return {"fully_authenticated": expected}\n')
    monkeypatch.setattr(watcher, 'SEAL', seal_path)
    monkeypatch.setattr(watcher, 'LAUNCHER', launcher)
    self_path = Path(watcher.__file__).resolve()
    sealed = {'files_sha256': {str(self_path): watcher.digest(self_path), str(launcher): watcher.digest(launcher)}}
    json_file(seal_path, sealed)
    expected = watcher.digest(seal_path)
    assert watcher.verify(expected) == {'fully_authenticated': expected}
    launcher.write_text('raise RuntimeError("changed code must not import")\n')
    with pytest.raises(ValueError, match='not bound'):
        watcher.verify(expected)


@pytest.mark.parametrize('expected', [None, '', 'NOT_READY', '0' * 64])
def test_verifier_rejects_wrong_or_missing_explicit_seal(tmp_path, monkeypatch, expected):
    monkeypatch.setattr(watcher, 'SEAL', json_file(tmp_path / 'seal.json', {}))
    with pytest.raises(ValueError, match='explicit Python v6 implementation seal hash differs'):
        watcher.verify(expected)
