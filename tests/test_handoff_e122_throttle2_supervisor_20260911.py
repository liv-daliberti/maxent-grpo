"""Verify CPU handoff ordering and that a failed gate never stops the old worker."""
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ops/exp_scaling'))
import handoff_e122_throttle2_supervisor_20260911 as m


def fixtures(monkeypatch, tmp_path):
    events = []
    monkeypatch.setattr(m, 'ART', tmp_path)
    monkeypatch.setattr(m, 'verify', lambda expected: {})
    monkeypatch.setattr(m.old, 'read', lambda path: {'job_id': '999', 'plan_sha256': 'pin'})
    monkeypatch.setattr(m.old, 'admission_locks', nullcontext)
    monkeypatch.setattr(m.c, 'locked', lambda path: nullcontext())
    monkeypatch.setattr(m, 'old_cpu', lambda plan: events.append('old_identity') or 'old')
    def cpu(jid, expected, held=False, dependency=False):
        assert jid == '999' and expected == 'pin' and dependency
        events.append('held_audit' if held else 'released_gate_audit')
        return 'JobState=PENDING Priority=5'
    monkeypatch.setattr(m, 'cpu_record', cpu)
    def command(argv):
        events.append(argv)
        return SimpleNamespace(returncode=0, stdout='', stderr='')
    monkeypatch.setattr(m.c, 'command', command)
    return events


def test_exact_old_cpu_cancel_follows_released_dependency_gate(monkeypatch, tmp_path):
    events = fixtures(monkeypatch, tmp_path)
    m.activate('pin')
    assert events == ['old_identity', 'held_audit', ['scontrol', 'release', '999'],
                      'released_gate_audit', 'old_identity', ['scancel', '31245514']]


def test_release_failure_preserves_old_cpu(monkeypatch, tmp_path):
    events = fixtures(monkeypatch, tmp_path)
    def failed(argv):
        events.append(argv)
        return SimpleNamespace(returncode=1, stdout='', stderr='failed')
    monkeypatch.setattr(m.c, 'command', failed)
    with pytest.raises(RuntimeError, match='Release uncertain'):
        m.activate('pin')
    assert ['scancel', '31245514'] not in events


def test_missing_successor_dependency_preserves_old_cpu(monkeypatch, tmp_path):
    events = fixtures(monkeypatch, tmp_path)
    def missing(jid, expected, held=False, dependency=False):
        if not held: raise RuntimeError('Exact predecessor gate absent')
        return 'held'
    monkeypatch.setattr(m, 'cpu_record', missing)
    with pytest.raises(RuntimeError, match='predecessor gate'):
        m.activate('pin')
    assert ['scancel', '31245514'] not in events


def test_changed_predecessor_blocks_all_scheduler_mutation(monkeypatch, tmp_path):
    events = fixtures(monkeypatch, tmp_path)
    def changed(plan): raise RuntimeError('Old CPU command changed')
    monkeypatch.setattr(m, 'old_cpu', changed)
    with pytest.raises(RuntimeError, match='Old CPU command'):
        m.activate('pin')
    assert not events


def test_full_submission_extraction_covers_wrap_arguments():
    rec = 'JobId=1 SubmitLine=sbatch --wrap=exec python /exact.py watch --pin abc WorkDir=/tmp Comment=x'
    assert m.submission(rec) == 'sbatch --wrap=exec python /exact.py watch --pin abc'
    with pytest.raises(RuntimeError, match='Missing full submission'):
        m.submission('JobId=1')
