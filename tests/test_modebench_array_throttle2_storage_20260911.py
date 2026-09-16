"""The already applied throttle display cannot change charged task identities."""
from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ops/exp_scaling'))
import modebench_array_throttle2_storage_20260911 as a


@pytest.mark.parametrize('text,expected', [('9%2',[9]), ('6-9%2',[6,7,8,9]), ('1,4-6%2',[1,4,5,6])])
def test_amended_display_preserves_every_task(text,expected,monkeypatch):
    monkeypatch.setattr(a,'verify_amendment',lambda:None)
    original=a.original.array_indices
    with a.compatible_array_parser():
        assert a.original.array_indices(text)==expected
    assert a.original.array_indices is original


@pytest.mark.parametrize('text',['9%3','9%0','9%02','6-10%2','6,6%2','6-9%2%2',''])
def test_unknown_throttle_or_malformed_task_stays_rejected(text,monkeypatch):
    monkeypatch.setattr(a,'verify_amendment',lambda:None)
    with a.compatible_array_parser(), pytest.raises(ValueError):
        a.original.array_indices(text)


def test_parser_restored_after_exception(monkeypatch):
    monkeypatch.setattr(a,'verify_amendment',lambda:None)
    original=a.original.array_indices
    with pytest.raises(RuntimeError),a.compatible_array_parser():
        raise RuntimeError('fixture')
    assert a.original.array_indices is original


def test_missing_amendment_never_reaches_admission(monkeypatch):
    def fail():raise ValueError('throttle2 evidence changed')
    monkeypatch.setattr(a,'verify_amendment',fail)
    monkeypatch.setattr(a.original,'storage_report',lambda **kwargs:pytest.fail('reached admission'))
    report=a.storage_report()
    assert not report['allowed'] and report['blocked_reason']=='unresolved_storage_safety'


def test_calls_unchanged_admission_and_reservations(monkeypatch):
    monkeypatch.setattr(a,'verify_amendment',lambda:None)
    expected={'allowed':True,'required_bytes':123,'evaluation_future_reserve_bytes':192*1024**3}
    def budget(**kwargs):
        assert kwargs['include_held_job_ids']==['111']
        assert a.original.array_indices('6-9%2')==[6,7,8,9]
        return expected.copy()
    monkeypatch.setattr(a.original,'storage_report',budget)
    result=a.storage_report(include_held_job_ids=['111'])
    assert all(result[key]==value for key,value in expected.items())
