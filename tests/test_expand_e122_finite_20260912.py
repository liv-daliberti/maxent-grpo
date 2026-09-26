import importlib.util
from pathlib import Path
import pytest
p=Path(__file__).resolve().parents[1]/'ops/exp_scaling/expand_e122_finite_20260912.py'
s=importlib.util.spec_from_file_location('finite_e122',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_unknown_and_cap_blocks_cannot_be_bypassed():
    for reason in ['unknown_scheduler_state','concurrency_cap','needs_operator_review','waiting_disk']:
        assert m.select_next({'blocked_reason':reason,'staged_held_job_ids':['31254498']},m.CANDIDATES) is None

def test_selection_never_releases_unregistered_or_already_released_jobs():
    status={'blocked_reason':None,'staged_held_job_ids':['999','31158700']}
    assert m.select_next(status,m.CANDIDATES)=='31158700'
    assert m.select_next(status,['31254498']) is None

def test_pending_array_keeps_full_reserve_and_unknown_writer_fails(monkeypatch):
    b=m.burst
    monkeypatch.setattr(b,'CAP',11)
    monkeypatch.setitem(b.ARRAYS,m.ARRAY[0],m.ARRAY[1:])
    monkeypatch.setattr(b.base,'canonical_registry',lambda:{})
    monkeypatch.setattr(b,'run',lambda _:f'UserId=user({m.os.getuid()}) WorkDir={m.ROOT} {m.ARRAY[2]}')
    monkeypatch.setattr(b,'digest',lambda _:'x')
    report=b.storage_budget({'jobs':[]},queue='31254520_6|PENDING|ReqNodeNotAvail|gres/gpu:a5000:2')
    assert report['required_bytes']==(64+125+16*11+32)*m.GIB
    with pytest.raises(ValueError,match='unregistered'):
        b.storage_budget({'jobs':[]},queue='31254520_7|PENDING|Resources|gres/gpu:1')
