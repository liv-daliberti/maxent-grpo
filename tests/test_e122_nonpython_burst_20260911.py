import importlib.util
from pathlib import Path
import pytest
P=Path(__file__).resolve().parents[1]/'ops/exp_scaling/e122_nonpython_burst_20260911.py'
s=importlib.util.spec_from_file_location('e122_nonpython_burst',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_python_never_selected_even_when_first_held():
    campaign={'jobs':[{'job_id':'1','cell':{'domain':'python_factors'}},{'job_id':'2','cell':{'domain':'graph_coloring'}}]}
    status={'blocked_reason':None,'staged_held_job_ids':['1','2']}
    assert m.nonpython_next(status,campaign)=='2'
    status['staged_held_job_ids']=['1'];assert m.nonpython_next(status,campaign) is None
    status.update(blocked_reason='concurrency_cap',staged_held_job_ids=['2']);assert m.nonpython_next(status,campaign) is None

def test_budget_keeps_all_pending_inference_tasks(monkeypatch):
    monkeypatch.setattr(m.base,'canonical_registry',lambda:{})
    monkeypatch.setattr(m,'run',lambda _:f'UserId=user({m.os.getuid()}) WorkDir={m.ROOT} {m.ARRAYS["31252690"][1]}')
    monkeypatch.setattr(m,'digest',lambda _:'x')
    q='31252690_0|RUNNING|None|gres/gpu:1\n31252690_1|PENDING|Resources|gres/gpu:1'
    report=m.storage_budget({'jobs':[]},queue=q)
    assert len(report['reservations'])==2
    assert report['required_bytes']==(64+125+128+64)*m.GIB
    with pytest.raises(ValueError,match='unregistered'):m.storage_budget({'jobs':[]},queue='9_0|RUNNING|None|gres/gpu:1')

def test_cap_and_finite_burst_are_explicit():
    assert m.CAP==8 and m.MAX_RELEASES==4
