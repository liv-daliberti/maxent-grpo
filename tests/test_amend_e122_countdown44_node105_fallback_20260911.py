from pathlib import Path
from contextlib import nullcontext
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ops/exp_scaling'))
import amend_e122_countdown44_node105_fallback_20260911 as m


def test_fallback_requires_definitive_account_rejection(monkeypatch,tmp_path):
    monkeypatch.setattr(m.reviewed,'ART',tmp_path)
    (tmp_path/'31158682.result.json').write_text('{}')
    result={'returncode':1,'error':None,'stderr':'Account may not be modified after submission'}
    monkeypatch.setattr(m.recovery,'read',lambda p:result)
    assert m.rejected_account_attempt().name=='31158682.result.json'
    result['returncode']=None
    with pytest.raises(RuntimeError):m.rejected_account_attempt()


def test_consumed_fallback_never_calls_scheduler(monkeypatch,tmp_path):
    monkeypatch.setattr(m,'ART',tmp_path)
    monkeypatch.setattr(m,'verify',lambda pin:{})
    monkeypatch.setattr(m.recovery,'admission_locks',nullcontext)
    monkeypatch.setattr(m.c,'locked',lambda path:nullcontext())
    (tmp_path/'apply.intent.json').write_text('{}')
    monkeypatch.setattr(m.c,'command',lambda *a:pytest.fail('Repeated mutation'))
    with pytest.raises(RuntimeError,match='never repeat'):m.apply('pin')
