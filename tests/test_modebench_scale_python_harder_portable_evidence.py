"""Scratch exact-evidence staging; never writes the real historical /tmp path."""
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('portable_evidence_under_test', ROOT/'artifacts/stage_modebench_scale_python_harder_evidence_20260912.py')
a = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(a)


@pytest.fixture
def s(tmp_path, monkeypatch):
    shared = tmp_path/'shared.json'; data = b'{"scratch":"exact original bytes"}\n'
    shared.write_bytes(data); shared.chmod(0o444)
    temp = tmp_path/'temp'; temp.mkdir()
    destination = temp/'historical-capacity/analysis.json'
    monkeypatch.setattr(a,'SHARED',shared); monkeypatch.setattr(a,'DESTINATION',destination)
    monkeypatch.setattr(a,'TEMP_ROOT',temp); monkeypatch.setattr(a,'SHA256',a.digest(data))
    monkeypatch.setattr(a.socket,'gethostname',lambda:'node202.synthetic')
    return SimpleNamespace(shared=shared,destination=destination,data=data,temp=temp)


def test_exact_atomic_copy_then_idempotent_read_with_truthful_host(s):
    first = a.stage()
    assert first['created_destination'] and first['status']=='created_exact_diagnostic_copy'
    assert first['observer_host']=='node202.synthetic' and first['shared_source']==str(s.shared)
    assert first['historical_destination']==str(s.destination) and first['sha256']==a.digest(s.data)
    assert s.destination.read_bytes()==s.data and s.destination.stat().st_mode&0o777==0o444
    stat_before=s.destination.stat(); second=a.stage()
    assert second['created_destination'] is False and second['status']=='verified_existing_exact_diagnostic'
    assert second['destination_stat']==first['destination_stat']
    assert s.destination.stat().st_ino==stat_before.st_ino and s.destination.stat().st_mtime_ns==stat_before.st_mtime_ns
    assert [p.name for p in s.destination.parent.iterdir()]==['analysis.json']


def test_matching_historical_owner_writable_file_is_preserved(s):
    s.destination.parent.mkdir(mode=0o700);s.destination.write_bytes(s.data);s.destination.chmod(0o644)
    before=s.destination.stat(); result=a.stage()
    assert not result['created_destination'] and s.destination.stat().st_mode&0o777==0o644
    assert s.destination.stat().st_mtime_ns==before.st_mtime_ns


def test_existing_mismatch_is_never_replaced(s):
    s.destination.parent.mkdir(mode=0o700);s.destination.write_bytes(b'different')
    with pytest.raises(ValueError,match='never overwrite'):a.stage()
    assert s.destination.read_bytes()==b'different'
    assert [p.name for p in s.destination.parent.iterdir()]==['analysis.json']


@pytest.mark.parametrize('kind',['wrong_hash','writable_shared','missing_shared','shared_symlink'])
def test_shared_source_authenticated_before_any_destination_creation(s,kind):
    if kind=='wrong_hash':s.shared.chmod(0o644);s.shared.write_bytes(b'changed');s.shared.chmod(0o444)
    elif kind=='writable_shared':s.shared.chmod(0o644)
    elif kind=='missing_shared':s.shared.unlink()
    else:
        target=s.shared.with_name('target');s.shared.rename(target);s.shared.symlink_to(target)
    with pytest.raises((ValueError,OSError)):a.stage()
    assert not s.destination.parent.exists()


@pytest.mark.parametrize('kind',['directory_symlink','file_symlink','fifo','public_directory','public_file'])
def test_destination_types_and_permissions_fail_without_overwrite(s,kind):
    if kind=='directory_symlink':
        other=s.temp/'elsewhere';other.mkdir();s.destination.parent.symlink_to(other,target_is_directory=True)
    else:
        s.destination.parent.mkdir(mode=0o700)
        if kind=='file_symlink':s.destination.symlink_to(s.shared)
        elif kind=='fifo':os.mkfifo(s.destination)
        elif kind=='public_directory':s.destination.parent.chmod(0o777)
        else:s.destination.write_bytes(s.data);s.destination.chmod(0o666)
    with pytest.raises((ValueError,OSError)):a.stage()
    assert s.shared.read_bytes()==s.data


def test_wrong_owner_fails_without_creating_destination(s,monkeypatch):
    monkeypatch.setattr(a.os,'getuid',lambda:99999999)
    with pytest.raises(ValueError,match='owned shared'):a.stage()
    assert not s.destination.parent.exists()


@pytest.mark.parametrize('host',['wash.cs.princeton.edu','node105.ionic.cs.princeton.edu'])
def test_unapproved_host_never_stages(s,monkeypatch,host):
    monkeypatch.setattr(a.socket,'gethostname',lambda:host)
    with pytest.raises(ValueError,match='authorized'):a.stage()
    assert not s.destination.parent.exists()


@pytest.mark.parametrize('matching',[True,False])
def test_atomic_publish_race_accepts_only_exact_competing_bytes(s,monkeypatch,matching):
    real_link=a.os.link
    def racing(src,dst,**kwargs):
        fd=os.open(dst,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o444,dir_fd=kwargs['dst_dir_fd'])
        try:os.write(fd,s.data if matching else b'conflict')
        finally:os.close(fd)
        return real_link(src,dst,**kwargs)
    monkeypatch.setattr(a.os,'link',racing)
    if matching:
        result=a.stage();assert result['created_destination'] is False
    else:
        with pytest.raises(ValueError,match='never overwrite'):a.stage()
    assert [p.name for p in s.destination.parent.iterdir()]==['analysis.json']
    assert s.destination.read_bytes()==(s.data if matching else b'conflict')


def test_no_arbitrary_destination_outside_fixed_temp_child(s,monkeypatch):
    monkeypatch.setattr(a,'DESTINATION',s.temp/'deep/extra/analysis.json')
    with pytest.raises(ValueError,match='fixed'):a.stage()
    assert not (s.temp/'deep').exists()
