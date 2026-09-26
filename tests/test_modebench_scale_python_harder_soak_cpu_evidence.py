"""Only scratch paths are staged; production helper and history stay unchanged."""
import hashlib
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'soak_cpu_evidence_under_test',
    ROOT/'artifacts/stage_modebench_scale_python_harder_soak_cpu_evidence_20260913.py')
a = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(a)


@pytest.fixture
def s(tmp_path, monkeypatch):
    # Private module instance: never rewrite the sealed predecessor on disk.
    helper = a.predecessor()
    shared = tmp_path/'shared.json'
    data = b'{"scratch":"same exact diagnostic"}\n'
    shared.write_bytes(data)
    shared.chmod(0o444)
    temp = tmp_path/'temp'
    temp.mkdir()
    destination = temp/'historical-capacity/analysis.json'
    monkeypatch.setattr(helper, 'SHARED', shared)
    monkeypatch.setattr(helper, 'DESTINATION', destination)
    monkeypatch.setattr(helper, 'TEMP_ROOT', temp)
    monkeypatch.setattr(helper, 'SHA256', helper.digest(data))
    monkeypatch.setattr(a, 'predecessor', lambda: helper)
    monkeypatch.setattr(a, 'host_identity', lambda: ('soak.cs.princeton.edu', os.getuid()))
    return SimpleNamespace(helper=helper, shared=shared, data=data, temp=temp, destination=destination)


def test_actual_sealed_predecessor_import_is_readonly_and_policy_stays_unchanged():
    before = a.PREDECESSOR.stat()
    helper = a.predecessor()
    assert helper.HOSTS == ('spin', 'node202', 'node203', 'node204')
    assert helper.contract() == a.EXACT_CONTRACT
    after = a.PREDECESSOR.stat()
    assert (before.st_ino, before.st_mtime_ns, before.st_mode) == (after.st_ino, after.st_mtime_ns, after.st_mode)
    assert hashlib.sha256(a.PREDECESSOR.read_bytes()).hexdigest() == a.PREDECESSOR_SHA256


def test_atomic_exact_copy_and_idempotent_existing_file(s):
    first = a.stage()
    assert first['schema'] == a.SCHEMA
    assert first['observer_host'] == 'soak.cs.princeton.edu'
    assert first['observer_uid'] == os.getuid()
    assert first['observer_user'] == a.USER
    assert first['predecessor']['sha256'] == a.PREDECESSOR_SHA256
    assert first['exact_evidence_contract'] == s.helper.contract()
    assert first['purpose'] == 'same_user_cpu_only_saved_output_audit_diagnostic'
    assert first['grader_invocations'] == 0
    assert first['model_sampling_performed'] is False
    assert first['historical_records_changed'] is False
    assert first['created_destination'] is True
    assert s.destination.read_bytes() == s.data
    assert s.destination.stat().st_mode & 0o777 == 0o444
    before = s.destination.stat()
    second = a.stage()
    assert second['created_destination'] is False
    assert second['status'] == 'verified_existing_exact_diagnostic'
    assert second['destination_stat'] == first['destination_stat']
    assert (s.destination.stat().st_ino, s.destination.stat().st_mtime_ns) == (before.st_ino, before.st_mtime_ns)
    assert [p.name for p in s.destination.parent.iterdir()] == ['analysis.json']
    assert s.helper.HOSTS == ('spin', 'node202', 'node203', 'node204')


def test_matching_owner_writable_historical_file_keeps_metadata(s):
    s.destination.parent.mkdir(mode=0o700)
    s.destination.write_bytes(s.data)
    s.destination.chmod(0o644)
    before = s.destination.stat()
    assert a.stage()['created_destination'] is False
    after = s.destination.stat()
    assert (before.st_ino, before.st_mtime_ns, before.st_mode) == (after.st_ino, after.st_mtime_ns, after.st_mode)


def test_existing_mismatch_is_never_overwritten(s):
    s.destination.parent.mkdir(mode=0o700)
    s.destination.write_bytes(b'other evidence')
    with pytest.raises(ValueError, match='never overwrite'):
        a.stage()
    assert s.destination.read_bytes() == b'other evidence'
    assert [p.name for p in s.destination.parent.iterdir()] == ['analysis.json']


@pytest.mark.parametrize('kind', ['wrong_hash', 'writable', 'absent', 'symlink'])
def test_shared_evidence_fails_before_destination_creation(s, kind):
    if kind == 'wrong_hash':
        s.shared.chmod(0o644)
        s.shared.write_bytes(b'different')
        s.shared.chmod(0o444)
    elif kind == 'writable':
        s.shared.chmod(0o644)
    elif kind == 'absent':
        s.shared.unlink()
    else:
        target = s.shared.with_name('target')
        s.shared.rename(target)
        s.shared.symlink_to(target)
    with pytest.raises((ValueError, OSError)):
        a.stage()
    assert not s.destination.parent.exists()


@pytest.mark.parametrize('kind', ['directory_symlink', 'file_symlink', 'fifo', 'public_directory', 'public_file'])
def test_unsafe_destination_rejected_without_replacement(s, kind):
    if kind == 'directory_symlink':
        other = s.temp/'elsewhere'
        other.mkdir()
        s.destination.parent.symlink_to(other, target_is_directory=True)
    else:
        s.destination.parent.mkdir(mode=0o700)
        if kind == 'file_symlink':
            s.destination.symlink_to(s.shared)
        elif kind == 'fifo':
            os.mkfifo(s.destination)
        elif kind == 'public_directory':
            s.destination.parent.chmod(0o777)
        else:
            s.destination.write_bytes(s.data)
            s.destination.chmod(0o666)
    with pytest.raises((ValueError, OSError)):
        a.stage()
    assert s.shared.read_bytes() == s.data


@pytest.mark.parametrize('matching', [True, False])
def test_concurrent_publication_only_accepts_exact_bytes(s, monkeypatch, matching):
    real_link = os.link

    def racing(src, dst, **kwargs):
        fd = os.open(dst, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444, dir_fd=kwargs['dst_dir_fd'])
        try:
            os.write(fd, s.data if matching else b'conflict')
        finally:
            os.close(fd)
        return real_link(src, dst, **kwargs)

    monkeypatch.setattr(a.os, 'link', racing)
    if matching:
        assert a.stage()['created_destination'] is False
    else:
        with pytest.raises(ValueError, match='never overwrite'):
            a.stage()
    assert s.destination.read_bytes() == (s.data if matching else b'conflict')
    assert [p.name for p in s.destination.parent.iterdir()] == ['analysis.json']


def test_directory_replacement_is_detected(s, monkeypatch):
    original_inspect = s.helper.inspect_destination
    replaced = False

    def inspect_then_replace(fd):
        nonlocal replaced
        result = original_inspect(fd)
        if not replaced:
            s.destination.parent.rename(s.temp/'original_directory')
            s.destination.parent.mkdir(mode=0o700)
            replaced = True
        return result

    monkeypatch.setattr(s.helper, 'inspect_destination', inspect_then_replace)
    with pytest.raises(ValueError, match='directory path changed'):
        a.stage()
    assert not s.destination.exists()
    assert (s.temp/'original_directory/analysis.json').read_bytes() == s.data


def test_deeper_destination_fails_before_writes(s, monkeypatch):
    monkeypatch.setattr(s.helper, 'DESTINATION', s.temp/'deep/extra/analysis.json')
    with pytest.raises(ValueError, match='fixed /tmp'):
        a.stage()
    assert not (s.temp/'deep').exists()


@pytest.mark.parametrize('host', ['spin.cs.princeton.edu', 'wash.cs.princeton.edu', 'node202', 'soak.attacker.example'])
def test_host_rejected_before_read_or_stage(monkeypatch, host):
    monkeypatch.setattr(a.socket, 'gethostname', lambda: host)
    monkeypatch.setattr(a.os, 'uname', lambda: SimpleNamespace(nodename=host))
    monkeypatch.setattr(a, 'predecessor', lambda: pytest.fail('predecessor must not be read on wrong host'))
    with pytest.raises(ValueError, match='exact soak'):
        a.stage()


@pytest.mark.parametrize('kind', ['disagreeing_hostname', 'uid', 'euid', 'user'])
def test_consistent_actual_host_and_same_user_required(monkeypatch, kind):
    monkeypatch.setattr(a.socket, 'gethostname', lambda: a.HOST)
    monkeypatch.setattr(a.os, 'uname', lambda: SimpleNamespace(nodename=a.HOST if kind != 'disagreeing_hostname' else 'spin.cs.princeton.edu'))
    monkeypatch.setattr(a.os, 'getuid', lambda: a.UID if kind != 'uid' else a.UID+1)
    monkeypatch.setattr(a.os, 'geteuid', lambda: a.UID if kind != 'euid' else a.UID+1)
    monkeypatch.setattr(a.pwd, 'getpwuid', lambda uid: SimpleNamespace(pw_name=a.USER if kind != 'user' else 'someone_else'))
    monkeypatch.setattr(a, 'predecessor', lambda: pytest.fail('no read on failed observer identity'))
    with pytest.raises(ValueError):
        a.stage()


@pytest.mark.parametrize('kind', ['bytes', 'writable', 'symlink'])
def test_predecessor_source_is_pinned_immutable_and_canonical(tmp_path, monkeypatch, kind):
    candidate = tmp_path/'predecessor.py'
    candidate.write_bytes(a.PREDECESSOR.read_bytes() if kind != 'bytes' else b'raise AssertionError("never execute changed bytes")')
    candidate.chmod(0o444 if kind != 'writable' else 0o644)
    if kind == 'symlink':
        target = candidate.with_name('target.py')
        candidate.rename(target)
        candidate.symlink_to(target)
    monkeypatch.setattr(a, 'PREDECESSOR', candidate)
    with pytest.raises(ValueError):
        a.predecessor()
