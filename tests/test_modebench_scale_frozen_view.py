"""Guard tests for the additive single-file PRoot source-view adapter."""
import hashlib
import importlib.util
import json
import os
import py_compile
from pathlib import Path
import shutil
import socket
import stat
import subprocess
import sys

import pytest

RUNNER = Path(__file__).resolve().parents[1] / 'artifacts/run_modebench_scale_frozen_view_20260912.py'
spec = importlib.util.spec_from_file_location('modebench_scale_frozen_view_runner', RUNNER)
view = importlib.util.module_from_spec(spec)
spec.loader.exec_module(view)
REAL_PROOT = view.PROOT
REAL_PROOT_SHA = view.PROOT_SHA256


@pytest.fixture
def fixture_view(tmp_path, monkeypatch):
    root = tmp_path / 'workspace'
    artifact_parent = root / 'artifacts'
    artifact_parent.mkdir(parents=True)
    canonical = root / 'src/oat_drgrpo/templates.py'
    canonical.parent.mkdir(parents=True)
    canonical.write_text("VALUE = 'neutral'\n")
    preserved = artifact_parent / 'preserved.py'
    preserved.write_text("VALUE = 'frozen!'\n")
    runner = artifact_parent / 'runner.py'
    runner.write_text('# immutable runner fixture\n')
    binary = tmp_path / 'proot'
    binary.write_text('# fake binary, never executed\n')
    binary.chmod(0o755)
    for name, value in {'ROOT': root, 'RUNNER': runner, 'CANONICAL': canonical,
                        'PRESERVED': preserved, 'PROOT': binary}.items():
        monkeypatch.setattr(view, name, value)
    for name, path in {'NEUTRAL_SHA256': canonical, 'OLD_SHA256': preserved,
                       'PROOT_SHA256': binary}.items():
        monkeypatch.setattr(view, name, view.file_sha(path))
    yield root, artifact_parent / 'fresh_view'
    # Restore directory access for pytest cleanup, including rejected fixtures.
    for directory in root.rglob('*'):
        if directory.is_dir():
            directory.chmod(0o700)


def seal(fixture_view):
    _, directory = fixture_view
    result = view.prepare(directory)
    return directory, Path(result['manifest'])


def rewrite_manifest(directory, mutate):
    path = directory / view.MANIFEST_NAME
    value = json.loads(path.read_text())
    mutate(value)
    path.chmod(0o600)
    path.write_text(json.dumps(value))
    path.chmod(0o444)
    checksum = directory / view.CHECKSUM_NAME
    checksum.chmod(0o600)
    checksum.write_text(view.file_sha(path) + '  ' + view.MANIFEST_NAME + '\n')
    checksum.chmod(0o444)


def test_prepare_preserves_host_and_seals_independent_copy(fixture_view):
    before = view.CANONICAL.read_bytes()
    directory, manifest = seal(fixture_view)
    value = view.authenticate(manifest)
    assert view.CANONICAL.read_bytes() == before
    assert value['prepared_host'] == socket.gethostname()
    assert value['mappings'] == [{'source': str(directory / view.COPY_NAME),
                                  'destination': str(view.CANONICAL),
                                  'sha256': view.OLD_SHA256}]
    assert (directory / view.COPY_NAME).stat().st_ino != view.PRESERVED.stat().st_ino
    assert stat.S_IMODE(directory.stat().st_mode) == 0o555
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o444 for path in directory.iterdir())
    assert set(value) == {'schema', 'artifact_root', 'workspace', 'prepared_at_utc',
                          'prepared_host', 'runner', 'proot', 'preserved_source',
                          'host_source', 'mappings', 'limitations'}
    with pytest.raises(ValueError, match='fresh view'):
        view.prepare(directory)


@pytest.mark.parametrize('target', ['CANONICAL', 'PRESERVED', 'PROOT', 'RUNNER'])
def test_changed_host_input_fails_closed(fixture_view, target):
    directory, manifest = seal(fixture_view)
    getattr(view, target).write_text('changed')
    with pytest.raises(ValueError, match='digest mismatch|manifest fields'):
        view.authenticate(manifest)


def test_rejects_changed_copy_even_with_original_mode(fixture_view):
    directory, manifest = seal(fixture_view)
    copy = directory / view.COPY_NAME
    copy.chmod(0o600)
    copy.write_text('changed')
    copy.chmod(0o444)
    with pytest.raises(ValueError, match='digest mismatch'):
        view.authenticate(manifest)


@pytest.mark.parametrize('mode_target', ['directory', 'copy', 'manifest', 'checksum'])
def test_rejects_unsealed_modes(fixture_view, mode_target):
    directory, manifest = seal(fixture_view)
    target = {'directory': directory, 'copy': directory / view.COPY_NAME,
              'manifest': manifest, 'checksum': directory / view.CHECKSUM_NAME}[mode_target]
    target.chmod(0o755 if mode_target == 'directory' else 0o644)
    with pytest.raises(ValueError, match='mode'):
        view.authenticate(manifest)


@pytest.mark.parametrize('mutation', [
    lambda m: m['mappings'].append(dict(m['mappings'][0])),
    lambda m: m['mappings'][0].update(destination='/tmp/elsewhere.py'),
    lambda m: m.update(unapproved_field=True),
    lambda m: m.update(workspace='/tmp'),
    lambda m: m['runner'].update(path='/tmp/runner.py'),
    lambda m: m['proot'].update(sha256='0' * 64),
])
def test_checksum_rewrite_cannot_expand_mapping_or_change_pins(fixture_view, mutation):
    directory, manifest = seal(fixture_view)
    rewrite_manifest(directory, mutation)
    with pytest.raises(ValueError, match='manifest fields'):
        view.authenticate(manifest)


def test_checksum_and_extra_file_guards(fixture_view):
    directory, manifest = seal(fixture_view)
    checksum = directory / view.CHECKSUM_NAME
    checksum.chmod(0o600)
    checksum.write_text('not a checksum\n')
    checksum.chmod(0o444)
    with pytest.raises(ValueError, match='checksum mismatch'):
        view.authenticate(manifest)
    directory.chmod(0o755)
    (directory / 'unexpected').touch()
    directory.chmod(0o555)
    with pytest.raises(ValueError, match='exactly'):
        view.authenticate(manifest)


def test_rejects_symlink_root_and_copy(fixture_view):
    root, directory = fixture_view
    link = root / 'artifacts/link'
    link.symlink_to(root / 'artifacts', target_is_directory=True)
    with pytest.raises(ValueError, match='canonical path'):
        view.prepare(link / 'new')
    directory, manifest = seal(fixture_view)
    directory.chmod(0o755)
    (directory / view.COPY_NAME).unlink()
    (directory / view.COPY_NAME).symlink_to(view.PRESERVED)
    directory.chmod(0o555)
    with pytest.raises(ValueError, match='canonical path'):
        view.authenticate(manifest)


def test_rejects_hard_link_copy(fixture_view):
    directory, manifest = seal(fixture_view)
    directory.chmod(0o755)
    (directory / view.COPY_NAME).unlink()
    os.link(view.PRESERVED, directory / view.COPY_NAME)
    directory.chmod(0o555)
    with pytest.raises(ValueError, match='one hard link'):
        view.authenticate(manifest)


@pytest.mark.parametrize('root_kind', ['relative', 'outside', 'artifacts_itself'])
def test_prepare_rejects_unapproved_root(fixture_view, root_kind, tmp_path):
    root, _ = fixture_view
    path = {'relative': Path('relative'), 'outside': tmp_path / 'elsewhere',
            'artifacts_itself': root / 'artifacts'}[root_kind]
    with pytest.raises(ValueError, match='canonical path|strictly under'):
        view.prepare(path)


def test_immutable_new_never_overwrites(tmp_path):
    path = tmp_path / 'existing'
    path.write_bytes(b'keep')
    with pytest.raises(FileExistsError):
        view.immutable_new(path, b'replacement')
    assert path.read_bytes() == b'keep'


@pytest.mark.parametrize('argv', [[], 'echo hi', ['relative-command'], ['/nonexistent'],
                                  ['/bin/echo', None], ['/bin/echo', 'bad\x00arg']])
def test_bad_command_argv_rejected(argv):
    with pytest.raises(ValueError):
        view.command_argv(argv)


@pytest.mark.parametrize('options', [
    ['-E'], ['-I'], ['-BE'], ['-X', 'pycache_prefix=/stale'],
    ['-Xpycache_prefix=/stale'], ['-W', 'ignore', '-I'],
    ['--check-hash-based-pycs', 'always', '-I'],
])
def test_python_cache_bypass_options_rejected(options):
    with pytest.raises(ValueError, match='cache'):
        view.command_argv([sys.executable, *options, '-c', 'pass'])


@pytest.mark.parametrize('option', ['-BXpycache_prefix=/stale', '-cprint(1)', '--unknown', '-W', '-X',
                                   '--check-hash-based-pycs', '--', '-'])
def test_unknown_or_combined_python_option_forms_fail_closed(option):
    with pytest.raises(ValueError, match='unsupported Python option'):
        view.command_argv([sys.executable, option, '-c', 'pass'])


def test_python_payload_is_not_mistaken_for_flags():
    command = [sys.executable, '-B', '-u', '-c', '-I']
    assert view.command_argv(command) == [sys.executable, '-B', *command[1:]]


@pytest.mark.parametrize('terminal', ['-c', '-m'])
def test_python_terminal_options_require_payload(terminal):
    with pytest.raises(ValueError, match='explicit payload'):
        view.command_argv([sys.executable, terminal])
    assert view.command_argv([sys.executable, '-u', terminal, 'payload']) == [
        sys.executable, '-B', '-u', terminal, 'payload']


def test_exec_argv_environment_are_explicit_and_private(fixture_view, monkeypatch):
    directory, manifest = seal(fixture_view)
    monkeypatch.setenv('PYTHONDONTWRITEBYTECODE', '0')
    monkeypatch.setenv('PYTHONPYCACHEPREFIX', '/stale/cache')
    monkeypatch.setenv('VIEW_TEST_KEEP', 'yes')
    literal = 'a; $(touch /tmp/do-not-run) and spaces'
    executable, argv, environment = view.execution(manifest, [sys.executable, '-c', literal])
    second = view.execution(manifest, [sys.executable, '-c', 'pass'])
    try:
        assert executable == str(view.PROOT)
        assert argv == [str(view.PROOT), '-r', '/', '-b',
                        str(directory / view.COPY_NAME) + ':' + str(view.CANONICAL),
                        '-w', str(view.ROOT), sys.executable, '-B', '-c', literal]
        assert environment['PYTHONDONTWRITEBYTECODE'] == '1'
        assert environment['VIEW_TEST_KEEP'] == 'yes'
        assert environment['PYTHONPYCACHEPREFIX'] != '/stale/cache'
        assert environment['PYTHONPYCACHEPREFIX'] != second[2]['PYTHONPYCACHEPREFIX']
        assert not Path(environment['PYTHONPYCACHEPREFIX']).exists()
        assert stat.S_IMODE(Path(environment['PYTHONPYCACHEPREFIX']).parent.stat().st_mode) == 0o700
        assert os.environ['PYTHONPYCACHEPREFIX'] == '/stale/cache'
    finally:
        shutil.rmtree(Path(environment['PYTHONPYCACHEPREFIX']).parent)
        shutil.rmtree(Path(second[2]['PYTHONPYCACHEPREFIX']).parent)


@pytest.mark.skipif(os.environ.get('MODEBENCH_FROZEN_VIEW_REAL_PROOT_TEST') != '1',
                    reason='explicit opt-in tiny real PRoot probe')
def test_real_tiny_view_import_and_child_inheritance(fixture_view, monkeypatch):
    assert REAL_PROOT.is_file() and view.file_sha(REAL_PROOT) == REAL_PROOT_SHA
    monkeypatch.setattr(view, 'PROOT', REAL_PROOT)
    monkeypatch.setattr(view, 'PROOT_SHA256', REAL_PROOT_SHA)
    before = view.CANONICAL.read_bytes()
    directory, manifest = seal(fixture_view)
    # Make host-neutral bytecode valid by timestamp and size for the mapped file.
    # A missing private cache prefix could silently import this stale code.
    monkeypatch.setattr(sys, 'pycache_prefix', None)
    py_compile.compile(str(view.CANONICAL), doraise=True)
    copy = directory / view.COPY_NAME
    assert len(copy.read_bytes()) == len(before)
    info = view.CANONICAL.stat()
    os.utime(copy, ns=(info.st_atime_ns, info.st_mtime_ns))
    assert Path(importlib.util.cache_from_source(str(view.CANONICAL))).is_file()
    script = (
        'import hashlib, importlib.util, json, os, pathlib, socket, subprocess, sys; '
        f'p=pathlib.Path({str(view.CANONICAL)!r}); '
        's=importlib.util.spec_from_file_location("probe_templates", p); '
        'm=importlib.util.module_from_spec(s); s.loader.exec_module(m); '
        'print(json.dumps({"value":m.VALUE,"file":str(pathlib.Path(m.__file__).resolve()),'
        '"sha":hashlib.sha256(p.read_bytes()).hexdigest(),'
        '"host":socket.gethostname(),"bytecode":sys.dont_write_bytecode,'
        '"child":subprocess.check_output(["/usr/bin/sha256sum",str(p)],text=True).split()[0]}))'
    )
    _, argv, environment = view.execution(manifest, [sys.executable, '-c', script])
    try:
        result = subprocess.run(argv, env=environment, text=True, capture_output=True, timeout=30)
        assert result.returncode == 0, result.stderr
        value = json.loads(result.stdout)
        assert value == {'value': 'frozen!', 'file': str(view.CANONICAL),
                         'sha': view.OLD_SHA256, 'child': view.OLD_SHA256,
                         'host': socket.gethostname(), 'bytecode': True}
        assert view.CANONICAL.read_bytes() == before
        assert view.authenticate(manifest)
    finally:
        shutil.rmtree(Path(environment['PYTHONPYCACHEPREFIX']).parent)
