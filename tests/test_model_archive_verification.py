"""Archive verification must fail closed before callers can consider cleanup."""
import copy
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('model_archive_verification', ROOT / 'ops/model_archive_verification.py')
v = importlib.util.module_from_spec(spec); spec.loader.exec_module(v)
COMMIT = 'a' * 40


@pytest.fixture
def exported(tmp_path):
    root = tmp_path / 'export'; root.mkdir()
    (root / 'weights').mkdir()
    (root / 'weights/model.safetensors').write_bytes(b'weights\0\x01\x02' * 1000)
    (root / 'config.json').write_bytes(b'{"model_type":"qwen2"}\n')
    (root / 'empty.txt').write_bytes(b'')
    return root


def test_manifest_stream_hashes_and_git_header(exported, monkeypatch):
    monkeypatch.setattr(v, 'CHUNK_BYTES', 7)
    manifest = v.build_manifest(exported, 'campaign/run/model')
    assert manifest['total_bytes'] == sum(p.stat().st_size for p in exported.rglob('*') if p.is_file())
    for item in manifest['files']:
        raw = Path(item['local_path']).read_bytes()
        assert item['sha256'] == hashlib.sha256(raw).hexdigest()
        assert item['git_blob_sha1'] == hashlib.sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest()
        assert item['path_in_repo'] == 'campaign/run/model/' + item['relative_path']
        assert set(item['stat']) == {'dev', 'ino', 'mtime_ns', 'ctime_ns'}
    assert v.verify_local_manifest(manifest)['status'] == 'verified'


@pytest.mark.parametrize('kind', ['file', 'directory', 'root'])
def test_symlinks_rejected_even_when_the_target_is_inside_export(exported, tmp_path, kind):
    if kind == 'file': (exported / 'link').symlink_to(exported / 'config.json')
    elif kind == 'directory': (exported / 'linked_dir').symlink_to(exported / 'weights', target_is_directory=True)
    else:
        link = tmp_path / 'linked_export'; link.symlink_to(exported, target_is_directory=True); exported = link
    with pytest.raises(v.ArchiveVerificationError, match='symlink'):
        v.build_manifest(exported)


@pytest.mark.parametrize('prefix', ['/absolute', '../outside', 'foo/../bar', 'foo//bar', 'foo\\bar', './foo'])
def test_repository_escapes_and_noncanonical_paths_rejected(exported, prefix):
    with pytest.raises(v.ArchiveVerificationError, match='unsafe'):
        v.build_manifest(exported, prefix)


@pytest.mark.parametrize('mutation', ['content', 'replacement', 'add_file', 'symlink'])
def test_local_manifest_rejects_later_mutation(exported, mutation):
    manifest = v.build_manifest(exported)
    path = exported / 'config.json'
    if mutation == 'content': path.write_bytes(b'changed')
    elif mutation == 'replacement':
        content = path.read_bytes(); replacement = path.with_name('replacement.tmp'); replacement.write_bytes(content); replacement.replace(path)
    elif mutation == 'add_file': (exported / 'new.bin').write_bytes(b'new')
    else:
        path.unlink(); path.symlink_to(exported / 'empty.txt')
    assert v.verify_local_manifest(manifest)['status'] == 'failed'


@pytest.mark.parametrize('mutation', ['grow', 'shrink', 'replace'])
def test_mutation_during_streaming_hash_rejected(tmp_path, monkeypatch, mutation):
    root = tmp_path / 'export'; root.mkdir(); path = root / 'weights'; path.write_bytes(b'x' * 32)
    original = v.os.read
    changed = False
    def read(fd, count):
        nonlocal changed
        value = original(fd, count)
        if not changed and value:
            changed = True
            if mutation == 'grow':
                with path.open('ab') as stream: stream.write(b'y')
            elif mutation == 'shrink': path.write_bytes(b'z')
            else:
                path.unlink(); path.write_bytes(b'x' * 32)
        return value
    monkeypatch.setattr(v.os, 'read', read); monkeypatch.setattr(v, 'CHUNK_BYTES', 8)
    with pytest.raises(v.ArchiveVerificationError, match='grew|shrank|changed|replaced'):
        v.build_manifest(root)


def test_regular_file_requirement_rejects_fifo(tmp_path):
    v.os.mkfifo(tmp_path / 'fifo')
    with pytest.raises(v.ArchiveVerificationError, match='non-regular'):
        v.build_manifest(tmp_path)


def remote_files(manifest):
    result = []
    for local in manifest['files']:
        lfs = SimpleNamespace(sha256=local['sha256'], size=local['size']) if local['path_in_repo'].endswith('.safetensors') else None
        result.append(SimpleNamespace(path=local['path_in_repo'], size=local['size'],
                                      lfs=lfs, blob_id=local['git_blob_sha1']))
    return result


class Api:
    def __init__(self, records): self.records = records; self.calls = []
    def get_paths_info(self, **kwargs):
        self.calls.append(kwargs)
        return [r for r in self.records if r.path in kwargs['paths']]


def test_remote_verifies_lfs_and_regular_git_blobs_at_exact_commit(exported):
    manifest = v.build_manifest(exported, 'model')
    api = Api(remote_files(manifest))
    report = v.verify_remote_manifest(api, 'owner/repo', COMMIT, manifest, batch_size=1)
    assert report['status'] == 'verified' and len(report['verified_files']) == 3
    assert {r['verification'] for r in report['verified_files']} == {'lfs_sha256', 'git_blob_sha1'}
    assert len(api.calls) == 3
    assert all(call['revision'] == COMMIT and call['repo_type'] == 'model' and call['expand'] is False for call in api.calls)
    assert all('files_metadata' not in call for call in api.calls)


@pytest.mark.parametrize('problem', ['missing', 'size', 'lfs_sha', 'lfs_size', 'git_blob', 'folder', 'duplicate', 'unexpected'])
def test_remote_missing_or_mismatched_metadata_fails(exported, problem):
    manifest = v.build_manifest(exported)
    records = remote_files(manifest)
    binary = next(r for r in records if r.lfs)
    regular = next(r for r in records if r.lfs is None)
    if problem == 'missing': records.remove(binary)
    elif problem == 'size': binary.size += 1
    elif problem == 'lfs_sha': binary.lfs.sha256 = 'b'*64
    elif problem == 'lfs_size': binary.lfs.size += 1
    elif problem == 'git_blob': regular.blob_id = 'b'*40
    elif problem == 'folder': del regular.size; del regular.blob_id
    elif problem == 'duplicate': records.append(binary)
    else: records.append(SimpleNamespace(path='not/requested', size=0, lfs=None, blob_id='b'*40))
    api = Api(records)
    if problem == 'unexpected': api.get_paths_info = lambda **kwargs: records
    report = v.verify_remote_manifest(api, 'owner/repo', COMMIT, manifest)
    assert report['status'] == 'failed' and report['errors']


@pytest.mark.parametrize('revision', ['main', 'v1', '', 'a'*39, 'a'*41, '../main', None])
def test_remote_requires_commit_not_branch(exported, revision):
    manifest = v.build_manifest(exported); api = Api(remote_files(manifest))
    with pytest.raises(v.ArchiveVerificationError, match='40-hex commit'):
        v.verify_remote_manifest(api, 'owner/repo', revision, manifest)
    assert not api.calls


def test_remote_lookup_exception_never_becomes_success(exported):
    manifest = v.build_manifest(exported)
    class MissingCommit:
        def get_paths_info(self, **kwargs):
            assert kwargs['revision'] == COMMIT
            raise RuntimeError('revision not found')
    report = v.verify_remote_manifest(MissingCommit(), 'owner/repo', COMMIT, manifest)
    assert report['status'] == 'failed' and not report['verified_files']


def test_tampered_manifest_local_escape_and_duplicate_rejected(exported):
    manifest = v.build_manifest(exported)
    altered = copy.deepcopy(manifest); altered['files'][0]['local_path'] = '/tmp/escaped'
    with pytest.raises(v.ArchiveVerificationError, match='escapes'):
        v.verify_local_manifest(altered)
    altered = copy.deepcopy(manifest); altered['files'].append(altered['files'][0])
    with pytest.raises(v.ArchiveVerificationError, match='duplicate'):
        v.verify_remote_manifest(Api([]), 'owner/repo', COMMIT, altered)
