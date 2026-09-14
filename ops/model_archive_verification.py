#!/usr/bin/env python3
"""Read-only local manifests and commit-pinned Hugging Face archive verification.

No upload, deletion, credential discovery, or repository mutation occurs here.
The caller supplies an authenticated API object and must require status=verified.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any

SCHEMA = 'model-archive-file-manifest-v1'
CHUNK_BYTES = 8 * 1024**2


class ArchiveVerificationError(ValueError):
    """A local path or manifest cannot safely identify stable regular files."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ArchiveVerificationError(message)


def _repo_path(value: str, *, allow_empty: bool = False) -> str:
    _require(isinstance(value, str), 'repository path must be text')
    if allow_empty and not value:
        return value
    path = PurePosixPath(value)
    _require(bool(value) and not path.is_absolute() and '\\' not in value
             and str(path) == value and all(p not in ('', '.', '..') for p in value.split('/')),
             f'unsafe repository-relative path: {value!r}')
    return value


def _identity(info: os.stat_result) -> dict[str, int]:
    return {'dev': info.st_dev, 'ino': info.st_ino, 'mtime_ns': info.st_mtime_ns,
            'ctime_ns': info.st_ctime_ns}


def _root(export_dir: str | Path) -> Path:
    source = Path(export_dir).absolute()
    _require(not source.is_symlink(), 'export directory is a symlink')
    result = source.resolve(strict=True)
    _require(result.is_dir(), 'export directory must be a directory')
    return result


def _inventory(root: Path) -> list[str]:
    files = []
    for current, directories, names in os.walk(root, followlinks=False, onerror=lambda error: (_ for _ in ()).throw(error)):
        for name in directories + names:
            path = Path(current) / name
            info = path.lstat()
            _require(not stat.S_ISLNK(info.st_mode), f'symlink beneath export directory: {path}')
            if name in directories:
                _require(stat.S_ISDIR(info.st_mode), f'non-directory entry: {path}')
            else:
                _require(stat.S_ISREG(info.st_mode), f'non-regular file: {path}')
                files.append(path.relative_to(root).as_posix())
    return sorted(files)


@contextmanager
def _open_regular_beneath(root: Path, relative: str):
    """Open each path component without following symlinks, including races."""
    parts = PurePosixPath(_repo_path(relative)).parts
    parent = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    descriptor = None
    try:
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            os.close(parent)
            parent = child
        descriptor = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        _require(stat.S_ISREG(os.fstat(descriptor).st_mode), f'not a regular file: {relative}')
        yield descriptor, parent, parts[-1]
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent)


def _hash_file(root: Path, relative: str, path_in_repo: str) -> dict[str, Any]:
    with _open_regular_beneath(root, relative) as (descriptor, parent, name):
        before = os.fstat(descriptor)
        sha256 = hashlib.sha256()
        blob = hashlib.sha1(f'blob {before.st_size}\0'.encode('ascii'))
        remaining = before.st_size
        while remaining:
            data = os.read(descriptor, min(CHUNK_BYTES, remaining))
            _require(bool(data), f'file shrank while hashing: {relative}')
            sha256.update(data); blob.update(data)
            remaining -= len(data)
        _require(not os.read(descriptor, 1), f'file grew while hashing: {relative}')
        after = os.fstat(descriptor)
        named = os.stat(name, dir_fd=parent, follow_symlinks=False)
        _require(stat.S_ISREG(named.st_mode) and before.st_size == after.st_size == named.st_size
                 and _identity(before) == _identity(after) == _identity(named),
                 f'file changed or was replaced while hashing: {relative}')
    return {'local_path': str(root / relative), 'relative_path': relative,
            'path_in_repo': path_in_repo, 'size': before.st_size,
            'sha256': sha256.hexdigest(), 'git_blob_sha1': blob.hexdigest(), 'stat': _identity(before)}


def build_manifest(export_dir: str | Path, repo_prefix: str = '') -> dict[str, Any]:
    """Stream every regular file once; reject symlinks, mutation and empty exports."""
    root = _root(export_dir)
    prefix = _repo_path(repo_prefix, allow_empty=True)
    before = _identity(root.stat())
    names = _inventory(root)
    _require(bool(names), 'export directory contains no regular files')
    records = [_hash_file(root, name, f'{prefix}/{name}' if prefix else name) for name in names]
    _require(before == _identity(root.stat()) and _inventory(root) == names,
             'export directory changed while building manifest')
    return {'schema': SCHEMA, 'export_dir': str(root), 'repo_prefix': prefix,
            'files': records, 'total_bytes': sum(r['size'] for r in records)}


def _validate_manifest(manifest: dict[str, Any]) -> tuple[Path, list[dict[str, Any]]]:
    _require(isinstance(manifest, dict) and manifest.get('schema') == SCHEMA, 'unknown archive manifest schema')
    root = Path(manifest.get('export_dir', ''))
    _require(root.is_absolute() and str(root) == os.path.normpath(str(root)), 'manifest export root must be absolute and normalized')
    prefix = _repo_path(manifest.get('repo_prefix', ''), allow_empty=True)
    files = manifest.get('files')
    _require(isinstance(files, list) and bool(files), 'manifest must contain files')
    seen = set()
    for record in files:
        _require(isinstance(record, dict), 'malformed file record')
        relative = _repo_path(record.get('relative_path'))
        remote = _repo_path(record.get('path_in_repo'))
        _require(remote == (f'{prefix}/{relative}' if prefix else relative), 'repository path differs from manifest prefix')
        _require(record.get('local_path') == str(root / relative), 'local file escapes or differs from export root')
        _require(relative not in seen, 'duplicate manifest file')
        seen.add(relative)
        _require(type(record.get('size')) is int and record['size'] >= 0, 'invalid manifest size')
        for key, length in (('sha256', 64), ('git_blob_sha1', 40)):
            _require(isinstance(record.get(key), str) and re.fullmatch(f'[0-9a-f]{{{length}}}', record[key]) is not None,
                     f'invalid {key}')
        identity = record.get('stat')
        _require(isinstance(identity, dict) and set(identity) == {'dev', 'ino', 'mtime_ns', 'ctime_ns'}
                 and all(type(v) is int for v in identity.values()), 'invalid local stat identity')
    _require(manifest.get('total_bytes') == sum(r['size'] for r in files), 'manifest total size differs')
    return root, files


def verify_local_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Rehash the export and compare original stat identities and full inventory."""
    root, files = _validate_manifest(manifest)
    errors, verified = [], []
    try:
        _require(_root(root) == root, 'export root changed')
        _require(_inventory(root) == sorted(r['relative_path'] for r in files), 'local file inventory changed')
        for record in files:
            current = _hash_file(root, record['relative_path'], record['path_in_repo'])
            if current != record:
                errors.append({'path': record['path_in_repo'], 'reason': 'local content or stat identity differs'})
            else:
                verified.append(record['path_in_repo'])
        _require(_inventory(root) == sorted(r['relative_path'] for r in files), 'local inventory changed during verification')
    except (OSError, ArchiveVerificationError) as error:
        errors.append({'path': str(root), 'reason': str(error)})
    return {'status': 'verified' if not errors else 'failed', 'verified_files': verified, 'errors': errors}


def _field(value: Any, name: str) -> Any:
    return value.get(name) if isinstance(value, dict) else getattr(value, name, None)


def verify_remote_manifest(api: Any, repo_id: str, commit_sha: str, manifest: dict[str, Any],
                           *, repo_type: str = 'model', batch_size: int = 100) -> dict[str, Any]:
    """Verify exact files at one immutable commit using Hub 0.36 RepoFile metadata.

    get_paths_info supplies LFS metadata without expand=True. Missing paths are
    silently omitted by the SDK, so this function checks every requested path.
    It does not infer verification from upload completion or a mutable branch.
    """
    _, files = _validate_manifest(manifest)
    _require(isinstance(commit_sha, str) and re.fullmatch('[0-9a-f]{40}', commit_sha) is not None,
             'an exact 40-hex commit SHA is required; branches and tags are forbidden')
    _require(isinstance(repo_id, str) and bool(repo_id.strip()), 'repository ID is required')
    _require(repo_type in ('model', 'dataset', 'space'), 'unsupported repository type')
    _require(type(batch_size) is int and 1 <= batch_size <= 1000, 'batch_size must be 1..1000')
    errors, verified = [], []
    for start in range(0, len(files), batch_size):
        records = {r['path_in_repo']: r for r in files[start:start+batch_size]}
        try:
            response = api.get_paths_info(repo_id=repo_id, paths=list(records), revision=commit_sha,
                                          repo_type=repo_type, expand=False)
        except Exception as error:
            errors.append({'paths': list(records), 'reason': f'remote metadata lookup failed: {type(error).__name__}'})
            continue
        returned = {}
        for remote in response:
            path = _field(remote, 'path')
            if not isinstance(path, str) or path not in records or path in returned:
                errors.append({'path': path if isinstance(path, str) else None, 'reason': 'unexpected or duplicate remote path'})
                continue
            returned[path] = remote
        for path, local in records.items():
            remote = returned.get(path)
            reason = None
            if remote is None:
                reason = 'remote file missing at pinned commit'
            elif type(_field(remote, 'size')) is not int or _field(remote, 'size') != local['size']:
                reason = 'remote size differs or path is not a file'
            elif _field(remote, 'lfs') is not None:
                lfs = _field(remote, 'lfs')
                if _field(lfs, 'sha256') != local['sha256'] or type(_field(lfs, 'size')) is not int or _field(lfs, 'size') != local['size']:
                    reason = 'remote LFS SHA256 or size differs'
            elif _field(remote, 'blob_id') != local['git_blob_sha1']:
                reason = 'remote Git blob SHA1 differs or is absent'
            if reason:
                errors.append({'path': path, 'reason': reason})
            else:
                verified.append({'path': path, 'size': local['size'],
                                 'verification': 'lfs_sha256' if _field(remote, 'lfs') is not None else 'git_blob_sha1'})
    return {'status': 'verified' if not errors else 'failed', 'repo_id': repo_id, 'repo_type': repo_type,
            'commit_sha': commit_sha, 'expected_files': len(files), 'verified_files': verified, 'errors': errors}
