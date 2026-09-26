"""CPU-only paper-data publisher tests; no real credentials or Hub calls."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location('archive_paper_data_tested', Path(__file__).resolve().parents[1] / 'ops/archive_paper_data.py')
a = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(a)


class FakeApi:
    def __init__(self):
        self.head = 'f' * 40
        self.trees = {self.head: {}}
        self.creates = []
        self.lookups = []
        self.omit = set()
        self.after_create = None
        self.fail_create = False
        self.private = False

    def model_info(self, repo_id):
        assert repo_id == a.REPO_ID
        return SimpleNamespace(private=self.private, sha=self.head)

    def create_commit(self, *, repo_id, repo_type, operations, num_threads, commit_message):
        assert repo_id == a.REPO_ID and repo_type == 'model' and num_threads == 1
        assert 1 <= len(operations) <= 50
        tree = copy.deepcopy(self.trees[self.head])
        for op in operations:
            raw = Path(op['path_or_fileobj']).read_bytes()
            tree[op['path_in_repo']] = {'path': op['path_in_repo'], 'size': len(raw), 'blob_id': hashlib.sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest(), 'lfs': None}
        self.head = f'{len(self.creates) + 1:040x}'
        self.trees[self.head] = tree
        self.creates.append(operations)
        if self.after_create:
            self.after_create(operations)
        if self.fail_create:
            raise RuntimeError('synthetic uncertain HTTP failure')
        return SimpleNamespace(oid=self.head)

    def get_paths_info(self, *, repo_id, paths, revision, repo_type, expand):
        assert repo_id == a.REPO_ID and repo_type == 'model' and expand is False
        assert len(paths) <= 50 and revision in self.trees
        self.lookups.append((revision, list(paths)))
        return [self.trees[revision][p] for p in paths if p in self.trees[revision] and p not in self.omit]


@pytest.fixture
def package(tmp_path, monkeypatch):
    root = tmp_path / 'repo'
    root.mkdir()
    monkeypatch.setattr(a, 'ROOT', root)
    payload = []
    for name, remote in [('dataset_info.json', 'data/ModeBench/toy/train/dataset_info.json'), ('results.json', 'results/snapshot/results.json')]:
        p = root / 'inputs' / name
        p.parent.mkdir(exist_ok=True)
        p.write_text(json.dumps({'fixture': name}))
        payload.append(a.verify._hash_file(root, p.relative_to(root).as_posix(), remote))
    metadata_rows = []
    for remote in sorted(a.METADATA_TARGETS):
        p = root / 'metadata' / remote
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('{}' if p.suffix == '.json' else '# Fixture\n')
        r = a.verify._hash_file(root, p.relative_to(root).as_posix(), remote)
        metadata_rows.append({k: r[k] for k in a.FILE_KEYS})
    plan = {'schema': 'paper-data-publication-allowlist-v1', 'repo_id': a.REPO_ID, 'public': True, 'files': payload, 'file_count': len(payload), 'total_bytes': sum(r['size'] for r in payload), 'upload_policy': {'max_files_per_commit': 50, 'upload_workers': 1, 'deletion_allowed': False}}
    plan_path = root / 'plan.json'
    plan_path.write_text(json.dumps(plan))
    monkeypatch.setattr(a, 'EXPECTED_PLAN_SHA256', a.digest(plan_path))
    monkeypatch.setattr(a, 'EXPECTED_COUNT', len(payload))
    metadata = {'schema': 'paper-data-metadata-publication-v1', 'data_upload_plan_sha256': a.EXPECTED_PLAN_SHA256, 'publish_after_data_payload_verified': True, 'files': metadata_rows}
    metadata_path = root / 'metadata.json'
    metadata_path.write_text(json.dumps(metadata))
    monkeypatch.setattr(a, 'EXPECTED_METADATA_SHA256', a.digest(metadata_path))
    return a.load_package(plan_path, metadata_path)


def add(**kw):
    return kw


def state():
    return a.ROOT / 'var/artifacts/data_upload_test'


def test_full_upload_verifies_payload_before_metadata_and_retains_files(package):
    api = FakeApi()
    before = {r['local_path']: Path(r['local_path']).read_bytes() for r in package['plan']['files']}
    result = a.publish(api, add, package, state())
    assert result['status'] == 'verified' and result['payload_files'] == 2
    assert len(api.creates) == 2
    assert all(x['path_in_repo'].startswith(('data/ModeBench/', 'results/snapshot/')) for x in api.creates[0])
    assert {x['path_in_repo'] for x in api.creates[1]} == a.METADATA_TARGETS | {'reproducibility/DATA_RELEASE.json'}
    assert result['data_commit_sha'] == f'{1:040x}'
    assert any(commit == result['data_commit_sha'] and len(paths) == 2 for commit, paths in api.lookups)
    assert all(Path(p).read_bytes() == raw for p, raw in before.items())
    assert a.read(state() / 'DATA_RELEASE.json')['commit_sha'] == result['data_commit_sha']


def test_verified_resume_does_not_reupload(package):
    api = FakeApi()
    first = a.publish(api, add, package, state())
    second = a.publish(api, add, package, state())
    assert len(api.creates) == 2
    assert first['data_commit_sha'] == second['data_commit_sha']


def test_source_change_rejected_before_any_commit(package):
    Path(package['plan']['files'][0]['local_path']).write_text('changed')
    api = FakeApi()
    with pytest.raises(a.DataArchiveError, match='source_content_changed'):
        a.publish(api, add, package, state())
    assert not api.creates


def test_source_symlink_rejected(package):
    row = package['plan']['files'][0]
    p = Path(row['local_path'])
    alias = p.parent / 'alias.json'
    alias.symlink_to(p)
    with pytest.raises(a.DataArchiveError, match='source_symlink'):
        a.current_source({**row, 'local_path': str(alias)})


@pytest.mark.parametrize('remote', ['experiments/evil.json', '../secret.json', 'data/../secret.json', '/data/file.json'])
def test_remote_path_outside_reviewed_namespaces_rejected(package, remote):
    row = copy.deepcopy(package['plan']['files'][0]); row['path_in_repo'] = remote
    with pytest.raises(ValueError):
        a.validate_rows([row])


def test_duplicate_remote_path_rejected(package):
    row = package['plan']['files'][0]
    with pytest.raises(a.DataArchiveError, match='remote_path_collision'):
        a.validate_rows([row, copy.deepcopy(row)])


def test_missing_remote_file_never_verifies(package):
    api = FakeApi(); row = package['plan']['files'][0]
    with pytest.raises(a.DataArchiveError, match='remote_content_verification_failed'):
        a.remote_proof(api, api.head, [row])


def test_wrong_remote_lfs_hash_never_verifies(package):
    api = FakeApi(); row = package['plan']['files'][0]
    api.trees[api.head][row['path_in_repo']] = {'path': row['path_in_repo'], 'size': row['size'], 'lfs': {'size': row['size'], 'sha256': '0' * 64}, 'blob_id': row['git_blob_sha1']}
    with pytest.raises(a.DataArchiveError, match='remote_content_verification_failed'):
        a.remote_proof(api, api.head, [row])


def test_postupload_source_change_stops_before_verified_receipt(package):
    api = FakeApi()
    api.after_create = lambda ops: Path(ops[0]['path_or_fileobj']).write_text('changed during upload')
    with pytest.raises(a.DataArchiveError, match='source_changed_during_upload'):
        a.publish(api, add, package, state())
    assert len(api.creates) == 1
    assert not (state() / 'batches/payload-0000/verified.json').exists()


def test_uncertain_commit_intent_is_not_blindly_retried(package):
    api = FakeApi(); api.fail_create = True
    with pytest.raises(RuntimeError):
        a.publish(api, add, package, state())
    api.fail_create = False
    with pytest.raises(a.DataArchiveError, match='uncertain_commit_requires_reconciliation'):
        a.publish(api, add, package, state())
    assert len(api.creates) == 1


def test_known_commit_verification_failure_resumes_without_duplicate_upload(package):
    api = FakeApi(); api.omit.add(package['plan']['files'][0]['path_in_repo'])
    with pytest.raises(a.DataArchiveError, match='remote_content_verification_failed'):
        a.publish(api, add, package, state())
    api.omit.clear()
    assert a.publish(api, add, package, state())['status'] == 'verified'
    assert len(api.creates) == 2


def test_resume_receipt_bound_to_plan(package):
    api = FakeApi(); a.publish(api, add, package, state())
    p = state() / 'batches/payload-0000/verified.json'
    receipt = a.read(p); receipt['plan_sha256'] = '0' * 64; a.save(p, receipt)
    with pytest.raises(a.DataArchiveError, match='saved_receipt_plan_binding_differs'):
        a.publish(api, add, package, state())
    assert len(api.creates) == 2


def test_package_mutation_rejected(package):
    with Path(package['plan_path']).open('a') as f:
        f.write('\n')
    api = FakeApi()
    with pytest.raises(a.DataArchiveError, match='package_changed_during_upload'):
        a.publish(api, add, package, state())
    assert not api.creates


def test_cli_sanitizes_sdk_exception(package, monkeypatch, capsys):
    token = a.ROOT / 'synthetic_token'
    token.write_text('SYNTHETIC_TEST_ONLY'); token.chmod(0o600)
    monkeypatch.setattr(sys, 'argv', ['archive_paper_data.py', '--plan', package['plan_path'], '--metadata', package['metadata_path'], '--publish', '--token-file', str(token)])
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(HfApi=lambda token: object(), CommitOperationAdd=add))
    def fail(*args):
        raise RuntimeError('https://signed.invalid/?secret=SYNTHETIC_TEST_ONLY')
    monkeypatch.setattr(a, 'publish', fail)
    assert a.main() == 1
    captured = capsys.readouterr()
    assert 'RuntimeError' in captured.err
    assert 'signed.invalid' not in captured.err and 'SYNTHETIC_TEST_ONLY' not in captured.err
