"""CPU-only paper-data publisher tests; no real credentials or Hub calls."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location('archive_paper_closing_tested', Path(__file__).resolve().parents[1] / 'ops/archive_paper_closing.py')
profile = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(profile)
a = profile.a


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
    old_releases = {}
    for key, letter in [('initial', 'a'), ('supplement', 'b'), ('evidence', 'c')]:
        old_plan_path = root / (key + '_plan.json')
        old_plan_path.write_text(json.dumps({'files': [{'path_in_repo': 'results/prior/' + key + '.json', 'sha256': letter * 64, 'size': 7}]}))
        old_receipt_path = root / (key + '_receipt.json')
        old_receipt_path.write_text(json.dumps({'status': 'verified', 'repo_id': a.REPO_ID, 'plan_sha256': a.digest(old_plan_path), 'data_commit_sha': letter * 40}))
        old_releases[key] = {'plan_path': str(old_plan_path), 'plan_sha256': a.digest(old_plan_path), 'receipt_path': str(old_receipt_path), 'receipt_sha256': a.digest(old_receipt_path), 'commit_sha': letter * 40}
    original_source = str(Path(payload[0]['local_path']).relative_to(root))
    reference = {'release': 'completion', 'path_in_repo': payload[0]['path_in_repo'], 'sha256': payload[0]['sha256'], 'size': payload[0]['size'], 'original_repository_path': original_source}
    models = []
    inventory_rows = []
    for n in range(8):
        run = 'var/data/fixture_' + str(n)
        exp = 'e118' if n < 7 else 'e119'
        draws = [{'draw_index': d, 'line': d + 1, 'data': copy.deepcopy(reference)} for d in range(4)]
        models.append({'run_repository_path': run, 'experiment': exp, 'domain': 'countdown', 'model_key': 'qwen05b', 'method': 'control', 'seed': n, 'verified_model_catalog_url': 'https://huggingface.co/od2961/maxent-grpo-models/blob/main/catalog.json', 'direct_model_download_url': None, 'model_download_status': 'consult_verified_model_catalog', 'terminal_update': 3072, 'evaluation_sources': [copy.deepcopy(reference)], 'selected_draw_records': draws, 'completion_receipt': copy.deepcopy(reference)})
        inventory_rows.append({'run_dir': str(root / run), 'source_experiment': exp, 'domain': 'countdown', 'model_key': 'qwen05b', 'arm': 'control', 'seed': n, 'endpoint_status': 'admitted', 'integrity_audit': {'status': 'admitted', 'step': 3072, 'sources': [{'path': str(root / original_source), 'sha256': reference['sha256']}], 'completion_receipt': {'path': str(root / original_source), 'sha256': reference['sha256']}, 'unique_draw_records': [{'draw_index': d, 'line': d + 1, 'path': str(root / original_source)} for d in range(4)]}})
    inventory_path = root / 'late_inventory.json'; inventory_path.write_text(json.dumps({'records': inventory_rows}))
    plan['late_completion_inventory'] = {'local_path': str(inventory_path), 'sha256': a.digest(inventory_path)}
    template = {'schema': 'paper-model-evidence-index-template-v1', 'repo_id': a.REPO_ID, 'scientific_model_records': 8, 'prior_scientific_model_records': 762, 'model_archive_entries': 8, 'missing_weight_exports': 0, 'models': models, 'current_scientific_assets': [{'data': {'release': key, 'path_in_repo': 'results/prior/' + key + '.json', 'sha256': letter * 64, 'size': 7}} for key, letter in [('initial', 'a'), ('supplement', 'b'), ('evidence', 'c')]]}
    template_path = root / 'model_evidence_index_template.json'; template_path.write_text(json.dumps(template))
    plan.update(old_releases=old_releases, evidence_index_template={'local_path': str(template_path), 'sha256': a.digest(template_path)})
    plan_path.write_text(json.dumps(plan)); monkeypatch.setattr(a, 'EXPECTED_PLAN_SHA256', a.digest(plan_path))
    metadata['data_upload_plan_sha256'] = a.EXPECTED_PLAN_SHA256; metadata_path.write_text(json.dumps(metadata)); monkeypatch.setattr(a, 'EXPECTED_METADATA_SHA256', a.digest(metadata_path))
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
    assert {x['path_in_repo'] for x in api.creates[1]} == a.METADATA_TARGETS | {'reproducibility/CLOSING_DATA_RELEASE.json', 'reproducibility/LATE_COMPLETION_EVIDENCE_INDEX.json', 'reproducibility/LATE_COMPLETION_EVIDENCE_INDEX.md'}
    assert result['data_commit_sha'] == f'{1:040x}'
    assert any(commit == result['data_commit_sha'] and len(paths) == 2 for commit, paths in api.lookups)
    assert all(Path(p).read_bytes() == raw for p, raw in before.items())
    assert a.read(state() / 'CLOSING_DATA_RELEASE.json')['commit_sha'] == result['data_commit_sha']


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


def test_addendum_does_not_overwrite_initial_release_metadata(package):
    api = FakeApi()
    originals = {'DATA.md': b'original data card', 'reproducibility/DATA_RELEASE.json': b'original revision', 'reproducibility/PAPER_DATA_MANIFEST.json': b'original manifest'}
    for path, raw in originals.items():
        api.trees[api.head][path] = {'path': path, 'size': len(raw), 'blob_id': hashlib.sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest(), 'lfs': None}
    before = copy.deepcopy(api.trees[api.head])
    result = a.publish(api, add, package, state())
    assert result['profile'] == 'paper_closing_supplement_20260911'
    assert all(api.trees[api.head][path] == value for path, value in before.items())
    assert not any(op['path_in_repo'] in originals for batch in api.creates for op in batch)
    registration = a.read(state() / 'registration.json')
    assert registration['engine_sha256'] == profile.ENGINE_SHA256
    assert registration['publisher_sha256'] == a.digest(profile.SOURCE)


def test_addendum_resume_rejects_another_profile(package):
    api = FakeApi()
    a.publish(api, add, package, state())
    path = state() / 'registration.json'
    record = a.read(path); record['profile'] = 'initial_release'; a.save(path, record)
    with pytest.raises(a.DataArchiveError, match='resume_registration_binding_differs'):
        a.publish(api, add, package, state())
    assert len(api.creates) == 2


def test_index_urls_bind_each_release_to_its_exact_revision(package):
    api = FakeApi(); result = a.publish(api, add, package, state())
    index = a.read(state() / 'LATE_COMPLETION_EVIDENCE_INDEX.json')
    fresh = index['models'][0]['evaluation_sources'][0]
    assert fresh['commit_sha'] == result['data_commit_sha']
    assert '/resolve/' + result['data_commit_sha'] + '/' in fresh['download_url']
    assert index['current_scientific_assets'][0]['data']['commit_sha'] == 'a' * 40
    assert index['current_scientific_assets'][1]['data']['commit_sha'] == 'b' * 40
    assert all(r['direct_model_download_url'] is None for r in index['models'])
    assert result['data_commit_sha'] != result['metadata_commit_sha']


def test_mutated_template_stops_before_any_upload(package):
    p = Path(package['plan']['evidence_index_template']['local_path'])
    p.write_text(p.read_text() + ' ')
    api = FakeApi()
    with pytest.raises(a.DataArchiveError, match='evidence_index_template_changed'):
        a.publish(api, add, package, state())
    assert not api.creates


def test_unbound_template_digest_rejected_even_after_repinning(package):
    pin = package['plan']['evidence_index_template']; p = Path(pin['local_path'])
    value = a.read(p); value['models'][0]['evaluation_sources'][0]['sha256'] = 'f' * 64
    a.save(p, value); pin['sha256'] = a.digest(p)
    with pytest.raises(a.DataArchiveError, match='unbound_evidence_reference'):
        profile.validate_template(package)


def test_changed_prior_verified_commit_is_rejected(package):
    pin = package['plan']['old_releases']['initial']; p = Path(pin['receipt_path'])
    value = a.read(p); value['data_commit_sha'] = 'c' * 40; a.save(p, value)
    pin['receipt_sha256'] = a.digest(p)
    with pytest.raises(a.DataArchiveError, match='prior_release_proof_invalid'):
        profile.validate_template(package)


def test_tampered_saved_index_blocks_metadata_reuse(package):
    api = FakeApi(); a.publish(api, add, package, state())
    p = state() / 'LATE_COMPLETION_EVIDENCE_INDEX.json'; value = a.read(p)
    value['models'][0]['evaluation_sources'][0]['download_url'] = 'https://example.invalid/substitution'
    a.save(p, value)
    with pytest.raises(a.DataArchiveError, match='saved_evidence_index_differs'):
        a.publish(api, add, package, state())
    assert len(api.creates) == 2


def test_initial_and_supplement_metadata_are_both_untouched(package):
    api = FakeApi()
    old_paths = ['DATA.md', 'SCIENTIFIC_DATA.md', 'reproducibility/DATA_RELEASE.json', 'reproducibility/SCIENTIFIC_DATA_RELEASE.json']
    for p in old_paths: api.trees[api.head][p] = {'path': p, 'size': 1, 'blob_id': 'a' * 40, 'lfs': None}
    before = copy.deepcopy(api.trees[api.head]); a.publish(api, add, package, state())
    assert all(api.trees[api.head][p] == before[p] for p in old_paths)
    assert not any(op['path_in_repo'] in old_paths for batch in api.creates for op in batch)


def test_readable_index_has_stable_model_anchors_and_immutable_data_links(package):
    api = FakeApi(); result = a.publish(api, add, package, state())
    text = (state() / 'LATE_COMPLETION_EVIDENCE_INDEX.md').read_text()
    index = a.read(state() / 'LATE_COMPLETION_EVIDENCE_INDEX.json')
    assert len({r['scientific_id'] for r in index['models']}) == 8
    for row in index['models']:
        assert 'id="' + row['evidence_index_anchor'] + '"' in text
        for source in row['evaluation_sources']: assert source['download_url'] in text
    assert text.count('**Weights unavailable**') == 0
    assert all('/resolve/' + result['data_commit_sha'] + '/' in source['download_url'] for row in index['models'] for source in row['evaluation_sources'])
    before = text; a.publish(api, add, package, state())
    assert (state() / 'LATE_COMPLETION_EVIDENCE_INDEX.md').read_text() == before


@pytest.mark.parametrize('field,value,error', [
    ('seed', 999, 'late_model_identity_differs'),
    ('terminal_update', 3000, 'terminal_evidence_missing'),
    ('direct_model_download_url', 'https://example.invalid/model', 'unverified_model_download_claim'),
])
def test_late_scientific_identity_and_availability_are_exact(package, field, value, error):
    pin = package['plan']['evidence_index_template']; p = Path(pin['local_path'])
    template = a.read(p); template['models'][0][field] = value; a.save(p, template); pin['sha256'] = a.digest(p)
    with pytest.raises(a.DataArchiveError, match=error): profile.validate_template(package)


def test_late_draw_line_substitution_is_rejected(package):
    pin = package['plan']['evidence_index_template']; p = Path(pin['local_path'])
    template = a.read(p); template['models'][0]['selected_draw_records'][0]['line'] = 999; a.save(p, template); pin['sha256'] = a.digest(p)
    with pytest.raises(a.DataArchiveError, match='late_draw_locations_differ'): profile.validate_template(package)


def test_late_admission_inventory_mutation_is_rejected(package):
    p = Path(package['plan']['late_completion_inventory']['local_path']); p.write_text(p.read_text() + ' ')
    with pytest.raises(a.DataArchiveError, match='late_inventory_changed'): profile.validate_template(package)
