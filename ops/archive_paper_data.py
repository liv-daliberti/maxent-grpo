#!/usr/bin/env python3
"""Publish the reviewed paper-data allowlist; never delete local or remote files."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_archive_verification as verify

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'var/artifacts/hf_paper_data_inventory_20260911/publication'
EXPECTED_PLAN_SHA256 = 'dfeadbb6bdc58d8f5e0d41245a4d1c23a2f2c74bb5f4fe71c1d00b0ea65e39ab'
EXPECTED_METADATA_SHA256 = 'abeb9260d81f53c23b5d3d033dc6524e3d9b0251c64f99f895f3c817d333ab11'
EXPECTED_COUNT = 1260
REPO_ID = 'od2961/maxent-grpo-models'
METADATA_TARGETS = {'DATA.md', 'data/README.md', 'results/README.md', 'reproducibility/README.md', 'reproducibility/PAPER_DATA_MANIFEST.json'}
FILE_KEYS = ('local_path', 'path_in_repo', 'size', 'sha256', 'git_blob_sha1')


class DataArchiveError(ValueError):
    pass


def require(ok, code):
    if not ok:
        raise DataArchiveError(code)


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    with temporary.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def source_path(path):
    p = Path(path)
    require(p.is_absolute() and p.is_relative_to(ROOT) and str(p) == os.path.normpath(str(p)), 'source_outside_workspace')
    require(p.resolve(strict=True) == p and not p.is_symlink(), 'source_symlink')
    require(not any(part.startswith('.') for part in p.relative_to(ROOT).parts), 'hidden_source_rejected')
    require(stat.S_ISREG(p.lstat().st_mode), 'source_not_regular')
    return p


def validate_rows(rows, *, metadata=False):
    require(isinstance(rows, list) and rows, 'empty_file_allowlist')
    seen = set()
    for row in rows:
        p = source_path(row['local_path'])
        remote = verify._repo_path(row['path_in_repo'])
        require(remote not in seen, 'remote_path_collision'); seen.add(remote)
        require((remote in METADATA_TARGETS if metadata else remote.startswith(('data/', 'results/', 'reproducibility/'))), 'remote_path_outside_allowlist')
        require(PurePosixPath(remote).name == p.name, 'local_remote_basename_differs')
        require(p.suffix in ('.json', '.jsonl', '.arrow', '.pdf', '.png', '.tex', '.csv', '.md'), 'unreviewed_source_extension')
        require(type(row['size']) is int and row['size'] >= 0, 'invalid_size')
        require(re.fullmatch('[0-9a-f]{64}', row['sha256']) is not None and re.fullmatch('[0-9a-f]{40}', row['git_blob_sha1']) is not None, 'invalid_digest')
        if 'stat' in row:
            require(set(row['stat']) == {'dev', 'ino', 'mtime_ns', 'ctime_ns'} and all(type(v) is int for v in row['stat'].values()), 'invalid_stat_identity')
    return seen


def load_package(plan_path, metadata_path):
    source_path(plan_path); source_path(metadata_path)
    require(digest(plan_path) == EXPECTED_PLAN_SHA256, 'plan_sha256_differs')
    require(digest(metadata_path) == EXPECTED_METADATA_SHA256, 'metadata_sha256_differs')
    plan, metadata = read(plan_path), read(metadata_path)
    require(plan['schema'] == 'paper-data-publication-allowlist-v1' and plan['repo_id'] == REPO_ID and plan['public'] is True, 'plan_identity_differs')
    require(plan['file_count'] == len(plan['files']) == EXPECTED_COUNT and plan['total_bytes'] == sum(r['size'] for r in plan['files']), 'plan_count_or_bytes_differs')
    require(plan['upload_policy']['max_files_per_commit'] == 50 and plan['upload_policy']['upload_workers'] == 1 and plan['upload_policy']['deletion_allowed'] is False, 'upload_policy_differs')
    require(metadata['schema'] == 'paper-data-metadata-publication-v1' and metadata['data_upload_plan_sha256'] == EXPECTED_PLAN_SHA256 and metadata['publish_after_data_payload_verified'] is True, 'metadata_plan_binding_differs')
    payload_paths = validate_rows(plan['files'])
    metadata_paths = validate_rows(metadata['files'], metadata=True)
    require(metadata_paths == METADATA_TARGETS and not (payload_paths & metadata_paths), 'metadata_collision_or_scope_differs')
    return {'plan': plan, 'metadata': metadata, 'plan_path': str(plan_path), 'metadata_path': str(metadata_path),
            'plan_sha256': EXPECTED_PLAN_SHA256, 'metadata_sha256': EXPECTED_METADATA_SHA256}


def package_unchanged(package):
    require(digest(package['plan_path']) == package['plan_sha256'] and digest(package['metadata_path']) == package['metadata_sha256'], 'package_changed_during_upload')


def current_source(row):
    p = source_path(row['local_path'])
    current = verify._hash_file(ROOT, p.relative_to(ROOT).as_posix(), row['path_in_repo'])
    require(all(current[k] == row[k] for k in ('size', 'sha256', 'git_blob_sha1')), 'source_content_changed')
    if 'stat' in row:
        require(current['stat'] == row['stat'], 'source_identity_changed')
    return {**row, 'stat': current['stat']}


def assert_source_identity(row):
    p = source_path(row['local_path']); info = p.lstat()
    require(info.st_size == row['size'] and verify._identity(info) == row['stat'], 'source_changed_during_upload')


def remote_proof(api, commit, rows):
    require(re.fullmatch('[0-9a-f]{40}', commit or '') is not None, 'immutable_commit_required')
    require(1 <= len(rows) <= 50, 'verification_batch_out_of_bounds')
    expected = {r['path_in_repo']: r for r in rows}
    require(len(expected) == len(rows), 'remote_path_collision')
    response = api.get_paths_info(repo_id=REPO_ID, paths=list(expected), revision=commit, repo_type='model', expand=False)
    returned = {}
    for item in response:
        name = verify._field(item, 'path')
        require(name in expected and name not in returned, 'unexpected_or_duplicate_remote_path')
        returned[name] = item

    class FrozenMetadata:
        def get_paths_info(self, *, repo_id, paths, revision, repo_type, expand):
            require(repo_id == REPO_ID and revision == commit and repo_type == 'model' and expand is False, 'metadata_lookup_identity_differs')
            return [returned[p] for p in paths if p in returned]

    evidence = []
    for row in rows:
        p = Path(row['local_path'])
        item = {k: row[k] for k in ('local_path', 'path_in_repo', 'size', 'sha256', 'git_blob_sha1', 'stat')}
        item['relative_path'] = p.name
        manifest = {'schema': verify.SCHEMA, 'export_dir': str(p.parent), 'repo_prefix': ('' if str(PurePosixPath(row['path_in_repo']).parent) == '.' else str(PurePosixPath(row['path_in_repo']).parent)), 'files': [item], 'total_bytes': row['size']}
        proof = verify.verify_remote_manifest(FrozenMetadata(), REPO_ID, commit, manifest)
        require(proof['status'] == 'verified' and not proof['errors'], 'remote_content_verification_failed')
        evidence.append({**proof['verified_files'][0], 'sha256': row['sha256'], 'git_blob_sha1': row['git_blob_sha1']})
    return {'status': 'verified', 'repo_id': REPO_ID, 'commit_sha': commit, 'files': evidence, 'verified_at_utc': now()}


def row_spec(rows):
    return [{k: row[k] for k in FILE_KEYS} for row in rows]


def verify_saved_receipt(receipt, package, batch_id, rows):
    require(receipt.get('schema') == 'verified-paper-data-batch-v1' and receipt.get('status') == 'verified', 'saved_receipt_schema_differs')
    require(receipt.get('plan_sha256') == package['plan_sha256'] and receipt.get('metadata_sha256') == package['metadata_sha256'], 'saved_receipt_plan_binding_differs')
    require(receipt.get('repo_id') == REPO_ID and receipt.get('batch_id') == batch_id and receipt.get('files') == row_spec(rows), 'saved_receipt_batch_binding_differs')
    require(receipt.get('source_identity_after_upload_verified') is True, 'saved_receipt_missing_source_check')
    commit = receipt.get('commit_sha')
    require(re.fullmatch('[0-9a-f]{40}', commit or '') is not None, 'saved_receipt_commit_invalid')
    proof = receipt.get('remote_proof', {})
    require(proof.get('status') == 'verified' and proof.get('repo_id') == REPO_ID and proof.get('commit_sha') == commit, 'saved_receipt_remote_identity_differs')
    files = proof.get('files', [])
    observed = {r['path']: r for r in files}
    require(len(files) == len(rows) == len(observed), 'saved_receipt_remote_coverage_differs')
    for row in rows:
        evidence = observed.get(row['path_in_repo'], {})
        require(evidence.get('size') == row['size'] and evidence.get('sha256') == row['sha256'] and evidence.get('git_blob_sha1') == row['git_blob_sha1'] and evidence.get('verification') in ('lfs_sha256', 'git_blob_sha1'), 'saved_receipt_remote_digest_differs')
    return receipt


def publish_batch(api, add_factory, package, state, batch_id, rows):
    directory = state / 'batches' / batch_id
    verified_path = directory / 'verified.json'
    if verified_path.exists():
        return verify_saved_receipt(read(verified_path), package, batch_id, rows)
    package_unchanged(package)
    intent_path, commit_path = directory / 'intent.json', directory / 'commit.json'
    if intent_path.exists():
        intent = read(intent_path)
        require(intent['plan_sha256'] == package['plan_sha256'] and intent['metadata_sha256'] == package['metadata_sha256'] and intent['batch_id'] == batch_id and intent['files'] == row_spec(rows), 'pending_intent_binding_differs')
        require(commit_path.exists(), 'uncertain_commit_requires_reconciliation')
        checked = intent['source_files']
        require(row_spec(checked) == row_spec(rows), 'pending_intent_source_binding_differs')
        for row in checked:
            assert_source_identity(row)
        commit_record = read(commit_path)
        require(commit_record['intent_sha256'] == digest(intent_path), 'commit_intent_binding_differs')
        commit = commit_record['commit_sha']
    else:
        require(not commit_path.exists(), 'commit_without_intent')
        checked = [current_source(r) for r in rows]
        package_unchanged(package)
        intent = {'schema': 'paper-data-upload-intent-v1', 'plan_sha256': package['plan_sha256'], 'metadata_sha256': package['metadata_sha256'], 'batch_id': batch_id, 'files': row_spec(rows), 'source_files': checked, 'at_utc': now()}
        save(intent_path, intent)
        result = api.create_commit(repo_id=REPO_ID, repo_type='model', operations=[add_factory(path_in_repo=r['path_in_repo'], path_or_fileobj=r['local_path']) for r in checked], num_threads=1, commit_message='Archive paper data ' + batch_id)
        commit = result.oid
        require(re.fullmatch('[0-9a-f]{40}', commit or '') is not None, 'uploaded_commit_invalid')
        save(commit_path, {'commit_sha': commit, 'intent_sha256': digest(intent_path), 'at_utc': now()})
        for row in checked:
            assert_source_identity(row)
    package_unchanged(package)
    proof = remote_proof(api, commit, checked)
    receipt = {'schema': 'verified-paper-data-batch-v1', 'status': 'verified', 'repo_id': REPO_ID, 'plan_sha256': package['plan_sha256'], 'metadata_sha256': package['metadata_sha256'], 'batch_id': batch_id, 'files': row_spec(rows), 'commit_sha': commit, 'source_identity_after_upload_verified': True, 'remote_proof': proof, 'verified_at_utc': now()}
    save(verified_path, receipt)
    print(json.dumps({'event': 'batch_verified', 'batch_id': batch_id, 'files': len(rows), 'commit_sha': commit}), flush=True)
    return receipt


@contextmanager
def single_writer(state):
    require(state.is_absolute() and state.is_relative_to(ROOT / 'var/artifacts') and state.resolve() == state, 'state_outside_artifacts')
    state.mkdir(parents=True, exist_ok=True)
    with (state / 'run.lock').open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def publish(api, add_factory, package, state):
    with single_writer(state):
        registration = {'schema': 'paper-data-publisher-registration-v1', 'repo_id': REPO_ID, 'plan_sha256': package['plan_sha256'], 'metadata_sha256': package['metadata_sha256'], 'publisher_sha256': digest(__file__), 'verifier_sha256': digest(verify.__file__)}
        registration_path = state / 'registration.json'
        if registration_path.exists():
            require(read(registration_path) == registration, 'resume_registration_binding_differs')
        else:
            require(not list(state.glob('batches/*/intent.json')), 'unregistered_existing_upload_intent')
            save(registration_path, registration)
        require(api.model_info(REPO_ID).private is False, 'repository_not_public')
        rows = package['plan']['files']
        receipts = []
        for offset in range(0, len(rows), 50):
            receipts.append(publish_batch(api, add_factory, package, state, f'payload-{offset//50:04d}', rows[offset:offset+50]))
        release_path = state / 'DATA_RELEASE.json'
        release_core = {'schema': 'paper-data-release-v1', 'repo_id': REPO_ID, 'file_count': len(rows), 'total_bytes': sum(r['size'] for r in rows), 'data_manifest_sha256': next(r['sha256'] for r in package['metadata']['files'] if r['path_in_repo'] == 'reproducibility/PAPER_DATA_MANIFEST.json'), 'data_plan_sha256': package['plan_sha256'], 'source_selection': 'Exact stable initial paper-data allowlist; historical sources retain their original analysis-selection boundaries.'}
        if release_path.exists():
            release = read(release_path)
            require(all(release.get(k) == value for k, value in release_core.items()), 'release_plan_binding_differs')
            commit = release['commit_sha']
        else:
            commit = api.model_info(REPO_ID).sha
            release = {**release_core, 'commit_sha': commit, 'verified_at_utc': now()}
        # One immutable tree must contain the complete data release, even while
        # independent model uploads continue to advance the repository branch.
        for offset in range(0, len(rows), 50):
            checked = [{**r, 'stat': r.get('stat', verify._identity(Path(r['local_path']).lstat()))} for r in rows[offset:offset+50]]
            remote_proof(api, commit, checked)
        if not release_path.exists():
            save(release_path, release)
        release_record = verify._hash_file(ROOT, release_path.relative_to(ROOT).as_posix(), 'reproducibility/DATA_RELEASE.json')
        metadata_rows = [{**r, 'stat': verify._identity(Path(r['local_path']).lstat())} for r in package['metadata']['files']] + [release_record]
        metadata_receipt = publish_batch(api, add_factory, package, state, 'metadata-0000', metadata_rows)
        result = {'schema': 'paper-data-publication-complete-v1', 'status': 'verified', 'repo_id': REPO_ID, 'plan_sha256': package['plan_sha256'], 'metadata_sha256': package['metadata_sha256'], 'payload_files': len(rows), 'payload_bytes': sum(r['size'] for r in rows), 'data_commit_sha': commit, 'metadata_commit_sha': metadata_receipt['commit_sha'], 'data_release_receipt_sha256': digest(release_path), 'verified_payload_batches': len(receipts), 'local_files_retained': True, 'completed_at_utc': now()}
        save(state / 'completed.json', result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=PACKAGE / 'upload_plan.json')
    parser.add_argument('--metadata', type=Path, default=PACKAGE / 'metadata_uploads.json')
    parser.add_argument('--state-dir', type=Path, default=PACKAGE / 'upload_state')
    parser.add_argument('--token-file', type=Path)
    parser.add_argument('--publish', action='store_true', help='Execute the user-authorized public upload; without this flag, validate the plan only.')
    args = parser.parse_args()
    try:
        package = load_package(args.plan.absolute(), args.metadata.absolute())
        if not args.publish:
            print(json.dumps({'status': 'plan_valid', 'files': len(package['plan']['files']), 'bytes': package['plan']['total_bytes'], 'plan_sha256': package['plan_sha256'], 'metadata_sha256': package['metadata_sha256']}))
            return 0
        require(args.token_file is not None, 'publish_requires_token_file')
        token_path = args.token_file.absolute()
        require(not token_path.is_symlink() and stat.S_ISREG(token_path.lstat().st_mode) and token_path.lstat().st_mode & 0o077 == 0, 'token_file_permissions_invalid')
        token = token_path.read_text().strip()
        require(bool(token), 'empty_token_file')
        os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
        logging.getLogger('huggingface_hub').setLevel(logging.CRITICAL)
        from huggingface_hub import HfApi, CommitOperationAdd
        result = publish(HfApi(token=token), CommitOperationAdd, package, args.state_dir.absolute())
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception as error:
        # SDK exception strings can contain signed URLs or request headers.
        message = {'event': 'stopped', 'error_type': type(error).__name__}
        if isinstance(error, DataArchiveError):
            message['reason'] = str(error)
        response = getattr(error, 'response', None)
        status_code = getattr(response, 'status_code', None)
        if type(status_code) is int:
            message['http_status'] = status_code
        print(json.dumps(message), file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
