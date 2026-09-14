#!/usr/bin/env python3
"""Publish the separately pinned scientific supplement; upload only, no deletion."""
from pathlib import Path
import hashlib
import importlib.util
import sys
sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve()
ROOT = SOURCE.parents[1]
ENGINE = ROOT / 'ops/archive_paper_data.py'
ENGINE_SHA256 = 'c0c852ee57b14192ea259ace5dee58a734eddccdbe92b7e16f8309eac1827faa'
if hashlib.sha256(ENGINE.read_bytes()).hexdigest() != ENGINE_SHA256:
    raise RuntimeError('reviewed_data_upload_engine_changed')
spec = importlib.util.spec_from_file_location('reviewed_addendum_upload_engine', ENGINE)
a = importlib.util.module_from_spec(spec); spec.loader.exec_module(a)
a.PACKAGE = ROOT / 'var/artifacts/hf_paper_data_inventory_20260911/scientific_addendum'
a.EXPECTED_PLAN_SHA256 = '31ab12622593b1c0b55967406f9fcfbec84b8f59e304c6b9b254829e1153a06f'
a.EXPECTED_METADATA_SHA256 = '6b6dc517760b88be28d70f0ba2df252cf4cdabdb541fef50a8c0bdd89fe3767a'
a.EXPECTED_COUNT = 317
a.METADATA_TARGETS = {'SCIENTIFIC_DATA.md', 'reproducibility/SCIENTIFIC_DATA_MANIFEST.json'}
RELEASE_TARGET = 'reproducibility/SCIENTIFIC_DATA_RELEASE.json'
MANIFEST_TARGET = 'reproducibility/SCIENTIFIC_DATA_MANIFEST.json'


def publish(api, add_factory, package, state):
    """Reuse the reviewed payload transaction; isolate supplement release metadata."""
    with a.single_writer(state):
        registration = {'schema': 'paper-data-publisher-registration-v1', 'repo_id': a.REPO_ID,
                        'plan_sha256': package['plan_sha256'], 'metadata_sha256': package['metadata_sha256'],
                        'publisher_sha256': a.digest(SOURCE), 'engine_sha256': ENGINE_SHA256,
                        'verifier_sha256': a.digest(a.verify.__file__), 'profile': 'scientific_addendum_20260911'}
        registration_path = state / 'registration.json'
        if registration_path.exists():
            a.require(a.read(registration_path) == registration, 'resume_registration_binding_differs')
        else:
            a.require(not list(state.glob('batches/*/intent.json')), 'unregistered_existing_upload_intent')
            a.save(registration_path, registration)
        a.require(api.model_info(a.REPO_ID).private is False, 'repository_not_public')
        rows = package['plan']['files']; receipts = []
        for offset in range(0, len(rows), 50):
            receipts.append(a.publish_batch(api, add_factory, package, state, f'payload-{offset//50:04d}', rows[offset:offset+50]))
        release_path = state / 'SCIENTIFIC_DATA_RELEASE.json'
        release_core = {'schema': 'paper-data-release-v1', 'repo_id': a.REPO_ID, 'file_count': len(rows),
                        'total_bytes': sum(r['size'] for r in rows),
                        'data_manifest_sha256': next(r['sha256'] for r in package['metadata']['files'] if r['path_in_repo'] == MANIFEST_TARGET),
                        'data_plan_sha256': package['plan_sha256'],
                        'source_selection': 'Explicit scientific supplement: E95, protocols, completed RLEP training pools, E121 integrity evidence, and the completed hosted-study snapshot; no new result selection.'}
        if release_path.exists():
            release = a.read(release_path)
            a.require(all(release.get(k) == value for k, value in release_core.items()), 'release_plan_binding_differs')
            commit = release['commit_sha']
        else:
            commit = api.model_info(a.REPO_ID).sha
            release = {**release_core, 'commit_sha': commit, 'verified_at_utc': a.now()}
        # Authenticate the entire supplement at one immutable tree, including on resume.
        for offset in range(0, len(rows), 50):
            checked = [{**r, 'stat': r.get('stat', a.verify._identity(Path(r['local_path']).lstat()))} for r in rows[offset:offset+50]]
            a.remote_proof(api, commit, checked)
        if not release_path.exists(): a.save(release_path, release)
        release_record = a.verify._hash_file(a.ROOT, release_path.relative_to(a.ROOT).as_posix(), RELEASE_TARGET)
        metadata_rows = [{**r, 'stat': a.verify._identity(Path(r['local_path']).lstat())} for r in package['metadata']['files']] + [release_record]
        receipt = a.publish_batch(api, add_factory, package, state, 'metadata-0000', metadata_rows)
        result = {'schema': 'paper-data-publication-complete-v1', 'status': 'verified', 'repo_id': a.REPO_ID,
                  'plan_sha256': package['plan_sha256'], 'metadata_sha256': package['metadata_sha256'],
                  'payload_files': len(rows), 'payload_bytes': sum(r['size'] for r in rows),
                  'data_commit_sha': commit, 'metadata_commit_sha': receipt['commit_sha'],
                  'data_release_receipt_sha256': a.digest(release_path), 'verified_payload_batches': len(receipts),
                  'local_files_retained': True, 'completed_at_utc': a.now(), 'profile': 'scientific_addendum_20260911'}
        a.save(state / 'completed.json', result)
        return result


a.publish = publish
if __name__ == '__main__':
    raise SystemExit(a.main())
