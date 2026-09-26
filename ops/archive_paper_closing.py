#!/usr/bin/env python3
"""Publish the separately pinned closing scientific supplement; upload only, no deletion."""
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
a.PACKAGE = ROOT / 'var/artifacts/paper_final_closing_20260911/supplemental_release'
a.EXPECTED_PLAN_SHA256 = '0cae67aff3d2bbccf48603560aad9c92e3deb4c61bf6d7c81f226411daea3d5d'
a.EXPECTED_METADATA_SHA256 = '5fa2004a9a9372a8bd6a717ddc9eae9b13c5bce962b67ea4ec41f5521169696f'
a.EXPECTED_COUNT = 45
a.METADATA_TARGETS = {'CLOSING_DATA.md', 'reproducibility/CLOSING_DATA_MANIFEST.json'}
RELEASE_TARGET = 'reproducibility/CLOSING_DATA_RELEASE.json'
MANIFEST_TARGET = 'reproducibility/CLOSING_DATA_MANIFEST.json'



_BASE_LOAD_PACKAGE = a.load_package
INDEX_TARGET = 'reproducibility/LATE_COMPLETION_EVIDENCE_INDEX.json'
READABLE_INDEX_TARGET = 'reproducibility/LATE_COMPLETION_EVIDENCE_INDEX.md'


def validate_template(package):
    """Authenticate all template references against the three pinned allowlists."""
    import re
    plan = package['plan']
    template_pin = plan['evidence_index_template']
    template_path = a.source_path(template_pin['local_path'])
    a.require(a.digest(template_path) == template_pin['sha256'], 'evidence_index_template_changed')
    template = a.read(template_path)
    a.require(template.get('schema') == 'paper-model-evidence-index-template-v1' and template.get('repo_id') == a.REPO_ID, 'evidence_index_schema_differs')
    a.require(len(template['models']) == template['scientific_model_records'] == 8, 'evidence_model_count_differs')
    a.require(len({r['run_repository_path'] for r in template['models']}) == 8, 'duplicate_scientific_model')
    a.require(template['model_archive_entries'] == 8 and template['missing_weight_exports'] == 0 and template['prior_scientific_model_records'] == 762, 'model_availability_counts_differ')
    refs = {'completion': {r['path_in_repo']: r for r in plan['files']}}
    revisions = {}
    a.require(set(plan['old_releases']) == {'initial', 'supplement', 'evidence'}, 'prior_release_set_differs')
    for key, pin in plan['old_releases'].items():
        old_path, receipt_path = a.source_path(pin['plan_path']), a.source_path(pin['receipt_path'])
        a.require(a.digest(old_path) == pin['plan_sha256'] and a.digest(receipt_path) == pin['receipt_sha256'], 'prior_release_pins_changed')
        old, receipt = a.read(old_path), a.read(receipt_path)
        a.require(receipt.get('status') == 'verified' and receipt.get('repo_id') == a.REPO_ID and receipt.get('plan_sha256') == pin['plan_sha256'] and receipt.get('data_commit_sha') == pin['commit_sha'] and re.fullmatch('[0-9a-f]{40}', pin['commit_sha']), 'prior_release_proof_invalid')
        refs[key] = {r['path_in_repo']: r for r in old['files']}
        revisions[key] = pin['commit_sha']
    def inspect(value):
        if isinstance(value, dict):
            if 'release' in value and 'path_in_repo' in value:
                row = refs.get(value['release'], {}).get(value['path_in_repo'])
                a.require(row is not None and value.get('sha256') == row['sha256'] and ('size' not in value or value['size'] == row['size']), 'unbound_evidence_reference')
            for item in value.values(): inspect(item)
        elif isinstance(value, list):
            for item in value: inspect(item)
    inspect(template)
    a.require(sum(r['model_download_status'] == 'weights_unavailable' for r in template['models']) == 0 and sum(r['model_download_status'] == 'consult_verified_model_catalog' for r in template['models']) == 8, 'actual_model_availability_counts_differ')
    for model in template['models']:
        a.require(model['direct_model_download_url'] is None and model['model_download_status'] in ('consult_verified_model_catalog', 'weights_unavailable'), 'unverified_model_download_claim')
        a.require(model['terminal_update'] == 3072 and model['evaluation_sources'], 'terminal_evidence_missing')
    inventory_pin = plan['late_completion_inventory']
    inventory_path = a.source_path(inventory_pin['local_path'])
    a.require(a.digest(inventory_path) == inventory_pin['sha256'], 'late_inventory_changed')
    inventory = a.read(inventory_path)
    a.require(len(inventory['records']) == 8, 'late_inventory_count_differs')
    expected = {str(Path(r['run_dir']).relative_to(a.ROOT)): r for r in inventory['records']}
    a.require(set(expected) == {r['run_repository_path'] for r in template['models']}, 'late_inventory_selection_differs')
    for model in template['models']:
        original = expected[model['run_repository_path']]
        audit = original['integrity_audit']
        a.require(original['endpoint_status'] == audit['status'] == 'admitted' and audit['step'] == model['terminal_update'] == 3072, 'late_endpoint_not_admitted')
        identity = (original['source_experiment'], original['domain'], original['model_key'], original['arm'], original['seed'])
        a.require(identity == (model['experiment'], model['domain'], model['model_key'], model['method'], model['seed']), 'late_model_identity_differs')
        source_digests = {r['path']: r['sha256'] for r in audit['sources']}
        got_sources = {str(a.ROOT / r['original_repository_path']): r['sha256'] for r in model['evaluation_sources']}
        a.require(source_digests == got_sources, 'late_evaluation_source_selection_differs')
        draws = model['selected_draw_records']
        a.require(sorted(r['draw_index'] for r in draws) == [0, 1, 2, 3], 'late_draw_set_differs')
        expected_draws = sorted((r['draw_index'], r['line'], r['path']) for r in audit['unique_draw_records'])
        got_draws = sorted((r['draw_index'], r['line'], str(a.ROOT / r['data']['original_repository_path'])) for r in draws)
        a.require(got_draws == expected_draws and all(type(r['line']) is int and r['line'] > 0 for r in draws), 'late_draw_locations_differ')
        a.require(model['completion_receipt']['sha256'] == audit['completion_receipt']['sha256'] and str(a.ROOT / model['completion_receipt']['original_repository_path']) == audit['completion_receipt']['path'], 'late_completion_receipt_differs')
    return template, revisions


def load_package(plan_path, metadata_path):
    package = _BASE_LOAD_PACKAGE(plan_path, metadata_path)
    validate_template(package)
    return package



def render_readable_index(index):
    """Eight dated completions with immutable evaluation and receipt links."""
    out = ['# Late-completion evaluation evidence', '',
           'These eight independently admitted completions were observed at the 2026-09-11 16:05 UTC cutoff: seven E118 and one E119. They supplement the earlier 762-record index; they do not change any earlier frozen comparison or select later runs.', '',
           '[Earlier 762-record index](PAPER_MODEL_EVIDENCE_INDEX.md) · [Closing data guide](../CLOSING_DATA.md) · [Verified model catalog](../catalog.json)', '',
           'Evaluation links below use exact immutable data commits. Each record contains four admitted draws (0–3) at optimizer update 3072. Model availability is determined by the separately verified catalog.', '',
           '| Study | Model | Domain | Method | Seed | Evaluation files | Completion proof |',
           '| --- | --- | --- | --- | ---: | --- | --- |']
    for model in sorted(index['models'], key=lambda r: (r['experiment'], r['model_key'], r['domain'], r['method'], r['seed'])):
        files = ' · '.join('[File ' + str(n + 1) + '](' + ref['download_url'] + ')' for n, ref in enumerate(model['evaluation_sources']))
        cells = ['<a id="' + model['evidence_index_anchor'] + '"></a>' + model['experiment'].upper(), model['model_key'], model['domain'], model['method'], str(model['seed']), files, '[Receipt](' + model['completion_receipt']['download_url'] + ')']
        out.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(out) + '\n'


def finalize_evidence_index(package, state, release):
    """Resolve only authenticated paths to each release's immutable revision."""
    template, revisions = validate_template(package)
    revisions['completion'] = release['commit_sha']
    def resolve(value):
        if isinstance(value, dict):
            result = {key: resolve(item) for key, item in value.items()}
            if 'release' in value and 'path_in_repo' in value:
                commit = revisions[value['release']]
                result['commit_sha'] = commit
                result['download_url'] = 'https://huggingface.co/' + a.REPO_ID + '/resolve/' + commit + '/' + value['path_in_repo']
            return result
        if isinstance(value, list): return [resolve(item) for item in value]
        return value
    index = resolve(template)
    for model in index['models']:
        model['scientific_id'] = hashlib.sha256(model['run_repository_path'].encode()).hexdigest()[:24]
        model['evidence_index_anchor'] = 'evidence-' + model['scientific_id']
    index.update(schema='paper-model-evidence-index-v1', generated_at_utc=release['verified_at_utc'], source_template_sha256=package['plan']['evidence_index_template']['sha256'], data_releases={key: {'commit_sha': commit, 'plan_sha256': package['plan_sha256'] if key == 'completion' else package['plan']['old_releases'][key]['plan_sha256']} for key, commit in revisions.items()})
    target = state / 'LATE_COMPLETION_EVIDENCE_INDEX.json'
    if target.exists():
        a.require(a.read(target) == index, 'saved_evidence_index_differs')
    else:
        a.save(target, index)
    readable = state / 'LATE_COMPLETION_EVIDENCE_INDEX.md'
    text = render_readable_index(index)
    if readable.exists(): a.require(readable.read_text() == text, 'saved_readable_index_differs')
    else: readable.write_text(text)
    return [a.verify._hash_file(a.ROOT, target.relative_to(a.ROOT).as_posix(), INDEX_TARGET), a.verify._hash_file(a.ROOT, readable.relative_to(a.ROOT).as_posix(), READABLE_INDEX_TARGET)]


def publish(api, add_factory, package, state):
    """Reuse the reviewed payload transaction; isolate supplement release metadata."""
    validate_template(package)
    with a.single_writer(state):
        registration = {'schema': 'paper-data-publisher-registration-v1', 'repo_id': a.REPO_ID,
                        'plan_sha256': package['plan_sha256'], 'metadata_sha256': package['metadata_sha256'],
                        'publisher_sha256': a.digest(SOURCE), 'engine_sha256': ENGINE_SHA256,
                        'verifier_sha256': a.digest(a.verify.__file__), 'index_template_sha256': package['plan']['evidence_index_template']['sha256'], 'profile': 'paper_closing_supplement_20260911'}
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
        release_path = state / 'CLOSING_DATA_RELEASE.json'
        release_core = {'schema': 'paper-data-release-v1', 'repo_id': a.REPO_ID, 'file_count': len(rows),
                        'total_bytes': sum(r['size'] for r in rows),
                        'data_manifest_sha256': next(r['sha256'] for r in package['metadata']['files'] if r['path_in_repo'] == MANIFEST_TARGET),
                        'data_plan_sha256': package['plan_sha256'],
                        'source_selection': 'Finite closing supplement: the selected three-model hosted comparison and completed scientific outputs; three retained numerical aggregation records; exact evaluation sources and completion proofs for eight independently admitted training completions observed by 2026-09-11T16:05Z. Previous releases and the 762-record index remain unchanged; later live edits and completions are outside scope.'}
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
        index_records = finalize_evidence_index(package, state, release)
        metadata_rows = [{**r, 'stat': a.verify._identity(Path(r['local_path']).lstat())} for r in package['metadata']['files']] + [release_record] + index_records
        receipt = a.publish_batch(api, add_factory, package, state, 'metadata-0000', metadata_rows)
        result = {'schema': 'paper-data-publication-complete-v1', 'status': 'verified', 'repo_id': a.REPO_ID,
                  'plan_sha256': package['plan_sha256'], 'metadata_sha256': package['metadata_sha256'],
                  'payload_files': len(rows), 'payload_bytes': sum(r['size'] for r in rows),
                  'data_commit_sha': commit, 'metadata_commit_sha': receipt['commit_sha'],
                  'data_release_receipt_sha256': a.digest(release_path), 'verified_payload_batches': len(receipts),
                  'local_files_retained': True, 'completed_at_utc': a.now(), 'profile': 'paper_closing_supplement_20260911'}
        a.save(state / 'completed.json', result)
        return result


a.load_package = load_package
a.publish = publish
if __name__ == '__main__':
    raise SystemExit(a.main())
