#!/usr/bin/env python3
"""Publish the separately pinned paper evidence completion; upload only, no deletion."""
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
a.PACKAGE = ROOT / 'var/artifacts/hf_paper_evidence_release_20260911/package'
a.EXPECTED_PLAN_SHA256 = '5f26031245467dfdee4c4eb4bd607d0bd28dec714427e9ffd948dcd6c143819e'
a.EXPECTED_METADATA_SHA256 = '6e595c6f614d37e50b217a9d86db57bd78a38165c45770345088f1d3653e34c1'
a.EXPECTED_COUNT = 324
a.METADATA_TARGETS = {'PAPER_EVIDENCE.md', 'reproducibility/PAPER_EVIDENCE_MANIFEST.json'}
RELEASE_TARGET = 'reproducibility/PAPER_EVIDENCE_RELEASE.json'
MANIFEST_TARGET = 'reproducibility/PAPER_EVIDENCE_MANIFEST.json'



_BASE_LOAD_PACKAGE = a.load_package
INDEX_TARGET = 'reproducibility/PAPER_MODEL_EVIDENCE_INDEX.json'
READABLE_INDEX_TARGET = 'reproducibility/PAPER_MODEL_EVIDENCE_INDEX.md'


def validate_template(package):
    """Authenticate all template references against the three pinned allowlists."""
    import re
    plan = package['plan']
    template_pin = plan['evidence_index_template']
    template_path = a.source_path(template_pin['local_path'])
    a.require(a.digest(template_path) == template_pin['sha256'], 'evidence_index_template_changed')
    template = a.read(template_path)
    a.require(template.get('schema') == 'paper-model-evidence-index-template-v1' and template.get('repo_id') == a.REPO_ID, 'evidence_index_schema_differs')
    a.require(len(template['models']) == template['scientific_model_records'] == 762, 'evidence_model_count_differs')
    a.require(len({r['run_repository_path'] for r in template['models']}) == 762, 'duplicate_scientific_model')
    a.require(template['model_archive_entries'] == 707 and template['missing_weight_exports'] == 55, 'model_availability_counts_differ')
    refs = {'completion': {r['path_in_repo']: r for r in plan['files']}}
    revisions = {}
    a.require(set(plan['old_releases']) == {'initial', 'supplement'}, 'prior_release_set_differs')
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
    a.require(sum(r['model_download_status'] == 'weights_unavailable' for r in template['models']) == 55 and sum(r['model_download_status'] == 'consult_verified_model_catalog' for r in template['models']) == 707, 'actual_model_availability_counts_differ')
    for model in template['models']:
        a.require(model['direct_model_download_url'] is None and model['model_download_status'] in ('consult_verified_model_catalog', 'weights_unavailable'), 'unverified_model_download_claim')
        a.require(model['terminal_update'] == 3072 and model['evaluation_sources'], 'terminal_evidence_missing')
    return template, revisions


def load_package(plan_path, metadata_path):
    package = _BASE_LOAD_PACKAGE(plan_path, metadata_path)
    validate_template(package)
    return package



def render_readable_index(index):
    """Human-readable study groups with stable per-record anchors and exact files."""
    groups = {}
    for model in index['models']: groups.setdefault(model['experiment'], []).append(model)
    models = {'qwen05b': 'Qwen2.5-0.5B-Instruct', 'qwen3b': 'Qwen2.5-3B-Instruct', 'falcon1b': 'Falcon3-1B-Instruct'}
    domains = {'graph_coloring': 'Graph Coloring', 'countdown': 'Countdown', 'python_factors': 'Python Factors', 'mathir': 'MathIR', 'pantry_plan': 'PantryPlan'}
    methods = {'control': 'Dr.GRPO', 'replay': 'Re:Dr', 'drgrpo': 'Dr.GRPO', 'replay_drgrpo': 'Re:Dr', 'maxrl': 'MaxRL', 'replay_maxrl': 'Re:Max', 'grpo_plain_control': 'GRPO', 'ucpo': 'UCPO', 'rlep_dr_sparse': 'Sparse RLEP-Dr', 'fresh_frequency': 'Fresh-frequency replay', 'semantic': 'Replay with fixed semantic regularization', 'semantic_only': 'Fixed semantic regularization', 'verified_support_discovery': 'Verified-support treatment', 'fixed_bank_survival': 'Fixed-bank survival'}
    def label(exp): return exp.upper().replace('R1', '-R1')
    def safe(value): return str(value).replace('|', r'\|').replace('\n', ' ')
    out = ['# Per-model evaluation evidence', '', 'Find a study below, then its domain, model, method and seed. Every evaluation-file link points to the exact immutable data revision. The JSON index records SHA256 digests and the four admitted draw-line locations at optimizer update 3072; export step 3073 is the following checkpoint label where applicable.', '', '**Availability:** 707 model entries use the separately verified model catalog; 55 E95 records have evaluation evidence but no downloadable weights. The three closing completions are labeled and do not alter older frozen comparisons.', '', '[Machine-readable index](PAPER_MODEL_EVIDENCE_INDEX.json) · [Evidence guide](../PAPER_EVIDENCE.md) · [Verified model catalog](../catalog.json)', '']
    out += [' · '.join('[' + label(exp) + '](#study-' + exp + ')' for exp in sorted(groups)), '']
    for exp in sorted(groups):
        out += ['<a id="study-' + exp + '"></a>', '## ' + label(exp), '']
        if exp == 'e95': out += ['These 55 admitted GRPO records have no downloadable model weights.', '']
        if exp == 'e112r1': out += ['The verified-support treatment is the registered bundled semantic-regularization and isolated-proposal comparison; its original protocol and paired analysis define the comparison scope.', '']
        if exp == 'e87': out += ['Historical one-seed descriptive probe, separate from balanced factorial evidence.', '']
        out += ['| Domain | Model | Method | Seed | Exact evaluation files | Model availability |', '| --- | --- | --- | ---: | --- | --- |']
        for model in sorted(groups[exp], key=lambda r: (r['domain'], r['model_key'], r['method'], r['seed'])):
            files = ' · '.join('[File ' + str(n + 1) + '](' + ref['download_url'] + ')' for n, ref in enumerate(model['evaluation_sources']))
            availability = '**Weights unavailable**' if model['model_download_status'] == 'weights_unavailable' else '[Check verified catalog](' + model['verified_model_catalog_url'] + ')'
            method = methods.get(model['method'], model['method']) + (' (closing completion)' if model.get('closing_completion_delta') else '')
            cells = ['<a id="' + model['evidence_index_anchor'] + '"></a>' + domains.get(model['domain'], model['domain']), models.get(model['model_key'], model['model_key']), method, model['seed'], files, availability]
            out.append('| ' + ' | '.join(safe(cell) for cell in cells) + ' |')
        out.append('')
    out += ['## Partial-checkpoint evidence', '', 'The JSON index separately maps the exact historical full files and newline-terminated prefixes used by partial training-curve snapshots. Their captured bytes and hashes are preserved; they are not relabeled as completed runs. Whole source files may contain multiple checkpoints; follow the recorded analysis boundaries.', '']
    return '\n'.join(out)


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
    target = state / 'PAPER_MODEL_EVIDENCE_INDEX.json'
    if target.exists():
        a.require(a.read(target) == index, 'saved_evidence_index_differs')
    else:
        a.save(target, index)
    readable = state / 'PAPER_MODEL_EVIDENCE_INDEX.md'
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
                        'verifier_sha256': a.digest(a.verify.__file__), 'index_template_sha256': package['plan']['evidence_index_template']['sha256'], 'profile': 'paper_evidence_completion_20260911'}
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
        release_path = state / 'PAPER_EVIDENCE_RELEASE.json'
        release_core = {'schema': 'paper-data-release-v1', 'repo_id': a.REPO_ID, 'file_count': len(rows),
                        'total_bytes': sum(r['size'] for r in rows),
                        'data_manifest_sha256': next(r['sha256'] for r in package['metadata']['files'] if r['path_in_repo'] == MANIFEST_TARGET),
                        'data_plan_sha256': package['plan_sha256'],
                        'source_selection': 'Missing admitted terminal evaluation sidecars, dated frozen scientific assets and selection records, three closing cells and their admission evidence, and exact historical partial-checkpoint source bytes; original analysis selection unchanged. Earlier releases are referenced separately.'}
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
                  'local_files_retained': True, 'completed_at_utc': a.now(), 'profile': 'paper_evidence_completion_20260911'}
        a.save(state / 'completed.json', result)
        return result


a.load_package = load_package
a.publish = publish
if __name__ == '__main__':
    raise SystemExit(a.main())
