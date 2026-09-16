#!/usr/bin/env python3
"""Archive admitted completed exports, verify remote bytes, then retire weights.

The run plan is an explicit allowlist. Original scientific receipts, exported
metadata, evaluations, source snapshots, and all unfinished runs remain local.
Credentials are supplied only through a separate private token file.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import math
from email.utils import parsedate_to_datetime
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_archive_verification as verify

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / 'var/data'
SHARED_LOCK = ROOT / 'var/artifacts/shared_storage_admission_20260911.lock'
DOWNLOAD_ROOT = Path('/tmp/maxent_hf_archive_20260911/verified_downloads')
LICENSE_ROOT = ROOT / 'var/artifacts/hf_model_archive_20260911/base_licenses'
WEIGHT_NAME = re.compile(r'(?:model(?:-\d{5}-of-\d{5})?\.safetensors|pytorch_model(?:-\d{5}-of-\d{5})?\.bin)')
MODELS = {'qwen05b': 'Qwen2.5-0.5B-Instruct', 'qwen3b': 'Qwen2.5-3B-Instruct', 'falcon1b': 'Falcon3-1B-Instruct'}
EXPERIMENTS = {'e118': 'E118', 'e119': 'E119', 'e120r1': 'E120-R1', 'e78': 'E78', 'e79': 'E79', 'e80r1': 'E80-R1'}
from archive_expanded_registry import ADDITIONAL_STUDIES, registry_for
EXPERIMENTS.update({key: value['name'] for key, value in ADDITIONAL_STUDIES.items()})
CATALOG_REFRESH_SECONDS = 120
CATALOG_MAX_RETRIES = 3
CATALOG_FALLBACK_WAIT_SECONDS = 300
CATALOG_MAX_WAIT_SECONDS = 900


def now():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp-{os.getpid()}')
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def emit(event, **fields):
    print(json.dumps({'at_utc': now(), 'event': event, **fields}), flush=True)


def queue_snapshot():
    result = subprocess.run(['squeue', '--json', '-u', 'od2961'], capture_output=True, text=True, check=True, timeout=30)
    value = json.loads(result.stdout)
    require(not value.get('errors') and isinstance(value.get('jobs'), list), 'scheduler evidence is unavailable')
    return value['jobs']


def assert_inactive(record, jobs):
    root, stamp = record['run_dir'], record['run_stamp']
    related = set(map(str, record['all_related_job_ids']))
    require(related, 'completed run lacks mapped scheduler identities')
    for job in jobs:
        text = '\n'.join(str(job.get(k, '')) for k in ('submit_line', 'command', 'name', 'current_working_directory'))
        if str(job.get('job_id')) in related or root in text or stamp in text:
            raise RuntimeError('model still has a queued or running consumer: ' + str(job.get('job_id')))


def check_pins(plan):
    for path, expected in plan['ledger_pins'].items():
        require(digest(path) == expected, 'run mapping changed; refresh archival admission: ' + path)


def check_record(record):
    require(record.get('audited_candidate') is True and record.get('endpoint_status') == 'admitted', 'only admitted completed runs may retire')
    root = Path(record['run_dir'])
    export = Path(record['terminal_export'])
    require(root.resolve(strict=True) == root and DATA_ROOT in root.parents, 'run path differs or escapes var/data')
    require(export.resolve(strict=True) == export and root in export.parents and not export.is_symlink(), 'export path differs or escapes completed run')
    require(Path(record['receipt_path']) == root / 'TRAINING_COMPLETE.json', 'completion receipt is outside its run')
    require(digest(record['receipt_path']) == record['receipt_sha256'], 'original completion receipt changed')
    receipt = read(record['receipt_path'])
    require(receipt.get('schema') == 'oat_zero_training_complete_v1', 'completion receipt schema differs')
    attempt = Path(receipt['terminal_attempt'])
    require(attempt.parent == root and attempt.name.startswith('debug_') and export.parent == attempt / 'saved_models', 'completion attempt/export hierarchy differs')
    require(receipt['terminal_export'] == str(export) and int(receipt['terminal_step']) == record['terminal_step'] >= 3072, 'terminal export/step differs')
    require(record['source_experiment'] in EXPERIMENTS, 'experiment not admitted by this archival plan')


def prepare(inventory_path, output, repo_id, public=False):
    source = read(inventory_path)
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    require(not (output / 'plan.json').exists(), 'archive plan already exists')
    records = []
    for original in source['records']:
        if not original.get('audited_candidate'):
            continue
        record = dict(original)
        exp = EXPERIMENTS[record['source_experiment']]
        model = MODELS[record['model_key']]
        arm = record['source_arm']
        record['repo_prefix'] = f"experiments/{exp}/{model}/{record['domain']}/{arm}/seed-{record['seed']}/step-{record['terminal_step']:05d}"
        record['archive_id'] = hashlib.sha256(record['repo_prefix'].encode()).hexdigest()[:24]
        records.append(record)
    require(records and len({r['terminal_export'] for r in records}) == len(records), 'empty or duplicate physical export plan')
    require(len({r['repo_prefix'] for r in records}) == len(records), 'remote model folder collision')
    records.sort(key=lambda r: (-r['terminal_bytes'], r['repo_prefix']))
    require(sum(r['terminal_bytes'] for r in records) < (9_000_000_000_000 if public else 990_000_000_000), 'archive exceeds reviewed storage allocation')
    plan = {'schema': 'completed-model-hf-archive-plan-v1', 'created_at_utc': now(), 'repo_id': repo_id,
        'private': not public, 'output_dir': str(output), 'inventory_path': str(Path(inventory_path).resolve()),
        'inventory_sha256': digest(inventory_path), 'ledger_pins': source['ledger_pins'],
        'models': records, 'expected_model_count': len(records), 'expected_total_export_bytes': sum(r['terminal_bytes'] for r in records),
        'retention': 'Remove only verified named weight files; retain original receipts, directories, config, tokenizer, index, evaluation, telemetry, and source snapshots.'}
    save(output / 'inventory.json', source)
    save(output / 'plan.json', plan)
    (output / 'plan.sha256').write_text(digest(output / 'plan.json') + '\n')
    emit('prepared', models=len(records), bytes=plan['expected_total_export_bytes'], plan=str(output / 'plan.json'))


def public_manifest(record, manifest):
    return {'schema': 'archived-experiment-model-v1', 'experiment': EXPERIMENTS[record['source_experiment']],
        'model': MODELS[record['model_key']], 'domain': record['domain'], 'method': record['source_arm'],
        'seed': record['seed'], 'terminal_step': record['terminal_step'], 'campaign_crosslist': record['campaign'],
        'training_completion_receipt_sha256': record['receipt_sha256'], 'endpoint_status': 'admitted',
        'original_run_dir': str(Path(record['run_dir']).relative_to(ROOT)),
        'original_terminal_export': str(Path(record['terminal_export']).relative_to(ROOT)),
        'repo_prefix': record['repo_prefix'], 'files': [{k: f[k] for k in ('relative_path', 'size', 'sha256', 'git_blob_sha1')} for f in manifest['files']],
        'total_bytes': manifest['total_bytes']}


def model_card(record):
    licensing = read(LICENSE_ROOT / record['model_key'] / 'PROVENANCE.json')['publication_fields']
    notice = licensing.get('prominent_documentation_phrase', '')
    if 'required_publication_template' in licensing:
        notice = licensing['required_publication_template'].replace('[name of relevant Derivative Work]', MODELS[record['model_key']] + ' ' + EXPERIMENTS[record['source_experiment']] + ' experimental model')
    scope = licensing.get('use_scope', 'Original base-model license applies; see LICENSE.')
    return f"""# {EXPERIMENTS[record['source_experiment']]} · {MODELS[record['model_key']]}

{notice}

{scope}

Domain: **{record['domain']}**. Method: **{record['source_arm']}**.
Seed: **{record['seed']}**. Terminal export step: **{record['terminal_step']}**.

This is an admitted completed experimental model from maxent-grpo. Model files
retain their original bytes. `ARCHIVE_MANIFEST.json` records file sizes, hashes,
completion provenance and original local placement. The archive catalog links
each model to its immutable upload commit. See `LICENSE`, any accompanying
`Notice`, and `MODIFICATIONS.md` for original attribution and fine-tuning changes.

Load this folder with Transformers using `subfolder={record['repo_prefix']!r}`
and the pinned revision from the catalog. For existing offline experiment tools,
restore the files to their original export directory before evaluation.
"""


def license_operations(record, manifest):
    from huggingface_hub import CommitOperationAdd
    directory = LICENSE_ROOT / record['model_key']
    license_file = directory / ('LICENSE' if (directory / 'LICENSE').exists() else 'LICENSE.txt')
    require(license_file.is_file() and (directory / 'PROVENANCE.json').is_file(), 'base-model license provenance is incomplete')
    provenance = read(directory / 'PROVENANCE.json')
    operations = [CommitOperationAdd(path_in_repo=record['repo_prefix'] + '/LICENSE', path_or_fileobj=str(license_file))]
    aup = directory / 'official_acceptable_use_policy.html'
    if aup.exists():
        operations.append(CommitOperationAdd(path_in_repo=record['repo_prefix'] + '/ACCEPTABLE_USE_POLICY.html', path_or_fileobj=str(aup)))
    if (directory / 'Notice').exists():
        operations.append(CommitOperationAdd(path_in_repo=record['repo_prefix'] + '/Notice', path_or_fileobj=str(directory / 'Notice')))
    names = [f['relative_path'] for f in manifest['files'] if WEIGHT_NAME.fullmatch(f['relative_path']) or f['relative_path'].endswith('config.json')]
    text = ('# Modified model files\n\n'
        + f"Base model: {provenance['repo_id']} at revision {provenance['revision']}.\n\n"
        + f"These files were modified by the {EXPERIMENTS[record['source_experiment']]} research fine-tuning run: "
        + f"{record['source_arm']}, {record['domain']}, seed {record['seed']}, terminal step {record['terminal_step']}.\n\n"
        + '\n'.join('- ' + name for name in names) + '\n\n'
        + 'The archived files preserve the exact outputs of that completed training run. '
        + 'Their names, sizes and SHA256 digests are in ARCHIVE_MANIFEST.json.\n'
        + ('\nThe incorporated Falcon Acceptable Use Policy is provided in ACCEPTABLE_USE_POLICY.html and at https://falconllm.tii.ae/falcon-acceptable-use-policy.html.\n' if aup.exists() else ''))
    operations.append(CommitOperationAdd(path_in_repo=record['repo_prefix'] + '/MODIFICATIONS.md', path_or_fileobj=text.encode()))
    return operations


def verify_large_remote_weight(api, repo_id, commit, item):
    """Use parallel Xet transfer in bounded temporary space, then hash all bytes."""
    from huggingface_hub import hf_hub_download
    import shutil
    import tempfile
    DOWNLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    with (DOWNLOAD_ROOT / 'download.lock').open('a+') as lock:
        # Serialize temporary weight materialization; other models can hash/upload.
        fcntl.flock(lock, fcntl.LOCK_EX)
        require(shutil.disk_usage(DOWNLOAD_ROOT).free >= item['size'] + 2 * 1024**3,
                'insufficient temporary space for independent remote verification')
        with tempfile.TemporaryDirectory(prefix='verification-', dir=DOWNLOAD_ROOT) as temporary:
            folder = Path(temporary)
            downloaded = Path(hf_hub_download(repo_id=repo_id, filename=item['path_in_repo'],
                revision=commit, repo_type='model', token=api.token, local_dir=str(folder)))
            require(downloaded.resolve(strict=True) == downloaded and folder in downloaded.parents
                    and downloaded.is_file() and not downloaded.is_symlink(), 'unexpected temporary download path')
            size, actual = downloaded.stat().st_size, digest(downloaded)
            require(size == item['size'] and actual == item['sha256'], 'downloaded remote weight differs from local original')
            return {'path': item['path_in_repo'], 'bytes': size, 'sha256': actual}


def stream_remote_weights(api, repo_id, commit, manifest, weight_names):
    """Independently download and hash every weight; large Xet files use local /tmp."""
    from huggingface_hub import hf_hub_url
    import requests
    results = []
    for item in manifest['files']:
        if item['relative_path'] not in weight_names:
            continue
        if item['size'] >= 64 * 1024**2:
            results.append(verify_large_remote_weight(api, repo_id, commit, item))
            continue
        url = hf_hub_url(repo_id, item['path_in_repo'], revision=commit, repo_type='model')
        h, size = hashlib.sha256(), 0
        with requests.get(url, headers={'Authorization': 'Bearer ' + api.token}, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=8 * 1024**2):
                size += len(chunk); h.update(chunk)
                require(size <= item['size'], 'remote weight exceeds admitted size')
        require(size == item['size'] and h.hexdigest() == item['sha256'], 'downloaded remote weight differs from local original')
        results.append({'path': item['path_in_repo'], 'bytes': size, 'sha256': h.hexdigest()})
    require(len(results) == len(weight_names) > 0, 'remote byte verification omitted a weight')
    return {'status': 'verified', 'repo_id': repo_id, 'commit_sha': commit, 'files': results, 'verified_at_utc': now()}


def weight_files(record, manifest):
    names = set(record['large_weight_files'])
    require(names and all(WEIGHT_NAME.fullmatch(n) for n in names), 'cleanup allowlist contains something other than model weights')
    actual = {f['relative_path'] for f in manifest['files'] if WEIGHT_NAME.fullmatch(f['relative_path'])}
    require(names == actual, 'model weight inventory differs from approved allowlist')
    return names


@contextmanager
def storage_lock():
    with SHARED_LOCK.open('a+') as stream:
        deadline = time.monotonic() + 60
        while True:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB); break
            except BlockingIOError:
                require(time.monotonic() < deadline, 'shared storage admission lock busy; try again')
                time.sleep(.5)
        yield


def retire_weights(plan, record, manifest, verification, state_dir):
    """Delete an exact weights-only allowlist after independent byte verification."""
    require(verification.get('remote_metadata', {}).get('status') == 'verified'
            and verification.get('remote_bytes', {}).get('status') == 'verified', 'both remote verification gates must pass')
    commit = verification['commit_sha']
    require(re.fullmatch('[0-9a-f]{40}', commit or '') is not None
            and verification['remote_metadata']['commit_sha'] == verification['remote_bytes']['commit_sha'] == commit,
            'verification does not bind one immutable commit')
    require(verification.get('repo_id') == plan['repo_id']
            and verification['remote_metadata'].get('repo_id') == plan['repo_id']
            and verification['remote_metadata'].get('repo_type') == 'model'
            and verification['remote_bytes'].get('repo_id') == plan['repo_id'], 'verification repository identity differs')
    require(manifest['repo_prefix'] == record['repo_prefix'], 'manifest prefix differs from admitted model')
    require(verification.get('manifest_sha256') == digest(state_dir / 'manifest.json'), 'verification manifest binding differs')
    upload = read(state_dir / 'upload.json')
    require(upload.get('repo_id') == plan['repo_id'] and upload.get('commit_sha') == commit, 'upload identity differs')
    expected_metadata = {f['path_in_repo']: f['size'] for f in manifest['files']}
    verified_metadata = verification['remote_metadata'].get('verified_files', [])
    require(len(verified_metadata) == len(expected_metadata)
            and verification['remote_metadata'].get('expected_files') == len(expected_metadata)
            and {f['path']: f['size'] for f in verified_metadata} == expected_metadata, 'remote metadata proof omits or changes model files')
    names = weight_files(record, manifest)
    expected = {f['path_in_repo']: (f['size'], f['sha256']) for f in manifest['files'] if f['relative_path'] in names}
    observed = {f['path']: (f['bytes'], f['sha256']) for f in verification['remote_bytes']['files']}
    require(expected == observed, 'remote byte proof differs from exact deletion files')
    # An interrupted retirement may have already unlinked some admitted weights.
    # Only a preexisting intent bound to this exact manifest permits that state.
    absent = {name for name in names if not (Path(manifest['export_dir']) / name).exists()}
    if absent:
        intent_path = state_dir / 'retirement_intent.json'
        require(intent_path.is_file(), 'weight missing without a recorded retirement intent')
        previous = read(intent_path)
        require(previous.get('repo_id') == plan['repo_id'] and previous.get('commit_sha') == commit
                and previous.get('repo_prefix') == record['repo_prefix']
                and previous.get('manifest_sha256') == digest(state_dir / 'manifest.json')
                and set(previous.get('retired_files', [])) == names, 'interrupted retirement intent differs')
    remaining_manifest = {**manifest, 'files': [f for f in manifest['files'] if f['relative_path'] not in absent]}
    remaining_manifest['total_bytes'] = sum(f['size'] for f in remaining_manifest['files'])
    local = verify.verify_local_manifest(remaining_manifest)
    require(local['status'] == 'verified', 'local export changed since upload')
    check_pins(plan); check_record(record)
    export = Path(manifest['export_dir'])
    require(export == Path(record['terminal_export']), 'deletion manifest points to another export')
    receipt = {'schema': 'local-model-archive-receipt-v1', 'repo_id': plan['repo_id'], 'commit_sha': commit,
        'repo_prefix': record['repo_prefix'], 'original_terminal_export': str(export),
        'manifest_path': str(state_dir / 'manifest.json'), 'manifest_sha256': digest(state_dir / 'manifest.json'),
        'verification_path': str(state_dir / 'verification.json'), 'verification_sha256': digest(state_dir / 'verification.json'),
        'completion_receipt_sha256': record['receipt_sha256'], 'prepared_at_utc': now(),
        'retired_files': sorted(names), 'local_metadata_preserved': True,
        'restore': f"python ops/archive_completed_models.py restore --receipt {record['run_dir']}/MODEL_ARCHIVE.json"}
    removed = [{'relative_path': f['relative_path'], 'bytes': f['size'], 'sha256': f['sha256']} for f in manifest['files'] if f['relative_path'] in absent]
    with storage_lock():
        check_pins(plan); check_record(record)
        assert_inactive(record, queue_snapshot())
        for item in remaining_manifest['files']:
            p = Path(item['local_path']); s = p.lstat()
            # Size and inode/mtime/ctime must match exactly; only the mount's
            # device number may differ, because the same unchanged export reads
            # a different st_dev from a different login node.
            require(stat.S_ISREG(s.st_mode) and s.st_size == item['size']
                    and verify.identity_matches(verify._identity(s), item['stat']),
                    'local file changed immediately before cleanup')
        save(state_dir / 'retirement_intent.json', receipt)
        save(Path(record['run_dir']) / 'MODEL_ARCHIVE.json', receipt)
        directory = os.open(export, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            for item in manifest['files']:
                if item['relative_path'] not in names or item['relative_path'] in absent:
                    continue
                current = os.stat(item['relative_path'], dir_fd=directory, follow_symlinks=False)
                # Last gate before the unlink. Size, inode, mtime and ctime must
                # match the manifest exactly; only the mount's device number may
                # differ, for the same reason as the two checks above.
                require(stat.S_ISREG(current.st_mode) and current.st_size == item['size']
                        and verify.identity_matches(verify._identity(current), item['stat']),
                        'weight changed before unlink')
                os.unlink(item['relative_path'], dir_fd=directory)
                removed.append({'relative_path': item['relative_path'], 'bytes': item['size'], 'sha256': item['sha256']})
                save(state_dir / 'retirement_progress.json', {'removed': removed, 'at_utc': now()})
        finally:
            os.close(directory)
    result = {**receipt, 'status': 'retired', 'retired_at_utc': now(), 'removed_files': removed,
        'removed_bytes': sum(r['bytes'] for r in removed)}
    save(state_dir / 'retired.json', result)
    save(Path(record['run_dir']) / 'MODEL_ARCHIVE.json', result)
    return result


def catalog(plan):
    records = []
    out = Path(plan['output_dir'])
    for model in plan['models']:
        directory = out / 'models' / model['archive_id']
        if not (directory / 'verification.json').exists():
            continue
        proof = read(directory / 'verification.json')
        records.append({'experiment': EXPERIMENTS[model['source_experiment']], 'model': MODELS[model['model_key']],
            'domain': model['domain'], 'method': model['source_arm'], 'seed': model['seed'], 'step': model['terminal_step'],
            'repo_prefix': model['repo_prefix'], 'commit_sha': proof['commit_sha'],
            'local_weights_retired': (directory / 'retired.json').exists(), 'export_bytes': model['terminal_bytes']})
    records.sort(key=lambda r: (r['experiment'], r['model'], r['domain'], r['method'], r['seed']))
    return {'schema': 'experiment-model-archive-catalog-v1', 'updated_at_utc': now(), 'repo_id': plan['repo_id'],
        'expected_model_count': len(plan['models']), 'verified_model_count': len(records), 'models': records}


def publish_catalog(api, plan):
    from huggingface_hub import CommitOperationAdd
    from archive_public_docs import render_readmes
    value = catalog(plan)
    studies, methods = registry_for(plan)
    sections = {}
    data_receipt = ROOT / 'var/artifacts/hf_paper_data_inventory_20260911/publication/upload_state/completed.json'
    if data_receipt.is_file():
        data = read(data_receipt)
        require(data.get('schema') == 'paper-data-publication-complete-v1'
                and data.get('status') == 'verified' and data.get('repo_id') == plan['repo_id']
                and data.get('plan_sha256') == 'dfeadbb6bdc58d8f5e0d41245a4d1c23a2f2c74bb5f4fe71c1d00b0ea65e39ab'
                and re.fullmatch('[0-9a-f]{40}', data.get('metadata_commit_sha', '')),
                'data publication receipt is not verified for this archive')
        sections = {'Data': 'data/README.md', 'Results': 'results/README.md',
                    'Reproducibility': 'reproducibility/README.md'}
    benchmark_revision = None
    benchmark_root = ROOT / 'var/artifacts/modebench_hf_release_20260911'
    benchmark_receipt = benchmark_root / 'publication_verified.json'
    if benchmark_receipt.is_file():
        benchmark = read(benchmark_receipt)
        require(benchmark.get('schema') == 'modebench-dataset-publication-v1'
                and benchmark.get('status') == 'verified' and benchmark.get('repo_type') == 'dataset'
                and benchmark.get('repo_id') == 'od2961/ModeBench'
                and benchmark.get('all_files_verified') is True
                and benchmark.get('all_public_downloaded_bytes_sha256_verified') is True
                and re.fullmatch('[0-9a-f]{40}', benchmark.get('commit_sha', ''))
                and benchmark.get('upload_plan_sha256') == digest(benchmark_root / 'upload_plan.json')
                    == 'b0eaa4f3fc8e3961acb5f49f71d539fd8cc0403b8b18dbff21414997847d8070',
                'benchmark publication receipt is not verified for the reviewed dataset')
        benchmark_revision = benchmark['commit_sha']
    pages = render_readmes(plan, value, additional_studies=studies,
                          additional_methods=methods, available_sections=sections,
                          benchmark_revision=benchmark_revision)
    save(Path(plan['output_dir']) / 'catalog.json', value)
    operations = [CommitOperationAdd(path_in_repo=path, path_or_fileobj=body.encode())
                  for path, body in pages.items()]
    operations.append(CommitOperationAdd(path_in_repo='catalog.json',
        path_or_fileobj=json.dumps(value, indent=2).encode()))
    api.create_commit(repo_id=plan['repo_id'], repo_type='model', operations=operations,
        commit_message=f"Index {len(value['models'])} verified experimental models")




def catalog_rate_limit_headers(error):
    """Keep diagnostics to bounded public timing fields, excluding URLs/secrets."""
    response = getattr(error, 'response', None)
    names = {'date': 'Date', 'ratelimit': 'RateLimit',
             'ratelimit-policy': 'RateLimit-Policy', 'retry-after': 'Retry-After'}
    result = {}
    for key, value in (getattr(response, 'headers', None) or {}).items():
        label = names.get(str(key).lower())
        text = str(value)
        if label and len(text) <= 512 and re.fullmatch(r'[A-Za-z0-9 ,;=._"()\-:+]*', text):
            result[label] = text
    return result


def catalog_retry_delay(error):
    """Read only public rate-limit timing; never expose HTTP exception strings."""
    headers = {k.lower(): v for k, v in catalog_rate_limit_headers(error).items()}
    delays = []
    retry_after = headers.get('retry-after', '')
    try:
        delay = float(retry_after)
    except ValueError:
        try:
            delay = parsedate_to_datetime(retry_after).timestamp() - time.time()
        except (ValueError, TypeError, OverflowError):
            delay = None
    if delay is not None and math.isfinite(delay) and delay >= 0:
        delays.append(delay)
    # Hugging Face uses t=seconds-until-reset in its RateLimit response field.
    for value in re.findall(r'(?:^|[;,])\s*t\s*=\s*(\d+(?:\.\d+)?)(?=\s*(?:[;,]|$))', headers.get('ratelimit', '')):
        delay = float(value)
        if math.isfinite(delay):
            delays.append(delay)
    return max(delays) + 2 if delays else CATALOG_FALLBACK_WAIT_SECONDS


def publish_catalog_with_retry(api, plan):
    """Retry only a rejected metadata publication, with a finite cooldown budget."""
    for attempt in range(CATALOG_MAX_RETRIES + 1):
        try:
            return publish_catalog(api, plan)
        except Exception as error:
            status = getattr(getattr(error, 'response', None), 'status_code', None)
            if status != 429 or attempt == CATALOG_MAX_RETRIES:
                raise
            delay = catalog_retry_delay(error)
            if delay > CATALOG_MAX_WAIT_SECONDS:
                emit('catalog_retry_wait_exceeds_bound', wait_seconds=delay,
                     max_wait_seconds=CATALOG_MAX_WAIT_SECONDS, headers=catalog_rate_limit_headers(error))
                raise
            emit('catalog_rate_limited', retry=attempt + 1,
                 max_retries=CATALOG_MAX_RETRIES, wait_seconds=delay,
                 headers=catalog_rate_limit_headers(error))
            # Keep each sleep responsive to interruption; retry no model action.
            remaining = delay
            while remaining > 0:
                chunk = min(30, remaining)
                time.sleep(chunk)
                remaining -= chunk


def archive_one(api, plan, record, *, keep_local):
    from huggingface_hub import CommitOperationAdd
    directory = Path(plan['output_dir']) / 'models' / record['archive_id']
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'retired.json').exists():
        return read(directory / 'retired.json')
    check_pins(plan); check_record(record); assert_inactive(record, queue_snapshot())
    save(directory / 'record.json', record)
    if (directory / 'manifest.json').exists():
        manifest = read(directory / 'manifest.json')
    else:
        emit('hashing', model=record['repo_prefix'], bytes=record['terminal_bytes'])
        manifest = verify.build_manifest(record['terminal_export'], record['repo_prefix'])
        save(directory / 'manifest.json', manifest)
    names = weight_files(record, manifest)
    if not (directory / 'upload.json').exists():
        metadata = public_manifest(record, manifest)
        save(directory / 'ARCHIVE_MANIFEST.json', metadata)
        operations = [CommitOperationAdd(path_in_repo=f['path_in_repo'], path_or_fileobj=f['local_path']) for f in manifest['files']]
        operations += [CommitOperationAdd(path_in_repo=record['repo_prefix'] + '/ARCHIVE_MANIFEST.json', path_or_fileobj=str(directory / 'ARCHIVE_MANIFEST.json')),
            CommitOperationAdd(path_in_repo=record['repo_prefix'] + '/README.md', path_or_fileobj=model_card(record).encode())]
        operations += license_operations(record, manifest)
        emit('uploading', model=record['repo_prefix'], bytes=manifest['total_bytes'])
        result = api.create_commit(repo_id=plan['repo_id'], repo_type='model', operations=operations, num_threads=2,
            commit_message='Archive ' + record['repo_prefix'].removeprefix('experiments/'))
        save(directory / 'upload.json', {'commit_sha': result.oid, 'repo_id': plan['repo_id'], 'at_utc': now()})
    upload = read(directory / 'upload.json')
    require(upload.get('repo_id') == plan['repo_id'], 'saved upload belongs to another repository')
    commit = upload['commit_sha']
    if (directory / 'verification.json').exists():
        proof = read(directory / 'verification.json')
        require(proof['commit_sha'] == commit, 'saved verification belongs to another upload')
    else:
        emit('verifying_remote', model=record['repo_prefix'], commit=commit)
        remote = verify.verify_remote_manifest(api, plan['repo_id'], commit, manifest)
        require(remote['status'] == 'verified', 'remote metadata verification failed: ' + json.dumps(remote['errors']))
        streamed = stream_remote_weights(api, plan['repo_id'], commit, manifest, names)
        proof = {'schema': 'verified-model-archive-v1', 'repo_id': plan['repo_id'], 'commit_sha': commit,
            'manifest_sha256': digest(directory / 'manifest.json'),
            'remote_metadata': remote, 'remote_bytes': streamed, 'verified_at_utc': now()}
        save(directory / 'verification.json', proof)
    if keep_local:
        emit('verified_kept_local', model=record['repo_prefix'], commit=commit)
        return proof
    emit('retiring_verified_weights', model=record['repo_prefix'])
    result = retire_weights(plan, record, manifest, proof, directory)
    emit('retired', model=record['repo_prefix'], bytes=result['removed_bytes'], commit=commit)
    return result


def run(plan_path, token_file, limit, keep_local, workers=1):
    from huggingface_hub import HfApi
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
    require(type(workers) is int and 1 <= workers <= 6, 'workers must be 1..6')
    plan_path = Path(plan_path)
    require(digest(plan_path) == plan_path.with_suffix('.sha256').read_text().strip(), 'archive plan changed')
    plan = read(plan_path)
    api = HfApi(token=Path(token_file).read_text().strip())
    require(api.model_info(plan['repo_id']).private == plan['private'], 'archive visibility differs from authorized plan')
    output = Path(plan['output_dir'])
    with (output / 'run.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check_pins(plan)
        pending = []
        for record in plan['models']:
            directory = output / 'models' / record['archive_id']
            if (directory / 'retired.json').exists() or (keep_local and (directory / 'verification.json').exists()):
                continue
            pending.append(record)
        if limit:
            pending = pending[:limit]
        failures = []
        last_catalog_refresh = None
        catalog_dirty = False
        next_record = iter(pending)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}

            def fill_slots():
                while len(futures) < workers:
                    record = next(next_record, None)
                    if record is None:
                        break
                    future = executor.submit(archive_one, api, plan, record, keep_local=keep_local)
                    futures[future] = record

            def report_running():
                save(output / 'status.json', {'status': 'draining' if failures else 'running', 'at_utc': now(),
                    'current_models': [r['repo_prefix'] for r in futures.values()],
                    'keep_local': keep_local, 'workers': workers, 'failures': failures})

            def record_failure(record, error, phase='model'):
                # Avoid signed download URLs/credentials from HTTP exception strings.
                failures.append({'model': record['repo_prefix'] if record else None,
                    'phase': phase, 'error_type': type(error).__name__,
                    'http_status': getattr(getattr(error, 'response', None), 'status_code', None)})
                emit('stopped_model' if record else 'stopped_catalog', **failures[-1])
                if isinstance(error, (RuntimeError, verify.ArchiveVerificationError)):
                    print(str(error), file=sys.stderr, flush=True)

            fill_slots()
            report_running()
            while futures:
                completed, _ = wait(tuple(futures), return_when=FIRST_COMPLETED)
                # Examine the entire completed set before replacing any worker.
                # Catalog publication can take time, so drain newly completed
                # futures too before deciding whether submissions may continue.
                while completed:
                    successful = []
                    for future in completed:
                        record = futures.pop(future)
                        try:
                            future.result()
                            successful.append(record)
                        except Exception as error:
                            record_failure(record, error)
                    if successful:
                        catalog_dirty = True
                    if successful and not failures and (last_catalog_refresh is None
                            or time.monotonic() - last_catalog_refresh >= CATALOG_REFRESH_SECONDS):
                        try:
                            publish_catalog_with_retry(api, plan)
                        except Exception as error:
                            record_failure(None, error, phase='catalog')
                        else:
                            last_catalog_refresh = time.monotonic()
                            catalog_dirty = False
                    completed = {future for future in futures if future.done()}
                if not failures:
                    fill_slots()
                report_running()
        if not failures and (catalog_dirty or last_catalog_refresh is None):
            try:
                publish_catalog_with_retry(api, plan)
            except Exception as error:
                record_failure(None, error, phase='catalog')
        if failures:
            save(output / 'status.json', {'status': 'stopped', 'at_utc': now(), 'failures': failures})
            raise SystemExit(1)
        result = catalog(plan)
        retired = [read(p) for p in output.glob('models/*/retired.json')]
        save(output / 'status.json', {'status': 'batch_finished', 'at_utc': now(), 'verified_models': len(result['models']),
            'retired_models': len(retired), 'removed_bytes': sum(r['removed_bytes'] for r in retired), 'expected_models': len(plan['models'])})
        emit('batch_finished', **read(output / 'status.json'))


def restore(receipt_path, token_file):
    from huggingface_hub import hf_hub_download
    receipt = read(receipt_path); manifest = read(receipt['manifest_path'])
    require(digest(receipt['manifest_path']) == receipt['manifest_sha256'], 'restore manifest changed')
    token = Path(token_file).read_text().strip() if token_file else False
    for item in manifest['files']:
        if item['relative_path'] not in receipt['retired_files']:
            continue
        target = Path(item['local_path'])
        if target.exists():
            require(digest(target) == item['sha256'], 'existing restore target differs')
            continue
        downloaded = hf_hub_download(repo_id=receipt['repo_id'], filename=item['path_in_repo'], revision=receipt['commit_sha'], token=token)
        require(digest(downloaded) == item['sha256'], 'downloaded restore file differs')
        import shutil
        temporary = target.with_name(target.name + '.restoring')
        with open(downloaded, 'rb') as source, temporary.open('xb') as destination:
            shutil.copyfileobj(source, destination, length=8 * 1024**2)
        require(digest(temporary) == item['sha256'], 'restored copy differs')
        os.replace(temporary, target)
    emit('restored', export=receipt['original_terminal_export'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='phase', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--inventory', required=True); p.add_argument('--output', required=True); p.add_argument('--repo-id', required=True); p.add_argument('--public', action='store_true')
    p = sub.add_parser('run'); p.add_argument('--plan', required=True); p.add_argument('--token-file', required=True); p.add_argument('--limit', type=int, default=0); p.add_argument('--keep-local', action='store_true'); p.add_argument('--workers', type=int, default=1)
    p = sub.add_parser('restore'); p.add_argument('--receipt', required=True); p.add_argument('--token-file')
    args = parser.parse_args()
    if args.phase == 'prepare': prepare(args.inventory, args.output, args.repo_id, args.public)
    elif args.phase == 'run': run(args.plan, args.token_file, args.limit, args.keep_local, args.workers)
    else: restore(args.receipt, args.token_file)


if __name__ == '__main__':
    main()
