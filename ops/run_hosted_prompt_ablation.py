#!/usr/bin/env python3
"""Collect the frozen prompt ablation using an explicitly supplied credential."""
from __future__ import annotations
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ops'))
from evaluate_native_prompt_ablation import atomic, file_sha, now, verify_snapshot
from prepare_hosted_prompt_ablation import BASE, SOURCES, CONDITION

AMENDMENT = 'execution_amendment_v2.json'


def credential(path=None):
    if path is None:
        key = os.environ.get('AZURE_OPENAI_API_KEY', '')
    else:
        path = Path(path)
        metadata = path.stat()
        if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) & 0o077:
            raise ValueError('Credential file must be owned by this user and inaccessible to other users')
        key = path.read_text().strip()
        if key.startswith('AZURE_OPENAI_API_KEY='):
            key = key.split('=', 1)[1].strip().strip('\"').strip("'")
    if not key or '\n' in key or '\r' in key:
        raise ValueError('Supply AZURE_OPENAI_API_KEY or an explicitly provided private credential file')
    return key


def validate_inventory(base):
    registry_path = base / 'hosted_analysis_runs.json'
    registry = json.loads(registry_path.read_text())
    if registry['experiment_condition'] != CONDITION or registry['ablation_manifest_sha256'] != file_sha(base / 'manifest.json'):
        raise ValueError('Ablation registry differs from frozen preparation')
    expected = {(slug, arm) for slug in SOURCES for arm in ('original', 'neutral')}
    seen = set()
    for entry in registry['runs']:
        run = Path(entry['run_dir'])
        key = (entry['model_id'], entry['arm'])
        if key not in expected or key in seen or entry['manifest_sha256'] != file_sha(run / 'manifest.json'):
            raise ValueError('Unexpected or changed hosted cohort')
        seen.add(key)
        manifest = json.loads((run / 'manifest.json').read_text())
        if (manifest['experiment_condition'] != CONDITION or manifest['prompt_arm'] != entry['arm']
                or not manifest['fresh_response_cohort'] or manifest['request_count'] != 1536):
            raise ValueError('Invalid fresh prompt arm')
        verify_snapshot(run, manifest)
    if seen != expected:
        raise ValueError('Missing paired cohort')
    amendment = json.loads((base / AMENDMENT).read_text())
    if (amendment.get('schema') != 'modebench-prompt-hint-ablation-execution-amendment-v2'
            or amendment.get('previous_amendment_sha256') != file_sha(base / 'execution_amendment.json')
            or amendment.get('root_manifest_sha256') != file_sha(base / 'manifest.json')
            or amendment.get('hosted_registry_sha256') != file_sha(registry_path)
            or amendment.get('orchestrator_sha256') != file_sha(Path(__file__))):
        raise ValueError('Prospective execution amendment does not bind this run')
    for name, digest in amendment['local_artifact_sha256'].items():
        if file_sha(base / name) != digest:
            raise ValueError('Amendment-bound local provenance changed: ' + name)
    return registry


def authenticated_completed(output):
    """Validate saved atomic grades with their own frozen native adapter; no HTTP.

    Status files and orchestration markers are never completion evidence. Every
    retained grade must bind a frozen request, row, and authentic raw response.
    """
    output = Path(output)
    manifest = json.loads((output / 'manifest.json').read_text())
    verify_snapshot(output, manifest)
    adapter_path = output / 'code/ops/evaluate_native_prompt_ablation.py'
    if manifest['code_sha256'].get('ops/evaluate_native_prompt_ablation.py') != file_sha(adapter_path):
        raise ValueError('Recovery requires the frozen native adapter')
    spec = importlib.util.spec_from_file_location('_ablation_recovery_' + file_sha(adapter_path), adapter_path)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    rows_list = native.read_jsonl(output / 'rows.jsonl')
    rows = {native.identity(row): row for row in rows_list}
    items = native.read_jsonl(output / 'requests.jsonl')
    requests = {item['sample_id']: item for item in items}
    group_list = native.read_jsonl(output / 'http_requests.jsonl')
    groups = {group['group_id']: group for group in group_list}
    if (len(rows) != len(rows_list) or len(requests) != len(items)
            or len(groups) != len(group_list) or len(items) != manifest['request_count']):
        raise ValueError('Duplicate or incomplete frozen recovery inventory')
    assigned = set()
    for group in group_list:
        if (group['request_sha256'] != native.sha(group['request'])
                or group['request']['model'] != manifest['model']
                or len(group['sample_ids']) != group['sample_count']):
            raise ValueError('Corrupt frozen recovery HTTP group')
        for sid in group['sample_ids']:
            if sid not in requests or sid in assigned:
                raise ValueError('Duplicate or unknown frozen recovery sample')
            assigned.add(sid)
            item = requests[sid]
            if (item['group_id'] != group['group_id']
                    or item['request_sha256'] != group['request_sha256']
                    or item['request'] != group['request']
                    or item['row_sha256'] != native.sha(rows[native.identity(item)])):
                raise ValueError('Frozen recovery sample/group identity mismatch')
    if assigned != set(requests):
        raise ValueError('Missing frozen recovery HTTP group')
    evidence, raw_cache, provider_ids = [], {}, set()
    for path in sorted((output / 'sample_receipts').glob('*.json')):
        record = json.loads(path.read_text())
        sid = record.get('sample_id')
        if sid not in requests or path.name != sid + '.json':
            raise ValueError('Unexpected saved recovery sample identity')
        item = requests[sid]
        native.validate_completed(output, item, groups[item['group_id']], record, raw_cache)
        if any(record.get(key) != item.get(key) for key in ('prompt_arm', 'experiment_condition')):
            raise ValueError('Saved sample belongs to a different prompt arm or experiment')
        provider_id = tuple(record['provider_sample_identity'])
        if provider_id in provider_ids:
            raise ValueError('Duplicate retained provider sample identity')
        provider_ids.add(provider_id)
        evidence.append({'sample_id': sid, 'sample_receipt': str(path.relative_to(output)),
                         'sample_receipt_sha256': file_sha(path), 'raw_receipt': record['raw_receipt'],
                         'raw_receipt_file_sha256': file_sha(output / record['raw_receipt']),
                         'provider_sample_identity': list(provider_id)})
    return evidence


def recover_preflight(entry):
    """Restore an authenticated preflight marker without requesting another draw."""
    output = Path(entry['run_dir'])
    evidence = authenticated_completed(output)
    if not evidence:
        return None
    if not 1 <= len(evidence) <= 1536:
        raise ValueError('Preflight recovery count must be in 1..1536')
    marker_path = output / 'preflight_result.json'
    old = json.loads(marker_path.read_text()) if marker_path.exists() else None
    manifest_sha = file_sha(output / 'manifest.json')
    if (old is not None and old.get('exit_code') == 0 and old.get('terminal_samples') == len(evidence)
            and old.get('authenticated_samples') == evidence and old.get('manifest_sha256') == manifest_sha):
        return old
    marker = {'model': entry['model_id'], 'arm': entry['arm'], 'stage': 'preflight',
              'exit_code': 0, 'terminal_samples': len(evidence), 'completed_at_utc': now(),
              'recovered_without_generation': True, 'authenticated_samples': evidence,
              'manifest_sha256': manifest_sha,
              'validation_adapter_sha256': file_sha(output / 'code/ops/evaluate_native_prompt_ablation.py'),
              'validation': 'Frozen native.validate_completed against frozen rows, requests, HTTP groups, and raw receipts.'}
    if old is not None:
        marker['previous_marker_sha256'] = file_sha(marker_path)
        marker['previous_marker'] = old
    atomic(marker_path, marker)
    return marker


def run(args):
    base = args.base.resolve()
    registry = validate_inventory(base)
    if args.stage == 'status':
        for entry in registry['runs']:
            path = Path(entry['run_dir']) / 'status.json'
            status = json.loads(path.read_text()) if path.exists() else {}
            print(json.dumps({'model': entry['model_id'], 'arm': entry['arm'], **status}), flush=True)
        return 0
    pending = []
    for entry in registry['runs']:
        marker = recover_preflight(entry)
        if args.stage == 'full' and marker is None:
            raise ValueError('Every frozen arm requires at least one authenticated retained preflight response')
        retained = marker['terminal_samples'] if marker is not None else 0
        if retained and (args.stage == 'preflight' or retained == 1536):
            print(json.dumps({'model': entry['model_id'], 'arm': entry['arm'],
                              'status': 'retained_complete', 'authenticated_samples': retained}), flush=True)
        else:
            pending.append(entry)
    if not pending:
        return 0
    child_env = os.environ.copy()
    child_env['AZURE_OPENAI_API_KEY'] = credential(args.credential_file)
    started = now()
    log_root = base / 'hosted_collection_logs' / (args.stage + '_' + started.replace(':', '-'))
    log_root.mkdir(parents=True)
    atomic(log_root / 'execution_intent.json', {
        'stage': args.stage, 'started_at_utc': started, 'fresh_response_cohort': True,
        'registry_sha256': file_sha(base / 'hosted_analysis_runs.json'),
        'execution_amendment_sha256': file_sha(base / AMENDMENT),
        'orchestrator_sha256': file_sha(Path(__file__)), 'workers_per_arm': 1 if args.stage == 'preflight' else args.workers,
        'arms_concurrent': True, 'credentials_recorded': False})
    processes = []
    for entry in pending:
        output = Path(entry['run_dir'])
        command = [sys.executable, str(output / 'code/ops/evaluate_native_prompt_ablation.py'), 'run',
                   '--model', entry['model'], '--output', str(output), '--profile', str(output / 'model_profile.json'),
                   '--workers', '1' if args.stage == 'preflight' else str(args.workers), '--request-timeout', '600',
                   '--max-attempts', '2' if args.stage == 'preflight' else '8']
        if args.stage == 'preflight':
            command.extend(['--max-new', '1'])
        log_path = log_root / (entry['model_id'] + '_' + entry['arm'] + '.log')
        handle = log_path.open('x')
        process = subprocess.Popen(command, cwd=ROOT, env=child_env, stdin=subprocess.DEVNULL,
                                   stdout=handle, stderr=subprocess.STDOUT)
        processes.append((entry, process, handle, log_path, command))
        atomic(log_root / (entry['model_id'] + '_' + entry['arm'] + '_started.json'),
               {'pid': process.pid, 'command': command, 'started_at_utc': now()})
    results = []
    while processes:
        remaining = []
        for entry, process, handle, log_path, command in processes:
            exit_code = process.poll()
            if exit_code is None:
                remaining.append((entry, process, handle, log_path, command))
                continue
            handle.close()
            output = Path(entry['run_dir'])
            status_path = output / 'status.json'
            status = json.loads(status_path.read_text()) if status_path.exists() else {}
            result = {'model': entry['model_id'], 'arm': entry['arm'], 'stage': args.stage,
                      'exit_code': exit_code, 'terminal_samples': status.get('completed_samples', 0),
                      'started_at_utc': started, 'completed_at_utc': now(), 'command': command,
                      'log': str(log_path.relative_to(base)), 'log_sha256': file_sha(log_path)}
            atomic(output / ('preflight_result.json' if args.stage == 'preflight' else 'collection_result.json'), result)
            results.append(result)
            print(json.dumps(result), flush=True)
        processes = remaining
        if processes:
            time.sleep(2)
    atomic(log_root / 'results.json', results)
    return int(any(row['exit_code'] != 0 for row in results))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['preflight', 'full', 'status'])
    parser.add_argument('--base', type=Path, default=BASE)
    parser.add_argument('--credential-file', type=Path)
    parser.add_argument('--workers', type=int, default=24)
    args = parser.parse_args()
    if not 1 <= args.workers <= 64:
        parser.error('workers must be in 1..64')
    with (args.base / '.hosted_collection.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
