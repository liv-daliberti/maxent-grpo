#!/usr/bin/env python3
"""Canary-gated, resumable orchestration for the authorized seven-model subset."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ops'))
from evaluate_chat_frontier_modebench import atomic, file_sha, now, verify_snapshot
from run_hosted_prompt_ablation import credential
from evaluate_hosted_reasoning_off import load_adapter, control_evidence
BASE = ROOT / 'artifacts/hosted_reasoning_off_32_20260911_v2'


def read(path):
    return json.loads(Path(path).read_text())


def registry_for(base):
    registry = read(base / 'experiment.json')
    amendment_path = base / 'deployment_amendments.json'
    if amendment_path.exists():
        amendment = read(amendment_path)
        if amendment['original_experiment_sha256'] != file_sha(base / 'experiment.json'):
            raise ValueError('Deployment amendment is bound to a different experiment')
        for replacement in amendment['replacements']:
            matches = [i for i, entry in enumerate(registry['runs']) if entry['slug'] == replacement['slug']]
            if len(matches) != 1:
                raise ValueError('Unknown amended deployment')
            index = matches[0]
            if registry['runs'][index] != replacement['original_entry']:
                raise ValueError('Original deployment binding differs')
            updated = replacement['replacement_entry']
            if updated['slug'] != replacement['slug'] or updated['model'] != registry['runs'][index]['model']:
                raise ValueError('Amendment changed deployment identity')
            registry['runs'][index] = updated
    return registry


def collector_running(run, proc_root=Path('/proc')):
    """Only report the saved collector as live when its full command still matches."""
    try:
        marker = read(run / 'active_collection.json')
        pid, command = marker.get('pid'), marker.get('command')
        if type(pid) is not int or pid <= 0 or not isinstance(command, list) or not command:
            return False
        if not all(isinstance(arg, str) for arg in command):
            return False
        process = proc_root / str(pid)
        actual = process.joinpath('cmdline').read_bytes().rstrip(b'\0').split(b'\0')
        if actual != [os.fsencode(arg) for arg in command]:
            return False
        # Zombies have exited even before their parent has collected the status.
        state = process.joinpath('stat').read_text().rsplit(')', 1)[1].split()[0]
        return state not in ('Z', 'X')
    except (OSError, ValueError, TypeError, AttributeError, IndexError):
        return False


def status(base):
    result = {'at_utc': now(), 'planned_terminal_generations': 26880, 'runs': []}
    for entry in registry_for(base)['runs']:
        run = Path(entry['run_directory'])
        current = read(run / 'status.json') if (run / 'status.json').exists() else {}
        canary = read(run / 'canary_result.json') if (run / 'canary_result.json').exists() else {}
        result['runs'].append({'slug': entry['slug'], 'model': entry['model'],
            'completed': current.get('completed_samples', 0), 'expected': 3840,
            'canary': canary.get('status', 'pending'), 'complete': current.get('complete', False),
            'failed_this_session': current.get('failed_groups_this_session', current.get('failed_samples_this_session', 0)),
            'updated_at_utc': current.get('updated_at_utc'),
            'collection_state': read(run / 'provider_control_violation.json').get('status') if (run / 'provider_control_violation.json').exists() else ('complete' if current.get('complete') else ('collecting' if collector_running(run) else 'stopped_incomplete')),
            'admissible_for_final_comparison': bool(current.get('complete')) and not (run / 'provider_control_violation.json').exists()})
    result['completed'] = sum(r['completed'] for r in result['runs'])
    return result


def run(base, credential_file=None, only_slug=None):
    registry = registry_for(base)
    if registry['schema'] != 'hosted-reasoning-off-first32-v1' or registry['planned_terminal_generations'] != 26880:
        raise ValueError('Wrong authorized scope')
    for entry in registry['runs']:
        output = Path(entry['run_directory'])
        if file_sha(output / 'manifest.json') != entry['manifest_sha256']:
            raise ValueError('Root registry binding changed')
        verify_snapshot(output, read(output / 'manifest.json'))
    if only_slug is not None:
        registry['runs'] = [entry for entry in registry['runs'] if entry['slug'] == only_slug]
        if len(registry['runs']) != 1:
            raise ValueError('Unknown single-deployment collection scope')
    key = credential(credential_file)
    child_env = os.environ.copy()
    child_env['AZURE_OPENAI_API_KEY'] = key
    child_env['AZURE_ANTHROPIC_API_KEY'] = key
    child_env['PYTHONUNBUFFERED'] = '1'
    stamp = now().replace(':', '-')
    log_root = base / 'collection_logs' / stamp
    log_root.mkdir(parents=True)
    active = []
    final = []

    def launch(entry, stage):
        output = Path(entry['run_directory'])
        command = [sys.executable, str(output / 'code/ops/evaluate_hosted_reasoning_off.py'),
                   '--output', str(output), '--workers', '1' if stage == 'canary' else '8', '--rpm', '60']
        if stage == 'canary':
            command += ['--max-new', '1']
        path = log_root / (entry['slug'] + '_' + stage + '.log')
        handle = path.open('w')
        process = subprocess.Popen(command, cwd=ROOT, env=child_env, stdin=subprocess.DEVNULL,
                                   stdout=handle, stderr=subprocess.STDOUT)
        entry_status = {'slug': entry['slug'], 'stage': stage, 'pid': process.pid,
                        'started_at_utc': now(), 'log': str(path.relative_to(base)), 'command': command}
        atomic(output / 'active_collection.json', entry_status)
        active.append((entry, stage, process, handle, path))
        print(json.dumps(entry_status), flush=True)

    for entry in registry['runs']:
        output = Path(entry['run_directory'])
        complete = read(output / 'status.json') if (output / 'status.json').exists() else {}
        if complete.get('complete'):
            continue
        marker_path = output / 'canary_result.json'
        if not marker_path.exists() and any((output / 'sample_receipts').glob('*.json')):
            # Recover the retained first canary after an orchestration interruption.
            # Never ask max-new=1 to generate the second slot merely to rebuild a marker.
            first = json.loads((output / 'requests.jsonl').read_text().splitlines()[0])
            sample_path = output / 'sample_receipts' / (first['sample_id'] + '.json')
            control_path = output / 'control_receipts' / (first['sample_id'] + '.json')
            if len(list((output / 'sample_receipts').glob('*.json'))) != 1 or not sample_path.exists() or not control_path.exists():
                raise ValueError('Unmarked canary inventory needs review before any additional calls')
            manifest = read(output / 'manifest.json')
            native = load_adapter(output, manifest)
            record = read(sample_path)
            if manifest['schema'] == 'frontier-modebench-native-chat-responses-v1':
                groups = {r['group_id']: r for r in map(json.loads, (output / 'http_requests.jsonl').read_text().splitlines())}
                native.validate_completed(output, first, groups[first['group_id']], record, {})
            else:
                native.validate_completed_record(output, first, record)
            control = read(control_path)
            raw_path = (output / control['raw_receipt']).resolve()
            if not raw_path.is_relative_to(output):
                raise ValueError('Canary raw receipt escapes cohort')
            raw = read(raw_path)
            if native.sha(raw) != control['raw_receipt_sha256'] or control['request_sha256'] != first['request_sha256']:
                raise ValueError('Canary control binding differs')
            evidence = control_evidence(first['request'], raw['response'])
            atomic(marker_path, {'slug': entry['slug'], 'stage': 'canary', 'status': 'passed',
                'recovered_without_api_call': True, 'terminal_samples': 1, 'completed_at_utc': now(),
                'control_receipt': str(control_path.relative_to(output)), 'control_receipt_sha256': file_sha(control_path),
                'sample_receipt_sha256': file_sha(sample_path), 'control_evidence': evidence})
        if marker_path.exists():
            marker = read(marker_path)
            if marker.get('status') != 'passed':
                final.append({'slug': entry['slug'], 'status': 'blocked_existing_failed_canary'})
                continue
            control_path = output / marker['control_receipt']
            if file_sha(control_path) != marker['control_receipt_sha256']:
                raise ValueError('Canary control evidence changed')
            launch(entry, 'collection')
        else:
            launch(entry, 'canary')
    while active:
        pending = []
        promote = []
        for entry, stage, process, handle, path in active:
            code = process.poll()
            if code is None:
                pending.append((entry, stage, process, handle, path))
                continue
            handle.close()
            output = Path(entry['run_directory'])
            current = read(output / 'status.json') if (output / 'status.json').exists() else {}
            result = {'slug': entry['slug'], 'stage': stage, 'exit_code': code,
                'completed_at_utc': now(), 'terminal_samples': current.get('completed_samples', 0),
                'log': str(path.relative_to(base)), 'log_sha256': file_sha(path)}
            if stage == 'canary':
                first = json.loads((output / 'requests.jsonl').read_text().splitlines()[0])
                control = output / 'control_receipts' / (first['sample_id'] + '.json')
                sample = output / 'sample_receipts' / (first['sample_id'] + '.json')
                passed = code == 0 and result['terminal_samples'] == 1 and control.exists() and sample.exists()
                result['status'] = 'passed' if passed else 'failed_closed'
                if passed:
                    result.update(control_receipt=str(control.relative_to(output)), control_receipt_sha256=file_sha(control),
                                  sample_receipt_sha256=file_sha(sample), control_evidence=read(control)['native_control_check'])
                    promote.append(entry)
                atomic(output / 'canary_result.json', result)
            else:
                result['status'] = 'complete' if code == 0 and current.get('complete') else 'incomplete_review_required'
                atomic(output / 'collection_result.json', result)
            final.append(result)
            print(json.dumps(result), flush=True)
        active = pending
        for entry in promote:
            launch(entry, 'collection')
        atomic(base / 'status.json', status(base))
        if active:
            time.sleep(5)
    atomic(log_root / 'results.json', final)
    atomic(base / 'status.json', status(base))
    return 0 if all(r['complete'] for r in status(base)['runs']) else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['run', 'status'])
    parser.add_argument('--base', type=Path, default=BASE)
    parser.add_argument('--credential-file', type=Path)
    parser.add_argument('--only-slug', choices=['deepseek_v4_pro'])
    args = parser.parse_args()
    if args.command == 'status':
        print(json.dumps(status(args.base.resolve()), indent=2))
    else:
        with (args.base / ('.orchestration.' + args.only_slug + '.lock' if args.only_slug else '.orchestration.lock')).open('a') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            raise SystemExit(run(args.base.resolve(), args.credential_file, args.only_slug))
