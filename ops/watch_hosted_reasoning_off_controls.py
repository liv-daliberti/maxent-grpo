#!/usr/bin/env python3
"""Audit nested native token counters while frozen collectors run; no API calls.

This complementary monitor covers every received HTTP200 usage/metadata object
without modifying any frozen request or evaluator. If native metadata contradicts
reasoning-off, stop that exact cohort process and exclude the entire incomplete
cohort. Terminal provider responses remain preserved and are never resampled.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ops'))
from evaluate_chat_frontier_modebench import atomic, file_sha, now
from run_hosted_reasoning_off import registry_for, status


def counters(value, prefix=''):
    result = []
    if isinstance(value, dict):
        for name, child in value.items():
            path = prefix + '/' + name
            if name in ('thinking_tokens', 'reasoning_tokens'):
                result.append({'field': path, 'value': child})
            else:
                result.extend(counters(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            result.extend(counters(child, prefix + '/' + str(index)))
    return result


def run(base):
    seen = set()
    audits = 0
    while True:
        for entry in registry_for(base)['runs']:
            output = Path(entry['run_directory'])
            for path in sorted((output / 'raw_responses').glob('*.json')):
                if path in seen:
                    continue
                receipt = json.loads(path.read_text())
                if receipt.get('http_status') != 200:
                    seen.add(path)
                    continue
                body = receipt.get('response') or {}
                native_counts = []
                for field in ('usage', 'usage_metadata', 'metadata'):
                    native_counts.extend(counters(body.get(field) or {}, field))
                bad = [count for count in native_counts if count['value'] not in (None, 0)]
                if bad and not (output / 'provider_control_violation.json').exists():
                    stopped = None
                    active = output / 'active_collection.json'
                    if active.exists():
                        process = json.loads(active.read_text())
                        pid = process['pid']
                        command_file = Path('/proc') / str(pid) / 'cmdline'
                        if command_file.exists():
                            command = command_file.read_bytes().split(b'\0')
                            if str(output).encode() in command and any(b'evaluate_hosted_reasoning_off.py' in arg for arg in command):
                                os.kill(pid, signal.SIGINT)
                                stopped = pid
                    atomic(output / 'provider_control_violation.json', {
                        'schema': 'hosted-reasoning-off-provider-control-violation-v1',
                        'recorded_at_utc': now(), 'model': entry['model'],
                        'status': 'collection_paused_control_not_reliably_honored',
                        'raw_receipt': str(path.relative_to(output)), 'raw_receipt_file_sha256': file_sha(path),
                        'sample_id': receipt.get('sample_id') or receipt.get('group_id'),
                        'request_sha256': receipt['request_sha256'], 'contradictory_native_counters': bad,
                        'stopped_collector_pid': stopped, 'automatic_retry_permitted': False,
                        'whole_cohort_admissible': False,
                        'reason': 'Native nested usage/metadata reports nonzero thinking/reasoning despite requested off; preserve all evidence and exclude whole incomplete cohort.'})
                seen.add(path)
                audits += 1
        current = status(base)
        current.update(native_control_audit={'received_http200_audited': audits,
            'monitor_source_sha256': file_sha(Path(__file__)),
            'scope': 'Recursively checks reasoning_tokens and thinking_tokens under native usage, usage_metadata and metadata.',
            'api_calls': 0})
        atomic(base / 'experiment_status.json', current)
        print(json.dumps({'at_utc': now(), 'completed': current['completed'], 'native_receipts_audited': audits,
                          'paused': [r['slug'] for r in current['runs'] if r['collection_state'].startswith('collection_paused')]}), flush=True)
        active = False
        for entry in registry_for(base)['runs']:
            output = Path(entry['run_directory'])
            marker = output / 'active_collection.json'
            if not marker.exists():
                continue
            pid = json.loads(marker.read_text())['pid']
            cmd = Path('/proc') / str(pid) / 'cmdline'
            if cmd.exists() and str(output).encode() in cmd.read_bytes().split(b'\0'):
                active = True
        if not active:
            return
        time.sleep(10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    args = parser.parse_args()
    run(args.base.resolve())
