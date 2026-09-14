#!/usr/bin/env python3
"""Run one frozen reasoning-off cohort with native receipt recovery and controls."""
from __future__ import annotations
import argparse
import asyncio
import fcntl
import importlib.util
import json
from pathlib import Path
import sys


def load_adapter(output, manifest):
    path = output / 'code' / manifest['native_runner']
    spec = importlib.util.spec_from_file_location('_reasoning_off_native', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def control_evidence(request, body):
    """Reject contradictory native evidence; never infer a hidden budget from silence."""
    if not isinstance(body, dict):
        raise ValueError('Missing native response body')
    token_counts = []

    def collect_usage(value):
        if isinstance(value, dict):
            for name, child in value.items():
                if name in ('reasoning_tokens', 'thinking_tokens'):
                    token_counts.append(child)
                else:
                    collect_usage(child)
        elif isinstance(value, list):
            for child in value:
                collect_usage(child)

    for field in ('usage', 'usage_metadata', 'metadata'):
        collect_usage(body.get(field) or {})
    if any(count not in (0, None) for count in token_counts):
        raise ValueError('Reasoning-off response reports nonzero native reasoning tokens')
    if 'input' in request:
        if request.get('reasoning', {}).get('effort') != 'none':
            raise ValueError('Frozen Responses request does not disable reasoning')
        returned = body.get('reasoning') or {}
        if returned.get('effort') != 'none':
            raise ValueError('Responses deployment did not echo reasoning effort none')
        if any(item.get('type') == 'reasoning' for item in body.get('output', [])):
            raise ValueError('Responses deployment returned a reasoning output block with none')
        mechanism = 'reasoning.effort=none; native echo verified'
    elif request.get('thinking') == {'type': 'disabled'}:
        if isinstance(body.get('content'), list):
            if any(block.get('type') in ('thinking', 'redacted_thinking') for block in body['content']):
                raise ValueError('Thinking-disabled Messages request returned thinking blocks')
        if body.get('thinking') not in (None, {'type': 'disabled'}):
            raise ValueError('Response contradicts disabled thinking request')
        mechanism = 'thinking.type=disabled; accepted native request, no contradictory exposed reasoning'
    elif request.get('reasoning_effort') == 'none':
        if body.get('reasoning_effort') not in (None, 'none'):
            raise ValueError('Native response contradicts reasoning_effort none')
        mechanism = 'reasoning_effort=none; accepted native request, no contradictory exposed reasoning'
    else:
        raise ValueError('Frozen request has no recognized true-off switch')
    for choice in body.get('choices', []):
        message = choice.get('message') or {}
        if message.get('reasoning_content') or message.get('reasoning'):
            raise ValueError('Reasoning-off Chat request returned nonempty reasoning content')
    return {'mechanism': mechanism, 'returned_model': body.get('model'),
            'native_reasoning_token_counts': token_counts,
            'reasoning_echo': body.get('reasoning'),
            'reasoning_effort_echo': body.get('reasoning_effort'),
            'thinking_echo': body.get('thinking'),
            'returned_temperature': body.get('temperature'), 'returned_top_p': body.get('top_p'),
            'limit': 'Absent provider metadata is not proof of zero hidden computation; native documentation plus accepted explicit control are used.'}


def refuse_uncertain_retries(output, requests):
    """Prevent a restart from silently repeating a potentially paid failed sample."""
    completed = {p.stem for p in (output / 'sample_receipts').glob('*.json')}
    attempts = {}
    events = output / 'events.jsonl'
    if not events.exists():
        return
    for line in events.read_text().splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get('event') == 'request_started':
            sid = event.get('sample_id') or event.get('group_id')
            attempts.setdefault(sid, []).append(event)
    for item in requests:
        sid = item['sample_id']
        if sid in completed or sid not in attempts:
            continue
        receipts = [json.loads(p.read_text()) for p in (output / 'raw_responses').glob(sid + '__*.json')]
        # Only a fully validated saved terminal receipt is certainly recoverable.
        # HTTP 200 alone can contain an unrecognized provider body.
        control_path = output / 'control_receipts' / (sid + '.json')
        if control_path.exists():
            control = json.loads(control_path.read_text())
            import hashlib
            for saved in receipts:
                digest = hashlib.sha256(json.dumps(saved, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
                if (saved.get('http_status') == 200 and digest == control.get('raw_receipt_sha256')
                        and control.get('request_sha256') == item['request_sha256']):
                    break
            else:
                raise ValueError('Saved control record has no matching terminal response: ' + sid)
            continue
        # Only explicit request/auth/rate rejections permit a deliberate resume.
        # Timeouts and upstream 5xx may have generated a billed response.
        if receipts and len(receipts) == len(attempts[sid]) and all(r.get('http_status') in (400, 401, 403, 404, 429) for r in receipts):
            continue
        raise ValueError('Uncertain possibly billed attempt requires separate accounting before resume: ' + sid)


def run(output, max_new=0, workers=8, rpm=60):
    output = Path(output).resolve()
    manifest = json.loads((output / 'manifest.json').read_text())
    if manifest.get('experiment_condition') != 'hosted-reasoning-off-first32-v1' or manifest['request_count'] != 3840:
        raise ValueError('Wrong experiment scope')
    native = load_adapter(output, manifest)
    for name, digest in manifest['artifact_sha256'].items():
        if native.file_sha(output / name) != digest:
            raise ValueError('Frozen experiment artifact changed: ' + name)
    for name, digest in manifest['code_sha256'].items():
        if native.file_sha(output / 'code' / name) != digest:
            raise ValueError('Frozen source changed: ' + name)
    if native.file_sha(Path(__file__)) != manifest['code_sha256']['ops/evaluate_hosted_reasoning_off.py']:
        raise ValueError('Use the original frozen reasoning-off wrapper')
    requests = [json.loads(line) for line in (output / 'requests.jsonl').read_text().splitlines()]
    refuse_uncertain_retries(output, requests)
    grouped = manifest['schema'] == 'frontier-modebench-native-chat-responses-v1'
    validator_name = 'validate_raw' if grouped else 'validate_raw_receipt'
    original_validator = getattr(native, validator_name)

    def validator(item, receipt, *args, **kwargs):
        result = original_validator(item, receipt, *args, **kwargs)
        evidence = control_evidence(item['request'], receipt['response'])
        sid = item.get('sample_id') or item.get('group_id')
        native.atomic(output / 'control_receipts' / (sid + '.json'), {
            'sample_id': sid, 'request_sha256': item['request_sha256'],
            'raw_receipt': receipt['relative_path'], 'raw_receipt_sha256': native.sha(receipt),
            'native_control_check': evidence})
        return result

    setattr(native, validator_name, validator)
    with (output / '.runner.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if grouped:
            return asyncio.run(native.run(output, manifest['model'], workers, 300, 1, max_new, rpm))
        if manifest['schema'] == 'frontier-modebench-anthropic-messages-v1':
            return asyncio.run(native.run(output, workers, 300, 1, max_new, 'x-api-key'))
        return asyncio.run(native.run(output, workers, 300, 1, max_new))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-new', type=int, default=0)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--rpm', type=float, default=60)
    args = parser.parse_args()
    if args.max_new < 0 or args.workers not in range(1, 17) or args.rpm <= 0:
        parser.error('Invalid collection limits')
    raise SystemExit(run(args.output, args.max_new, args.workers, args.rpm))
