#!/usr/bin/env python3
"""Resumable, inference-only Azure Responses evaluation of frozen ModeBench tests.

The credential is read from hidden terminal input or AZURE_OPENAI_API_KEY and
never saved. Requests, raw HTTP receipts, grades, errors and code are retained.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import fcntl
import getpass
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / 'artifacts/frontier_modebench_gpt56sol_20260911'
ENDPOINT = 'https://liv.services.ai.azure.com/openai/v1/responses'
SCHEMA = 'frontier-modebench-responses-v1'


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def append(path, value):
    with Path(path).open('a') as handle:
        handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())


def read_rows(output):
    return [json.loads(line) for line in (output / 'rows.jsonl').read_text().splitlines()]


def prepare(output, max_output_tokens):
    """Freeze every request before observing any test response."""
    if (output / 'manifest.json').exists():
        manifest = json.loads((output / 'manifest.json').read_text())
        if manifest['max_output_tokens'] != max_output_tokens:
            raise ValueError('Existing run has a different output budget')
        for name, digest in manifest['artifact_sha256'].items():
            if file_sha(output / name) != digest:
                raise ValueError('Frozen artifact changed: ' + name)
        return manifest
    from frontier_modebench_contract import make_messages, profile_metadata
    rows = read_rows(output)
    counts = Counter((r['level'], r['domain']) for r in rows)
    if len(counts) != 15 or set(counts.values()) != {128}:
        raise ValueError('Expected all 15 cells with 128 test prompts each')
    if len({(r['level'], r['domain'], r['row_index']) for r in rows}) != len(rows):
        raise ValueError('Duplicate prompt identity')
    snapshot = output / 'code'
    sources = [Path(__file__), ROOT / 'ops/frontier_modebench_contract.py']
    sources += sorted((ROOT / 'src/oat_drgrpo').rglob('*.py'))
    code_hashes = {}
    for source in sources:
        relative = source.relative_to(ROOT)
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        code_hashes[str(relative)] = file_sha(destination)
    row_lookup = {(r['level'], r['domain'], r['row_index']): r for r in rows}
    requests = []
    # Interleave cells and complete K=8 groups uniformly through the test set.
    for row_index in range(128):
        for sample_index in range(8):
            for level, domain in sorted(counts):
                row = row_lookup[level, domain, row_index]
                payload = {'model': 'gpt-5.6-sol',
                           'input': make_messages(level, domain, row),
                           'reasoning': {'effort': 'medium'},
                           'max_output_tokens': max_output_tokens, 'store': False}
                item = {'level': level, 'domain': domain, 'row_index': row_index,
                        'sample_index': sample_index, 'row_sha256': sha(row),
                        'request': payload, 'request_sha256': sha(payload)}
                item['sample_id'] = f'L{level}_{domain}_{row_index:03d}_{sample_index}'
                requests.append(item)
    requests_path = output / 'requests.jsonl'
    with requests_path.open('w') as handle:
        for item in requests:
            handle.write(json.dumps(item, sort_keys=True) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    manifest = {
        'schema': SCHEMA, 'prepared_at_utc': now(), 'endpoint': ENDPOINT,
        'model': 'gpt-5.6-sol', 'reasoning_effort': 'medium',
        'max_output_tokens': max_output_tokens,
        'temperature': 'service default; exact returned value saved per response',
        'top_p': 'service default; exact returned value saved per response',
        'seed': 'not supported by Responses; independent stateless calls',
        'sample_count': 8, 'prompt_count': len(rows), 'request_count': len(requests),
        'training': False, 'tools': [], 'conversation_state': False,
        'profiles': {f'{level}/{domain}': profile_metadata(level, domain)
                     for level, domain in sorted(counts)},
        'artifact_sha256': {name: file_sha(output / name)
                            for name in ('datasets.json', 'rows.jsonl', 'requests.jsonl')},
        'code_sha256': code_hashes,
        'notes': [
            'Static inference concentration; does not identify training-induced collapse.',
            'All provider-exposed response fields saved; private reasoning is not exposed.',
            'No local-model syntax-constrained sampling; hosted interface adaptation.',
            'One K8 draw per prompt; not the four-draw local training protocol.',
            'API failures are missing data, not wrong model answers.',
            'Transport timeouts may have consumed tokens; every retry is logged.',
        ],
    }
    atomic(output / 'manifest.json', manifest)
    return manifest


def response_text(body):
    return ''.join(part.get('text', '') for item in body.get('output', [])
                   if item.get('type') == 'message'
                   for part in item.get('content', []) if part.get('type') == 'output_text')


def validate_raw_receipt(item, receipt, *, relative_path=None):
    """Apply the same paid-response identity checks to fresh and recovered data."""
    if receipt.get('sample_id') != item['sample_id']:
        raise ValueError('Raw receipt sample mismatch')
    if receipt.get('request_sha256') != item['request_sha256']:
        raise ValueError('Raw receipt request mismatch')
    if relative_path is not None and receipt.get('relative_path') != relative_path:
        raise ValueError('Raw receipt path mismatch')
    body = receipt.get('response')
    if receipt.get('http_status') != 200 or not isinstance(body, dict) or body.get('status') not in ('completed', 'incomplete'):
        raise ValueError('Raw receipt is not a received model answer')
    if body.get('model') != item['request']['model']:
        raise ValueError('Unexpected returned model: ' + str(body.get('model')))


def validate_completed_record(output, item, record):
    """Authenticate a durable grade against its exact paid HTTP response."""
    for name in ('sample_id', 'level', 'domain', 'row_index', 'sample_index', 'request_sha256', 'row_sha256'):
        if record.get(name) != item[name]:
            raise ValueError('Saved sample identity mismatch: ' + name)
    relative = record.get('raw_receipt', '')
    path = Path(relative)
    prefix = item['sample_id'] + '__'
    suffix = path.name[len(prefix):-len('.json')] if path.name.startswith(prefix) and path.name.endswith('.json') else ''
    if path.parent != Path('raw_responses') or not suffix.isdigit() or int(suffix) < 1:
        raise ValueError('Saved sample has an invalid raw receipt path')
    receipt = json.loads((output / path).read_text())
    validate_raw_receipt(item, receipt, relative_path=relative)
    if record.get('raw_receipt_sha256') is not None and record['raw_receipt_sha256'] != sha(receipt):
        raise ValueError('Saved raw receipt digest mismatch')
    body = receipt['response']
    expected = {'text': response_text(body), 'response_status': body.get('status'),
                'incomplete_details': body.get('incomplete_details'),
                'response_id': body.get('id'), 'model': body.get('model'),
                'usage': body.get('usage'), 'reasoning': body.get('reasoning'),
                'temperature': body.get('temperature'), 'top_p': body.get('top_p'),
                'service_tier': body.get('service_tier'), 'latency_seconds': receipt['latency_seconds']}
    if any(record.get(name) != value for name, value in expected.items()):
        raise ValueError('Saved sample differs from its raw HTTP receipt')
    if not isinstance(record.get('verified'), bool) or record['verified'] != (record.get('canonical_key') is not None):
        raise ValueError('Saved sample has inconsistent verifier fields')


def grade_receipt(item, receipt, row, grade_response):
    body = receipt['response']
    text = response_text(body)
    grade = grade_response(item['level'], item['domain'], row, text)
    return {k: v for k, v in item.items() if k != 'request'} | {
        'text': text, **grade, 'response_status': body.get('status'),
        'incomplete_details': body.get('incomplete_details'),
        'response_id': body.get('id'), 'model': body.get('model'),
        'usage': body.get('usage'), 'reasoning': body.get('reasoning'),
        'temperature': body.get('temperature'), 'top_p': body.get('top_p'),
        'service_tier': body.get('service_tier'),
        'latency_seconds': receipt['latency_seconds'],
        'raw_receipt': receipt['relative_path'], 'raw_receipt_sha256': sha(receipt), 'recorded_at_utc': now(),
    }


async def run(output, workers, request_timeout, max_attempts, max_new):
    import httpx
    manifest = json.loads((output / 'manifest.json').read_text())
    for name, digest in manifest['artifact_sha256'].items():
        if file_sha(output / name) != digest:
            raise ValueError('Frozen artifact changed: ' + name)
    for name, digest in manifest['code_sha256'].items():
        if file_sha(output / 'code' / name) != digest:
            raise ValueError('Frozen code changed: ' + name)
    # The saved grading source is the source used, including after resume.
    sys.path.insert(0, str(output / 'code/src'))
    sys.path.insert(0, str(output / 'code/ops'))
    from frontier_modebench_contract import grade_response
    key = os.environ.get('AZURE_OPENAI_API_KEY') or getpass.getpass('Azure API key (hidden): ')
    if not key:
        raise ValueError('No API credential')
    rows = {(r['level'], r['domain'], r['row_index']): r for r in read_rows(output)}
    requests = [json.loads(line) for line in (output / 'requests.jsonl').read_text().splitlines()]
    for folder in ('raw_responses', 'sample_receipts'):
        (output / folder).mkdir(exist_ok=True)
    completed = {}
    request_lookup = {item['sample_id']: item for item in requests}
    for path in (output / 'sample_receipts').glob('*.json'):
        record = json.loads(path.read_text())
        sample_id = record.get('sample_id')
        if sample_id not in request_lookup or path.name != sample_id + '.json' or sample_id in completed:
            raise ValueError('Unexpected or duplicate saved sample identity')
        validate_completed_record(output, request_lookup[sample_id], record)
        completed[sample_id] = record
    pending = [item for item in requests if item['sample_id'] not in completed]
    if max_new:
        pending = pending[:max_new]
    queue = asyncio.Queue()
    for item in pending:
        queue.put_nowait(item)
    failures = []
    stop = asyncio.Event()
    started = time.monotonic()
    initial_count = len(completed)
    last_status = 0.0

    def save_status(force=False):
        nonlocal last_status
        if not force and time.monotonic() - last_status < 10:
            return
        last_status = time.monotonic()
        usage = Counter()
        for record in completed.values():
            for name in ('input_tokens', 'output_tokens', 'total_tokens'):
                usage[name] += (record.get('usage') or {}).get(name, 0)
        status = {'updated_at_utc': now(), 'completed_samples': len(completed),
                  'expected_samples': len(requests), 'failed_samples_this_session': len(failures),
                  'new_samples_this_session': len(completed) - initial_count,
                  'elapsed_seconds_this_session': time.monotonic() - started,
                  'response_status_counts': dict(Counter(x['response_status'] for x in completed.values())),
                  'usage': dict(usage), 'complete': len(completed) == len(requests),
                  'cells': dict(Counter(f"L{x['level']}/{x['domain']}" for x in completed.values()))}
        atomic(output / 'status.json', status)
        print(json.dumps(status), flush=True)

    async def process(item, client):
        sample_id = item['sample_id']
        old = sorted((output / 'raw_responses').glob(sample_id + '__*.json'))
        # Recover a received answer without repeating a paid API call.
        received = [json.loads(path.read_text()) for path in old]
        valid = [r for r in received if r.get('http_status') == 200 and
                 isinstance(r.get('response'), dict) and
                 r['response'].get('status') in ('completed', 'incomplete')]
        receipt = valid[0] if valid else None
        if receipt is not None:
            try:
                matching_path = next(path for path, candidate in zip(old, received) if candidate is receipt)
                validate_raw_receipt(item, receipt, relative_path=str(matching_path.relative_to(output)))
            except ValueError:
                stop.set()
                raise
        if receipt is None:
            for attempt in range(len(old) + 1, len(old) + max_attempts + 1):
                relative = f'raw_responses/{sample_id}__{attempt:02d}.json'
                append(output / 'events.jsonl', {'event': 'request_started', 'at_utc': now(),
                                               'sample_id': sample_id, 'attempt': attempt,
                                               'request_sha256': item['request_sha256']})
                begin = time.monotonic()
                receipt = {'sample_id': sample_id, 'attempt': attempt,
                           'request_sha256': item['request_sha256'], 'relative_path': relative,
                           'started_at_utc': now()}
                retry_after = 0.0
                try:
                    response = await client.post(ENDPOINT, json=item['request'])
                    try:
                        body = response.json()
                    except ValueError:
                        body = {'non_json_response': response.text}
                    headers = {k: v for k, v in response.headers.items()
                               if k.lower() not in ('set-cookie', 'authorization', 'api-key')}
                    receipt.update(http_status=response.status_code, response=body, headers=headers)
                    try:
                        retry_after = float(response.headers.get('retry-after', 0))
                    except ValueError:
                        pass
                except (httpx.TimeoutException, httpx.TransportError) as error:
                    receipt.update(http_status=None, error_type=type(error).__name__,
                                   error=str(error), possibly_billed=True)
                receipt.update(latency_seconds=time.monotonic() - begin, received_at_utc=now())
                receipt = json.loads(json.dumps(receipt).replace(key, '[REDACTED]'))
                atomic(output / relative, receipt)
                body = receipt.get('response', {})
                if receipt.get('http_status') == 200 and body.get('status') in ('completed', 'incomplete'):
                    try:
                        validate_raw_receipt(item, receipt, relative_path=relative)
                    except ValueError:
                        stop.set()
                        raise
                    break
                append(output / 'errors.jsonl', receipt)
                if receipt.get('http_status') in (400, 401, 403, 404):
                    stop.set()
                    raise RuntimeError(f"Non-retryable API error {receipt['http_status']}; see {relative}")
                receipt = None
                if attempt < len(old) + max_attempts:
                    await asyncio.sleep(min(60, max(retry_after, 2 ** min(attempt - len(old), 6))))
            if receipt is None:
                raise RuntimeError('Retry budget exhausted: ' + sample_id)
        validate_raw_receipt(item, receipt)
        row = rows[item['level'], item['domain'], item['row_index']]
        record = await asyncio.to_thread(grade_receipt, item, receipt, row, grade_response)
        atomic(output / 'sample_receipts' / (sample_id + '.json'), record)
        append(output / 'samples.jsonl', record)
        completed[sample_id] = record
        save_status()

    async def worker(client):
        while not stop.is_set():
            try:
                item = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                await process(item, client)
            except Exception as error:
                entry = {'sample_id': item['sample_id'], 'at_utc': now(),
                         'error_type': type(error).__name__, 'error': str(error).replace(key, '[REDACTED]')}
                failures.append(entry)
                append(output / 'runner_errors.jsonl', entry)
                print(json.dumps(entry), flush=True)
            finally:
                queue.task_done()

    append(output / 'events.jsonl', {'event': 'session_started', 'at_utc': now(),
                                   'workers': workers, 'pending_samples': len(pending),
                                   'timeout_seconds': request_timeout, 'pid': os.getpid()})
    save_status(True)
    async with httpx.AsyncClient(headers={'api-key': key}, follow_redirects=False,
                                 timeout=httpx.Timeout(request_timeout, connect=30),
                                 limits=httpx.Limits(max_connections=workers,
                                                    max_keepalive_connections=workers)) as client:
        await asyncio.gather(*(worker(client) for _ in range(workers)))
    # Rebuild canonical JSONL from atomic receipts so interrupted append cannot
    # create duplicate or truncated records in the completed export.
    temporary = output / 'samples.jsonl.tmp'
    with temporary.open('w') as handle:
        for item in requests:
            if item['sample_id'] in completed:
                handle.write(json.dumps(completed[item['sample_id']], sort_keys=True) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(output / 'samples.jsonl')
    save_status(True)
    append(output / 'events.jsonl', {'event': 'session_finished', 'at_utc': now(),
                                   'completed_samples': len(completed), 'failures': len(failures)})
    return 0 if not failures and (max_new or len(completed) == len(requests)) else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run'))
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--max-output-tokens', type=int, default=8192)
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--request-timeout', type=float, default=240)
    parser.add_argument('--max-attempts', type=int, default=8)
    parser.add_argument('--max-new', type=int, default=0, help='Optional bounded integration run; zero means all')
    args = parser.parse_args()
    if not 1 <= args.workers <= 64:
        parser.error('workers must be 1..64')
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / '.runner.lock').open('w') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.command == 'prepare':
            sys.path.insert(0, str(ROOT / 'ops'))
            print(json.dumps(prepare(args.output, args.max_output_tokens), indent=2))
            return 0
        return asyncio.run(run(args.output, args.workers, args.request_timeout,
                               args.max_attempts, args.max_new))


if __name__ == '__main__':
    raise SystemExit(main())
