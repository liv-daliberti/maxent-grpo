#!/usr/bin/env python3
"""Collect the approved mixed-temperature extension with retained native preflights.

Run the frozen copy under collection_v1/code/ops. Credentials are read only from
AZURE_OPENAI_API_KEY or hidden terminal input. Each received model answer is
retained once; failed transports consume a bounded, durable attempt budget.
"""
from __future__ import annotations
import argparse
import asyncio
from collections import Counter, defaultdict
import fcntl
import getpass
import json
import os
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_frontier_modebench as native
from prepare_gpt56_prompt_expansion import (APPROVED_PLAN_SHA256, DEFAULT_OUTPUT, SCHEMA,
    SERVED_MODEL, TEMPERATURES, identity, read_lines, require, slug, write_lines)
atomic, append, file_sha, now, sha = native.atomic, native.append, native.file_sha, native.now, native.sha


def validate_inventory(output):
    output = Path(output).resolve()
    manifest = json.loads((output / 'manifest.json').read_text())
    require(manifest.get('schema') == SCHEMA, 'Unexpected expanded collection schema')
    require(manifest.get('approved_plan_sha256') == APPROVED_PLAN_SHA256, 'Approved plan binding differs')
    require(file_sha(output / 'approved_plan.json') == APPROVED_PLAN_SHA256, 'Approved plan changed')
    require(manifest['endpoint'] == native.ENDPOINT and manifest['model'] == 'gpt-5.6-sol', 'Collection endpoint/model differs')
    require(manifest['reasoning_effort'] == 'none' and tuple(manifest['temperatures']) == TEMPERATURES, 'Collection controls differ')
    require(manifest['prompt_count'] == 360 and manifest['request_count'] == 14400 and manifest['sample_count'] == 8, 'Collection counts differ')
    require(manifest['max_attempts_per_slot'] == 8, 'Collection attempt limit differs')
    for name, digest in manifest['artifact_sha256'].items():
        require(file_sha(output / name) == digest, 'Frozen artifact changed: ' + name)
    for name, digest in manifest['code_sha256'].items():
        require(file_sha(output / 'code' / name) == digest, 'Frozen source changed: ' + name)
    plan = json.loads((output / 'approved_plan.json').read_text())
    for name, binding in plan['artifacts'].items():
        require(file_sha(output / ('approved_' + name)) == binding['sha256'], 'Approved plan artifact changed')
    selection = {identity(s): s for s in read_lines(output / 'approved_selection.jsonl') if s['cohort'] == 'additional_360'}
    rows = read_lines(output / 'rows.jsonl')
    row_map = {identity(row): row for row in rows}
    require(len(rows) == len(row_map) == len(selection) == 360 and row_map.keys() == selection.keys(), 'Expanded rows differ from approved additional cohort')
    for key, row in row_map.items():
        require(sha(row) == selection[key]['row_sha256'], 'Expanded row bytes differ from approved row')
    requests = read_lines(output / 'requests.jsonl')
    require(len(requests) == len({x['sample_id'] for x in requests}) == 14400, 'Expanded requests have missing or duplicate slots')
    approved = {(x['temperature'], x['sample_id']): x for x in read_lines(output / 'approved_new_request_hashes.jsonl')}
    seen, groups = set(), defaultdict(list)
    for item in requests:
        temperature = item['temperature_condition']
        key = (temperature, item['original_sample_id'])
        require(key in approved and key not in seen, 'Unapproved or duplicate expanded request')
        seen.add(key)
        expected = approved[key]
        require(item['sample_id'] == slug(temperature).upper() + '__' + item['original_sample_id'], 'Expanded sample identity differs')
        for name in ('level', 'domain', 'row_index', 'sample_index', 'row_sha256', 'reference_request_sha256'):
            require(item[name] == expected[name], 'Expanded request binding differs: ' + name)
        require(item['request_sha256'] == sha(item['request']) == expected['planned_request_sha256'], 'Expanded payload differs from approved request')
        require(item['row_sha256'] == sha(row_map[identity(item)]), 'Expanded request row binding differs')
        require(item['request']['temperature'] == temperature and item['request']['reasoning']['effort'] == 'none', 'Expanded request controls differ')
        groups[(temperature, *identity(item))].append(item['sample_index'])
    require(len(groups) == 1800 and all(sorted(v) == list(range(8)) for v in groups.values()), 'Expanded eight-draw groups are incomplete')
    for start in range(0, len(requests), 5):
        group = requests[start:start+5]
        require(tuple(x['temperature_condition'] for x in group) == TEMPERATURES and len({x['original_sample_id'] for x in group}) == 1, 'Temperature dispatch order is not interleaved')
    require(manifest['preflight_sample_ids'] == [x['sample_id'] for x in requests[:5]], 'Preflight registration differs')
    return manifest, row_map, requests


def validate_raw_receipt(item, receipt, *, relative_path=None):
    native.validate_raw_receipt(item, receipt, relative_path=relative_path)
    body = receipt['response']
    require(body.get('temperature') == item['temperature_condition'], 'Returned temperature differs')
    require(body.get('reasoning', {}).get('effort') == 'none', 'Returned reasoning effort differs')
    require(body.get('top_p') == 0.98, 'Returned top_p differs')
    headers = {k.lower(): v for k, v in receipt.get('headers', {}).items()}
    require(headers.get('x-ms-served-model') == SERVED_MODEL, 'Returned served-model snapshot differs')


def validate_completed_record(output, item, record):
    native.validate_completed_record(output, item, record)
    for name in ('temperature_condition', 'original_sample_id', 'reference_request_sha256'):
        require(record.get(name) == item[name], 'Saved expanded sample identity differs: ' + name)
    receipt = json.loads((output / record['raw_receipt']).read_text())
    validate_raw_receipt(item, receipt, relative_path=record['raw_receipt'])


def load_completed(output, requests):
    lookup = {item['sample_id']: item for item in requests}
    completed = {}
    for path in sorted((output / 'sample_receipts').glob('*.json')):
        record = json.loads(path.read_text())
        sample_id = record.get('sample_id')
        require(sample_id in lookup and path.name == sample_id + '.json' and sample_id not in completed, 'Unexpected or duplicate saved sample')
        validate_completed_record(output, lookup[sample_id], record)
        completed[sample_id] = record
    return completed


def audit_attempts(output, requests):
    """A durable dispatch without a receipt is ambiguous and cannot auto-retry."""
    lookup = {item['sample_id']: item for item in requests}
    starts = {}
    events = output / 'events.jsonl'
    if events.exists():
        for event in read_lines(events):
            if event.get('event') != 'request_started':
                continue
            key = event['sample_id'], event['attempt']
            require(key not in starts, 'Duplicate durable dispatch attempt')
            require(key[0] in lookup and event['request_sha256'] == lookup[key[0]]['request_sha256'], 'Dispatch request binding differs')
            starts[key] = event
    receipts = defaultdict(list)
    for path in sorted((output / 'raw_responses').glob('*.json')):
        raw = json.loads(path.read_text())
        sample_id, attempt = raw.get('sample_id'), raw.get('attempt')
        require(sample_id in lookup and isinstance(attempt, int) and 1 <= attempt <= 8, 'Unexpected raw receipt identity/attempt')
        key = sample_id, attempt
        require(key in starts, 'Raw receipt lacks its durable dispatch intent')
        require(raw['relative_path'] == str(path.relative_to(output)), 'Raw receipt path differs')
        require(path.name == f'{sample_id}__{attempt:02d}.json', 'Raw receipt filename differs')
        require(raw.get('request_sha256') == lookup[sample_id]['request_sha256'], 'Raw receipt request differs')
        receipts[sample_id].append(raw)
    received_keys = {(sample_id, raw['attempt']) for sample_id, values in receipts.items() for raw in values}
    require(received_keys == set(starts), 'Ambiguous interrupted dispatch: request_started lacks a durable raw receipt; inspect before any retry')
    for sample_id, values in receipts.items():
        require(sorted(raw['attempt'] for raw in values) == list(range(1, len(values) + 1)), 'Raw attempt sequence has a gap')
        answered = [raw for raw in values if raw.get('http_status') == 200]
        require(len(answered) <= 1, 'Duplicate received model answers for one sample slot')
        if answered:
            validate_raw_receipt(lookup[sample_id], answered[0], relative_path=answered[0]['relative_path'])
    return receipts


def gate(output, manifest, requests, *, create=False):
    first = requests[:5]
    samples = []
    for item in first:
        path = output / 'sample_receipts' / (item['sample_id'] + '.json')
        record = json.loads(path.read_text())
        validate_completed_record(output, item, record)
        raw_path = output / record['raw_receipt']
        samples.append({'sample_id': item['sample_id'], 'temperature': item['temperature_condition'],
                        'request_sha256': item['request_sha256'], 'sample_receipt_sha256': file_sha(path),
                        'raw_receipt': record['raw_receipt'], 'raw_receipt_sha256': file_sha(raw_path)})
    expected = {'schema': 'gpt56-temperature-expansion-preflight-gate-v1',
                'status': 'authorized_supported_controls', 'manifest_sha256': file_sha(output / 'manifest.json'),
                'approved_plan_sha256': APPROVED_PLAN_SHA256, 'preflight_samples': samples,
                'registered_requests': 14400, 'preflights_count_toward_registered_requests': True,
                'returned_controls_required': manifest['returned_controls_required']}
    path = output / 'gate.json'
    if path.exists():
        actual = json.loads(path.read_text())
        actual.pop('created_at_utc', None)
        require(actual == expected, 'Preflight gate changed or source receipts differ')
    elif create:
        atomic(path, expected | {'created_at_utc': now()})
    else:
        raise ValueError('Full collection requires five retained authenticated preflights')


def export_views(output, manifest, requests):
    """Create authenticated per-temperature hardlink views for offline grading."""
    completed = load_completed(output, requests)
    require(len(completed) == 14400, 'Arm views require all 14,400 collected samples')
    audit_attempts(output, requests)
    for temperature in TEMPERATURES:
        view = output / 'arms' / slug(temperature)
        arm = [item for item in requests if item['temperature_condition'] == temperature]
        view.mkdir(parents=True, exist_ok=True)
        if (view / 'manifest.json').exists():
            saved = json.loads((view / 'manifest.json').read_text())
            for name, digest in saved['artifact_sha256'].items():
                require(file_sha(view / name) == digest, 'Existing arm view artifact differs')
            continue
        for name in ('rows.jsonl', 'datasets.json'):
            if not (view / name).exists():
                shutil.copyfile(output / name, view / name)
        if not (view / 'requests.jsonl').exists():
            write_lines(view / 'requests.jsonl', arm)
        require(read_lines(view / 'requests.jsonl') == arm, 'Arm view request subsequence differs')
        sources = []
        for item in arm:
            sample_id = item['sample_id']
            sample_path = Path('sample_receipts') / (sample_id + '.json')
            raw_path = Path(completed[sample_id]['raw_receipt'])
            for relative in (sample_path, raw_path):
                destination = view / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    require(file_sha(destination) == file_sha(output / relative), 'Existing view receipt differs')
                else:
                    os.link(output / relative, destination)
            sources.append({'sample_id': sample_id, 'sample_receipt': str(sample_path),
                            'sample_receipt_sha256': file_sha(output / sample_path),
                            'raw_receipt': str(raw_path), 'raw_receipt_sha256': file_sha(output / raw_path)})
        atomic(view / 'parent_collection_binding.json', {'schema': 'gpt56-temperature-expansion-arm-binding-v1',
               'parent_collection': str(output), 'parent_manifest_sha256': file_sha(output / 'manifest.json'),
               'parent_requests_sha256': file_sha(output / 'requests.jsonl'), 'temperature': temperature, 'samples': sources})
        for name, digest in manifest['code_sha256'].items():
            destination = view / 'code' / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                shutil.copyfile(output / 'code' / name, destination)
            require(file_sha(destination) == digest, 'Arm frozen code differs')
        arm_manifest = dict(manifest)
        arm_manifest.update(schema='frontier-modebench-responses-v1', expansion_view_schema='gpt56-temperature-expansion-arm-view-v1',
                            temperature=temperature, prompt_count=360, request_count=2880,
                            artifact_sha256={name: file_sha(view / name) for name in ('rows.jsonl', 'datasets.json', 'requests.jsonl', 'parent_collection_binding.json')})
        atomic(view / 'manifest.json', arm_manifest)
        atomic(view / 'status.json', {'complete': True, 'completed_samples': 2880, 'expected_samples': 2880})
        if not (view / 'samples.jsonl').exists():
            write_lines(view / 'samples.jsonl', [completed[item['sample_id']] for item in arm])
    return {'status': 'exported', 'arm_count': 5, 'samples_per_arm': 2880}


async def collect(output, stage, workers, request_timeout, credential, *, client_factory=None):
    import httpx
    manifest, rows, requests = validate_inventory(output)
    for folder in ('raw_responses', 'sample_receipts'):
        (output / folder).mkdir(exist_ok=True)
    completed = load_completed(output, requests)
    history = audit_attempts(output, requests)
    if stage == 'full':
        gate(output, manifest, requests)
    eligible = requests[:5] if stage == 'preflight' else requests
    pending = [item for item in eligible if item['sample_id'] not in completed]
    max_attempts = 2 if stage == 'preflight' else manifest['max_attempts_per_slot']
    if not pending:
        if stage == 'preflight':
            gate(output, manifest, requests, create=True)
        return 0
    require(bool(credential), 'Missing API credential')
    sys.path.insert(0, str(output / 'code/src'))
    sys.path.insert(0, str(output / 'code/ops'))
    from frontier_modebench_contract import grade_response
    queue = asyncio.Queue()
    for item in pending:
        queue.put_nowait(item)
    failures, stop = [], asyncio.Event()
    started, initial_count, last_status = time.monotonic(), len(completed), 0.0

    def save_status(force=False):
        nonlocal last_status
        if not force and time.monotonic() - last_status < 10:
            return
        last_status = time.monotonic()
        usage = Counter()
        for record in completed.values():
            for key in ('input_tokens', 'output_tokens', 'total_tokens'):
                usage[key] += (record.get('usage') or {}).get(key, 0)
        status = {'updated_at_utc': now(), 'completed_samples': len(completed), 'expected_samples': 14400,
                  'complete': len(completed) == 14400, 'stage': stage, 'failed_samples_this_session': len(failures),
                  'new_samples_this_session': len(completed)-initial_count, 'elapsed_seconds_this_session': time.monotonic()-started,
                  'response_status_counts': dict(Counter(x['response_status'] for x in completed.values())), 'usage': dict(usage),
                  'temperatures': dict(Counter(str(x['temperature_condition']) for x in completed.values())),
                  'cells': dict(Counter(f"T{x['temperature_condition']}/L{x['level']}/{x['domain']}" for x in completed.values()))}
        atomic(output / 'status.json', status)
        print(json.dumps(status), flush=True)

    async def process(item, client):
        sample_id = item['sample_id']
        old = history.get(sample_id, [])
        answers = [raw for raw in old if raw.get('http_status') == 200]
        receipt = answers[0] if answers else None
        if receipt is None:
            for attempt in range(len(old)+1, max_attempts+1):
                if stop.is_set():
                    raise RuntimeError('Collection stopped before dispatch')
                relative = f'raw_responses/{sample_id}__{attempt:02d}.json'
                append(output / 'events.jsonl', {'event': 'request_started', 'at_utc': now(), 'sample_id': sample_id,
                       'attempt': attempt, 'request_sha256': item['request_sha256'], 'temperature': item['temperature_condition']})
                begin, retry_after = time.monotonic(), 0.0
                receipt = {'sample_id': sample_id, 'attempt': attempt, 'request_sha256': item['request_sha256'],
                           'relative_path': relative, 'started_at_utc': now()}
                try:
                    response = await client.post(manifest['endpoint'], json=item['request'])
                    try:
                        body = response.json()
                    except ValueError:
                        body = {'non_json_response': response.text}
                    headers = {k: v for k, v in response.headers.items() if k.lower() not in ('set-cookie', 'authorization', 'api-key')}
                    receipt.update(http_status=response.status_code, response=body, headers=headers)
                    try:
                        retry_after = float(response.headers.get('retry-after', 0))
                    except ValueError:
                        pass
                except (httpx.TimeoutException, httpx.TransportError) as error:
                    receipt.update(http_status=None, error_type=type(error).__name__, error=str(error), possibly_billed=True)
                receipt.update(latency_seconds=time.monotonic()-begin, received_at_utc=now())
                receipt = json.loads(json.dumps(receipt).replace(credential, '[REDACTED]'))
                atomic(output / relative, receipt)
                history.setdefault(sample_id, []).append(receipt)
                if receipt.get('http_status') == 200:
                    try:
                        validate_raw_receipt(item, receipt, relative_path=relative)
                    except Exception:
                        stop.set()
                        raise
                    break
                append(output / 'errors.jsonl', receipt)
                if receipt.get('http_status') in (400, 401, 403, 404):
                    stop.set()
                    raise RuntimeError(f"Non-retryable API error {receipt['http_status']}; see {relative}")
                receipt = None
                if attempt < max_attempts:
                    await asyncio.sleep(min(60, max(retry_after, 2 ** min(attempt, 6))))
            if receipt is None:
                raise RuntimeError('Lifetime retry budget exhausted: ' + sample_id)
        validate_raw_receipt(item, receipt)
        record = await asyncio.to_thread(native.grade_receipt, item, receipt, rows[identity(item)], grade_response)
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
                entry = {'sample_id': item['sample_id'], 'at_utc': now(), 'error_type': type(error).__name__,
                         'error': str(error).replace(credential, '[REDACTED]')}
                failures.append(entry)
                append(output / 'runner_errors.jsonl', entry)
                print(json.dumps(entry), flush=True)
            finally:
                queue.task_done()

    append(output / 'events.jsonl', {'event': 'session_started', 'at_utc': now(), 'stage': stage, 'workers': workers,
           'pending_samples': len(pending), 'timeout_seconds': request_timeout, 'pid': os.getpid(),
           'manifest_sha256': file_sha(output / 'manifest.json')})
    save_status(True)
    factory = client_factory or httpx.AsyncClient
    async with factory(headers={'api-key': credential}, follow_redirects=False,
                       timeout=httpx.Timeout(request_timeout, connect=30),
                       limits=httpx.Limits(max_connections=workers, max_keepalive_connections=workers)) as client:
        await asyncio.gather(*(worker(client) for _ in range(workers)))
    temporary = output / 'samples.jsonl.tmp'
    with temporary.open('w') as handle:
        for item in requests:
            if item['sample_id'] in completed:
                handle.write(json.dumps(completed[item['sample_id']], sort_keys=True) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(output / 'samples.jsonl')
    save_status(True)
    append(output / 'events.jsonl', {'event': 'session_finished', 'at_utc': now(), 'stage': stage,
           'completed_samples': len(completed), 'failures': len(failures)})
    if stage == 'preflight' and not failures and all(x['sample_id'] in completed for x in requests[:5]):
        gate(output, manifest, requests, create=True)
    return 0 if not failures and all(x['sample_id'] in completed for x in eligible) else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('preflight', 'full', 'all', 'validate', 'export-views'))
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--workers', type=int, default=64)
    parser.add_argument('--request-timeout', type=float, default=600)
    args = parser.parse_args()
    require(1 <= args.workers <= 64 and args.request_timeout > 0, 'Invalid worker/timeout setting')
    output = args.output.resolve()
    with (output / '.runner.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest, rows, requests = validate_inventory(output)
        if args.stage == 'validate':
            completed = load_completed(output, requests)
            audit_attempts(output, requests)
            print(json.dumps({'status': 'valid', 'registered_samples': len(requests), 'completed_samples': len(completed)}))
            return 0
        if args.stage == 'export-views':
            print(json.dumps(export_views(output, manifest, requests)))
            return 0
        frozen = output / 'code/ops/run_gpt56_prompt_expansion.py'
        require(file_sha(__file__) == file_sha(frozen), 'Execute the frozen collector matching this manifest')
        credential = os.environ.get('AZURE_OPENAI_API_KEY') or getpass.getpass('Azure API key (hidden): ')
        require(bool(credential), 'Missing API credential')
        stages = ('preflight', 'full') if args.stage == 'all' else (args.stage,)
        for stage in stages:
            result = asyncio.run(collect(output, stage, min(args.workers, 5) if stage == 'preflight' else args.workers,
                                         args.request_timeout, credential))
            if result:
                return result
        if stages[-1] == 'full':
            print(json.dumps(export_views(output, manifest, requests)))
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
