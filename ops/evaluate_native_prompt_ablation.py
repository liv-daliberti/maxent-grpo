#!/usr/bin/env python3
"""Prompt-ablation transport: the native runner with GPT-5.6 Sol admitted.

Only model admission and its Responses transport mapping differ from
ops/evaluate_chat_frontier_modebench.py. Request payloads remain frozen.

Frozen ModeBench evaluation using native Chat Completions or Responses.

No training. Credentials are read from hidden terminal input or
AZURE_OPENAI_API_KEY, never saved. Each HTTP request/attempt, native response,
exposed reasoning, answer-only grade, and source snapshot is retained. Profiles
are explicit immutable JSON, so provider-specific reasoning settings cannot be
silently changed on resume. Multi-choice requests are disabled after a provider
preflight returned unrelated answers in choices beyond index zero.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
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
REFERENCE = ROOT / 'artifacts/frontier_modebench_gpt56sol_20260911'
MODELS = ('FW-Kimi-K3', 'grok-4.3', 'DeepSeek-V4-Pro', 'gpt-5.4', 'gpt-5.6-sol')
BASE_URL = 'https://liv.services.ai.azure.com/openai/v1/'
SCHEMA = 'frontier-modebench-native-chat-responses-v1'
SENSITIVE_HEADERS = {'authorization', 'api-key', 'x-api-key', 'set-cookie'}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def append(path, value):
    with Path(path).open('a') as handle:
        handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path, records):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def identity(item):
    return item['level'], item['domain'], item['row_index']


def default_profile(model, max_output_tokens=8192):
    if model not in MODELS:
        raise ValueError('Unsupported deployment')
    protocol = 'responses' if model in ('gpt-5.4', 'gpt-5.6-sol') else 'chat_completions'
    params = ({'max_output_tokens': max_output_tokens, 'reasoning': {'effort': 'medium'}, 'store': False}
              if protocol == 'responses' else {'max_tokens': max_output_tokens})
    return {'model': model, 'protocol': protocol,
            'endpoint': BASE_URL + ('responses' if protocol == 'responses' else 'chat/completions'),
            'request_parameters': params}


def validate_profile(profile):
    model = profile.get('model')
    expected = default_profile(model)
    if profile.get('protocol') != expected['protocol'] or profile.get('endpoint') != expected['endpoint']:
        raise ValueError('Protocol or endpoint differs from the supported deployment')
    if set(profile) - {'model', 'protocol', 'endpoint', 'request_parameters', 'notes', 'preflight_sha256'}:
        raise ValueError('Unknown profile metadata')
    p = profile.get('request_parameters')
    if not isinstance(p, dict):
        raise ValueError('Profile requires request_parameters')
    allowed = ({'max_output_tokens', 'reasoning', 'store', 'temperature', 'top_p'}
               if profile['protocol'] == 'responses' else
               {'max_tokens', 'max_completion_tokens', 'reasoning_effort', 'thinking', 'temperature', 'top_p', 'n', 'chat_template_kwargs'})
    if set(p) - allowed:
        raise ValueError('Unsupported or prompt-overriding request parameter')
    budgets = [p[k] for k in ('max_tokens', 'max_output_tokens', 'max_completion_tokens') if k in p]
    if len(budgets) != 1 or not isinstance(budgets[0], int) or isinstance(budgets[0], bool) or budgets[0] <= 0:
        raise ValueError('Exactly one positive native output token budget is required')
    if profile['protocol'] == 'responses' and p.get('store') is not False:
        raise ValueError('Responses profile must explicitly disable storage')
    n = p.get('n', 1)
    if type(n) is not int or n != 1:
        raise ValueError('Multiple choices are disabled following the Kimi n=8 preflight anomaly')
    return profile


def native_request(messages, profile):
    validate_profile(profile)
    if [m.get('role') for m in messages] != ['system', 'user'] or any(not isinstance(m.get('content'), str) for m in messages):
        raise ValueError('Expected the frozen system/user prompt strings')
    field = 'input' if profile['protocol'] == 'responses' else 'messages'
    return {'model': profile['model'], field: messages, **profile['request_parameters']}


def verify_snapshot(output, manifest):
    for name, digest in manifest['artifact_sha256'].items():
        if file_sha(output / name) != digest:
            raise ValueError('Frozen artifact changed: ' + name)
    for name, digest in manifest['code_sha256'].items():
        if file_sha(output / 'code' / name) != digest:
            raise ValueError('Frozen code changed: ' + name)


def prepare(output, profile, reference=REFERENCE):
    output, reference = Path(output), Path(reference)
    validate_profile(profile)
    if output.resolve() == reference.resolve():
        raise ValueError('Output must be separate from the reference run')
    if (output / 'manifest.json').exists():
        manifest = json.loads((output / 'manifest.json').read_text())
        if manifest['model_profile'] != profile:
            raise ValueError('Existing run has a different model or immutable request profile')
        if manifest['reference_run'] != str(reference.resolve()):
            raise ValueError('Existing run uses a different reference')
        verify_snapshot(output, manifest)
        return manifest
    original = json.loads((reference / 'manifest.json').read_text())
    verify_snapshot(reference, original)
    rows = read_jsonl(reference / 'rows.jsonl')
    row_lookup = {identity(row): row for row in rows}
    if len(row_lookup) != 1920 or set(Counter((r['level'], r['domain']) for r in rows).values()) != {128}:
        raise ValueError('Expected all 1,920 test prompts with 128 in each of 15 cells')
    original_requests = read_jsonl(reference / 'requests.jsonl')
    if len(original_requests) != 15360 or len({r['sample_id'] for r in original_requests}) != 15360:
        raise ValueError('Expected 15,360 unique reference samples')
    output.mkdir(parents=True, exist_ok=True)
    for name in ('rows.jsonl', 'datasets.json'):
        if (output / name).exists() and file_sha(output / name) != file_sha(reference / name):
            raise ValueError('Existing unfrozen input differs from reference: ' + name)
        shutil.copyfile(reference / name, output / name)
    hashes = {}
    for name in original['code_sha256']:
        target = output / 'code' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(reference / 'code' / name, target)
        hashes[name] = file_sha(target)
    runner_path = 'ops/evaluate_chat_frontier_modebench.py'
    shutil.copyfile(Path(__file__), output / 'code' / runner_path)
    hashes[runner_path] = file_sha(output / 'code' / runner_path)
    n = profile['request_parameters'].get('n', 1)
    samples, groups = [], {}
    seen_prompt_messages = {}
    for old in original_requests:
        if old['row_sha256'] != sha(row_lookup[identity(old)]) or old['request_sha256'] != sha(old['request']):
            raise ValueError('Reference request identity mismatch')
        payload = native_request(old['request']['input'], profile)
        pid = identity(old)
        if pid in seen_prompt_messages and seen_prompt_messages[pid] != old['request']['input']:
            raise ValueError('Reference samples do not share identical prompt text')
        seen_prompt_messages[pid] = old['request']['input']
        group_id = old['sample_id'] if n == 1 else f"L{old['level']}_{old['domain']}_{old['row_index']:03d}_n8"
        item = {k: v for k, v in old.items() if k not in ('request', 'request_sha256')}
        item.update(request=payload, request_sha256=sha(payload), group_id=group_id,
                    choice_index=old['sample_index'] if n == 8 else 0,
                    reference_request_sha256=old['request_sha256'])
        samples.append(item)
        if group_id not in groups:
            groups[group_id] = {'group_id': group_id, 'request': payload, 'request_sha256': sha(payload),
                                'sample_ids': [], 'sample_count': n, 'protocol': profile['protocol']}
        groups[group_id]['sample_ids'].append(item['sample_id'])
    if any(len(g['sample_ids']) != n for g in groups.values()):
        raise ValueError('Incomplete HTTP sampling group')
    write_jsonl(output / 'requests.jsonl', samples)
    write_jsonl(output / 'http_requests.jsonl', groups.values())
    atomic(output / 'model_profile.json', profile)
    manifest = {'schema': SCHEMA, 'prepared_at_utc': now(), 'model': profile['model'],
        'endpoint': profile['endpoint'], 'protocol': profile['protocol'], 'model_profile': profile,
        'model_profile_sha256': sha(profile), 'sample_count': 8, 'prompt_count': len(rows),
        'request_count': len(samples), 'http_request_count': len(groups), 'samples_per_http_request': n,
        'max_output_tokens': next(profile['request_parameters'][k] for k in ('max_output_tokens', 'max_tokens', 'max_completion_tokens') if k in profile['request_parameters']),
        'requested_settings': profile['request_parameters'],
        'reasoning_effort': profile['request_parameters'].get('reasoning_effort', profile['request_parameters'].get('reasoning', {}).get('effort', 'provider default; not specified')),
        'temperature': profile['request_parameters'].get('temperature', 'provider default; not specified'),
        'top_p': profile['request_parameters'].get('top_p', 'provider default; not specified'),
        'seed': 'not specified; provider RNG independence cannot be audited', 'training': False,
        'tools': [], 'conversation_state': False, 'profiles': original['profiles'],
        'reference_run': str(reference.resolve()), 'reference_manifest_sha256': file_sha(reference / 'manifest.json'),
        'reference_artifact_sha256': original['artifact_sha256'], 'code_sha256': hashes,
        'artifact_sha256': {name: file_sha(output / name) for name in ('rows.jsonl', 'datasets.json', 'requests.jsonl', 'http_requests.jsonl', 'model_profile.json')},
        'notes': [
            'Exact frozen GPT-5.6 Sol benchmark system/user strings and verifier source are reused.',
            'Only assistant final-answer content is graded; provider-exposed reasoning is retained in raw responses.',
            'Native usage is preserved exactly; no assumption that native total equals prompt plus completion tokens.',
            'For n=8, one HTTP response contains eight choices; choice_index identifies samples and native usage is attributed once to choice zero.',
            'Provider-specific reasoning settings are not matched compute budgets across models.',
            'Independent stateless HTTP requests for n=1; grouped provider generations for n=8.',
            'Requested settings and returned settings are separate; absent returned fields remain null.',
            'Transport errors may have consumed tokens; every attempt and retry is retained.',
            'Synthetic local Python worker warmup precedes grading without altering verifier timeouts.',
            'Inference-time concentration does not identify training-induced collapse or zero probability for unseen modes.'
        ]}
    atomic(output / 'manifest.json', manifest)
    return manifest


def choice_for(body, choice_index):
    matches = [c for c in body.get('choices', []) if c.get('index') == choice_index]
    if len(matches) != 1:
        raise ValueError('Missing or duplicate native choice index')
    return matches[0]


def response_text(body, protocol, choice_index=0):
    if protocol == 'responses':
        return ''.join(p.get('text', '') for item in body.get('output', []) if item.get('type') == 'message'
                       for p in item.get('content', []) if p.get('type') == 'output_text')
    content = choice_for(body, choice_index).get('message', {}).get('content')
    if content is None:
        return ''
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return ''.join(p.get('text', '') for p in content if isinstance(p, dict) and p.get('type') in ('text', 'output_text'))
    raise ValueError('Unsupported assistant content shape')


def response_status(body, protocol, choice_index=0):
    if protocol == 'responses':
        return body.get('status') if body.get('status') in ('completed', 'incomplete') else None
    reason = choice_for(body, choice_index).get('finish_reason')
    if reason == 'length':
        return 'incomplete'
    return 'completed' if reason in ('stop', 'content_filter', 'tool_calls', 'function_call') else None


def validate_raw(group, receipt, relative_path=None):
    if receipt.get('group_id') != group['group_id'] or receipt.get('request_sha256') != group['request_sha256']:
        raise ValueError('Raw receipt request identity mismatch')
    if relative_path is not None and receipt.get('relative_path') != relative_path:
        raise ValueError('Raw receipt path mismatch')
    body = receipt.get('response')
    if receipt.get('http_status') != 200 or not isinstance(body, dict):
        raise ValueError('Receipt is not a native HTTP 200 model response')
    if body.get('model') != group['request']['model'] or not isinstance(body.get('id'), str) or not body['id']:
        raise ValueError('Unexpected returned model or missing response ID')
    if group['protocol'] == 'chat_completions':
        choices = body.get('choices')
        if not isinstance(choices, list) or len(choices) != group['sample_count'] or {c.get('index') for c in choices} != set(range(group['sample_count'])):
            raise ValueError('Returned choice count or indices differ from the frozen request')
        if any(c.get('message', {}).get('role') != 'assistant' for c in choices):
            raise ValueError('Expected assistant choices')
    for i in range(group['sample_count']):
        if response_status(body, group['protocol'], i) is None:
            raise ValueError('Unexpected native completion status')
        response_text(body, group['protocol'], i)


def response_fields(item, group, receipt):
    body, index = receipt['response'], item['choice_index']
    status = response_status(body, group['protocol'], index)
    native = body.get('usage')
    if group['protocol'] == 'responses':
        usage = native
        stop = body.get('status')
        incomplete = body.get('incomplete_details')
    else:
        # Map native names without altering their numeric meaning. Some providers
        # exclude reasoning from completion_tokens but include it in total_tokens.
        usage = {new: native[old] for new, old in [('input_tokens', 'prompt_tokens'), ('output_tokens', 'completion_tokens'), ('total_tokens', 'total_tokens')]
                 if isinstance(native, dict) and old in native}
        stop = choice_for(body, index).get('finish_reason')
        incomplete = {'reason': stop} if status == 'incomplete' else None
    return {'text': response_text(body, group['protocol'], index), 'response_status': status,
        'incomplete_details': incomplete, 'stop_reason': stop, 'response_id': body['id'],
        'provider_sample_identity': [body['id'], index], 'model': body['model'],
        'usage': usage if index == 0 else None, 'native_usage': native,
        'usage_scope': 'http_response', 'usage_attributed_to_sample': index == 0,
        'reasoning': body.get('reasoning'), 'requested_settings': group['request'] | {},
        'returned_settings': {k: body.get(k) for k in ('reasoning', 'reasoning_effort', 'thinking', 'temperature', 'top_p', 'service_tier')},
        'temperature': body.get('temperature'), 'top_p': body.get('top_p'), 'service_tier': body.get('service_tier'),
        'latency_seconds': receipt['latency_seconds']}


def grade_receipt(item, group, receipt, row, grade_response):
    fields = response_fields(item, group, receipt)
    # Do not duplicate prompt strings in each grade; request hash links exact text.
    fields['requested_settings'] = {k: v for k, v in group['request'].items() if k not in ('messages', 'input')}
    grade = grade_response(item['level'], item['domain'], row, fields['text'])
    return {k: v for k, v in item.items() if k != 'request'} | fields | grade | {
        'raw_receipt': receipt['relative_path'], 'raw_receipt_sha256': sha(receipt), 'recorded_at_utc': now()}


def validate_completed(output, item, group, record, raw_cache):
    for k in ('sample_id', 'level', 'domain', 'row_index', 'sample_index', 'row_sha256', 'request_sha256', 'group_id', 'choice_index'):
        if item[k] != record.get(k):
            raise ValueError('Saved sample identity mismatch: ' + k)
    relative = record.get('raw_receipt', '')
    path = Path(relative)
    gid, sep, attempt = path.stem.rpartition('__')
    if path.parent != Path('raw_responses') or path.suffix != '.json' or gid != group['group_id'] or not sep or not attempt.isdigit() or int(attempt) < 1:
        raise ValueError('Invalid saved raw receipt path')
    if relative not in raw_cache:
        raw_cache[relative] = json.loads((output / path).read_text())
    receipt = raw_cache[relative]
    validate_raw(group, receipt, relative)
    if record.get('raw_receipt_sha256') != sha(receipt):
        raise ValueError('Saved raw receipt digest mismatch')
    fields = response_fields(item, group, receipt)
    fields['requested_settings'] = {k: v for k, v in group['request'].items() if k not in ('messages', 'input')}
    if any(record.get(k) != v for k, v in fields.items()):
        raise ValueError('Saved grade differs from native response fields')
    if not isinstance(record.get('verified'), bool) or record['verified'] != (record.get('canonical_key') is not None):
        raise ValueError('Inconsistent verifier fields')


def index_raw(output, groups):
    index = defaultdict(list)
    for path in (output / 'raw_responses').glob('*.json'):
        gid, sep, attempt = path.stem.rpartition('__')
        if not sep or not attempt.isdigit() or int(attempt) < 1 or gid not in groups:
            raise ValueError('Unexpected raw receipt filename: ' + path.name)
        index[gid].append(path)
    return {gid: sorted(paths, key=lambda p: int(p.stem.rsplit('__', 1)[1])) for gid, paths in index.items()}


def load_frozen_grader(output):
    frozen = (output / 'code').resolve()
    for name, module in list(sys.modules.items()):
        if name == 'frontier_modebench_contract' or name == 'oat_drgrpo' or name.startswith('oat_drgrpo.'):
            source = getattr(module, '__file__', None)
            if source and not Path(source).resolve().is_relative_to(frozen):
                raise ValueError('Refusing previously imported unfrozen verifier module: ' + name)
    sys.path.insert(0, str(frozen / 'src'))
    sys.path.insert(0, str(frozen / 'ops'))
    from frontier_modebench_contract import grade_response
    return grade_response


def warm_python_worker():
    from oat_drgrpo.python_modebench_process import _SHARED_VERIFIER
    candidate = 'lambda n: 2'
    spec = {'verifier': 'python_factor_function', 'python_version': 'factor-v1', 'cases': [6, 8]}
    results = []
    for _ in range(3):
        _SHARED_VERIFIER._start()
        time.sleep(1.25)
        result = _SHARED_VERIFIER.validate(candidate, spec)
        results.append(result is not None and result.outputs == (2, 2))
        if results[-1]:
            return {'event': 'python_worker_warmed', 'at_utc': now(), 'synthetic_fixture': True,
                    'attempt_results': results, 'parent_timeout_seconds': _SHARED_VERIFIER.timeout_seconds,
                    'child_candidate_timeout_seconds': 0.25}
    raise RuntimeError('Frozen Python worker failed synthetic warmup')


class RequestPacer:
    def __init__(self, rpm):
        self.interval = 60 / rpm if rpm else 0
        self.next_start = 0.0
        self.lock = asyncio.Lock()

    async def wait(self):
        if not self.interval:
            return
        async with self.lock:
            wait = max(0, self.next_start - time.monotonic())
            self.next_start = max(self.next_start, time.monotonic()) + self.interval
        if wait:
            await asyncio.sleep(wait)


def retry_delay(headers, attempt):
    headers = {k.lower(): v for k, v in headers.items()}
    try:
        value = float(headers.get('retry-after', 0))
    except (TypeError, ValueError):
        try:
            value = max(0, (parsedate_to_datetime(headers['retry-after']) - datetime.now(timezone.utc)).total_seconds())
        except (KeyError, TypeError, ValueError):
            value = 0
    try:
        value = max(value, float(headers.get('retry-after-ms', 0)) / 1000)
    except (TypeError, ValueError):
        pass
    return max(value, min(60, 2 ** min(attempt, 6)))


async def run(output, model, workers=64, request_timeout=300, max_attempts=8, max_new=0, rpm=None, profile=None):
    import httpx
    output = Path(output)
    manifest = json.loads((output / 'manifest.json').read_text())
    if manifest['model'] != model:
        raise ValueError('Run model differs from frozen deployment')
    if profile is not None and profile != manifest['model_profile']:
        raise ValueError('Run profile differs from frozen request profile')
    validate_profile(manifest['model_profile'])
    verify_snapshot(output, manifest)
    grade_response = load_frozen_grader(output)
    key = os.environ.get('AZURE_OPENAI_API_KEY') or getpass.getpass('Azure API key (hidden): ')
    if not key:
        raise ValueError('No API credential')
    rows = {identity(r): r for r in read_jsonl(output / 'rows.jsonl')}
    requests = read_jsonl(output / 'requests.jsonl')
    request_lookup = {r['sample_id']: r for r in requests}
    group_list = read_jsonl(output / 'http_requests.jsonl')
    groups = {g['group_id']: g for g in group_list}
    for g in group_list:
        if g['request_sha256'] != sha(g['request']) or g['request']['model'] != model:
            raise ValueError('Corrupt HTTP request identity')
        if len(g['sample_ids']) != g['sample_count']:
            raise ValueError('Incomplete HTTP sample group')
        for sid in g['sample_ids']:
            item = request_lookup[sid]
            if item['group_id'] != g['group_id'] or item['request_sha256'] != g['request_sha256'] or item['request'] != g['request'] or item['row_sha256'] != sha(rows[identity(item)]):
                raise ValueError('Frozen sample/group identity mismatch')
    for folder in ('raw_responses', 'sample_receipts'):
        (output / folder).mkdir(exist_ok=True)
    raw_index = index_raw(output, groups)  # One directory listing for the entire run.
    raw_cache, completed = {}, {}
    provider_ids = {}
    for path in (output / 'sample_receipts').glob('*.json'):
        record = json.loads(path.read_text())
        sid = record.get('sample_id')
        if sid not in request_lookup or path.name != sid + '.json' or sid in completed:
            raise ValueError('Unexpected saved sample identity')
        item = request_lookup[sid]
        validate_completed(output, item, groups[item['group_id']], record, raw_cache)
        ident = tuple(record['provider_sample_identity'])
        if ident in provider_ids and provider_ids[ident] != sid:
            raise ValueError('Provider returned duplicate sample identity across independent requests')
        provider_ids[ident] = sid
        completed[sid] = record
    pending = [g for g in group_list if any(sid not in completed for sid in g['sample_ids'])]
    if max_new:
        if max_new % manifest['samples_per_http_request']:
            raise ValueError('max_new must be divisible by samples_per_http_request')
        pending = pending[:max_new // manifest['samples_per_http_request']]
    if pending and any(request_lookup[sid]['domain'] == 'python_factors' for g in pending for sid in g['sample_ids']):
        append(output / 'events.jsonl', await asyncio.to_thread(warm_python_worker))
    queue = asyncio.Queue()
    for g in pending:
        queue.put_nowait(g)
    failures, stop = [], asyncio.Event()
    started, initial_count, last_status = time.monotonic(), len(completed), 0.0
    pacer = RequestPacer(rpm if rpm is not None else (90 if model == 'FW-Kimi-K3' else 0))

    def save_status(force=False):
        nonlocal last_status
        if not force and time.monotonic() - last_status < 15:
            return
        last_status = time.monotonic()
        usage = Counter()
        for record in completed.values():
            for name, value in (record.get('usage') or {}).items():
                if isinstance(value, (float, int)) and not isinstance(value, bool):
                    usage[name] += value
        status = {'updated_at_utc': now(), 'model': model, 'completed_samples': len(completed),
            'expected_samples': len(requests), 'completed_http_groups': sum(all(sid in completed for sid in g['sample_ids']) for g in group_list),
            'expected_http_groups': len(groups), 'failed_groups_this_session': len(failures),
            'new_samples_this_session': len(completed) - initial_count,
            'elapsed_seconds_this_session': time.monotonic() - started,
            'response_status_counts': dict(Counter(r['response_status'] for r in completed.values())),
            'usage': dict(usage), 'usage_note': 'Native fields mapped without altering provider token accounting; group usage counted once at choice zero.',
            'complete': len(completed) == len(requests),
            'cells': dict(Counter(f"L{r['level']}/{r['domain']}" for r in completed.values()))}
        atomic(output / 'status.json', status)
        print(json.dumps(status), flush=True)

    async def process(group, client):
        gid = group['group_id']
        old = raw_index.get(gid, [])
        receipt = None
        for path in old:
            relative = str(path.relative_to(output))
            saved = raw_cache.get(relative)
            if saved is None:
                saved = json.loads(path.read_text())
            if saved.get('group_id') != gid or saved.get('request_sha256') != group['request_sha256'] or saved.get('relative_path') != relative:
                stop.set()
                raise ValueError('Historical raw receipt identity mismatch')
            if saved.get('http_status') == 200:
                try:
                    validate_raw(group, saved, relative)
                except ValueError:
                    stop.set()
                    raise
                if receipt is None:
                    receipt = saved
        if receipt is None:
            first_attempt = max((int(p.stem.rsplit('__', 1)[1]) for p in old), default=0) + 1
            for offset in range(max_attempts):
                await pacer.wait()
                if stop.is_set():
                    raise RuntimeError('Stopped before starting another paid request')
                attempt = first_attempt + offset
                relative = f'raw_responses/{gid}__{attempt:02d}.json'
                append(output / 'events.jsonl', {'event': 'request_started', 'at_utc': now(), 'group_id': gid,
                    'sample_ids': group['sample_ids'], 'attempt': attempt, 'request_sha256': group['request_sha256']})
                begin = time.monotonic()
                receipt = {'group_id': gid, 'sample_ids': group['sample_ids'], 'attempt': attempt,
                    'request_sha256': group['request_sha256'], 'relative_path': relative, 'started_at_utc': now()}
                try:
                    response = await client.post(manifest['endpoint'], json=group['request'])
                    body_text = response.text
                    try:
                        body = response.json()
                    except ValueError:
                        body = {'non_json_response': body_text}
                    header_items = [(k, '[REDACTED]' if k.lower() in SENSITIVE_HEADERS else v) for k, v in response.headers.multi_items()]
                    receipt.update(http_status=response.status_code, response=body, http_body_text=body_text,
                                   headers=dict(header_items), header_items=header_items)
                except (httpx.TimeoutException, httpx.TransportError) as error:
                    receipt.update(http_status=None, error_type=type(error).__name__, error=str(error), possibly_billed=True)
                receipt.update(latency_seconds=time.monotonic() - begin, received_at_utc=now())
                receipt = json.loads(json.dumps(receipt).replace(key, '[REDACTED]'))
                atomic(output / relative, receipt)
                raw_index.setdefault(gid, []).append(output / relative)
                if receipt.get('http_status') == 200:
                    try:
                        validate_raw(group, receipt, relative)
                    except ValueError:
                        stop.set()
                        raise
                    break
                append(output / 'errors.jsonl', receipt)
                status = receipt.get('http_status')
                if status is not None and status not in (408, 409, 425, 429) and not 500 <= status <= 599:
                    stop.set()
                    raise RuntimeError(f'Non-retryable API error {status}; see {relative}')
                delay = retry_delay(receipt.get('headers', {}), offset + 1)
                receipt = None
                if offset + 1 < max_attempts:
                    await asyncio.sleep(delay)
            if receipt is None:
                raise RuntimeError('Retry budget exhausted: ' + gid)
        validate_raw(group, receipt)
        for sid in group['sample_ids']:
            if sid in completed:
                continue
            item = request_lookup[sid]
            ident = (receipt['response']['id'], item['choice_index'])
            if ident in provider_ids and provider_ids[ident] != sid:
                stop.set()
                raise ValueError('Duplicate provider sample identity across requests')
            provider_ids[ident] = sid
            record = await asyncio.to_thread(grade_receipt, item, group, receipt, rows[identity(item)], grade_response)
            atomic(output / 'sample_receipts' / (sid + '.json'), record)
            append(output / 'samples.jsonl', record)
            completed[sid] = record
        save_status()

    async def worker(client):
        while not stop.is_set():
            try:
                group = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                await process(group, client)
            except Exception as error:
                entry = {'group_id': group['group_id'], 'at_utc': now(), 'error_type': type(error).__name__,
                         'error': str(error).replace(key, '[REDACTED]')}
                failures.append(entry)
                append(output / 'runner_errors.jsonl', entry)
                print(json.dumps(entry), flush=True)
            finally:
                queue.task_done()

    append(output / 'events.jsonl', {'event': 'session_started', 'at_utc': now(), 'workers': workers,
        'pending_http_groups': len(pending), 'request_timeout_seconds': request_timeout, 'pid': os.getpid(),
        'model_profile_sha256': manifest['model_profile_sha256'], 'rpm_limit': 60 / pacer.interval if pacer.interval else None})
    save_status(True)
    try:
        async with httpx.AsyncClient(headers={'api-key': key}, follow_redirects=False,
            timeout=httpx.Timeout(request_timeout, connect=30),
            limits=httpx.Limits(max_connections=workers, max_keepalive_connections=workers)) as client:
            await asyncio.gather(*(worker(client) for _ in range(workers)))
    finally:
        write_jsonl(output / 'samples.jsonl', (completed[item['sample_id']] for item in requests if item['sample_id'] in completed))
        save_status(True)
        append(output / 'events.jsonl', {'event': 'session_finished', 'at_utc': now(),
            'completed_samples': len(completed), 'failures': len(failures), 'stopped': stop.is_set()})
    return 0 if not failures and (max_new or len(completed) == len(requests)) else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run'))
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path, default=REFERENCE)
    parser.add_argument('--profile', type=Path, help='Immutable provider-specific request profile JSON; defaults to preflight baseline.')
    parser.add_argument('--max-output-tokens', type=int, default=8192, help='Default-profile output budget, ignored when --profile is supplied.')
    parser.add_argument('--workers', type=int, default=64)
    parser.add_argument('--request-timeout', type=float, default=300)
    parser.add_argument('--max-attempts', type=int, default=8)
    parser.add_argument('--max-new', type=int, default=0, help='Bounded integration samples; zero runs the full set. Must be a multiple of n.')
    parser.add_argument('--rpm', type=float, help='Request-start rate; default 90 for Kimi, unlimited for other deployments.')
    args = parser.parse_args()
    if not 1 <= args.workers <= 64 or args.max_attempts < 1 or args.request_timeout <= 0 or args.max_new < 0 or (args.rpm is not None and args.rpm < 0):
        parser.error('Invalid concurrency, retry, rate, or sample limit')
    profile = json.loads(args.profile.read_text()) if args.profile else default_profile(args.model, args.max_output_tokens)
    if profile.get('model') != args.model:
        parser.error('Profile deployment differs from --model')
    validate_profile(profile)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / '.runner.lock').open('w') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.command == 'prepare':
            manifest = prepare(args.output, profile, args.reference)
            print(json.dumps({k: manifest[k] for k in ('model', 'model_profile', 'prompt_count', 'request_count', 'http_request_count', 'samples_per_http_request')}, indent=2))
            return 0
        # No profile means use the manifest's exact options, but deployment must match.
        return asyncio.run(run(args.output, args.model, args.workers, args.request_timeout,
                               args.max_attempts, args.max_new, args.rpm, profile if args.profile else None))


if __name__ == '__main__':
    raise SystemExit(main())
