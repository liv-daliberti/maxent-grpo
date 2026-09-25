#!/usr/bin/env python3
"""Collect GPT-5.6 Sol on Python under both wordings, direct through OpenAI.

The hosted appendix substitutes a revised Python wording for one deployment,
Claude Opus 5, because the provider refused most of the original requests. That
substitution is marked and carries a Python-excluded companion, but nothing
measures what the revised wording does to any *other* deployment. This condition
measures it on the one deployment we can reach.

Both arms run here. Levels 1--3 of the published GPT-5.6 Sol cohort were served
by Azure at ``liv.services.ai.azure.com``; this runs against ``api.openai.com``,
which may serve a different build. Collecting only the revised arm and comparing
it to the published cells would therefore confound wording with deployment --
the same class of objection this condition exists to answer. So the original
wording is re-collected here too, byte-identical to the published requests, and
every comparison drawn from this directory is within this directory.

What this condition does and does not establish:

* It is a paired within-deployment contrast of two wordings on the same 384
  frozen Python rows, the same eight draws per row, and the same grader.
* It is **not** a replacement for any published GPT-5.6 Sol cell. The published
  Azure cells stay exactly as they are; nothing here feeds the hosted cohort
  tables, the PCMD cohort table, or any macro average.
* It does not identify *why* a wording changes an outcome. Task wording and the
  presence of a system message change together, as they do in the Opus 5
  condition it is built to qualify.

The credential is read from ``OPENAI_API_KEY`` and is never written to disk,
echoed, or stored in any receipt.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / 'ops', ROOT / 'src'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

ENDPOINT = 'https://api.openai.com/v1/responses'
MODEL = 'gpt-5.6-sol'
DOMAIN = 'python_factors'
LEVELS = (1, 2, 3)
SAMPLES = 8
PROMPTS_PER_LEVEL = 128
ARMS = ('original', 'revised')
SCHEMA = 'gpt56sol-python-wording-parity-openai-direct-v1'
SOURCE = ROOT / 'artifacts/frontier_modebench_gpt56sol_20260911'
DEFAULT_OUTPUT = ROOT / 'artifacts/frontier_modebench_gpt56sol_python_parity_20260918'

#: The revised arm reuses the wording of the Opus 5 condition verbatim, so the
#: two conditions test the same intervention rather than two similar ones.
PROMPT_PREFIX = 'For the inputs n in '
PROMPT_SUFFIX = (
    ', write one arithmetic expression in Python notation lambda n: EXPR. '
    'For every listed input, the result d must be an integer satisfying '
    '1 < d < n and n % d == 0. Use only integer literals, n, +, -, *, //, %, '
    'comparisons, Boolean operators, and conditional expressions. '
    'Any proper divisor meeting these conditions is acceptable. '
    'Return only the one-line lambda inside \\boxed{}.'
)
REVISED_SOURCE = ROOT / 'ops/prepare_frontier_python_plain_prompt.py'


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def file_sha(path: Path) -> str:
    with Path(path).open('rb') as handle:
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def require(condition, message: str) -> None:
    if not condition:
        raise ValueError(message)


def public_cases(row: dict) -> list[int]:
    """The public input list, read from the frozen task rather than the prompt text."""
    matches = re.findall(r'for each n in (\[[0-9, ]+\])\.', row['problem'])
    require(len(matches) == 1, 'Expected one public input list in the original problem')
    cases = json.loads(matches[0])
    answer = row['answer']
    answer = json.loads(answer) if isinstance(answer, str) else answer
    require(cases and all(type(n) is int and n > 2 for n in cases)
            and answer.get('cases') == cases
            and answer.get('verifier') == 'python_factor_function'
            and answer.get('python_version') == 'factor-v1',
            'Public input list differs from the frozen mathematical task')
    return cases


def revised_prompt(row: dict) -> str:
    return PROMPT_PREFIX + json.dumps(public_cases(row)) + PROMPT_SUFFIX


def _verify_revised_wording() -> str:
    """The revised wording must still be the one the Opus 5 condition froze."""
    text = REVISED_SOURCE.read_text(encoding='utf-8')
    require(f'PROMPT_PREFIX = {PROMPT_PREFIX!r}' in text,
            'Revised prompt prefix differs from the frozen Opus 5 condition')
    for fragment in ('write one arithmetic expression in Python notation lambda n: EXPR',
                     'Any proper divisor meeting these conditions is acceptable',
                     'Return only the one-line lambda inside'):
        require(fragment in text and fragment in PROMPT_SUFFIX,
                f'Revised prompt fragment differs from the frozen condition: {fragment}')
    return file_sha(REVISED_SOURCE)


def load_source() -> tuple[dict, dict, dict]:
    """Rows and frozen requests of the published cohort, re-authenticated here."""
    manifest = json.loads((SOURCE / 'manifest.json').read_text())
    require(manifest['model'] == MODEL, 'Source cohort is a different deployment')
    require('azure' in manifest['endpoint'],
            'Source cohort is no longer the Azure deployment this condition contrasts with')
    rows, requests = {}, {}
    for row in read_jsonl(SOURCE / 'rows.jsonl'):
        if row.get('domain') != DOMAIN or row.get('level') not in LEVELS:
            continue
        rows[(row['level'], row['problem'])] = row
    ordered = defaultdict(dict)
    for record in read_jsonl(SOURCE / 'requests.jsonl'):
        if record.get('domain') != DOMAIN or record.get('level') not in LEVELS:
            continue
        ordered[(record['level'], record['row_index'])][record['sample_index']] = record
    require(len(ordered) == PROMPTS_PER_LEVEL * len(LEVELS),
            'Expected 128 Python prompts at each of three levels in the source cohort')
    for key, samples in ordered.items():
        require(sorted(samples) == list(range(SAMPLES)),
                f'{key}: source cohort does not carry all eight draws')
    return manifest, rows, ordered


def prepare(output: Path) -> dict:
    """Freeze every request in both arms before a single response is observed."""
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'manifest.json'
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    revised_source_sha = _verify_revised_wording()
    source_manifest, rows, ordered = load_source()

    by_row = {}
    for (level, row_index), samples in ordered.items():
        first = samples[0]
        payload = first['request']
        require(sha(payload) == first['request_sha256'], 'Source request digest mismatch')
        require(payload['model'] == MODEL and payload.get('store') is False,
                'Source request is not the frozen deployment payload')
        messages = payload['input']
        require(len(messages) == 2 and messages[0]['role'] == 'system'
                and messages[1]['role'] == 'user',
                'Original Python request is not a system-plus-user pair')
        row = rows.get((level, messages[1]['content']))
        require(row is not None, f'level {level} row {row_index}: no frozen row matches the request')
        require(sha(row) == first['row_sha256'], 'Source row digest mismatch')
        for index, record in samples.items():
            require(record['request_sha256'] == first['request_sha256']
                    and record['row_sha256'] == first['row_sha256'],
                    f'level {level} row {row_index}: draws do not share one frozen request')
        by_row[(level, row_index)] = (row, payload)

    requests = []
    # Interleave arm, level and draw so a partial run still covers every cell.
    for row_index in range(PROMPTS_PER_LEVEL):
        for sample_index in range(SAMPLES):
            for level in LEVELS:
                row, original = by_row[(level, row_index)]
                revised = copy.deepcopy(original)
                revised['input'] = [{'role': 'user', 'content': revised_prompt(row)}]
                for arm, payload in (('original', copy.deepcopy(original)), ('revised', revised)):
                    require(payload['model'] == original['model']
                            and payload['reasoning'] == original['reasoning']
                            and payload['max_output_tokens'] == original['max_output_tokens']
                            and payload['store'] == original['store'],
                            'Only the wording and system message may differ between arms')
                    requests.append({
                        'arm': arm, 'level': level, 'domain': DOMAIN, 'row_index': row_index,
                        'sample_index': sample_index, 'row_sha256': sha(row),
                        'sample_id': f'{arm}_L{level}_{DOMAIN}_{row_index:03d}_{sample_index}',
                        'request': payload, 'request_sha256': sha(payload)})

    expected = len(ARMS) * len(LEVELS) * PROMPTS_PER_LEVEL * SAMPLES
    require(len(requests) == expected, f'Expected {expected} frozen requests')
    require(len({r['sample_id'] for r in requests}) == expected, 'Duplicate sample identity')
    original_digests = {r['request_sha256'] for r in requests if r['arm'] == 'original'}
    published = {record['request_sha256'] for samples in ordered.values()
                 for record in samples.values()}
    require(original_digests == published,
            'The original arm is not byte-identical to the published requests')
    require(not original_digests & {r['request_sha256'] for r in requests if r['arm'] == 'revised'},
            'A revised request collides with an original one')

    (output / 'requests.jsonl').write_text(
        ''.join(json.dumps(r, sort_keys=True) + '\n' for r in requests))
    manifest = {
        'schema': SCHEMA, 'prepared_at_utc': now(), 'endpoint': ENDPOINT,
        'provider': 'openai_direct', 'model': MODEL, 'domain': DOMAIN,
        'levels': list(LEVELS), 'arms': list(ARMS), 'sample_count': SAMPLES,
        'prompts_per_level': PROMPTS_PER_LEVEL, 'requests': len(requests),
        'requests_sha256': sha(sorted(r['request_sha256'] for r in requests)),
        'reasoning_effort': source_manifest.get('reasoning_effort'),
        'max_output_tokens': source_manifest.get('max_output_tokens'),
        'source_cohort': {
            'directory': str(SOURCE.relative_to(ROOT)),
            'endpoint': source_manifest['endpoint'],
            'manifest_sha256': file_sha(SOURCE / 'manifest.json'),
            'rows_sha256': file_sha(SOURCE / 'rows.jsonl'),
            'requests_sha256': file_sha(SOURCE / 'requests.jsonl'),
        },
        'revised_wording_source': {
            'path': str(REVISED_SOURCE.relative_to(ROOT)), 'sha256': revised_source_sha},
        'provider_differs_from_published_cells': True,
        'published_cells_provider': source_manifest['endpoint'],
        'claims_not_made': [
            'This condition replaces no published GPT-5.6 Sol cell.',
            'No result here enters a hosted cohort table, the PCMD cohort table, or any macro average.',
            'Wording and system-message presence change together; neither is isolated.',
            'The published cells were served by a different provider, so only within-directory '
            'comparisons are drawn from this run.',
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    return manifest


async def _one(client, item, sem, max_attempts, lock, handle, raw_handle, done):
    async with sem:
        for attempt in range(max_attempts):
            try:
                response = await client.post(ENDPOINT, json=item['request'])
                if response.status_code in (429, 500, 502, 503, 504):
                    await asyncio.sleep(min(2 ** attempt, 30))
                    continue
                response.raise_for_status()
                body = response.json()
                record = {
                    **{k: item[k] for k in ('arm', 'level', 'domain', 'row_index', 'sample_index',
                                            'sample_id', 'row_sha256', 'request_sha256')},
                    'recorded_at_utc': now(),
                    'served_model': body.get('model'),
                    'response_id': body.get('id'),
                    'status': body.get('status'),
                    'usage': body.get('usage'),
                    'text': _extract_text(body),
                    'incomplete_details': body.get('incomplete_details'),
                }
                async with lock:
                    handle.write(json.dumps(record, sort_keys=True) + '\n')
                    # The refusal audit reads native provider fields, so the whole
                    # body is retained rather than the text this run extracts.
                    raw_handle.write(json.dumps(
                        {'sample_id': item['sample_id'], 'body': body}, sort_keys=True) + '\n')
                    handle.flush()
                    raw_handle.flush()
                    done.append(1)
                    if len(done) % 250 == 0:
                        print(json.dumps({'event': 'progress', 'done': len(done)}), flush=True)
                return
            except Exception as exc:  # network/transport; retried with backoff
                if attempt == max_attempts - 1:
                    async with lock:
                        handle.write(json.dumps(
                            {'sample_id': item['sample_id'],
                             'error': f'{type(exc).__name__}: {exc}'[:300]}, sort_keys=True) + '\n')
                        handle.flush()
                    return
                await asyncio.sleep(min(2 ** attempt, 30))


def _extract_text(body: dict) -> str:
    chunks = []
    for item in body.get('output') or []:
        if item.get('type') != 'message':
            continue
        for part in item.get('content') or []:
            if part.get('type') in ('output_text', 'text') and isinstance(part.get('text'), str):
                chunks.append(part['text'])
    return ''.join(chunks)


async def _run(output: Path, workers: int, timeout: float, max_attempts: int, limit: int) -> None:
    import httpx
    key = os.environ.get('OPENAI_API_KEY')
    if not key:
        raise SystemExit('OPENAI_API_KEY is not set; the credential is never read from a file')
    requests = read_jsonl(output / 'requests.jsonl')
    responses_path = output / 'responses.jsonl'
    seen = set()
    if responses_path.exists():
        for line in responses_path.read_text().splitlines():
            try:
                record = json.loads(line)
            except Exception:
                continue
            if 'error' not in record:
                seen.add(record['sample_id'])
    pending = [r for r in requests if r['sample_id'] not in seen]
    if limit:
        pending = pending[:limit]
    print(json.dumps({'event': 'start', 'pending': len(pending),
                      'already_done': len(seen)}), flush=True)
    sem = asyncio.Semaphore(workers)
    lock = asyncio.Lock()
    done: list[int] = []
    started = time.time()
    with responses_path.open('a') as handle, (output / 'raw_responses.jsonl').open('a') as raw:
        async with httpx.AsyncClient(headers={'Authorization': f'Bearer {key}'},
                                     timeout=timeout) as client:
            await asyncio.gather(*[_one(client, item, sem, max_attempts, lock, handle, raw, done)
                                   for item in pending])
    print(json.dumps({'event': 'complete', 'written': len(done),
                      'elapsed_s': round(time.time() - started, 1)}), flush=True)


def grade(output: Path) -> dict:
    """Apply the original executable verifier to every saved response."""
    from frontier_modebench_contract import grade_response

    _, rows, ordered = load_source()
    by_row = {}
    for (level, row_index), samples in ordered.items():
        message = samples[0]['request']['input'][1]['content']
        by_row[(level, row_index)] = rows[(level, message)]
    graded, errors = [], 0
    for record in read_jsonl(output / 'responses.jsonl'):
        if 'text' not in record:
            errors += 1
            continue
        row = by_row[(record['level'], record['row_index'])]
        verdict = grade_response(record['level'], DOMAIN, row, record['text'])
        graded.append({**record, **verdict})
    (output / 'audited_primary_samples.jsonl').write_text(
        ''.join(json.dumps(g, sort_keys=True) + '\n' for g in graded))
    summary = {'schema': SCHEMA, 'graded': len(graded), 'transport_errors': errors,
               'served_models': dict(Counter(g.get('served_model') for g in graded)),
               'verified': sum(1 for g in graded if g['verified'])}
    (output / 'grading_summary.json').write_text(
        json.dumps(summary, indent=2, sort_keys=True) + '\n')
    return summary


def normalize(output: Path) -> dict:
    """Post-hoc, formatting-only sensitivity analysis, under the frozen normalizer."""
    from frontier_modebench_normalization import normalize_and_grade

    _, rows, ordered = load_source()
    by_row = {}
    for (level, row_index), samples in ordered.items():
        message = samples[0]['request']['input'][1]['content']
        by_row[(level, row_index)] = rows[(level, message)]
    out = []
    for record in read_jsonl(output / 'audited_primary_samples.jsonl'):
        row = dict(by_row[(record['level'], record['row_index'])])
        row['level'], row['domain'] = record['level'], DOMAIN
        strict = {k: record[k] for k in ('verified', 'canonical_key', 'graded_text')}
        verdict = normalize_and_grade(row, record['text'], strict_grade=strict)
        out.append({**{k: record[k] for k in ('arm', 'level', 'domain', 'row_index',
                                              'sample_index', 'sample_id', 'served_model')},
                    'strict_verified': strict['verified'],
                    'verified': verdict['verified'],
                    'canonical_key': verdict['canonical_key'],
                    'transformations': verdict.get('transformations', [])})
    (output / 'normalized_samples.jsonl').write_text(
        ''.join(json.dumps(o, sort_keys=True) + '\n' for o in out))
    return {'normalized': len(out),
            'rescued': sum(1 for o in out if o['verified'] and not o['strict_verified'])}


def _cell_metrics(draws: dict[int, list[dict]], key: str) -> dict:
    """Accuracy, distinct@8 and correct-pair collision over complete prompt groups."""
    correct = distinct = pairs = colliding = responses = 0
    eligible = 0
    for group in draws.values():
        responses += len(group)
        keys = [item[key] for item in group if item['verified']]
        correct += len(keys)
        distinct += len(set(keys))
        counts = Counter(keys)
        total = len(keys) * (len(keys) - 1) // 2
        same = sum(n * (n - 1) // 2 for n in counts.values())
        pairs += total
        colliding += same
        if len(keys) >= 2:
            eligible += 1
    return {
        'prompts': len(draws), 'responses': responses, 'correct_responses': correct,
        'distinct_correct_modes': distinct, 'correct_pairs': pairs,
        'colliding_correct_pairs': colliding, 'collision_eligible_prompts': eligible,
        'accuracy': correct / responses if responses else None,
        'distinct8': distinct / len(draws) if draws else None,
        'correct_pair_collision': colliding / pairs if pairs else None,
        'pcmd': 1 - colliding / pairs if pairs else None,
    }


def summarize(output: Path) -> dict:
    """Per arm and level, under both gradings, from the complete saved cohort."""
    strict = read_jsonl(output / 'audited_primary_samples.jsonl')
    normalized = {record['sample_id']: record
                  for record in read_jsonl(output / 'normalized_samples.jsonl')}
    cells = {}
    for grading, records in (('strict', strict),
                             ('normalized', [normalized[r['sample_id']] for r in strict])):
        grouped = defaultdict(lambda: defaultdict(list))
        for record in records:
            grouped[(record['arm'], record['level'])][record['row_index']].append(record)
        for (arm, level), draws in grouped.items():
            cells[f'{grading}/{arm}/level{level}'] = _cell_metrics(draws, 'canonical_key')
    payload = {'schema': SCHEMA, 'created_at_utc': now(),
               'manifest_sha256': file_sha(output / 'manifest.json'),
               'primary_samples_sha256': file_sha(output / 'audited_primary_samples.jsonl'),
               'normalized_samples_sha256': file_sha(output / 'normalized_samples.jsonl'),
               'definitions': {
                   'accuracy': 'Verified responses over all sampled responses, refusals included.',
                   'distinct8': 'Mean distinct verified canonical modes per prompt, zero included.',
                   'pcmd': 'One minus the pooled correct-pair collision within prompts.',
               },
               'cells': cells}
    (output / 'summary.json').write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n')
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stage', choices=('prepare', 'run', 'grade', 'normalize', 'summarize'))
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--timeout', type=float, default=300.0)
    parser.add_argument('--max-attempts', type=int, default=5)
    parser.add_argument('--limit', type=int, default=0,
                        help='Collect at most this many pending requests; 0 collects all')
    args = parser.parse_args()
    if args.stage == 'prepare':
        result = prepare(args.output)
        print(json.dumps({k: result[k] for k in ('schema', 'requests', 'arms', 'levels')}))
    elif args.stage == 'run':
        prepare(args.output)
        asyncio.run(_run(args.output, args.workers, args.timeout,
                         args.max_attempts, args.limit))
    else:
        print(json.dumps({'prepare': prepare, 'grade': grade, 'normalize': normalize,
                          'summarize': summarize}[args.stage](args.output),
                         indent=2, sort_keys=True)[:2000])


if __name__ == '__main__':
    main()
