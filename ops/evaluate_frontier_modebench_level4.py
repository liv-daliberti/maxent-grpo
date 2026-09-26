#!/usr/bin/env python3
"""Collect GPT-5.6 Sol on Level 4, direct through OpenAI.

This is a separate runner rather than a flag on
``ops/evaluate_frontier_modebench.py`` for two reasons. That collector asserts
all fifteen Level 1--3 cells and targets the Azure deployment whose served
snapshot the manuscript's hosted appendix documents; neither assertion should be
loosened to accommodate a different provider and an unadmitted level.

Two properties of this run differ from every other hosted cell in the paper and
must travel with it wherever it is reported:

* **Provider.** Levels 1--3 were served by Azure at
  ``liv.services.ai.azure.com`` under snapshot ``gpt-5.6-sol-2026-07-09``. This
  runs against ``api.openai.com`` directly, which may serve a different build.
  The served model string is recorded per response so the two can be compared
  rather than assumed equal.
* **Admission.** Level 4 has not completed composite admission. These cells
  measure a frozen deployment on a dataset that is not an admitted benchmark
  level, and no difficulty equivalence with Levels 1--3 is claimed.

The credential is read from ``OPENAI_API_KEY`` and is never written to disk,
echoed, or stored in any receipt.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / 'ops', ROOT / 'src'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

ENDPOINT = 'https://api.openai.com/v1/responses'
MODEL = 'gpt-5.6-sol'
LEVEL = 4
DOMAINS = ('graph_coloring', 'countdown', 'python_factors', 'mathir', 'pantry')
ROWS = ROOT / 'artifacts/modebench_base_level_grid_multifamily_20260914/rows/level4'
DEFAULT_OUTPUT = ROOT / 'artifacts/frontier_modebench_gpt56sol_level4_20260914'
SCHEMA = 'frontier-modebench-level4-openai-direct-v1'
SAMPLES = 8


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def load_rows() -> list[dict]:
    rows = []
    for domain in DOMAINS:
        path = ROWS / f'{domain}.jsonl'
        lines = path.read_text().splitlines()
        if len(lines) != 128:
            raise ValueError(f'{domain}: expected 128 rows, found {len(lines)}')
        for index, line in enumerate(lines):
            row = json.loads(line)
            row['_domain'] = 'pantry_plan' if domain == 'pantry' else domain
            row['_row_index'] = index
            rows.append(row)
    return rows


def prepare(output: Path, max_output_tokens: int) -> dict:
    """Freeze every request before a single response is observed."""
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'manifest.json'
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    from frontier_modebench_contract import make_messages, profile_metadata

    rows = load_rows()
    requests = []
    # Interleave so that a partial run still covers every cell evenly.
    for row_index in range(128):
        for sample_index in range(SAMPLES):
            for domain in DOMAINS:
                key = 'pantry_plan' if domain == 'pantry' else domain
                row = next(r for r in rows if r['_domain'] == key and r['_row_index'] == row_index)
                payload = {'model': MODEL,
                           'input': make_messages(LEVEL, key, row),
                           'reasoning': {'effort': 'medium'},
                           'max_output_tokens': max_output_tokens, 'store': False}
                requests.append({
                    'level': LEVEL, 'domain': key, 'row_index': row_index,
                    'sample_index': sample_index, 'row_sha256': sha(row),
                    'sample_id': f'L4_{key}_{row_index:03d}_{sample_index}',
                    'request': payload, 'request_sha256': sha(payload)})
    (output / 'requests.jsonl').write_text(
        ''.join(json.dumps(r, sort_keys=True) + '\n' for r in requests))
    manifest = {
        'schema': SCHEMA, 'prepared_at_utc': now(), 'endpoint': ENDPOINT,
        'provider': 'openai_direct',
        'provider_differs_from_levels_1_3': True,
        'levels_1_3_provider': 'azure:liv.services.ai.azure.com snapshot gpt-5.6-sol-2026-07-09',
        'model': MODEL, 'reasoning_effort': 'medium',
        'max_output_tokens': max_output_tokens, 'sample_count': SAMPLES,
        'level4_composite_admission_completed': False,
        'claims_not_made': [
            'No difficulty equivalence between Level 4 and any other level is claimed.',
            'No composite release admission is claimed.',
            'Level 4 and Levels 1-3 hosted cells were served by different providers.',
        ],
        'profile': {d: profile_metadata(LEVEL, 'pantry_plan' if d == 'pantry' else d)
                    for d in DOMAINS},
        'requests': len(requests),
        'requests_sha256': sha([r['request_sha256'] for r in requests]),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    return manifest


async def _one(client, item, sem, max_attempts, out_lock, handle, done):
    import httpx
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
                    **{k: item[k] for k in ('level', 'domain', 'row_index', 'sample_index',
                                            'sample_id', 'row_sha256', 'request_sha256')},
                    'recorded_at_utc': now(),
                    'served_model': body.get('model'),
                    'response_id': body.get('id'),
                    'status': body.get('status'),
                    'usage': body.get('usage'),
                    'text': _extract_text(body),
                    'incomplete_details': body.get('incomplete_details'),
                }
                async with out_lock:
                    handle.write(json.dumps(record, sort_keys=True) + '\n')
                    handle.flush()
                    done.append(1)
                    if len(done) % 250 == 0:
                        print(json.dumps({'event': 'progress', 'done': len(done)}), flush=True)
                return
            except Exception as exc:  # network/transport; retried with backoff
                if attempt == max_attempts - 1:
                    async with out_lock:
                        handle.write(json.dumps({**{k: item[k] for k in ('sample_id',)},
                                                 'error': f'{type(exc).__name__}: {exc}'[:300]},
                                                sort_keys=True) + '\n')
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


async def _run(output: Path, workers: int, timeout: float, max_attempts: int, limit: int):
    import httpx
    key = os.environ.get('OPENAI_API_KEY')
    if not key:
        raise SystemExit('OPENAI_API_KEY is not set; the credential is never read from a file')
    requests = [json.loads(line) for line in (output / 'requests.jsonl').read_text().splitlines()]
    responses_path = output / 'responses.jsonl'
    seen = set()
    if responses_path.exists():
        for line in responses_path.read_text().splitlines():
            try:
                seen.add(json.loads(line)['sample_id'])
            except Exception:
                continue
    pending = [r for r in requests if r['sample_id'] not in seen]
    if limit:
        pending = pending[:limit]
    print(json.dumps({'event': 'start', 'pending': len(pending), 'already_done': len(seen)}), flush=True)
    sem = asyncio.Semaphore(workers)
    lock = asyncio.Lock()
    done: list[int] = []
    started = time.time()
    with responses_path.open('a') as handle:
        async with httpx.AsyncClient(headers={'Authorization': f'Bearer {key}'},
                                     timeout=timeout) as client:
            await asyncio.gather(*[_one(client, item, sem, max_attempts, lock, handle, done)
                                   for item in pending])
    print(json.dumps({'event': 'complete', 'written': len(done),
                      'elapsed_s': round(time.time() - started, 1)}), flush=True)


def grade(output: Path) -> dict:
    """Apply the original executable verifier to every saved response."""
    from frontier_modebench_contract import grade_response

    rows = {(r['_domain'], r['_row_index']): r for r in load_rows()}
    graded, errors = [], 0
    for line in (output / 'responses.jsonl').read_text().splitlines():
        record = json.loads(line)
        if 'text' not in record:
            errors += 1
            continue
        row = rows[(record['domain'], record['row_index'])]
        verdict = grade_response(LEVEL, record['domain'], row, record['text'])
        graded.append({**record, **verdict})
    path = output / 'audited_primary_samples.jsonl'
    path.write_text(''.join(json.dumps(g, sort_keys=True) + '\n' for g in graded))
    served = Counter(g.get('served_model') for g in graded)
    summary = {'schema': SCHEMA, 'graded': len(graded), 'transport_errors': errors,
               'served_models': dict(served),
               'verified': sum(1 for g in graded if g['verified'])}
    (output / 'grading_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    return summary


def normalize(output: Path) -> dict:
    """Post-hoc, formatting-only sensitivity analysis, as at Levels 1-3.

    The primary grade stays strict; this reruns the identical responses through
    the frozen formatting normalizer so presentation differences are not scored
    as capability. It matters most in Python, where the deployment emits
    LaTeX-escaped Python that a strict grader rejects outright.
    """
    from frontier_modebench_normalization import normalize_and_grade

    rows = {(r['_domain'], r['_row_index']): r for r in load_rows()}
    out = []
    for line in (output / 'audited_primary_samples.jsonl').read_text().splitlines():
        record = json.loads(line)
        row = dict(rows[(record['domain'], record['row_index'])])
        row['level'], row['domain'] = LEVEL, record['domain']
        strict = {k: record[k] for k in ('verified', 'canonical_key', 'graded_text')}
        verdict = normalize_and_grade(row, record['text'], strict_grade=strict)
        out.append({**{k: record[k] for k in ('level', 'domain', 'row_index', 'sample_index',
                                              'sample_id', 'served_model')},
                    'strict_verified': strict['verified'],
                    'verified': verdict['verified'],
                    'canonical_key': verdict['canonical_key'],
                    'transformations': verdict.get('transformations', [])})
    (output / 'normalized_secondary_samples.jsonl').write_text(
        ''.join(json.dumps(o, sort_keys=True) + '\n' for o in out))
    from collections import Counter, defaultdict
    per = defaultdict(lambda: [0, 0, 0])
    for o in out:
        per[o['domain']][0] += 1
        per[o['domain']][1] += o['strict_verified']
        per[o['domain']][2] += o['verified']
    summary = {'schema': SCHEMA + '-normalized', 'analysis':
               'Post hoc, formatting-only sensitivity analysis; strict successes retained exactly.',
               'per_domain': {d: {'draws': n, 'strict_verified': sv, 'normalized_verified': nv}
                              for d, (n, sv, nv) in sorted(per.items())},
               'rescued_by_normalization': sum(1 for o in out
                                               if o['verified'] and not o['strict_verified'])}
    (output / 'normalized_summary.json').write_text(
        json.dumps(summary, indent=2, sort_keys=True) + '\n')
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run', 'grade', 'normalize'))
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--max-output-tokens', type=int, default=8192)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--request-timeout', type=float, default=240)
    parser.add_argument('--max-attempts', type=int, default=6)
    parser.add_argument('--limit', type=int, default=0, help='bounded smoke run; 0 means all')
    args = parser.parse_args()
    if args.command == 'prepare':
        manifest = prepare(args.output, args.max_output_tokens)
        print(json.dumps({'event': 'prepared', 'requests': manifest['requests']}))
    elif args.command == 'run':
        prepare(args.output, args.max_output_tokens)
        asyncio.run(_run(args.output, args.workers, args.request_timeout,
                         args.max_attempts, args.limit))
    elif args.command == 'grade':
        print(json.dumps(grade(args.output)))
    else:
        print(json.dumps(normalize(args.output)))


if __name__ == '__main__':
    main()
