#!/usr/bin/env python3
"""Save temperature-only GPT-5.6 Sol capability probes, outside the benchmark."""
from __future__ import annotations

import argparse
import asyncio
import getpass
import math
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ops'))
from evaluate_chat_frontier_modebench import atomic, file_sha, now, sha

OUTPUT = ROOT / 'artifacts/frontier_temperature_20260911/gpt56_control_probe'
ENDPOINT = 'https://liv.services.ai.azure.com/openai/v1/responses'
SENSITIVE = {'authorization', 'api-key', 'x-api-key', 'set-cookie'}


def validate_cached_grid(result, temperatures, efforts):
    expected = {(effort, temperature) for effort in efforts for temperature in temperatures}
    actual = [(r['reasoning_effort'], r['requested_temperature']) for r in result['results']]
    allowed = expected | ({('none', 1.5)} if 'none' not in efforts else set())
    if len(actual) != len(set(actual)) or not expected <= set(actual) <= allowed:
        raise ValueError('Existing probe results use a different grid; choose a new --output directory')


async def main():
    import httpx
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--temperatures', type=float, nargs='+', default=[0.5, 1.0, 1.5, 2.0])
    parser.add_argument('--reasoning-efforts', choices=['none', 'medium'], nargs='+', default=['medium'])
    args = parser.parse_args()
    if (len(set(args.temperatures)) != len(args.temperatures)
            or any(not math.isfinite(t) or t < 0 for t in args.temperatures)
            or len(set(args.reasoning_efforts)) != len(args.reasoning_efforts)):
        parser.error('Use unique, finite, nonnegative temperatures and unique reasoning settings')
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'probe_results.json').exists():
        result = json.loads((output / 'probe_results.json').read_text())
        validate_cached_grid(result, args.temperatures, args.reasoning_efforts)
        print(json.dumps(result, indent=2))
        return
    code_path = output / Path(__file__).name
    if code_path.exists() and file_sha(code_path) != file_sha(__file__):
        raise ValueError('Saved probe source differs')
    if not code_path.exists():
        code_path.write_bytes(Path(__file__).read_bytes())
    key = os.environ.get('AZURE_OPENAI_API_KEY') or getpass.getpass('Azure API key (hidden): ')
    if not key:
        raise ValueError('Missing credential')
    plan = {'created_at_utc': now(), 'model': 'gpt-5.6-sol', 'endpoint': ENDPOINT,
            'temperatures': args.temperatures, 'reasoning_efforts': args.reasoning_efforts,
            'top_p': 'omitted in every probe', 'max_output_tokens': 8192,
            'fixture': 'synthetic arithmetic transport probe, not a test-set observation',
            'fallback': 'Only if a medium request rejects temperature, probe T=1.5 with reasoning=none to identify an alternative configuration; do not collect a benchmark in that configuration automatically.',
            'source_sha256': file_sha(code_path)}
    if not (output / 'probe_plan.json').exists():
        atomic(output / 'probe_plan.json', plan)

    async def probe(client, temperature, effort):
        name = f'{effort}_t{str(temperature).replace(".", "p")}'
        path = output / (name + '.json')
        if path.exists():
            return json.loads(path.read_text())
        payload = {'model': 'gpt-5.6-sol', 'input': [{'role': 'user', 'content': 'What is 2 + 2? Return only the integer.'}],
                   'reasoning': {'effort': effort}, 'temperature': temperature,
                   'max_output_tokens': 8192, 'store': False}
        receipt = {'probe': name, 'request': payload, 'request_sha256': sha(payload), 'started_at_utc': now()}
        atomic(output / (name + '_request.json'), receipt)
        start = time.monotonic()
        try:
            response = await client.post(ENDPOINT, json=payload)
            try:
                body = response.json()
            except ValueError:
                body = {'non_json_response': response.text}
            receipt.update(http_status=response.status_code, response=body,
                           http_body_text=response.text,
                           headers={k: '[REDACTED]' if k.lower() in SENSITIVE else v for k, v in response.headers.items()})
        except (httpx.TimeoutException, httpx.TransportError) as error:
            receipt.update(http_status=None, error_type=type(error).__name__, error=str(error), possibly_billed=True)
        receipt.update(received_at_utc=now(), latency_seconds=time.monotonic() - start)
        receipt = json.loads(json.dumps(receipt).replace(key, '[REDACTED]'))
        atomic(path, receipt)
        return receipt

    async with httpx.AsyncClient(headers={'api-key': key}, timeout=httpx.Timeout(180, connect=30), follow_redirects=False) as client:
        receipts = await asyncio.gather(*(probe(client, temperature, effort)
            for effort in args.reasoning_efforts for temperature in args.temperatures))
        rejected_temperature = any(r.get('http_status') == 400 and 'temperature' in json.dumps(r.get('response', {})).lower() for r in receipts)
        if rejected_temperature and 'none' not in args.reasoning_efforts:
            receipts.append(await probe(client, 1.5, 'none'))
    result = {'completed_at_utc': now(), 'benchmark_calls': 0, 'capability_probes': len(receipts),
              'plan_sha256': file_sha(output / 'probe_plan.json'),
              'results': [{'probe': r['probe'], 'http_status': r.get('http_status'),
                           'requested_temperature': r['request']['temperature'],
                           'reasoning_effort': r['request']['reasoning']['effort'],
                           'returned_temperature': (r.get('response') or {}).get('temperature'),
                           'returned_top_p': (r.get('response') or {}).get('top_p'),
                           'returned_reasoning_effort': ((r.get('response') or {}).get('reasoning') or {}).get('effort'),
                           'response_status': (r.get('response') or {}).get('status'),
                           'error': (r.get('response') or {}).get('error'),
                           'receipt_sha256': file_sha(output / (r['probe'] + '.json'))} for r in receipts]}
    atomic(output / 'probe_results.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    asyncio.run(main())
