#!/usr/bin/env python3
"""Fold one recovery cell's request receipts into a single cost record.

A cell writes one receipt per request, which is what makes it resumable, but it
also means thousands of small files. They are read once here, per cell, and
summarised into ``costs.json`` so no later analysis has to touch them again.
Each receipt is checked against the hash the cell recorded for it.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'ops') not in sys.path:
    sys.path.insert(0, str(ROOT / 'ops'))

from followup_metrics import atomic_new, file_sha  # noqa: E402

BASE = ROOT / 'artifacts/modebench_recovery_five_domain_20260917'


def phase(uid):
    """Which part of the protocol a request belongs to, from its identifier."""
    if uid.startswith('dev_t'):
        return 'calibration'
    if uid.startswith('test_temp_'):
        return 'temperature'
    if uid.startswith('test_diverse_'):
        return 'diversity_prompt'
    if uid.startswith('recovery_'):
        return 'recovery/' + uid.split('_')[2]
    raise ValueError('unrecognised request identifier: ' + uid)


def aggregate(cell, rewrite):
    result = json.loads((cell / 'result.json').read_text())
    if result['status'] != 'complete':
        raise ValueError('cell is not complete: ' + str(cell))
    out = cell / 'costs.json'
    if out.exists():
        if not rewrite:
            return json.loads(out.read_text())
        out.unlink()
    totals = collections.defaultdict(lambda: {'requests': 0, 'responses': 0,
                                              'logical_input_tokens': 0, 'output_tokens': 0,
                                              'generation_wall_seconds': 0.0})
    seen = set()
    for receipt in result['request_receipts']:
        path = Path(receipt['path'])
        if path in seen:
            continue
        seen.add(path)
        if file_sha(path) != receipt['sha256']:
            raise ValueError('receipt changed since the cell completed: ' + str(path))
        record = json.loads(path.read_text())
        entry = totals[phase(record['request']['uid'])]
        entry['requests'] += 1
        entry['responses'] += len(record['samples'])
        entry['logical_input_tokens'] += record['logical_input_tokens']
        entry['output_tokens'] += sum(s['output_tokens'] for s in record['samples'])
        entry['generation_wall_seconds'] += record['generation_wall_seconds']
    payload = {'schema': 'modebench-recovery-costs-v1', 'identity': result['identity'],
               'checkpoint': result['checkpoint']['label'], 'domain': result['domain'],
               'arm': result['checkpoint']['arm'], 'seed': result['checkpoint']['seed'],
               'receipts': len(seen), 'phases': dict(totals),
               'note': 'The ordinary strategy generates nothing here: its portfolio is the '
                       'retained terminal draw block, so it has no phase of its own.'}
    atomic_new(out, payload)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rewrite', action='store_true')
    args = parser.parse_args()
    cells = sorted(p.parent for p in (BASE / 'results').glob('*/result.json'))
    done = []
    for cell in cells:
        payload = aggregate(cell, args.rewrite)
        done.append({'checkpoint': payload['checkpoint'], 'receipts': payload['receipts']})
        print(json.dumps(done[-1]), flush=True)
    print(json.dumps({'cells': len(done)}))


if __name__ == '__main__':
    main()
