#!/usr/bin/env python3
"""Correct the eight Level-1 E124 cells that can never start.

``validate_zero_math_args`` treats a named ModeBench domain as a declaration that
the Level-2 prompt/syntax contract applies. Level 1 is the native interface by
design -- ``interface()`` documents it as such and always sets the syntax profile
to ``none`` -- yet it named a domain anyway, so eight cells were rejected at
startup with "Level-2 prompt/syntax contract mismatch" and requeued by the
watchdog without ever training. Level-1 Graph Coloring survived only because its
Level-2 contract happens to be the Level-1 pair ``('qwen_boxed', 'none')``.

The correction is scientifically inert. ``modebench_domain`` is read only by
``guided_sampling_params``, which returns its input unchanged whenever the profile
is ``none``; at Level 1 the profile is always ``none``. Training, evaluation and
sampling are identical before and after.

Graph Coloring is deliberately left untouched: its contract is already satisfied
and one of its cells is training.

Run with --apply. Afterwards the eight jobs must be resubmitted, because their
recorded environments no longer match the plan.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / 'var/artifacts/e124_qwen7b_three_level'


def seal(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()

    plan = json.loads((ART / 'plan.json').read_text())
    broken = [c for c in plan['cells']
              if c['level'] == 1
              and c['environment'].get('OAT_ZERO_MODEBENCH_DOMAIN', 'none') != 'none'
              and c['domain'] != 'graph_coloring']
    print(json.dumps({'cells_to_correct': [c['cell_id'] for c in broken]}, indent=2))
    if not args.apply:
        print('\ndry run; pass --apply')
        return 0
    if not broken:
        print('nothing to correct')
        return 0

    for name in ('plan.json', 'systems/plan.json'):
        source = ART / name
        backup = source.with_suffix('.pre_level1_correction.json')
        if not backup.exists():
            shutil.copy2(source, backup)

    for cell in broken:
        cell['environment']['OAT_ZERO_MODEBENCH_DOMAIN'] = 'none'
        cell['environment_sha256'] = seal(cell['environment'])
    plan['level1_domain_correction'] = {
        'schema': 'e124_level1_modebench_domain_correction_v1',
        'at': datetime.now(timezone.utc).isoformat(),
        'cells': [c['cell_id'] for c in broken],
        'change': "OAT_ZERO_MODEBENCH_DOMAIN set to 'none' on Level-1 cells that named a domain",
        'why': ('Level 1 runs no guided syntax, but naming a domain makes '
                'validate_zero_math_args demand the Level-2 prompt/syntax contract, which Level 1 '
                'does not satisfy. These cells failed at startup and never trained.'),
        'scientifically_inert': ('modebench_domain is consumed only by guided_sampling_params, which '
                                 'returns its input unchanged when the syntax profile is "none". '
                                 'Level 1 always has profile "none".'),
        'untouched': 'Level-1 Graph Coloring, already contract-satisfying and training',
    }
    previous = plan.pop('plan_sha256')
    plan['plan_sha256'] = seal({k: v for k, v in plan.items() if k != 'plan_sha256'})
    (ART / 'plan.json').write_text(json.dumps(plan, indent=2, sort_keys=True) + '\n')

    bench = json.loads((ART / 'systems/plan.json').read_text())
    bench.pop('plan_sha256')
    bench['manifest_sha256'] = digest(ART / 'plan.json')
    bench['plan_sha256'] = seal(bench)
    (ART / 'systems/plan.json').write_text(json.dumps(bench, indent=2, sort_keys=True) + '\n')

    tx_path = ART / 'transaction.json'
    tx = json.loads(tx_path.read_text())
    tx['plan_sha256'] = plan['plan_sha256']
    tx['auxiliary_pins'][str((ART / 'systems/plan.json').resolve())] = digest(ART / 'systems/plan.json')
    for cell in broken:
        row = tx['rows'].get(cell['cell_id'])
        if row:
            row['status'] = 'needs_resubmission'
            row['correction'] = 'level1 modebench domain cleared; recorded job carries the rejected environment'
    tx_path.write_text(json.dumps(tx, indent=2, sort_keys=True) + '\n')

    print(json.dumps({'corrected': [c['cell_id'] for c in broken],
                      'plan_sha256': f'{previous[:12]} -> {plan["plan_sha256"][:12]}',
                      'next': 'cancel and resubmit those eight jobs, then release them'}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
