#!/usr/bin/env python3
"""Emit a figure-source manifest for the completed multi-family grid cells.

Re-runnable while collection is in flight: it lists whatever receipts exist
and reports how many of the planned cells are still outstanding, so the PCMD
payload can be rebuilt from a partial grid without pretending it is complete.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COLLECTION = ROOT / 'artifacts/modebench_base_level_grid_multifamily_20260914'
OUT = COLLECTION / 'figure_source.json'


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _one(values: set[str]) -> str:
    known = sorted(v for v in values if v)
    if len(known) != 1:
        raise ValueError(f'receipts disagree on the evaluator: {known}')
    return known[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection', type=Path, default=COLLECTION)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()

    plan = json.loads((args.collection / 'plan.json').read_text())
    receipts, missing = [], []
    evaluators: set[str] = set()
    for cell in plan['cells']:
        path = Path(cell['output'])
        if not path.is_file():
            missing.append(cell['cell_id']); continue
        receipt = json.loads(path.read_text())
        if receipt.get('status') != 'complete':
            missing.append(cell['cell_id']); continue
        code = receipt.get('identity', {}).get('code_sha256') or {}
        if isinstance(code, dict):
            code = code.get('ops/evaluate_modebench_base_grid.py', '')
        evaluators.add(code)
        receipts.append({'domain': cell['domain'], 'level': cell['level'],
                         'model_label': cell['model_label'],
                         'path': str(path.relative_to(ROOT)), 'sha256': file_sha(path)})
    receipts.sort(key=lambda r: (r['model_label'], r['level'], r['domain']))
    args.output.write_text(json.dumps({
        'schema': 'modebench-multifamily-figure-source-v1',
        'collection': str(args.collection.relative_to(ROOT)),
        # Later collections record the evaluator per receipt rather than in the
        # plan; take it from the receipts, which is the run's own provenance.
        'evaluator_sha256': plan.get('evaluator_sha256') or _one(evaluators),
        'planned_cells': len(plan['cells']),
        'complete_cells': len(receipts),
        'outstanding_cells': sorted(missing),
        'receipts': receipts,
    }, indent=1, sort_keys=True) + '\n')
    print(json.dumps({'event': 'manifest', 'complete': len(receipts),
                      'planned': len(plan['cells']), 'outstanding': len(missing)}))


if __name__ == '__main__':
    main()
