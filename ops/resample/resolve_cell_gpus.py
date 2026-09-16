#!/usr/bin/env python3
"""Fill in each cohort cell's training GPU model from the launch ledgers.

A decoding measurement is only comparable to the one it is being checked
against when it runs on the same GPU model, so every cell must carry one. The
curated ledgers record ``gpu`` for the older experiments but not for E118, whose
per-scale sub-ledgers instead preserve the raw scheduler record; the requested
generic resource there names the model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / 'var/artifacts/pmd_independent_resample_20260915/cohort_manifest.json'
LEDGER_DIRS = (ROOT / 'paper/audits/training_curves_20260912/ledgers', ROOT / 'var/artifacts')
GRES = re.compile(r'gres/gpu:([a-z][a-z0-9]*)=')


def ledger_payloads():
    for directory in LEDGER_DIRS:
        for path in sorted(directory.glob('*jobs.json')):
            try:
                yield path, json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                continue


def gpu_map() -> dict[str, tuple[str, str]]:
    """Map run_dir to (gpu model, the ledger that named it)."""
    resolved: dict[str, tuple[str, str]] = {}
    for path, payload in ledger_payloads():
        runs = payload.get('runs')
        if not isinstance(runs, list):
            continue
        for run in runs:
            run_dir = run.get('run_dir')
            if not run_dir:
                continue
            gpu = run.get('gpu')
            if not gpu:
                hit = GRES.search(str(run.get('held_scheduler_record', '')))
                gpu = hit.group(1) if hit else None
            if not gpu:
                continue
            previous = resolved.get(str(run_dir))
            if previous and previous[0] != gpu:
                raise SystemExit(f'ledgers disagree on the GPU for {run_dir}: '
                                 f'{previous[0]} ({previous[1]}) vs {gpu} ({path.name})')
            resolved[str(run_dir)] = (gpu, path.name)
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    resolved = gpu_map()
    missing = []
    for cell in manifest['cells']:
        hit = resolved.get(cell['run_dir'])
        if hit is None:
            if not cell.get('gpu'):
                missing.append(cell['run_dir'])
            continue
        gpu, source = hit
        if cell.get('gpu') and cell['gpu'] != gpu:
            raise SystemExit(f'{cell["run_dir"]}: manifest says {cell["gpu"]}, {source} says {gpu}')
        cell['gpu'] = gpu
        cell['gpu_source_ledger'] = source
    if missing:
        raise SystemExit(f'{len(missing)} cells have no ledger GPU; first: {missing[0]}')
    counts: dict[str, int] = {}
    for cell in manifest['cells']:
        counts[cell['gpu']] = counts.get(cell['gpu'], 0) + 1
    manifest['cells_by_gpu'] = dict(sorted(counts.items()))
    with args.manifest.open('w') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write('\n')
    print(json.dumps({'cells': len(manifest['cells']), 'by_gpu': manifest['cells_by_gpu']}, indent=2))


if __name__ == '__main__':
    main()
