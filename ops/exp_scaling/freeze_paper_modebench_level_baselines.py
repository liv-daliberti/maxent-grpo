#!/usr/bin/env python3
"""Freeze the untrained Level-1 and Level-2 checkpoints the level figure starts from.

The frozen Level1/Level2 comparison snapshot decides which cells are registered
and where they wrote; this reads the untrained step of those same registered run
directories. It is a separate artifact because the comparison snapshot is bound
by several analyses that have nothing to do with a baseline, and because the
verified-sample archive that supplies Level-1 and Level-2 PCMD elsewhere admits
its own census, which does not cover every untrained checkpoint on disk.

A cell whose untrained checkpoint cannot be read is a gap here and nothing else:
it never removes that cell's endpoint, which is a different measurement.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for directory in ('ops', 'ops/exp_scaling'):
    if str(ROOT / directory) not in sys.path:
        sys.path.insert(0, str(ROOT / directory))

from build_paper_modebench_level_comparison import (  # noqa: E402
    DOMAINS, LEVELS, METHODS, METRICS, SEEDS,
)
from mode_diversity import DEFAULT_MIN_DEFINED_PROMPTS  # noqa: E402
from modebench_checkpoint_pmd import checkpoint_pmd, definition  # noqa: E402
from snapshot_evaluation_coverage import read_cell  # noqa: E402

SCHEMA = 'modebench-level-baseline-frozen-snapshot-v1'
BASELINE_STEP = 0
SNAPSHOT = ROOT / 'paper/results/modebench_level_comparison_snapshot.json'
OUTPUT = ROOT / 'paper/results/modebench_level_baseline_snapshot.json'


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def registered_cells(snapshot: dict) -> list[dict]:
    cells = {}
    for cell in snapshot['availability']:
        key = (cell['level'], cell['domain'], cell['method'], int(cell['seed']))
        if key in cells:
            raise RuntimeError(f'duplicate registered comparison cell: {key}')
        cells[key] = {'level': key[0], 'domain': key[1], 'method': key[2], 'seed': key[3],
                      'run_dir': cell['run_dir']}
    required = {(level, domain, method, seed) for level in LEVELS for domain in DOMAINS
                for method in METHODS for seed in SEEDS}
    if set(cells) != required:
        raise RuntimeError('the comparison snapshot does not enumerate the 200 registered cells')
    return [cells[key] for key in sorted(cells)]


def draw_rows(cell: dict, checkpoint: dict) -> list[dict]:
    if checkpoint['draw_count'] != 4:
        raise RuntimeError('admitted untrained checkpoint lacks four draws')
    rows = []
    for draw in checkpoint['draws']:
        meta = draw['metadata']
        if (meta['prompt_count'], meta['sample_count'], meta['temperature']) != (128, 8, 1):
            raise RuntimeError('untrained draw violates the fixed 128-prompt K8 temperature1 contract')
        rows.append({
            'level': cell['level'], 'domain': cell['domain'], 'method': cell['method'],
            'seed': cell['seed'], 'step': BASELINE_STEP, 'draw_index': draw['draw_index'],
            'evaluation_kind': meta['evaluation_kind'], 'sample_count': meta['sample_count'],
            'metrics': draw['metrics'], 'evaluation_metadata': meta, 'origins': draw['origins'],
        })
    return rows


def compose(cells: list[dict], coverage: list[dict], min_defined: int, sources: dict) -> dict:
    admission, evaluations, pmd_cells = [], [], []
    for cell, read in zip(cells, coverage):
        checkpoint = read['complete_checkpoints'].get(str(BASELINE_STEP))
        record = {field: cell[field] for field in ('level', 'domain', 'method', 'seed')}
        record.update(admitted=checkpoint is not None, endpoint=None)
        if checkpoint is not None:
            rows = draw_rows(cell, checkpoint)
            record['endpoint'] = {metric: sum(row['metrics'][field] for row in rows) / 4
                                  for metric, field in METRICS.items()}
            evaluations.extend(rows)
            pmd_cells.append({**{field: cell[field] for field in
                                 ('level', 'domain', 'method', 'seed')},
                              **checkpoint_pmd(checkpoint, min_defined)})
        admission.append(record)
    return {
        'schema': SCHEMA,
        'levels': list(LEVELS),
        'model': 'Qwen2.5-0.5B-Instruct',
        'baseline_step': BASELINE_STEP,
        'collected_at_utc': now(),
        'admission_rule': (
            'A cell contributes an untrained point when its registered run directory holds '
            'one complete, unconflicted four-draw fixed-seed sampled K=8 evaluation at step 0. '
            'The untrained step is admitted separately from the endpoint, so a cell that '
            'cannot be measured before training keeps the measurement made after it.'
        ),
        'pmd_definition': definition(min_defined, digest),
        'sources': sources,
        'baseline_admission': admission,
        'baseline_evaluations': evaluations,
        'baseline_pmd_cells': pmd_cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, default=SNAPSHOT)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--min-defined', type=int, default=DEFAULT_MIN_DEFINED_PROMPTS)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error('--workers must be between 1 and 8')
    snapshot = json.loads(args.snapshot.read_text())
    if snapshot.get('schema') != 'modebench-level-comparison-frozen-snapshot-v1':
        raise RuntimeError('wrong frozen Level1/Level2 comparison snapshot')
    cells = registered_cells(snapshot)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        coverage = list(pool.map(lambda cell: read_cell(cell['run_dir']), cells))
    sources = {str(args.snapshot.relative_to(ROOT)): digest(args.snapshot.read_bytes())}
    for path in (Path(__file__).resolve(), ROOT / 'ops/exp_scaling/snapshot_evaluation_coverage.py'):
        sources[str(path.relative_to(ROOT))] = digest(path.read_bytes())
    record = compose(cells, coverage, args.min_defined, sources)
    write_json(args.output, record)
    admitted = sum(row['admitted'] for row in record['baseline_admission'])
    print(json.dumps({'event': 'frozen', 'output': str(args.output),
                      'registered_cells': len(cells), 'admitted_baseline_cells': admitted}))


if __name__ == '__main__':
    main()
