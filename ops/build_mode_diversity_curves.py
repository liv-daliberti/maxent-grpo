#!/usr/bin/env python3
"""Build PCMD training curves from the extracted verified-key container.

Consumes the compact archive written by ``extract_mode_diversity_curves.py``
and reports pairwise correct-mode diversity at every saved checkpoint, so the
manuscript's training curves can run on a breadth axis that does not move with
correctness. Each prompt pools its four draws at a step; a prompt contributes
only where it returns at least two verified responses, and a curve point is
suppressed when too few prompts do.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'ops') not in sys.path:
    sys.path.insert(0, str(ROOT / 'ops'))

from followup_metrics import atomic_new, file_sha  # noqa: E402
from mode_diversity import DEFAULT_MIN_DEFINED_PROMPTS, mode_diversity  # noqa: E402

SCHEMA = 'paper-mode-diversity-curves-v1'
ARCHIVE = ROOT / 'var/artifacts/mode_diversity_curves/verified_keys_by_step.jsonl.gz'
OUT = ROOT / 'paper/results/mode_diversity_curves.json'
CELL_FIELDS = ('level', 'scale', 'domain', 'method', 'seed')


def build(archive: Path = ARCHIVE, min_defined: int = DEFAULT_MIN_DEFINED_PROMPTS) -> dict:
    # cell -> step -> prompt -> pooled verified-mode counts
    cells: dict[tuple, dict[int, dict[str, Counter]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(Counter)))
    seen: set[tuple] = set()
    duplicates = 0
    with gzip.open(archive, 'rt') as handle:
        for line in handle:
            record = json.loads(line)
            if record.get('record_kind') != 'source':
                continue
            cell = tuple(record[k] for k in CELL_FIELDS)
            for draw in record['draws']:
                stamp = (cell, draw['step'], draw['draw_index'])
                if stamp in seen:
                    # The same draw can be reachable through more than one
                    # registered source file; count it once.
                    duplicates += 1
                    continue
                seen.add(stamp)
                by_prompt = cells[cell][draw['step']]
                for prompt, keys in draw['prompts'].items():
                    by_prompt[prompt].update(k for k in keys if k is not None)

    curves = []
    for cell, steps in sorted(cells.items()):
        points = []
        for step in sorted(steps):
            values = [v for v in (mode_diversity(c) for c in steps[step].values()) if v is not None]
            prompts = len(steps[step])
            points.append({
                'step': step,
                'pmd': statistics.fmean(values) if values else None,
                'defined_prompts': len(values),
                'prompts': prompts,
                'reportable': len(values) >= min_defined,
            })
        curves.append(dict(zip(CELL_FIELDS, cell)) | {'points': points})

    return {
        'schema': SCHEMA,
        'archive': {'path': str(archive.relative_to(ROOT)), 'sha256': file_sha(archive)},
        'builder': {'path': 'ops/build_mode_diversity_curves.py',
                    'sha256': file_sha(Path(__file__).resolve())},
        'definition': {'metric': 'pairwise correct-mode diversity (PCMD)',
                       'min_defined_prompts': min_defined,
                       'aggregation': "pooled over a step's four draws, unweighted mean over defined prompts"},
        'curves': curves,
        'coverage': {'cells': len(curves), 'duplicate_draws_skipped': duplicates,
                     'steps': sorted({p['step'] for c in curves for p in c['points']})},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=ARCHIVE)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--min-defined', type=int, default=DEFAULT_MIN_DEFINED_PROMPTS)
    args = parser.parse_args()
    payload = build(args.archive, args.min_defined)
    atomic_new(args.output, payload)
    cov = payload['coverage']
    print(json.dumps({'event': 'built', 'output': str(args.output),
                      'cells': cov['cells'], 'steps': len(cov['steps']),
                      'duplicate_draws_skipped': cov['duplicate_draws_skipped']}))


if __name__ == '__main__':
    main()
