#!/usr/bin/env python3
"""Extract (pass@8, PMD) for each hosted deployment cell, for the levels figure.

The hosted comparison already publishes correct-pair collision per
model/level/domain; PMD is one minus that, on the same success-conditional
definition the local grid uses. Frontier models have no parameter count, so
they are emitted separately rather than being placed on the scale ramp.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'paper/results/frontier_comparison_20260911.json'
OUT = ROOT / 'paper/results/mode_diversity_frontier_points.json'
# the hosted comparison spells PantryPlan with an underscore
DOMAIN = {'graph_coloring': 'graph_coloring', 'countdown': 'countdown',
          'python_factors': 'python_factors', 'mathir': 'mathir',
          'pantry_plan': 'pantry', 'pantry': 'pantry'}
MIN_DEFINED = 30


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    raw = json.loads(args.source.read_text())

    cells = []
    for model in raw['models']:
        label = model.get('label') or model.get('model')
        for key, cell in model['cells'].items():
            metrics = cell.get('metrics', {})
            collision = (metrics.get('correct_pair_collision') or {}).get('estimate')
            pass8 = (metrics.get('pass8') or {}).get('estimate')
            if collision is None or pass8 is None:
                continue
            counts = cell.get('counts', {})
            pairs = counts.get('correct_pairs')
            cells.append({
                'model': label,
                'level': f"level{cell['level']}",
                'domain': DOMAIN.get(cell['domain'], cell['domain']),
                'pass8': pass8,
                'pmd': 1.0 - collision,
                'correct_pairs': pairs,
                # a hosted cell is reportable when enough prompts yielded a
                # verified pair, matching the local grid's support rule
                'reportable': bool(pairs and pairs >= MIN_DEFINED),
            })
    payload = {
        'schema': 'mode-diversity-frontier-points-v1',
        'definition': {'pmd': 'one minus correct-pair collision',
                       'min_correct_pairs': MIN_DEFINED,
                       'note': 'inference-only deployments; no parameter count'},
        'source': {'path': str(args.source.relative_to(ROOT)),
                   'sha256': hashlib.sha256(args.source.read_bytes()).hexdigest()},
        'models': sorted({c['model'] for c in cells}),
        'cells': sorted(cells, key=lambda c: (c['model'], c['level'], c['domain'])),
    }
    args.output.write_text(json.dumps(payload, indent=1, sort_keys=True) + '\n')
    rep = sum(1 for c in cells if c['reportable'])
    print(json.dumps({'event': 'built', 'models': len(payload['models']),
                      'cells': len(cells), 'reportable': rep}))


if __name__ == '__main__':
    main()
