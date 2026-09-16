#!/usr/bin/env python3
"""Report what the base grid currently says about scale and training recipe.

Means are taken over cells COMMON to every model being compared. Averaging each
model over whatever cells it happens to have finished mixes domains that differ
far more in PCMD than the models do, which can invent or hide a trend while the
grid is still filling in.
"""
from __future__ import annotations
import argparse
import collections
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ops'))
import evaluate_modebench_base_grid as grid

PAYLOAD = ROOT / 'paper/results/mode_diversity_base_grid.json'


def paired_means(index, models, keys):
    out = {}
    for model in models:
        cells = [index[(model,) + key] for key in keys]
        out[model] = {'pmd': sum(c['pmd'] for c in cells) / len(cells),
                      'pass8': sum(c['pass8'] for c in cells) / len(cells)}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--payload', type=Path, default=PAYLOAD)
    parser.add_argument('--min-common', type=int, default=3,
                        help='families with fewer shared cells are reported as too sparse')
    args = parser.parse_args()
    payload = json.loads(args.payload.read_text())
    cells = payload['cells']

    index = {(c['model_label'], c['domain'], c['level']): c for c in cells if c['reportable']}
    run = collections.Counter((c['model_label'], c['level']) for c in cells)

    print('== coverage (reportable / run)')
    levels = sorted({c['level'] for c in cells})
    header = ''.join(f'{lvl.replace("level", "L"):>12s}' for lvl in levels)
    print(f'{"model":10s} {"params":>7s}{header}')
    for model in sorted(grid.MODEL_PARAMS, key=lambda m: grid.MODEL_PARAMS[m]):
        row = f'{model:10s} {grid.MODEL_PARAMS[model]:6.2f}B'
        for level in levels:
            total = run[(model, level)]
            rep = sum(1 for c in cells
                      if c['model_label'] == model and c['level'] == level and c['reportable'])
            row += f'{rep:>7d}/{total:<4d}' if total else f'{"-":>12s}'
        print(row)

    families = collections.defaultdict(list)
    for model, family in grid.MODEL_FAMILY.items():
        families[family].append(model)

    print('\n== within-family scale trend (paired on cells shared by every scale)')
    for family in sorted(families):
        models = sorted((m for m in families[family] if any(k[0] == m for k in index)),
                        key=lambda m: grid.MODEL_PARAMS[m])
        if len(models) < 2:
            print(f'-- {family}: fewer than two scales measured'); continue
        common = set.intersection(*({k[1:] for k in index if k[0] == m} for m in models))
        if len(common) < args.min_common:
            print(f'-- {family}: only {len(common)} shared cells across {len(models)} scales '
                  f'(need {args.min_common}); trend not reported'); continue
        means = paired_means(index, models, sorted(common))
        print(f'-- {family}  ({len(common)} shared cells, {len(models)} scales)')
        for model in models:
            m = means[model]
            print(f'   {grid.MODEL_PARAMS[model]:6.2f}B {model:10s} '
                  f'PCMD={m["pmd"]:.3f}  pass8={m["pass8"]:.3f}')

    print('\n== cross-family at matched scale (paired on cells shared by the listed models)')
    for low, high, label in ((6.0, 8.0, '~7B'), (10.0, 15.0, '10-15B'), (0.1, 2.0, 'under 2B')):
        models = sorted((m for m, p in grid.MODEL_PARAMS.items()
                         if low <= p <= high and any(k[0] == m for k in index)),
                        key=lambda m: grid.MODEL_PARAMS[m])
        if len(models) < 2:
            continue
        common = set.intersection(*({k[1:] for k in index if k[0] == m} for m in models))
        if len(common) < args.min_common:
            print(f'-- {label}: only {len(common)} shared cells across {len(models)} models; '
                  'not reported')
            continue
        means = paired_means(index, models, sorted(common))
        print(f'-- {label}  ({len(common)} shared cells)')
        for model in models:
            m = means[model]
            print(f'   {grid.MODEL_PARAMS[model]:6.2f}B {model:10s} {grid.MODEL_FAMILY[model]:9s} '
                  f'PCMD={m["pmd"]:.3f}  pass8={m["pass8"]:.3f}')


if __name__ == '__main__':
    main()
