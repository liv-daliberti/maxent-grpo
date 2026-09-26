#!/usr/bin/env python3
"""Raise the draw budget on cells whose trained policy starves the measure.

\\pmd{} needs two *correct* responses to one prompt before it is defined. A
policy that has concentrated hard returns few correct responses per prompt, so
the same budget that defines the measure comfortably on the untrained
checkpoint can leave the trained one with almost no eligible prompt -- not
because breadth is unmeasurable there, but because the sample is too small to
catch the successes that remain.

That asymmetry is an artifact of the budget, not of the policy: the untrained
side of the difference is measured at the step-0 budget while the trained side
sits at the cohort default. Raising the trained side to match removes it. The
estimator is unbiased at every K, so the raise changes the precision of the
number and not the quantity it estimates.

Only ``eval_mode_coverage_draws`` moves. The prompt set, template, context,
seed policy, action support and checkpoint are the run's own, and the cell
records the raise so a reader can see which cells were measured this way.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REASON = 'match the step-0 budget on the same cell'


def select(cells: list[dict], scale: str, domain: str, method: str,
           level: str) -> list[int]:
    """Indices of the cells this raise applies to, in manifest order."""
    return [i for i, c in enumerate(cells)
            if c.get('scale') == scale and c.get('domain') == domain
            and c.get('method') == method and c.get('level') == level]


def raise_cell(cell: dict, draws: int) -> str:
    """Apply the raise, or say why the cell was left alone."""
    config = cell['eval_config']
    current = int(config['eval_mode_coverage_draws'])
    if current == draws:
        return 'already at budget'
    if current > draws:
        return f'already above: {current} draws'
    cell['budget_raised'] = {'from_draws': current, 'to_draws': draws,
                             'reason': REASON}
    config['eval_mode_coverage_draws'] = draws
    return f'{current} -> {draws}'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--scale', required=True)
    parser.add_argument('--domain', required=True)
    parser.add_argument('--method', required=True)
    parser.add_argument('--level', default='level1')
    parser.add_argument('--draws', type=int, default=128)
    parser.add_argument('--apply', action='store_true',
                        help='write the manifest; default reports only')
    args = parser.parse_args()

    payload = json.loads(args.manifest.read_text())
    cells = payload['cells']
    indices = select(cells, args.scale, args.domain, args.method, args.level)
    if not indices:
        raise SystemExit(f'{args.manifest.name}: no cell matches that selection')
    for i in indices:
        print(f'  cell {i} s{cells[i]["seed"]}: {raise_cell(cells[i], args.draws)}')
    if args.apply:
        args.manifest.write_text(json.dumps(payload, indent=1, sort_keys=True) + '\n')
    contiguous = indices == list(range(min(indices), max(indices) + 1))
    print(json.dumps({'event': 'raise', 'applied': args.apply,
                      'manifest': args.manifest.name, 'draws': args.draws,
                      'cells': indices,
                      'array': (f'{min(indices)}-{max(indices)}' if contiguous
                                else ','.join(str(i) for i in indices))},
                     sort_keys=True))


if __name__ == '__main__':
    main()
