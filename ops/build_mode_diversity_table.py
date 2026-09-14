#!/usr/bin/env python3
"""Emit the appendix table body pairing the registered endpoints with PMD.

One row per domain and level, one column group per model scale. Each group
reports \\texttt{pass@8} and \\pmd{} so a reader can see directly that the two
do not track each other, and so the registered endpoint stays visible next to
the statistic that carries the breadth axis. Cells without enough verified
pairs print an em dash rather than a number.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAYLOAD = ROOT / 'paper/results/mode_diversity_base_grid.json'
OUT = ROOT / 'paper/results/mode_diversity_base_grid_table_body.tex'

MODELS = ('05b', '3b', '7b', '14b')
DOMAINS = ('graph_coloring', 'countdown', 'python_factors', 'mathir', 'pantry')
DOMAIN_LABELS = {'graph_coloring': 'Graph', 'countdown': 'Countdown',
                 'python_factors': 'Python', 'mathir': 'MathIR', 'pantry': 'Pantry'}
LEVELS = ('level1', 'level2', 'level3')


def fmt(value: float | None, *, reportable: bool = True) -> str:
    if value is None or not reportable:
        return '---'
    text = f'{value:.3f}'
    return text[1:] if text.startswith('0.') else text


def build(payload: dict) -> str:
    index = {(c['model_label'], c['level'], c['domain']): c for c in payload['cells']}
    lines = []
    for domain in DOMAINS:
        for position, level in enumerate(LEVELS):
            label = DOMAIN_LABELS[domain] if position == 0 else ''
            cells = [f'{label}', f'{position + 1}']
            for model in MODELS:
                cell = index.get((model, level, domain))
                if cell is None:
                    cells += ['---', '---']
                    continue
                cells.append(fmt(cell['pass8']))
                cells.append(fmt(cell['pmd'], reportable=cell['reportable']))
            lines.append(' & '.join(cells) + r' \\')
        if domain != DOMAINS[-1]:
            lines.append(r'\addlinespace')
    # The rule closes the body file rather than the manuscript: a body ending
    # in a bare row separator leaves \bottomrule stranded outside the \cr.
    lines.append(r'    \bottomrule')
    return '\n'.join(lines) + '\n'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--payload', type=Path, default=PAYLOAD)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    payload = json.loads(args.payload.read_text())
    args.output.write_text(build(payload))
    gaps = sum(1 for c in payload['cells'] if not c['reportable'])
    print(json.dumps({'event': 'built', 'output': str(args.output),
                      'rows': len(DOMAINS) * len(LEVELS), 'gap_cells': gaps}))


if __name__ == '__main__':
    main()
