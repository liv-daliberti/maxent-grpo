#!/usr/bin/env python3
"""Emit the hosted PCMD table body: one row per level, five domain columns."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAYLOAD = ROOT / 'paper/results/mode_diversity_hosted.json'
OUT = ROOT / 'paper/results/mode_diversity_hosted_table_body.tex'
DOMAINS = ('graph_coloring', 'countdown', 'python_factors', 'mathir', 'pantry_plan')


def fmt(value: float | None) -> str:
    if value is None:
        return '---'
    text = f'{value:.3f}'
    return text[1:] if text.startswith('0.') else text


def build(payload: dict) -> str:
    index = {(c['level'], c['domain']): c for c in payload['cells']}
    lines = []
    for level in (1, 2, 3):
        cells = [str(level)]
        for domain in DOMAINS:
            cell = index.get((level, domain))
            if cell is None or not cell['reportable']:
                cells.append('---')
            else:
                cells.append(f"{fmt(cell['pmd'])} ({cell['effective_modes']:.2f})")
        lines.append(' & '.join(cells) + r' \\')
    lines.append(r'    \bottomrule')
    return '\n'.join(lines) + '\n'


def build_cohort(payload: dict) -> str:
    """One row per deployment: macro PCMD, effective modes, and per-level macro."""
    lines = []
    for model in sorted(payload['models'], key=lambda m: -(m['macro_pmd'] or -1)):
        macro = model['macro_pmd']
        neff = 1.0 / (1.0 - macro) if macro is not None and macro < 1 else None
        cells = [model['label'], fmt(macro), f'{neff:.2f}' if neff else '---']
        for level in (1, 2, 3):
            level_cells = [c['pmd'] for c in model['cells']
                           if c['level'] == level and c['reportable']]
            cells.append(fmt(sum(level_cells) / len(level_cells)) if level_cells else '---')
        lines.append(' & '.join(cells) + r' \\')
    lines.append(r'    \bottomrule')
    return '\n'.join(lines) + '\n'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--payload', type=Path, default=PAYLOAD)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--cohort-payload', type=Path,
                        default=ROOT / 'paper/results/mode_diversity_hosted_cohort.json')
    parser.add_argument('--cohort-output', type=Path,
                        default=ROOT / 'paper/results/mode_diversity_hosted_cohort_table_body.tex')
    args = parser.parse_args()
    args.output.write_text(build(json.loads(args.payload.read_text())))
    args.cohort_output.write_text(build_cohort(json.loads(args.cohort_payload.read_text())))
    print(json.dumps({'event': 'built', 'output': str(args.output),
                      'cohort_output': str(args.cohort_output)}))


if __name__ == '__main__':
    main()
