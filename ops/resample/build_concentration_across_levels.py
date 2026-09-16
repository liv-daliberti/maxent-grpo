#!/usr/bin/env python3
"""Final minus initial \\pmd{} at every construction level, from the resampling.

The bound concentration analysis answers this question for Levels 1 and 2 only,
because it reads a training run's own saved outputs and runs exist only at those
levels. The independently seeded resampling measures the same checkpoints on
every level's held-out split, and the step-0 cohort measures the untrained
checkpoint through the same surface, so the difference is available all the way
up the ladder.

It is not the same quantity, and the caption has to say so. Levels 1 and 2 are a
policy trained and measured at one level. Levels 3 to 5 are the same Level-1 and
Level-2 policies read on a harder construction they never trained on, so the
difference there is transfer: what the training did to breadth, seen on problems
it did not see. Both are ``final minus initial'' on a fixed prompt set with a
fixed sampling surface, which is what makes them comparable at all.

Pairing is by seed where seeds exist. The untrained checkpoint has no training
seed -- there is one of it -- so every trained seed is differenced against that
single baseline rather than pretending to a paired design. The spread reported
is therefore the spread across training seeds, not a paired interval.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'var/artifacts/pmd_independent_resample_20260915'
DEFAULT_SOURCE = RUN / 'mode_diversity_resampled.json'
DEFAULT_OUTPUT = ROOT / 'paper/results/concentration_across_levels.json'
LEVELS = ('level1', 'level2', 'level3', 'level4', 'level5')
# Student-t 95% half-widths, two-sided, for n-1 degrees of freedom.
T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365}


def summarise(values: list[float]) -> dict:
    """Mean and a nominal interval over training seeds, or neither if too few."""
    n = len(values)
    out = {'n': n, 'mean': statistics.fmean(values) if values else None,
           'ci95': None, 'values': sorted(values)}
    if n >= 2:
        half = T95.get(n, 1.96) * statistics.stdev(values) / math.sqrt(n)
        out['ci95'] = [out['mean'] - half, out['mean'] + half]
    return out


def build(source: Path) -> dict:
    payload = json.loads(source.read_text())
    cells = payload['cells']
    baseline: dict[tuple[str, str, str], dict] = {}
    trained: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for cell in cells:
        key = (cell['level'], cell['domain'], cell['scale'])
        if cell['method'] == 'base':
            baseline[key] = cell
        else:
            trained[key + (cell['method'],)].append(cell)

    blocks = []
    for (level, domain, scale, method), group in sorted(trained.items()):
        base = baseline.get((level, domain, scale))
        block = {
            'kind': 'before_after', 'level': level, 'domain': domain,
            'scale': scale, 'method': method,
            # Every cross-level cell is a Level-1-trained policy read on another
            # level's split, so only Level 1 is trained-and-evaluated together.
            # Calling Level 2 same-level would misdescribe the whole ladder.
            'trained_level': 'level1',
            'transfer': level != 'level1',
        }
        if base is None:
            block.update({'status': 'no_untrained_baseline', 'summary': summarise([])})
            blocks.append(block)
            continue
        initial = base['resampled']
        # Both sides must define the measure. A checkpoint that never returns two
        # correct responses to one prompt has no breadth to compare against, and
        # a zero here would be a claim the data does not make.
        # PCMD is defined wherever a prompt has two correct responses; the
        # support bar is a precision threshold, not a definedness one. A cell
        # under the bar is therefore measured and marked provisional rather than
        # discarded -- that is the convention the base-grid figure already uses
        # -- and only a cell with no eligible prompt at all has nothing to say.
        usable = [c for c in group if c['resampled']['defined_prompts'] >= 2]
        deltas = [c['resampled']['pmd'] - initial['pmd'] for c in usable
                  if initial['defined_prompts'] >= 2]
        provisional = bool(deltas) and not (
            initial['reportable'] and all(c['resampled']['reportable'] for c in usable))
        block.update({
            'provisional': provisional,
            'status': 'measured' if deltas else (
                'initial_below_support' if initial['defined_prompts'] < 2
                else 'final_below_support'),
            'initial_pmd': initial['pmd'],
            'initial_defined_prompts': initial['defined_prompts'],
            'initial_reportable': initial['reportable'],
            'seeds': sorted(c['seed'] for c in usable),
            'initial_eligible_prompts': initial['defined_prompts'],
            'final_eligible_prompts': sorted(c['resampled']['defined_prompts'] for c in usable),
            'summary': summarise(deltas),
        })
        blocks.append(block)
    measured = [b for b in blocks if b['status'] == 'measured']
    return {
        'schema': 'paper-concentration-across-levels-v1',
        'status': 'analyzed',
        'source': {'path': str(source if not str(source).startswith(str(ROOT)) else source.relative_to(ROOT)),
                   'schema': payload.get('schema')},
        'builder': {'path': str(Path(__file__).resolve().relative_to(ROOT))},
        'definition': {
            'quantity': 'terminal PCMD minus untrained PCMD on the level\'s held-out split',
            'pairing': 'one untrained checkpoint per cell; spread is across training seeds',
            'trained_level': 'level1',
            'transfer_levels': ['level2', 'level3', 'level4', 'level5'],
            'transfer_note': 'Every cell trains at Level 1. Level 1 is trained and '
                             'evaluated together; Levels 2-5 read that same policy on '
                             'constructions it never trained on, so they are transfer.',
        },
        'coverage': {
            'blocks': len(blocks), 'measured': len(measured),
            'levels': sorted({b['level'] for b in blocks}),
            'domains': sorted({b['domain'] for b in blocks}),
            'scales': sorted({b['scale'] for b in blocks}),
        },
        'blocks': blocks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1, sort_keys=True) + '\n')
    print(json.dumps({'event': 'built', 'output': str(args.output),
                      **payload['coverage']}, sort_keys=True))


if __name__ == '__main__':
    main()
