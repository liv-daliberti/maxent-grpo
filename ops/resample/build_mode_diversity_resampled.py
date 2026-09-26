#!/usr/bin/env python3
"""Recompute terminal PCMD from the independently seeded resampling.

This is the same estimator the manuscript reports -- pool a prompt's draws,
take ``1 - sum n_m(n_m-1)/(K(K-1))`` over verified responses, average over
prompts that admit it -- applied to responses whose thirty-two sampling streams
are disjoint by construction rather than eleven streams counted as thirty-two.

Each arm is summarised over its seeds, and every cell is reported beside the
registered value it replaces, so the question a reader actually has -- does the
breadth claim survive independent sampling? -- is answered by a difference they
can see rather than by an assurance.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / 'ops', ROOT / 'ops' / 'exp_scaling'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mode_diversity import (  # noqa: E402
    DEFAULT_MIN_DEFINED_PROMPTS, effective_modes, mode_diversity,
)

SCHEMA = 'paper-mode-diversity-resampled-terminal-v1'
RUN = ROOT / 'var/artifacts/pmd_independent_resample_20260915'
REGISTERED = ROOT / 'paper/results/mode_diversity_training.json'
OUT = ROOT / 'paper/results/mode_diversity_terminal_resampled.json'


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def cell_metrics(receipt: dict, min_defined: int) -> dict:
    """PCMD, pass@8 and distinct@8 for one cell, pooling each prompt's draws."""
    pooled: dict[int, Counter] = defaultdict(Counter)
    pass8: list[float] = []
    distinct8: list[float] = []
    for draw in receipt['draws']:
        for prompt in draw['prompts']:
            verified = [key for key, reward in zip(prompt['answer_keys'], prompt['rewards'])
                        if key is not None and float(reward) > 0]
            pooled[int(prompt['prompt_index'])].update(verified)
            pass8.append(1.0 if verified else 0.0)
            distinct8.append(float(len(set(verified))))
    diversities = [value for value in
                   (mode_diversity(counts) for counts in pooled.values())
                   if value is not None]
    prompts = len(pooled)
    return {
        'pmd': statistics.fmean(diversities) if diversities else None,
        'effective_modes': effective_modes(statistics.fmean(diversities)) if diversities else None,
        'defined_prompts': len(diversities),
        'prompts': prompts,
        'support': len(diversities) / prompts if prompts else 0.0,
        'reportable': len(diversities) >= min_defined,
        'pass8': statistics.fmean(pass8) if pass8 else None,
        'distinct8': statistics.fmean(distinct8) if distinct8 else None,
    }


def registered_index() -> dict[tuple, dict]:
    if not REGISTERED.is_file():
        return {}
    payload = json.loads(REGISTERED.read_text())
    index = {}
    for row in payload['seeds']:
        key = (row['scale'], row['level'], row['domain'], row['method'], int(row['seed']))
        index[key] = row.get('after')
    return index


def build(receipts_dir: Path, min_defined: int) -> dict:
    registered = registered_index()
    cells = []
    for path in sorted(receipts_dir.glob('*.json.gz')):
        with gzip.open(path, 'rt') as handle:
            receipt = json.loads(handle.read())
        if receipt['mode'] != 'independent':
            continue
        identity = receipt['cell']
        plan = receipt['seed_plan']
        k = int(receipt['eval_config']['eval_mode_coverage_k'])
        draws = int(receipt['eval_config']['eval_mode_coverage_draws'])
        if plan['distinct_child_streams_per_prompt'] != k * draws:
            raise SystemExit(
                f'{path.name}: {plan["distinct_child_streams_per_prompt"]} distinct streams '
                f'per prompt, expected {k * draws}; this receipt does not carry the '
                'independent sampling it claims')
        metrics = cell_metrics(receipt, min_defined)
        key = (identity['scale'], identity['level'], identity['domain'],
               identity['method'], int(identity['seed']))
        before = registered.get(key)
        cells.append({
            **{field: identity[field] for field in
               ('scale', 'level', 'domain', 'method', 'seed', 'run_dir', 'gpu')},
            'receipt': str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
            'receipt_sha256': file_sha(path),
            'resampled': metrics,
            'registered': before,
            'pmd_shift': (None if not before or before.get('pmd') is None
                          or metrics['pmd'] is None
                          else metrics['pmd'] - before['pmd']),
        })
    arms: dict[tuple, list] = defaultdict(list)
    for cell in cells:
        arms[(cell['scale'], cell['level'], cell['domain'], cell['method'])].append(cell)
    arm_rows = []
    for (scale, level, domain, method), members in sorted(arms.items()):
        usable = [c for c in members if c['resampled']['reportable']]
        shifts = [c['pmd_shift'] for c in members if c['pmd_shift'] is not None]
        arm_rows.append({
            'scale': scale, 'level': level, 'domain': domain, 'method': method,
            'seeds': len(members), 'reportable_seeds': len(usable),
            'pmd_resampled': (statistics.fmean(c['resampled']['pmd'] for c in usable)
                              if usable else None),
            # A registered entry can exist and still carry no PCMD: the training
            # run measured that cell on a budget too small to leave two correct
            # responses on any prompt. Averaging over it would read a missing
            # measurement as a value, so the arm reports no registered mean
            # unless every usable seed has one.
            'pmd_registered': (statistics.fmean(c['registered']['pmd'] for c in usable)
                               if usable and all(c['registered'] and
                                                 c['registered'].get('pmd') is not None
                                                 for c in usable) else None),
            'pmd_shift_mean': statistics.fmean(shifts) if shifts else None,
            'pmd_shift_range': [min(shifts), max(shifts)] if shifts else None,
            'pass8_resampled': (statistics.fmean(c['resampled']['pass8'] for c in usable)
                                if usable else None),
            'reportable': bool(usable),
        })
    return {
        'schema': SCHEMA,
        'definition': {
            'metric': 'pairwise correct-mode diversity (PCMD)',
            'estimator': 'PCMD = 1 - sum_m n_m (n_m - 1) / (K (K - 1)) over a prompt\'s verified responses',
            'aggregation': 'pooled over a cell\'s four draws, unweighted mean over defined prompts, then over seeds',
            'sampling': ('aligned eight-wide RNG blocks hashed from (domain, problem, draw), '
                         'the policy registered for the frozen base-model grid'),
            'streams_per_prompt': 'thirty-two disjoint, against eleven overlapping in the registered draws',
            'min_defined_prompts': min_defined,
        },
        'builder': {'path': str(Path(__file__).resolve().relative_to(ROOT)),
                    'sha256': file_sha(Path(__file__).resolve())},
        'registered_comparison': {'path': str(REGISTERED.relative_to(ROOT)),
                                  'sha256': file_sha(REGISTERED) if REGISTERED.is_file() else None},
        'coverage': {'cells': len(cells), 'arms': len(arm_rows),
                     'reportable_arms': sum(1 for a in arm_rows if a['reportable'])},
        'arms': arm_rows,
        'cells': cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipts', type=Path, default=RUN / 'receipts/independent')
    parser.add_argument('--min-defined', type=int, default=DEFAULT_MIN_DEFINED_PROMPTS)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    payload = build(args.receipts, args.min_defined)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write('\n')
    print(json.dumps({'coverage': payload['coverage'], 'output': str(args.output)}, indent=2))


if __name__ == '__main__':
    main()
