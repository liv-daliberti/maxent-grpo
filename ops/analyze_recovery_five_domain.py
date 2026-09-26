#!/usr/bin/env python3
"""Recovery after a withdrawn option, in all five domains.

The estimands mirror App.~\\ref{app:pantry-adaptation}: how often a saved
portfolio already holds a valid answer, how often one is recovered within eight
extra calls, how many calls that costs, and what the two inference-time
strategies -- a temperature tuned on held-back problems, and a diversity prompt
-- do instead. Every cell is a checkpoint; the contrast is the replay arm minus
the fresh objective it was added to, on the same prompts and withdrawals.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'ops') not in sys.path:
    sys.path.insert(0, str(ROOT / 'ops'))

from followup_metrics import atomic_new, file_sha  # noqa: E402
import portfolio_withdrawals as pw  # noqa: E402

BASE = ROOT / 'artifacts/modebench_recovery_five_domain_20260917'
STRATEGIES = ('ordinary', 'temperature', 'diversity_prompt')
METRICS = ('zero_call_recovery', 'recovered', 'recovery_calls', 'initial_correct',
           'initial_distinct', 'recovery_output_tokens')
REPLICATES = 20000
SEED = 20260917


def load():
    inputs = json.loads((BASE / 'inputs.json').read_text())
    if inputs.get('outcomes_read') is not False:
        raise ValueError('frozen inputs must record an outcome-blind freeze')
    cells = {}
    for path in sorted((BASE / 'results').glob('*/result.json')):
        result = json.loads(path.read_text())
        if result['status'] != 'complete':
            continue
        checkpoint = result['checkpoint']
        if checkpoint not in inputs['checkpoints']:
            raise ValueError('a completed cell is not in the frozen cohort: ' + checkpoint['label'])
        costs = path.parent / 'costs.json'
        cells[(result['domain'], checkpoint['arm'], checkpoint['seed'])] = {
            'result': result,
            'costs': json.loads(costs.read_text()) if costs.is_file() else None}
    return inputs, cells


def bootstrap(by_prompt):
    ids = sorted(by_prompt)
    sums = np.array([by_prompt[i][0] for i in ids], dtype=float)
    counts = np.array([by_prompt[i][1] for i in ids], dtype=float)
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(ids), size=(REPLICATES, len(ids)))
    estimates = np.sort(sums[draws].sum(axis=1) / counts[draws].sum(axis=1))
    return {'estimate': float(sums.sum() / counts.sum()),
            'ci95': [float(estimates[int(math.floor(0.025 * REPLICATES))]),
                     float(estimates[int(math.ceil(0.975 * REPLICATES)) - 1])],
            'prompts': len(ids), 'withdrawals': int(counts.sum())}


def records(cell, strategy):
    """One value per (prompt, withdrawal) for this cell and strategy."""
    out = {}
    for row in cell['result']['records']:
        if row['strategy'] != strategy:
            continue
        out[(row['prompt_index'], json.dumps(row['withdrawal']))] = row
    return out


def analyze(rewrite):
    if rewrite:
        (BASE / 'results.json').unlink(missing_ok=True)
    inputs, cells = load()
    domains = sorted({key[0] for key in cells})
    summary, contrasts = [], []
    for domain in pw.DOMAINS:
        if domain not in domains:
            continue
        seeds = sorted({key[2] for key in cells if key[0] == domain})
        for arm in ('control', 'replay'):
            for strategy in STRATEGIES:
                values = collections.defaultdict(list)
                for seed in seeds:
                    cell = cells.get((domain, arm, seed))
                    if cell is None:
                        continue
                    for row in records(cell, strategy).values():
                        for metric in METRICS:
                            values[metric].append(float(row[metric]))
                if not values:
                    continue
                entry = {'domain': domain, 'arm': arm, 'strategy': strategy, 'seeds': seeds,
                         'withdrawals': len(values['recovered'])}
                entry.update({metric: statistics.fmean(values[metric]) for metric in METRICS})
                summary.append(entry)
        for strategy in STRATEGIES:
            row = {'domain': domain, 'strategy': strategy, 'seeds': seeds, 'delta': {}}
            for metric in METRICS:
                by_prompt = collections.defaultdict(lambda: [0.0, 0])
                for seed in seeds:
                    replay, control = cells.get((domain, 'replay', seed)), cells.get((domain, 'control', seed))
                    if not replay or not control:
                        continue
                    left, right = records(replay, strategy), records(control, strategy)
                    for key in left:
                        if key not in right:
                            continue
                        bucket = by_prompt[key[0]]
                        bucket[0] += float(left[key][metric]) - float(right[key][metric])
                        bucket[1] += 1
                if by_prompt:
                    row['delta'][metric] = bootstrap(by_prompt)
            contrasts.append(row)

    # The replay arm's own strategies, against its ordinary portfolio: this is
    # the question the inference-time controls are there to answer.
    strategy_contrasts = []
    for domain in pw.DOMAINS:
        seeds = sorted({key[2] for key in cells if key[0] == domain})
        for arm in ('control', 'replay'):
            for strategy in ('temperature', 'diversity_prompt'):
                for metric in ('zero_call_recovery', 'recovered', 'recovery_calls'):
                    by_prompt = collections.defaultdict(lambda: [0.0, 0])
                    for seed in seeds:
                        cell = cells.get((domain, arm, seed))
                        if cell is None:
                            continue
                        left, right = records(cell, strategy), records(cell, 'ordinary')
                        for key in left:
                            if key not in right:
                                continue
                            bucket = by_prompt[key[0]]
                            bucket[0] += float(left[key][metric]) - float(right[key][metric])
                            bucket[1] += 1
                    if by_prompt:
                        strategy_contrasts.append({'domain': domain, 'arm': arm,
                                                   'strategy': strategy, 'metric': metric,
                                                   **bootstrap(by_prompt)})

    costs = []
    for (domain, arm, seed), cell in sorted(cells.items()):
        if cell['costs'] is None:
            continue
        costs.append({'domain': domain, 'arm': arm, 'seed': seed,
                      'chosen_temperature': cell['result']['chosen_temperature'],
                      'phases': cell['costs']['phases'], 'receipts': cell['costs']['receipts']})

    results = {
        'schema': 'modebench-recovery-five-domain-results-v1',
        'inputs_sha256': file_sha(BASE / 'inputs.json'),
        'protocol': {key: inputs[key] for key in ('cohort', 'split', 'withdrawals_per_problem',
                                                  'temperature_grid', 'max_recovery_calls',
                                                  'strategies', 'interface', 'grading', 'counts')},
        'cells': sorted('/'.join(str(part) for part in key) for key in cells),
        'complete_cells': len(cells), 'expected_cells': len(inputs['checkpoints']),
        'summary': summary, 'contrasts': contrasts, 'strategy_contrasts': strategy_contrasts,
        'costs': costs,
        'bootstrap': {'replicates': REPLICATES, 'seed': SEED,
                      'unit': 'paired whole-prompt resampling; a prompt\'s withdrawals and seeds '
                              'move together'},
    }
    atomic_new(BASE / 'results.json', results)
    print(json.dumps({'stage': 'complete', 'cells': len(cells),
                      'expected': len(inputs['checkpoints']),
                      'summary_rows': len(summary), 'contrasts': len(contrasts)}))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rewrite', action='store_true')
    analyze(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
