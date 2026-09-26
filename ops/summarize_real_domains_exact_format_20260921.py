#!/usr/bin/env python3
"""Bound QA diversity from exhaustive listed-letter/EOS likelihoods.

All other strings remain unenumerated. Their probability is retained as an
explicit residual, not silently assigned zero or discarded by renormalizing.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pcmd_bounds(probabilities, residual):
    """Extremize conditional diversity over allocations of unenumerated mass.

    All known correct modes, including zero-mass ones, must be represented.
    Concentrating residual on the largest mode minimizes diversity; filling
    the smallest modes maximizes it. If no correct mass is known, conditional
    diversity may be undefined, so this diagnostic reports no bound.
    """
    probabilities = sorted(probabilities)
    if residual < 0 or any(p < 0 for p in probabilities):
        raise ValueError('negative probability mass')
    known = sum(probabilities)
    if not probabilities or known <= 0:
        return None, None
    total = known + residual
    lower = 1 - ((probabilities[-1]+residual)/total)**2 - sum((p/total)**2 for p in probabilities[:-1])
    filled = probabilities[:]
    remaining = residual
    for index in range(len(filled)-1):
        cost = (filled[index+1]-filled[index])*(index+1)
        addition = min(remaining, cost)
        for j in range(index+1):
            filled[j] += addition/(index+1)
        remaining -= addition
        if remaining <= 0:
            break
    if remaining > 0:
        filled = [p+remaining/len(filled) for p in filled]
    upper = 1 - sum((p/total)**2 for p in filled)
    return max(0.0, lower), max(0.0, upper)


def task_metrics(rows):
    modes = defaultdict(float)
    total = 0.0
    support = set()
    for row, score in rows:
        probability = math.exp(score['sum_logprob'])
        total += probability
        if row['accepted']:
            support.add(row['canonical_key'])
            modes[row['canonical_key']] += probability
    if total > 1.00002:
        raise ValueError('enumerated disjoint sequence probabilities exceed one')
    residual = max(0.0, 1.0-total)
    success = sum(modes.values())
    pcmd_lower, pcmd_upper = pcmd_bounds(list(modes.values()), residual)
    result = {
        'enumerated_probability': total, 'unresolved_probability': residual,
        'success_lower_bound': success,
        'success_upper_bound': min(1.0, success+residual),
        'known_support': len(support), 'mode_probabilities_in_enumerated_format': dict(modes),
        'pcmd_lower_bound': pcmd_lower, 'pcmd_upper_bound': pcmd_upper,
        'pcmd_conditioned_on_enumerated_correct_format': 1-sum((p/success)**2 for p in modes.values()) if success > 0 else None,
    }
    for k in (8, 32):
        lower = sum(-math.expm1(k*math.log1p(-min(p, 1-1e-16))) for p in modes.values())
        # Adding total mass r to any modes can increase this concave discovery
        # functional by at most k*r; true support supplies an additional cap.
        result[f'ed{k}_lower_bound'] = lower
        result[f'ed{k}_upper_bound'] = min(float(len(support)), lower+k*residual)
    return result


def summarize(config_path, scores_path, output):
    if output.exists():
        raise FileExistsError(output)
    config, result = json.loads(config_path.read_text()), json.loads(scores_path.read_text())
    if result['config_sha256'] != sha(config_path) or result['status'] != 'complete':
        raise ValueError('score/config identity mismatch or incomplete result')
    rows = {r['row_id']: r for r in config['rows']}
    sequences = {(r['task_id'], tuple(r['response_token_ids'])) for r in config['rows']}
    if len(sequences) != len(config['rows']):
        raise ValueError('duplicate response events would double-count probability')
    expected = {(c['name'], r['row_id']) for c in config['checkpoints'] for r in config['rows']}
    native = [r for r in result['scores'] if r['adapter_precision'] == 'native']
    if len(native) != len(expected) or {(r['checkpoint'], r['row_id']) for r in native} != expected:
        raise ValueError('missing, duplicate or unexpected scores')
    grouped = defaultdict(list)
    for score in native:
        row = rows[score['row_id']]
        grouped[score['checkpoint'], row['task_id']].append((row, score))
    tasks = [{'checkpoint': checkpoint, 'task_id': task, 'split': pairs[0][0]['split'], **task_metrics(pairs)}
             for (checkpoint, task), pairs in grouped.items()]
    cohorts = {}
    labels = ['train', 'dev'] + config.get('secondary_diagnostic_cohorts', [])
    if len(labels) != len(set(labels)):
        raise ValueError('duplicate diagnostic cohort labels')
    for split in labels:
        arms = {}
        for checkpoint in (c['name'] for c in config['checkpoints']):
            selected = [r for r in tasks if r['checkpoint'] == checkpoint and
                        (r['split'] == split if split in ('train', 'dev') else
                         config['mechanistic_cohorts'][r['task_id']] == split)]
            if not selected:
                continue
            metrics = ('success_lower_bound', 'success_upper_bound', 'pcmd_lower_bound', 'pcmd_upper_bound', 'ed8_lower_bound', 'ed8_upper_bound',
                       'ed32_lower_bound', 'ed32_upper_bound', 'unresolved_probability')
            arms[checkpoint] = {'tasks': len(selected), **{k: statistics.mean(r[k] for r in selected) if all(r[k] is not None for r in selected) else None for k in metrics},
                                'max_unresolved_probability': max(r['unresolved_probability'] for r in selected)}
        contrasts = {}
        names = config.get('contrast_checkpoints', {'maxrl': 'maxrl_32', 'remax': 'remax_32'})
        if names['remax'] in arms and names['maxrl'] in arms:
            re, mx = arms[names['remax']], arms[names['maxrl']]
            for key in ('success', 'pcmd', 'ed8', 'ed32'):
                contrasts[key] = {'lower_bound': re[key+'_lower_bound']-mx[key+'_upper_bound'],
                                  'upper_bound': re[key+'_upper_bound']-mx[key+'_lower_bound']} if all(a[key+bound] is not None for a in (re,mx) for bound in ('_lower_bound','_upper_bound')) else None
        trajectory = {}
        for checkpoint in arms:
            if not checkpoint.startswith('maxrl_'):
                continue
            step = checkpoint.removeprefix('maxrl_')
            treatment = 'remax_' + step
            if treatment not in arms:
                raise ValueError('unpaired checkpoint in diagnostic trajectory')
            re, mx = arms[treatment], arms[checkpoint]
            trajectory[step] = {key: ({'lower_bound': re[key+'_lower_bound']-mx[key+'_upper_bound'],
                                     'upper_bound': re[key+'_upper_bound']-mx[key+'_lower_bound']} if all(a[key+bound] is not None for a in (re,mx) for bound in ('_lower_bound','_upper_bound')) else None)
                                for key in ('success', 'pcmd', 'ed8', 'ed32')}
        cohorts[split] = {'checkpoints': arms, 'remax_minus_maxrl_bounds': contrasts,
                         'primary_diagnostic_cohort': split in ('train', 'dev'),
                         'fixed_checkpoint_trajectory_bounds': trajectory}
    payload = {'schema': 'real-domains-exact-format-bounds-20260921-v1', 'status': 'complete',
               'interpretation': 'Diagnostic teacher-forced probabilities under the HF training engine. Bounds retain all unenumerated output probability. PCMD bounds use the complete finite correct-topic support and absolute event masses; they are population quantities, not thresholded sample estimates. These bounds do not quantify training-seed or inference-kernel uncertainty or establish held-out efficacy.',
               'sources': {str(config_path.resolve()): sha(config_path), str(scores_path.resolve()): sha(scores_path)},
               'reducer_sha256': sha(__file__),
               'cohorts': cohorts, 'tasks': tasks}
    output.write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
    print(json.dumps(cohorts, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--scores', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summarize(args.config, args.scores, args.output)
