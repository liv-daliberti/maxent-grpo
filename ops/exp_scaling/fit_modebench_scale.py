#!/usr/bin/env python3
"""Fit scale candidates on development only and audit untouched test receipts."""
from __future__ import annotations

import argparse
from collections import Counter
import math
from pathlib import Path
import random
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
for directory in ('ops', 'ops/exp_scaling', 'src'):
    sys.path.insert(0, str(ROOT / directory))
from fit_modebench_level3 import (atomic_new, sha, file_sha, cell_histogram,
    serialize_cells, allocate_cells, select_rows, ordered_cells, weight_grid)
from materialize_modebench_scale import (DEFAULT_ROOT, DOMAINS, LEVELS, TOLERANCES,
    SELECTION_SEED, authenticate, read, require, labels, deserialize_cells, union_histogram)

SCHEMA = 'modebench_scale_development_recipe_v1'


def choose_mixture(domain, pools, scores, histograms, baseline):
    """Rank only whole-pool forecasts; inspect one selected DEV set afterward."""
    require(set(pools) == set(scores) == set(range(4)), 'four tiers required')
    require(set(baseline) == set(TOLERANCES) and all(
        isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 1
        for v in baseline.values()) and baseline['pass1'] <= baseline['pass8'], 'invalid fixed target')
    union = union_histogram(histograms)
    seen = set()
    for tier in range(4):
        require(cell_histogram(domain, pools[tier]) == union, 'complete registered calibration cells required')
        hashes = {sha(row) for row in pools[tier]}
        require(len(hashes) == len(pools[tier]) and not hashes & seen, 'duplicate pool rows')
        seen.update(hashes)
        require(set(scores[tier]) == hashes, 'scores must cover every pool row exactly')
        require(all(set(v) == set(TOLERANCES) and all(type(x) in (int, float) and math.isfinite(x)
                    and 0 <= x <= 1 for x in v.values()) and v['pass1'] <= v['pass8']
                    for v in scores[tier].values()), 'invalid row scores')
    indexed = ordered_cells(domain, pools, SELECTION_SEED)
    means = {tier: {cell: {metric: statistics.mean(scores[tier][digest][metric]
                   for _, _, digest in items) for metric in TOLERANCES}
                   for cell, items in indexed[tier].items()} for tier in range(4)}
    best, count = None, 0
    for weights in weight_grid():
        count += 1
        forecasts, differences = {}, {}
        for split in ('dev', 'eval'):
            target = histograms[split]
            allocation = allocate_cells(target, weights, SELECTION_SEED)
            forecasts[split] = {metric: math.fsum(counts[tier] * means[tier][cell][metric]
                    for cell, counts in allocation.items() for tier in range(4)) / sum(target.values())
                    for metric in TOLERANCES}
            differences[split] = {metric: forecasts[split][metric] - baseline[metric]
                                  for metric in TOLERANCES}
        errors = [abs(differences[split][metric]) / TOLERANCES[metric]
                  for split in ('dev', 'eval') for metric in TOLERANCES]
        objective = (max(errors), math.fsum(x*x for x in errors), weights)
        if best is None or objective < best[0]:
            best = objective, weights, forecasts, differences
    require(count == 1771, 'complete grid20 search required')
    selected = select_rows(domain, pools, histograms['dev'], best[1], SELECTION_SEED, indexed)
    metrics = {metric: statistics.mean(scores[row['difficulty']][row['row_sha256']][metric]
               for row in selected) for metric in TOLERANCES}
    differences = {**best[3], 'selected_dev': {metric: metrics[metric] - baseline[metric]
                   for metric in TOLERANCES}}
    gates = {split: {metric: abs(value) <= TOLERANCES[metric] for metric, value in delta.items()}
             for split, delta in differences.items()}
    return {'weights': list(best[1]), 'forecast_metrics': best[2], 'selected_dev_metrics': metrics,
            'differences': differences, 'gates': gates, 'objective': list(best[0][:2]),
            'selected_dev_row_sha256': [item['row_sha256'] for item in selected],
            'development_fit_pass': all(all(row.values()) for row in gates.values()),
            'weight_combinations_considered': count, 'selected_development_sets_scored': 1,
            'selected_evaluation_sets_scored': 0,
            'algorithm': 'dual_full_pool_cell_forecasts_then_one_fixed_development_selection',
            'calibration_cells': serialize_cells(union)}


def receipt_scores(path, rows, protocol, level, domain, phase, regrade=False):
    import evaluate_modebench_scale as evaluator
    from evaluate_modebench_level3 import summarize
    receipt = read(path)
    evaluator.validate_seed_receipt(receipt, rows)
    for field, expected in {'domain': domain, 'level': level, 'split': phase,
                            'model_label': LEVELS[level]}.items():
        require(receipt.get(field) == expected, 'receipt ' + field + ' mismatch')
    identity = receipt['identity']
    require(identity['seeds'] == labels(level, phase), 'unregistered draw labels')
    require(identity['source']['rows_sha256'] == sha(rows) and
            identity['source']['row_offset'] == 0 and identity['source']['row_limit'] == 0,
            'full source rows required')
    actual_model = {k: v for k, v in identity['model'].items() if k != 'vllm_version'}
    require(actual_model == protocol['models'][LEVELS[level]], 'wrong model snapshot')
    values = {}
    if regrade:
        from oat_drgrpo.math_grader import validated_modebench_outcome_key
    for index, (row, result) in enumerate(zip(rows, receipt['prompt_results'])):
        require(result['row_index'] == index and result['row_sha256'] == sha(row), 'row order/hash mismatch')
        for draw in result['draws']:
            attempts = draw['attempts']
            require(len(attempts) == 8, 'exactly eight attempts required')
            for attempt in attempts:
                require(type(attempt['verified']) is bool and attempt['verified'] ==
                        (attempt['canonical_key'] is not None), 'invalid canonical success flag')
                if regrade:
                    key = validated_modebench_outcome_key(attempt['text'], row['answer'])
                    require(sha(key) == sha(attempt['canonical_key']), 'original grader disagrees')
            verified = sum(a['verified'] for a in attempts)
            distinct = len({sha(a['canonical_key']) for a in attempts if a['verified']})
            require(draw['verified_count'] == verified and draw['pass1'] == verified/8
                    and draw['pass8'] == float(verified > 0) and draw['distinct8'] == distinct,
                    'draw metrics disagree with saved attempts')
        for metric in ('pass1', 'pass8', 'distinct8'):
            require(result[metric] == statistics.mean(d[metric] for d in result['draws']),
                    'prompt metrics disagree with draws')
        values[sha(row)] = {metric: result[metric] for metric in TOLERANCES}
    require(receipt['metrics'] == summarize(receipt['prompt_results']), 'aggregate metrics disagree')
    return receipt, values


def fit_domain(campaign, level, domain, publish=True):
    import json
    campaign = Path(campaign).resolve()
    protocol = authenticate(campaign / 'protocol.json')
    require(level in LEVELS and domain in DOMAINS, 'unknown level/domain')
    pools, scores, pins, runtimes = {}, {}, {}, []
    pool_identity_path = campaign / level / 'pools' / domain / 'identity.json'
    pool_identity = read(pool_identity_path)
    require(pool_identity.get('schema') == 'modebench_scale_development_pools_v1'
            and pool_identity.get('level') == level and pool_identity.get('domain') == domain
            and pool_identity.get('protocol_sha256') == file_sha(campaign / 'protocol.json'),
            'development pool identity differs from registered campaign')
    pins[str(pool_identity_path)] = file_sha(pool_identity_path)
    for tier in range(4):
        pool_path = campaign / level / 'pools' / domain / f'difficulty_{tier}.jsonl'
        pools[tier] = [json.loads(line) for line in pool_path.read_text().splitlines()]
        certificate = pool_identity['tiers'][str(tier)]
        require(certificate['rows_sha256'] == sha(pools[tier])
                and certificate['rows'] == len(pools[tier])
                and certificate['semantic_disjoint'] is True and certificate['prompt_disjoint'] is True,
                'development pool changed after structural verification')
        receipt_path = campaign / level / 'results/development' / domain / f'difficulty_{tier}.json'
        receipt, scores[tier] = receipt_scores(receipt_path, pools[tier], protocol, level, domain, 'dev')
        require(Path(receipt['identity']['source']['path']).resolve() == pool_path, 'receipt scored wrong pool')
        require(all(row['scale_candidate_tier'] == tier for row in pools[tier]), 'wrong candidate tier')
        runtimes.append(receipt['identity'].get('runtime'))
        pins.update({str(pool_path): file_sha(pool_path), str(receipt_path): file_sha(receipt_path)})
    require(all(runtime == runtimes[0] for runtime in runtimes), 'candidate tiers used different runtime settings')
    histograms = {s: deserialize_cells(h) for s, h in protocol['histograms'][domain].items()}
    result = choose_mixture(domain, pools, scores, histograms, protocol['targets'][domain]['metrics'])
    result.update(schema=SCHEMA, level=level, domain=domain,
                  protocol_sha256=file_sha(campaign / 'protocol.json'), input_sha256=pins,
                  target=protocol['targets'][domain], runtime=runtimes[0],
                  fitter_sha256=file_sha(__file__))
    if publish:
        atomic_new(campaign / level / 'recipes' / (domain + '.json'), result)
    return result


def bootstrap_delta(values, target, seed, repetitions=2000):
    rng = random.Random(seed)
    estimates = sorted(statistics.mean(rng.choices(values, k=len(values))) - target
                       for _ in range(repetitions))
    return [estimates[int(repetitions * .025)], estimates[int(repetitions * .975)]]


def confirm_domain(campaign, level, domain):
    """Regrade all test attempts, preserving a failed confirmation as a result."""
    from materialize_modebench_harder_v2 import load_rows
    campaign = Path(campaign).resolve()
    protocol = authenticate(campaign / 'protocol.json')
    base = campaign / level / 'dataset' / domain
    dataset = read(base / 'identity.json')
    require(dataset.get('schema') == 'modebench_scale_frozen_domain_v1'
            and dataset.get('status') == 'frozen_pending_heldout_confirmation'
            and dataset.get('level') == level and dataset.get('domain') == domain
            and dataset.get('protocol_sha256') == file_sha(campaign / 'protocol.json'),
            'frozen dataset identity differs from registered level/domain/protocol')
    recipe_path = campaign / level / 'recipes' / (domain + '.json')
    require(dataset['recipe_sha256'] == file_sha(recipe_path), 'frozen recipe changed')
    recipe = read(recipe_path)
    require(recipe['development_fit_pass'] and recipe == fit_domain(campaign, level, domain, publish=False),
            'passing reproducible development recipe required')
    rows = load_rows(base / 'eval', 'multi_answer')
    require(len(rows) == 128 and sha(rows) == dataset['splits']['eval']['rows_sha256'], 'frozen test split changed')
    path = campaign / level / 'results/confirmation' / (domain + '.json')
    receipt, scores = receipt_scores(path, rows, protocol, level, domain, 'eval', regrade=True)
    require(Path(receipt['identity']['source']['path']).resolve() in (base / 'eval', base / 'eval.jsonl'),
            'wrong confirmation source')
    require(receipt['identity'].get('runtime') == recipe['runtime'], 'confirmation changed development runtime')
    target = protocol['targets'][domain]['metrics']
    delta = {metric: receipt['metrics'][metric] - target[metric] for metric in TOLERANCES}
    gates = {metric: abs(delta[metric]) <= TOLERANCES[metric] for metric in TOLERANCES}
    result = {'schema': 'modebench_scale_confirmation_v1', 'level': level, 'domain': domain,
              'difficulty_matched': all(gates.values()), 'gates': gates, 'differences': delta,
              'target': target, 'metrics': receipt['metrics'],
              'candidate_prompt_bootstrap_delta_95': {metric: bootstrap_delta(
                  [score[metric] for score in scores.values()], target[metric], SELECTION_SEED)
                  for metric in TOLERANCES},
              'uncertainty': 'prompt bootstrap conditional on historical fixed target; not an equivalence test',
              'original_grader_replayed_attempts': len(rows) * 32,
              'receipt_sha256': file_sha(path), 'recipe_sha256': file_sha(recipe_path),
              'dataset_identity_sha256': file_sha(base / 'identity.json')}
    atomic_new(campaign / level / 'confirmation' / (domain + '.json'), result)
    return result


def main():
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('fit', 'confirm'))
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--level', choices=LEVELS, required=True)
    parser.add_argument('--domain', choices=DOMAINS, required=True)
    args = parser.parse_args()
    result = (fit_domain if args.action == 'fit' else confirm_domain)(args.root, args.level, args.domain)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
