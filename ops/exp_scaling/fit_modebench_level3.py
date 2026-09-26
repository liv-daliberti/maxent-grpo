#!/usr/bin/env python3
"""Fit a development-only Level 3 mixture with outcome-independent row ordering."""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / 'ops', ROOT / 'ops/exp_scaling', ROOT / 'src'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from materialize_modebench_level3 import DOMAINS, generator, reference_rows, row_hash
from modebench_level3_allocation import allocate_cells

SCHEMA = 'modebench_level3_development_recipe_v2'
SELECTION_ALGORITHM = 'full_pool_cell_forecast_then_controlled_allocation_and_fixed_hash_order_v2'
SELECTION_SEED = 6391701
GRID = 20
TOLERANCES = {'pass1': 0.04, 'pass8': 0.08}


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_new(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as handle:
            json.dump(payload, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def cell(domain, row):
    support = int(row['answer_mode_count'])
    return (support, str(row['answer_mode_family'])) if domain == 'pantry' else (support,)


def cell_histogram(domain, rows):
    return Counter(cell(domain, row) for row in rows)


def serialize_cells(histogram):
    return [{'cell': list(key), 'rows': count} for key, count in sorted(histogram.items())]


def hamilton(count, weights, key=(), seed=SELECTION_SEED):
    """Allocate integer counts using exact arithmetic and outcome-free tie breaks."""
    if len(weights) != 4 or any(w < 0 or int(w) != w for w in weights) or sum(weights) != GRID:
        raise ValueError('four nonnegative integer weights must sum to 20')
    allocated = [count * int(w) // GRID for w in weights]
    ranking = sorted(range(4), key=lambda d: (-(count * int(weights[d]) % GRID), sha([seed, list(key), d])))
    for difficulty in ranking[:count - sum(allocated)]:
        allocated[difficulty] += 1
    return allocated


def weight_grid():
    for a in range(GRID + 1):
        for b in range(GRID - a + 1):
            for c in range(GRID - a - b + 1):
                yield (a, b, c, GRID - a - b - c)


def ordered_cells(domain, pools, seed):
    indexed = {}
    for difficulty, rows in sorted(pools.items()):
        cells = defaultdict(list)
        for index, row in enumerate(rows):
            cells[cell(domain, row)].append((index, row, sha(row)))
        for key, values in cells.items():
            values.sort(key=lambda item: sha([seed, domain, list(key), difficulty, item[2]]))
        indexed[difficulty] = cells
    return indexed


def select_rows(domain, pools, target, weights, seed=SELECTION_SEED, indexed=None):
    indexed = indexed or ordered_cells(domain, pools, seed)
    selected = []
    allocation = allocate_cells(target, weights, seed)
    for key, count in sorted(target.items()):
        for difficulty, required in enumerate(allocation[key]):
            available = indexed[difficulty].get(key, [])
            if len(available) < required:
                raise ValueError(f'pool {difficulty} lacks cell {key}: {required} requested')
            selected.extend({'difficulty': difficulty, 'pool_row_index': index,
                             'row_sha256': digest, 'row': row}
                            for index, row, digest in available[:required])
    selected.sort(key=lambda item: sha([seed, 'selected_order', item['row_sha256']]))
    return selected


def choose_mixture(domain, pools, scores, target, baseline, seed=SELECTION_SEED):
    """Rank weight recipes on full-pool cell forecasts, then score one row set.

    Selected-row residuals never enter ranking or tie breaks. A selected-set
    gate failure is reported for this optimum; it cannot trigger another weight
    choice, row ordering, or seed search.
    """
    indexed = ordered_cells(domain, pools, seed)
    count = sum(target.values())
    if count <= 0:
        raise ValueError('mixture fitting requires a nonempty target')
    cell_means = {}
    for difficulty in range(4):
        cell_means[difficulty] = {}
        for key, required in target.items():
            available = indexed[difficulty].get(key, [])
            if len(available) < required:
                raise ValueError(f'pool {difficulty} lacks cell {key}: {required} requested')
            cell_means[difficulty][key] = {
                metric: statistics.mean(scores[difficulty][digest][metric]
                                        for _, _, digest in available)
                for metric in TOLERANCES
            }
    best = None
    seen_allocations = set()
    combinations = 0
    for weights in weight_grid():
        combinations += 1
        allocation = allocate_cells(target, weights, seed)
        signature = tuple((key, allocation[key]) for key in sorted(target))
        seen_allocations.add(signature)
        expected = {
            metric: math.fsum(allocation[key][difficulty] * cell_means[difficulty][key][metric]
                              for key in target for difficulty in range(4)) / count
            for metric in TOLERANCES
        }
        expected_delta = {metric: expected[metric] - baseline[metric] for metric in TOLERANCES}
        scaled = [abs(expected_delta[metric]) / TOLERANCES[metric] for metric in TOLERANCES]
        objective = (max(scaled), sum(value * value for value in scaled), weights)
        if best is None or objective < best[0]:
            best = (objective, weights, expected, expected_delta)
    if best is None:
        raise ValueError('mixture weight grid is empty')
    selected = select_rows(domain, pools, target, best[1], seed, indexed)
    metrics = {metric: statistics.mean(scores[item['difficulty']][item['row_sha256']][metric]
                                       for item in selected)
               for metric in TOLERANCES}
    delta = {metric: metrics[metric] - baseline[metric] for metric in TOLERANCES}
    gates = {
        'expected': {metric: abs(best[3][metric]) <= tolerance for metric, tolerance in TOLERANCES.items()},
        'selected': {metric: abs(delta[metric]) <= tolerance for metric, tolerance in TOLERANCES.items()},
    }
    return {'weights': best[1], 'selected': selected, 'metrics': metrics,
            'delta': delta, 'expected_metrics': best[2], 'expected_delta': best[3],
            'gates': gates, 'objective': list(best[0][:2]),
            'unique_development_selections_considered': len(seen_allocations),
            'unique_cell_allocations_considered': len(seen_allocations),
            'selected_development_sets_scored': 1,
            'weight_combinations_considered': combinations}


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def receipt_rows(receipt):
    source = receipt['identity']['source']
    if source['kind'] == 'jsonl':
        path = Path(source['path'])
        if source.get('file_sha256') != file_sha(path):
            raise ValueError('receipt JSONL file hash mismatch')
        rows = read_jsonl(path)
    elif source['kind'] == 'saved_dataset':
        from datasets import load_from_disk
        data = load_from_disk(source['path'])
        if hasattr(data, 'keys'):
            data = data['multi_answer']
        rows = [dict(row) for row in data]
    else:
        raise ValueError('unsupported receipt row source')
    if source.get('row_offset', 0) or source.get('row_limit', 0):
        raise ValueError('mixture fitting requires full development receipts')
    if sha(rows) != source['rows_sha256'] or len(rows) != source['selected_rows']:
        raise ValueError('receipt source rows mismatch')
    return rows


def load_receipt(path, domain, level, model):
    receipt = json.loads(Path(path).read_text())
    if (receipt.get('status') != 'complete' or receipt.get('domain') != domain
            or receipt.get('level') != level or receipt.get('split') != 'dev'
            or receipt.get('model_label') != model):
        raise ValueError(f'expected complete {domain}/{level}/{model} development receipt: {path}')
    if receipt['identity_sha256'] != sha(receipt['identity']):
        raise ValueError('receipt identity hash mismatch')
    if receipt.get('information_boundary', {}).get('evaluation_prompts_loaded', True):
        raise ValueError('confirmation outcomes cannot enter development fitting')
    seeds = receipt['identity'].get('seeds', [])
    if len(seeds) != 4 or len(set(seeds)) != 4:
        raise ValueError('development fitting requires four independent sampling seeds')
    rows = receipt_rows(receipt)
    results = receipt['prompt_results']
    if len(results) != len(rows):
        raise ValueError('receipt does not score every source row')
    scores = {}
    for row, result in zip(rows, results):
        digest = sha(row)
        if result['row_sha256'] != digest or digest in scores:
            raise ValueError('receipt row hash/order is invalid or duplicated')
        draws = result.get('draws', [])
        if len(draws) != 4 or [draw.get('seed') for draw in draws] != seeds:
            raise ValueError('development row must contain the four registered draws in order')
        derived = {'pass1': [], 'pass8': []}
        for draw in draws:
            attempts = draw.get('attempts', [])
            if len(attempts) != 8:
                raise ValueError('each development draw must contain eight attempts')
            if any(type(attempt.get('verified')) is not bool or
                   attempt['verified'] != (attempt.get('canonical_key') is not None)
                   for attempt in attempts):
                raise ValueError('development verification flags disagree with canonical keys')
            count = sum(attempt['verified'] for attempt in attempts)
            for metric, value in (('pass1', count / 8), ('pass8', float(count > 0))):
                if not math.isclose(float(draw.get(metric, -1)), value, abs_tol=1e-12):
                    raise ValueError('development draw metric disagrees with recorded attempts')
                derived[metric].append(value)
        for metric in TOLERANCES:
            value = result[metric]
            if not math.isfinite(value) or not math.isclose(value, statistics.mean(derived[metric]), abs_tol=1e-12):
                raise ValueError('development row metric disagrees with its four draws')
        scores[digest] = {metric: result[metric] for metric in TOLERANCES}
    for metric in TOLERANCES:
        if not math.isclose(receipt['metrics'][metric], statistics.mean(r[metric] for r in results), abs_tol=1e-12):
            raise ValueError('receipt aggregate metric mismatch')
    return receipt, rows, scores


def local_dependency_sources(initial_paths):
    """Hash transitive local imports without importing or executing the modules."""
    search_roots = (ROOT / 'ops/exp_scaling', ROOT / 'ops', ROOT / 'src')
    pending, found = list(initial_paths), set()
    while pending:
        path = Path(pending.pop()).resolve()
        if path in found:
            continue
        if not path.is_file() or not path.is_relative_to(ROOT):
            raise ValueError(f'generator source is not a local file: {path}')
        found.add(path)
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
                names.extend(node.module + '.' + alias.name for alias in node.names if alias.name != '*')
            if isinstance(node, ast.ImportFrom) and node.level:
                base = path.parent
                for _ in range(node.level - 1):
                    base = base.parent
                module = base.joinpath(*(node.module or '').split('.'))
                relatives = [module]
                relatives.extend(module / alias.name for alias in node.names if alias.name != '*')
                for relative in relatives:
                    match = next((candidate for candidate in
                                  (relative.with_suffix('.py'), relative / '__init__.py')
                                  if candidate.is_file()), None)
                    if match is not None:
                        pending.append(match)
            for name in names:
                relative = Path(*name.split('.'))
                for root in search_roots:
                    candidates = (root / relative.with_suffix('.py'), root / relative / '__init__.py')
                    match = next((candidate for candidate in candidates if candidate.is_file()), None)
                    if match is not None:
                        pending.append(match)
                        break
    return {str(path.relative_to(ROOT)): file_sha(path) for path in sorted(found)}


def generator_sources(domain):
    main = Path(sys.modules[generator(domain).__module__].__file__).resolve()
    return local_dependency_sources([main, ROOT / 'ops/exp_scaling/materialize_modebench_level3.py',
                                     ROOT / 'ops/exp_scaling/modebench_level3_allocation.py'])


def fit_recipe(baseline_path, score_paths, domain, output=None, seed=SELECTION_SEED):
    if len(score_paths) != 4:
        raise ValueError('exactly four scored difficulty pools are required')
    baseline, baseline_rows, _ = load_receipt(baseline_path, domain, 'level1', '05b')
    reference = reference_rows(domain, 'dev')
    target = cell_histogram(domain, reference)
    if len(reference) != 128:
        raise ValueError('expected 128-row Level 2 development reference')
    if len(reference) % len(baseline_rows):
        raise ValueError('baseline support histogram cannot scale to development size')
    scale = len(reference) // len(baseline_rows)
    baseline_support = Counter(int(row['answer_mode_count']) for row in baseline_rows)
    expected_support = Counter({key: count * scale for key, count in baseline_support.items()})
    if Counter(int(row['answer_mode_count']) for row in reference) != expected_support:
        raise ValueError('baseline and candidate reference support histograms do not match')
    pools, scores, provenance = {}, {}, {}
    model_identity = None
    for path in score_paths:
        receipt, rows, row_scores = load_receipt(path, domain, 'level3', '3b')
        identity = receipt['identity']
        if (identity['interface'] != baseline['identity']['interface']
                or identity['seeds'] != baseline['identity']['seeds']
                or identity['code_sha256'] != baseline['identity']['code_sha256']):
            raise ValueError('baseline and pool must use identical interfaces, seeds and verifier/evaluator code')
        if model_identity is not None and identity['model'] != model_identity:
            raise ValueError('candidate pools used different model identities')
        model_identity = identity['model']
        difficulties = {int(row['level3_difficulty']) for row in rows}
        if len(difficulties) != 1:
            raise ValueError('each receipt must score one difficulty pool')
        difficulty = difficulties.pop()
        if difficulty not in range(4) or difficulty in pools:
            raise ValueError('difficulty pools must cover 0,1,2,3 exactly once')
        if cell_histogram(domain, rows) != target:
            raise ValueError('pool support/family distribution differs from Level 2 reference')
        pool_path = Path(identity['source']['path']).resolve()
        pool_manifest_path = pool_path.with_suffix('.identity.json')
        pool_manifest = json.loads(pool_manifest_path.read_text())
        if pool_manifest['rows_sha256'] != row_hash(rows):
            raise ValueError('candidate pool manifest row hash mismatch')
        main_path = Path(sys.modules[generator(domain).__module__].__file__)
        if pool_manifest['source_sha256'] != file_sha(main_path):
            raise ValueError('candidate pool generator changed; freeze a consistent generation recipe')
        pools[difficulty], scores[difficulty] = rows, row_scores
        provenance[str(difficulty)] = {
            'rows_path': str(pool_path), 'rows_file_sha256': file_sha(pool_path),
            'rows_sha256': row_hash(rows), 'rows': len(rows),
            'pool_identity_path': str(pool_manifest_path),
            'pool_identity_sha256': file_sha(pool_manifest_path),
            'receipt_path': str(Path(path).resolve()), 'receipt_sha256': file_sha(path),
            'receipt_identity_sha256': receipt['identity_sha256'],
        }
    best = choose_mixture(domain, pools, scores, target, baseline['metrics'], seed)
    selected_rows = [item['row'] for item in best['selected']]
    fit_pass = all(passed for checks in best['gates'].values() for passed in checks.values())
    result = {
        'schema': SCHEMA, 'domain': domain,
        'decision': 'development_fit_pass_pending_confirmation' if fit_pass else 'development_fit_failed_revise_candidates',
        'development_fit_pass': fit_pass,
        'weights': [weight / GRID for weight in best['weights']],
        'weight_units': list(best['weights']), 'weight_denominator': GRID,
        'selection': {'seed': seed, 'algorithm': SELECTION_ALGORITHM,
                      'weight_objective': 'full_pool_cell_means_at_exact_controlled_allocation',
                      'selected_residual_used_for_ranking': False,
                      'on_selected_gate_failure': 'fail_without_alternate_weights_or_hash_seeds',
                      'pantry_cells_include_family': domain == 'pantry',
                      'outcome_independent_within_pool_cell_order': True},
        'development': {
            'rows': len(selected_rows), 'rows_sha256': row_hash(selected_rows),
            'cells': serialize_cells(target),
            'baseline_rows': len(baseline_rows),
            'baseline_metrics': {metric: baseline['metrics'][metric] for metric in TOLERANCES},
            'selected_metrics': best['metrics'], 'differences': best['delta'],
            'expected_metrics': best['expected_metrics'], 'expected_differences': best['expected_delta'],
            'gates': best['gates'],
            'tolerances': TOLERANCES, 'objective': best['objective'],
            'weight_combinations_considered': best['weight_combinations_considered'],
            'unique_development_selections_considered': best['unique_development_selections_considered'],
            'unique_cell_allocations_considered': best['unique_cell_allocations_considered'],
            'selected_development_sets_scored': best['selected_development_sets_scored'],
            'actual_difficulty_counts': dict(Counter(item['difficulty'] for item in best['selected'])),
            'reference_rows_sha256': row_hash(reference),
        },
        'selected_development': [{key: value for key, value in item.items() if key != 'row'} for item in best['selected']],
        'provenance': {
            'baseline_receipt_path': str(Path(baseline_path).resolve()),
            'baseline_receipt_sha256': file_sha(baseline_path),
            'baseline_identity_sha256': baseline['identity_sha256'],
            'pools': provenance, 'generator_sources_sha256': generator_sources(domain),
            'fitter_source_sha256': file_sha(Path(__file__)),
            'interface': baseline['identity']['interface'],
            'seeds': baseline['identity']['seeds'],
            'baseline_model': baseline['identity']['model'], 'candidate_model': model_identity,
            'evaluator_and_verifier_code_sha256': baseline['identity']['code_sha256'],
        },
        'information_boundary': {
            'development_only': True, 'confirmation_outcomes_used': False,
            'interpretation': 'Weights are ranked only by full-pool cell forecasts; both forecast and selected-development tolerances must pass, and fresh confirmation is required for a difficulty-match claim.',
        },
    }
    if output is not None:
        atomic_new(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True, type=Path)
    parser.add_argument('--scores', required=True, nargs=4, type=Path)
    parser.add_argument('--domain', required=True, choices=DOMAINS)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--selection-seed', type=int, default=SELECTION_SEED)
    args = parser.parse_args()
    result = fit_recipe(args.baseline, args.scores, args.domain, args.output, args.selection_seed)
    print(json.dumps({'domain': args.domain, 'decision': result['decision'],
                      'weights': result['weights'], 'development': result['development']}, sort_keys=True))


if __name__ == '__main__':
    main()
