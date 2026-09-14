#!/usr/bin/env python3
"""Fit fresh candidate pools to fixed empirical controls in the V3 campaign.

The numerical kernel ranks all grid-20 weights using full-pool forecasts for
both required support histograms. It scores exactly one development selection
after ranking; evaluation selections and historical candidate outcomes never
enter this computation. Historical Level1 outcomes are explicit fixed targets.
"""
from __future__ import annotations

from collections import Counter
import argparse
import fcntl
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'ops/exp_scaling'))
import fit_modebench_level3 as mixture

ALGORITHM = 'dual_full_pool_dev_eval_forecasts_then_one_fixed_dev_selection_v3'
SELECTION_SEED = 6391701
GRID = 20
TOLERANCES = {'pass1': .04, 'pass8': .08}
FORECAST_SPLITS = ('dev', 'eval')
DOMAINS = ('graph_coloring', 'python_factors')
SCHEMA = 'modebench_level3_fixed_reference_development_recipe_v3'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_targets(targets):
    require(set(targets) == set(FORECAST_SPLITS), 'both DEV and EVAL histograms are required')
    for split, histogram in targets.items():
        require(histogram and all(isinstance(key, tuple) and len(key) == 1
                    and type(key[0]) is int and key[0] > 0
                    and type(value) is int and value > 0
                    for key, value in histogram.items()), 'positive exact support-cell counts required')
        require(sum(histogram.values()) == 128, split + ': exact 128-row histogram required')
    return Counter({key: max(targets[split].get(key, 0) for split in FORECAST_SPLITS)
                    for key in set().union(*targets.values())})


def choose_fixed_reference_mixture(domain, pools, scores, targets, baseline,
                                   seed=SELECTION_SEED):
    """Rank forecasts for two histograms, then evaluate one DEV selection.

    Objective: minimum maximum normalized absolute forecast error across the
    four split/metric combinations, then their squared-error sum, then the
    lexicographic integer weights. Selected residuals never break ties.
    """
    require(domain in DOMAINS, 'only the two prospectively revised domains may be fitted')
    require(type(seed) is int and seed == SELECTION_SEED, 'the selected-row seed is fixed')
    require(mixture.GRID == GRID and mixture.TOLERANCES == TOLERANCES
            and mixture.SELECTION_SEED == SELECTION_SEED, 'frozen numerical helpers differ')
    union = validate_targets(targets)
    require(set(pools) == set(scores) == set(range(4))
            and all(type(key) is int for key in pools)
            and all(type(key) is int for key in scores), 'exactly four distinct integer tiers required')
    require(set(baseline) == set(TOLERANCES)
            and all(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1
                    for value in baseline.values()), 'finite fixed pass1/pass8 reference rates required')
    require(baseline['pass1'] <= baseline['pass8'], 'fixed reference pass1 cannot exceed pass8')

    all_identities = set()
    for tier in range(4):
        require(mixture.cell_histogram(domain, pools[tier]) == union,
                'every pool must have exactly the registered union histogram')
        hashes = [mixture.sha(row) for row in pools[tier]]
        require(len(set(hashes)) == len(hashes) and not (set(hashes) & all_identities),
                'duplicate rows in or across calibration pools')
        all_identities.update(hashes)
        require(set(scores[tier]) == set(hashes), 'scores must cover exactly every pool row')
        for values in scores[tier].values():
            require(set(values) == set(TOLERANCES) and all(
                type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1
                for value in values.values()), 'finite per-row pass1/pass8 rates required')
            require(values['pass1'] <= values['pass8'], 'per-attempt success cannot exceed pass8')

    indexed = mixture.ordered_cells(domain, pools, seed)
    cell_means = {
        tier: {key: {metric: statistics.mean(scores[tier][digest][metric]
                       for _, _, digest in indexed[tier][key]) for metric in TOLERANCES}
               for key in sorted(union)}
        for tier in range(4)
    }
    best = None
    unique_allocations = set()
    count = 0
    for weights in mixture.weight_grid():
        count += 1
        require(len(weights) == 4 and all(type(w) is int and w >= 0 for w in weights)
                and sum(weights) == GRID, 'frozen weight grid is invalid')
        forecasts, differences, signatures = {}, {}, []
        for split in FORECAST_SPLITS:
            target = targets[split]
            allocation = mixture.allocate_cells(target, weights, seed)
            signatures.append(tuple((key, allocation[key]) for key in sorted(target)))
            forecasts[split] = {
                metric: math.fsum(allocation[key][tier] * cell_means[tier][key][metric]
                    for key in sorted(target) for tier in range(4)) / 128
                for metric in TOLERANCES
            }
            differences[split] = {metric: forecasts[split][metric] - baseline[metric]
                                  for metric in TOLERANCES}
        unique_allocations.add(tuple(signatures))
        scaled = [abs(differences[split][metric]) / TOLERANCES[metric]
                  for split in FORECAST_SPLITS for metric in TOLERANCES]
        objective = (max(scaled), math.fsum(value * value for value in scaled), weights)
        if best is None or objective < best[0]:
            best = (objective, weights, forecasts, differences)
    require(count == 1771 and best is not None, 'the complete 1771-weight grid is required')

    # This is the sole selected subset, created only after the weight decision.
    selected = mixture.select_rows(domain, pools, targets['dev'], best[1], seed, indexed)
    require(len(selected) == 128 and mixture.cell_histogram(domain,
        [item['row'] for item in selected]) == targets['dev'], 'exact DEV selection required')
    metrics = {metric: statistics.mean(scores[item['difficulty']][item['row_sha256']][metric]
        for item in selected) for metric in TOLERANCES}
    delta = {metric: metrics[metric] - baseline[metric] for metric in TOLERANCES}
    gates = {f'expected_{split}': {metric: abs(best[3][split][metric]) <= tolerance
        for metric, tolerance in TOLERANCES.items()} for split in FORECAST_SPLITS}
    gates['selected'] = {metric: abs(delta[metric]) <= tolerance
                        for metric, tolerance in TOLERANCES.items()}
    return {
        'weights': best[1], 'selected': selected, 'metrics': metrics, 'delta': delta,
        'expected_dev_metrics': best[2]['dev'], 'expected_eval_metrics': best[2]['eval'],
        'expected_dev_differences': best[3]['dev'], 'expected_eval_differences': best[3]['eval'],
        'gates': gates, 'objective': list(best[0][:2]),
        'unique_cell_allocations_considered': len(unique_allocations),
        'selected_development_sets_scored': 1, 'selected_evaluation_sets_scored': 0,
        'weight_combinations_considered': count, 'algorithm': ALGORITHM,
        'calibration_union_cells': mixture.serialize_cells(union),
        'forecast_cells': {split: mixture.serialize_cells(targets[split]) for split in FORECAST_SPLITS},
        'selected_residual_used_for_ranking': False,
        'historical_candidate_outcomes_used': False,
        'historical_level1_confirmation_used_as_fixed_target': True,
        'development_fit_pass': all(value for row in gates.values() for value in row.values()),
    }


def fit_recipe(registration_path, registration_sha256, domain, score_paths=None):
    """Authenticate fresh candidate receipts and the declared historical target.

    This in-memory operation permits independent exact reconstruction. Only
    main() publishes, under one exclusive fit intent with no automatic retries.
    """
    import modebench_level3_v3_common as common
    require(domain in DOMAINS, 'only registered revised domains can be fitted')
    registration = common.validate_registration(registration_path, registration_sha256)
    require(registration['files_sha256'].get(str(Path(__file__).resolve())) == common.digest(__file__),
            'fixed-reference fitter was not prospectively registered')
    revision = registration['candidate_revisions'][domain]
    expected_name = {'graph_coloring': 'graph_v8', 'python_factors': 'python_v7'}[domain]
    require(revision['name'] == expected_name
            and Path(revision['pool_root']) == common.POOL_ROOTS[domain], 'candidate revision route differs')
    expected_paths = {common.RESULTS / f'calibration_3b_{expected_name}_d{tier}.json': tier
                      for tier in range(4)}
    score_paths = list(expected_paths) if score_paths is None else [Path(p).resolve() for p in score_paths]
    require(len(score_paths) == 4 and set(score_paths) == set(expected_paths),
            'exactly the four registered new candidate development receipts are required')
    require(revision.get('development_receipts') == {str(tier): str(path) for path, tier in expected_paths.items()},
            'candidate receipt paths were not prospectively registered')
    for kind in ('generator', 'materializer'):
        path = Path(revision[kind + '_path']).resolve()
        require(registration['files_sha256'].get(str(path)) == revision[kind + '_sha256']
                == common.digest(path), 'registered candidate source changed: ' + kind)
    references = common.fixed_references()
    target = references[domain]
    # No reference receipt is relabeled DEV or rewritten for the old fitter.
    inherited = common.authenticate_inherited()
    histograms = common.reference_histograms(domain)
    forecast_targets = {split: histograms[split] for split in FORECAST_SPLITS}
    union = common.calibration_histogram(domain)
    pools, scores, provenance, outcome_pins = {}, {}, {}, {}
    semantic_identities = set()
    from modebench_level3_discrete import row_identity
    for path in sorted(score_paths):
        tier = expected_paths[path]
        outcome_pins[str(path)] = common.digest(path)
        receipt, rows, row_scores = common.validate_new_candidate_receipt(path, domain)
        identity = receipt['identity']
        require(identity['model'] == inherited['models']['3b']
                and identity['interface'] == target['receipt']['identity']['interface']
                and identity['code_sha256'] == target['receipt']['identity']['code_sha256'],
                'candidate model, interface or evaluator/grader code differs from the fixed study')
        pool_path = common.POOL_ROOTS[domain] / 'pools' / domain / f'difficulty_{tier}.jsonl'
        source = identity['source']
        require(source['kind'] == 'jsonl' and Path(source['path']) == pool_path
                and source['file_sha256'] == common.digest(pool_path), 'candidate pool route/hash differs')
        require(mixture.cell_histogram(domain, rows) == union
                and all(type(row.get('level3_difficulty')) is int and row['level3_difficulty'] == tier
                        for row in rows), 'candidate tier or union support coverage differs')
        semantic = {row_identity(domain, row) for row in rows}
        require(len(semantic) == len(rows) and not (semantic & semantic_identities),
                'candidate pools repeat semantic problem identities')
        semantic_identities.update(semantic)
        certificate_path = pool_path.with_suffix('.identity.json')
        outcome_pins[str(certificate_path)] = common.digest(certificate_path)
        certificate = common.read(certificate_path)
        require(certificate['candidate_revision'] == expected_name
                and certificate['registration_path'] == str(common.REGISTRATION)
                and certificate['registration_sha256'] == registration_sha256
                and certificate['rows_sha256'] == mixture.row_hash(rows)
                and certificate['source_sha256'] == revision['generator_sha256']
                and certificate.get('checks')
                and all(value is True for value in certificate['checks'].values()),
                'candidate structural certificate differs from the registered law')
        for item in (pool_path, certificate_path):
            current = common.digest(item)
            require(str(item) not in outcome_pins or outcome_pins[str(item)] == current,
                    'candidate evidence changed during initial authentication')
            outcome_pins[str(item)] = current
        pools[tier], scores[tier] = rows, row_scores
        provenance[str(tier)] = {
            'rows_path': str(pool_path), 'rows_file_sha256': common.digest(pool_path),
            'rows_sha256': mixture.row_hash(rows), 'rows': len(rows),
            'pool_identity_path': str(certificate_path), 'pool_identity_sha256': common.digest(certificate_path),
            'receipt_path': str(path), 'receipt_sha256': outcome_pins[str(path)],
            'receipt_identity_sha256': receipt['identity_sha256'],
        }
    best = choose_fixed_reference_mixture(domain, pools, scores, forecast_targets, target['metrics'])
    selected_rows = [item['row'] for item in best['selected']]
    development = {
        'rows': 128, 'rows_sha256': mixture.row_hash(selected_rows),
        'cells': mixture.serialize_cells(histograms['dev']),
        'reference_rows': 128, 'baseline_metrics': target['metrics'],
        'selected_metrics': best['metrics'], 'differences': best['delta'],
        'tolerances': dict(TOLERANCES), 'actual_difficulty_counts': dict(Counter(
            item['difficulty'] for item in best['selected'])),
        **{key: best[key] for key in ('expected_dev_metrics', 'expected_eval_metrics',
            'expected_dev_differences', 'expected_eval_differences', 'gates', 'objective',
            'weight_combinations_considered', 'unique_cell_allocations_considered',
            'selected_development_sets_scored', 'selected_evaluation_sets_scored',
            'calibration_union_cells', 'forecast_cells')},
    }
    result = {
        'schema': SCHEMA, 'domain': domain, 'development_fit_pass': best['development_fit_pass'],
        'decision': ('development_fit_pass_pending_fresh_candidate_confirmation' if best['development_fit_pass']
                     else 'development_fit_failed_revise_candidates'),
        'candidate_revision': dict(revision),
        'registration': {'path': str(common.REGISTRATION), 'sha256': registration_sha256},
        'weights': [weight / GRID for weight in best['weights']],
        'weight_units': list(best['weights']), 'weight_denominator': GRID,
        'selection': {'seed': SELECTION_SEED, 'algorithm': ALGORITHM,
            'weight_objective': 'minimax_normalized_dev_eval_forecast_then_squared_sum_then_weights',
            'selected_residual_used_for_ranking': False,
            'on_selected_gate_failure': 'fail_without_alternate_weights_or_hash_seeds',
            'outcome_independent_within_pool_cell_order': True},
        'development': development,
        'selected_development': [{key: value for key, value in item.items() if key != 'row'}
                                 for item in best['selected']],
        'provenance': {'reference_target': target['provenance'], 'pools': provenance,
            'fitter_source_sha256': common.digest(__file__),
            'generator_sources_sha256': {str(Path(revision[kind + '_path'])): revision[kind + '_sha256']
                                        for kind in ('generator', 'materializer')},
            'interface': target['receipt']['identity']['interface'], 'seeds': list(common.DEV_LABELS),
            'candidate_model': inherited['models']['3b'],
            'evaluator_and_verifier_code_sha256': target['receipt']['identity']['code_sha256']},
        'information_boundary': {
            'reference_semantics': 'fixed_measured_level1_benchmark',
            'historical_level1_confirmation_used_as_fixed_reference': True,
            'historical_level3_confirmation_used_for_mixture_fitting': False,
            'new_candidate_fitting_uses_development_outcomes_only': True,
            'fresh_candidate_confirmation_outcomes_used': False,
            'adaptive_confirmation_round': 2, 'all_five_fresh_same_round': False,
            'statistical_equivalence_claimed': False,
        },
    }
    common.verify_pins(outcome_pins)
    common.validate_registration(registration_path, registration_sha256)
    return result


def main():
    import modebench_level3_v3_common as common
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, default=common.REGISTRATION)
    parser.add_argument('--registration-sha256', required=True)
    parser.add_argument('--domain', required=True, choices=DOMAINS)
    parser.add_argument('--scores', nargs=4, type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    expected_output = common.CAMPAIGN / 'recipes' / (args.domain + '.json')
    require(args.output.resolve() == expected_output, 'only the canonical new recipe may be published')
    expected_output.parent.mkdir(parents=True, exist_ok=True)
    intent = expected_output.with_suffix('.fit_intent.json')
    with expected_output.with_suffix('.fit.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not expected_output.exists() and not intent.exists(),
                'existing recipe or ambiguous fit intent requires review; never refit automatically')
        common.validate_registration(args.registration, args.registration_sha256)
        common.atomic_new(intent, {'schema': 'modebench_level3_v3_fixed_reference_fit_intent_v1',
            'domain': args.domain, 'registration_sha256': args.registration_sha256,
            'source_sha256': common.digest(__file__), 'output': str(expected_output)})
        result = fit_recipe(args.registration, args.registration_sha256, args.domain, args.scores)
        common.atomic_new(expected_output, result)
        print(json.dumps({'path': str(expected_output), 'sha256': common.digest(expected_output),
                          'development_fit_pass': result['development_fit_pass'], 'weights': result['weights']}))


if __name__ == '__main__':
    main()
