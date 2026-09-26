#!/usr/bin/env python3
"""Audit all-five-domain empirical Level 3/3B versus Level 1/0.5B matching.

--pairs-json contains [{"domain": ..., "baseline": "receipt.json",
"candidate": "receipt.json"}, ...]. Optional baseline_sha256/candidate_sha256
pin receipt files; baseline_dataset/candidate_dataset pin saved source datasets.
--dataset-root is required for confirmation and authenticates frozen recipes,
model/interface/code settings, and both source datasets. Confirmation requires
128 rows and four n=8 draws on each side. --development emits a provisional
report with relaxed sizes and permits omission of --dataset-root.

The bootstrap resamples independent prompts, keeping all draws for each prompt
together. Intervals describe uncertainty and do not assert statistical equivalence.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import statistics
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'ops') not in sys.path:
    sys.path.insert(0, str(ROOT / 'ops'))
from evaluate_modebench_level3 import DOMAINS, SCHEMA as RECEIPT_SCHEMA, atomic_new, file_sha, frozen_interface, sha

SCHEMA = 'modebench-level3-observed-match-audit-v1'
RECIPE_SCHEMA = 'modebench_level3_development_recipe_v2'
SELECTION_ALGORITHM = 'full_pool_cell_forecast_then_controlled_allocation_and_fixed_hash_order_v2'
TOLERANCES = {'pass1': .04, 'pass8': .08}
METRICS = ('pass1', 'pass8', 'distinct8')
DEFAULT_CONFIRMATION_SEEDS = (6319000, 6319001, 6319002, 6319003)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def close(actual: Any, expected: float, context: str) -> None:
    require(isinstance(actual, (int, float)) and math.isfinite(actual)
            and abs(actual - expected) <= 1e-12, f'{context}: metric mismatch')


def valid_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def materializer_row_hash(rows: list[dict]) -> str:
    """Match materialize_modebench_harder_v2.row_hash (distinct from sha(rows))."""
    import hashlib
    return hashlib.sha256('\n'.join(json.dumps(row, sort_keys=True, separators=(',', ':'))
                                    for row in rows).encode()).hexdigest()


def validate_receipt(receipt: dict[str, Any], *, domain: str, role: str,
                     development: bool = False) -> list[dict[str, float]]:
    label, level = ('05b', 'level1') if role == 'baseline' else ('3b', 'level3')
    split = 'dev' if development else 'eval'
    prefix = f'{domain}/{role}'
    require(receipt.get('schema') == RECEIPT_SCHEMA, f'{prefix}: unknown receipt schema')
    require(receipt.get('status') == 'complete', f'{prefix}: incomplete receipt')
    for field, expected in (('domain', domain), ('model_label', label), ('level', level), ('split', split)):
        require(receipt.get(field) == expected, f'{prefix}: wrong {field}; expected {expected}')
    identity = receipt.get('identity', {})
    require(sha(identity) == receipt.get('identity_sha256'), f'{prefix}: identity hash mismatch')
    for field, expected in (('schema', RECEIPT_SCHEMA), ('domain', domain), ('level', level), ('split', split)):
        require(identity.get(field) == expected, f'{prefix}: inconsistent identity {field}')
    require(identity.get('model', {}).get('label') == label, f'{prefix}: inconsistent model identity')
    interface = identity.get('interface', {})
    require(interface == frozen_interface(domain, interface.get('name')), f'{prefix}: interface is not frozen')
    require(sha(interface) == identity.get('interface_sha256'), f'{prefix}: interface hash mismatch')
    seeds = identity.get('seeds', [])
    require(isinstance(seeds, list) and seeds and all(type(seed) is int and seed >= 0 for seed in seeds)
            and len(seeds) == len(set(seeds)), f'{prefix}: invalid sampling seeds')
    require(development or len(seeds) == 4, f'{prefix}: confirmation requires exactly four draws')
    require(receipt.get('sampling') == {**interface, 'seeds': seeds}, f'{prefix}: inconsistent sampling metadata')
    boundary = receipt.get('information_boundary', {})
    require(boundary.get('evaluation_prompts_loaded') is (not development), f'{prefix}: invalid split boundary')
    if not development:
        require(boundary.get('confirmation_explicitly_authorized') is True, f'{prefix}: confirmation was not authorized')
    require(boundary.get('treatment_training_started') is False, f'{prefix}: trained model evidence is not initial difficulty')
    rows = receipt.get('prompt_results', [])
    require(isinstance(rows, list) and len(rows) > 0, f'{prefix}: missing prompt results')
    require(development or len(rows) == 128, f'{prefix}: confirmation requires 128 prompts')
    source = identity.get('source', {})
    require(source.get('selected_rows') == len(rows), f'{prefix}: source row count mismatch')
    require(valid_sha(source.get('rows_sha256')) and valid_sha(source.get('all_rows_sha256')),
            f'{prefix}: invalid source identity')
    offset = source.get('row_offset', 0)
    require(type(offset) is int and offset >= 0, f'{prefix}: invalid source row offset')
    if not development:
        require(source.get('total_rows') == 128 and offset == 0 and source.get('row_limit', 0) == 0,
                f'{prefix}: confirmation source was sliced')
    require([row.get('row_index') for row in rows] == list(range(offset, offset + len(rows))),
            f'{prefix}: repeated or missing prompt indices')
    require(len({row.get('row_sha256') for row in rows}) == len(rows), f'{prefix}: duplicate source rows')
    values = []
    for index, row in enumerate(rows):
        context = f'{prefix}/row{index}'
        require(all(valid_sha(row.get(field)) for field in ('row_sha256', 'problem_sha256', 'spec_sha256')),
                f'{context}: invalid row identity')
        draws = row.get('draws', [])
        require(len(draws) == len(seeds) and [draw.get('seed') for draw in draws] == seeds,
                f'{context}: missing or repeated draw')
        recalculated_draws = []
        for draw in draws:
            attempts = draw.get('attempts', [])
            require(len(attempts) == 8, f'{context}: each draw must have eight attempts')
            for attempt in attempts:
                require(type(attempt.get('verified')) is bool and
                        attempt['verified'] == (attempt.get('canonical_key') is not None),
                        f'{context}: verified/canonical key disagreement')
                require(isinstance(attempt.get('text'), str), f'{context}: missing sampled text')
                require(type(attempt.get('token_count')) is int and 0 <= attempt['token_count'] <= interface['max_tokens'],
                        f'{context}: invalid token count')
            correct = sum(attempt['verified'] for attempt in attempts)
            require(draw.get('verified_count') == correct, f'{context}: incorrect verified_count')
            distinct = len({sha(a['canonical_key']) for a in attempts if a['verified']})
            metrics = {'pass1': correct / 8, 'pass8': float(correct > 0), 'distinct8': float(distinct)}
            for metric, value in metrics.items():
                close(draw.get(metric), value, f'{context}/draw{draw["seed"]}/{metric}')
            recalculated_draws.append(metrics)
        averages = {metric: statistics.mean(draw[metric] for draw in recalculated_draws) for metric in METRICS}
        for metric, value in averages.items():
            close(row.get(metric), value, f'{context}/{metric}')
        values.append(averages)
    require(receipt.get('metrics', {}).get('rows') == len(values), f'{prefix}: summary row count mismatch')
    for metric in METRICS:
        close(receipt.get('metrics', {}).get(metric), statistics.mean(row[metric] for row in values),
              f'{prefix}/summary/{metric}')
    return values


def percentile(sorted_values: list[float], probability: float) -> float:
    position = probability * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (position - lower)


def bootstrap_differences(baseline: list[dict[str, float]], candidate: list[dict[str, float]],
                          *, seed: int, replicates: int) -> dict[str, Any]:
    """Independent prompt bootstrap; replicate draws stay grouped within prompts."""
    require(replicates >= 100, 'at least 100 bootstrap replicates are required')
    require(bool(baseline) and bool(candidate), 'bootstrap requires both prompt populations')
    rng = random.Random(seed)
    differences = {metric: [] for metric in METRICS}
    for _ in range(replicates):
        left = rng.choices(baseline, k=len(baseline))
        right = rng.choices(candidate, k=len(candidate))
        for metric in METRICS:
            differences[metric].append(sum(row[metric] for row in right) / len(right)
                                       - sum(row[metric] for row in left) / len(left))
    return {metric: {'90_percent': [percentile(sorted(values), .05), percentile(sorted(values), .95)],
                     '95_percent': [percentile(sorted(values), .025), percentile(sorted(values), .975)]}
            for metric, values in differences.items()}


def verify_dataset(receipt: dict[str, Any], path: Path, expected: dict | None = None) -> dict[str, Any]:
    from datasets import load_from_disk
    dataset = load_from_disk(str(path))
    if hasattr(dataset, 'keys'):
        dataset = dataset['multi_answer']
    all_rows = [dict(row) for row in dataset]
    source = receipt['identity']['source']
    offset, limit = source.get('row_offset', 0), source.get('row_limit', 0)
    selected = all_rows[offset:offset + limit if limit else None]
    require(sha(all_rows) == source['all_rows_sha256'], f'{path}: full source rows do not match receipt')
    require(sha(selected) == source['rows_sha256'], f'{path}: selected source rows do not match receipt')
    require(len(selected) == len(receipt['prompt_results']), f'{path}: selected source length mismatch')
    for row, recorded in zip(selected, receipt['prompt_results']):
        spec = json.loads(row['answer']) if isinstance(row['answer'], str) else row['answer']
        require((sha(row), sha(row['problem']), sha(spec)) ==
                (recorded['row_sha256'], recorded['problem_sha256'], recorded['spec_sha256']),
                f'{path}: row/spec/prompt identity mismatch')
        require({k: v for k, v in row.items() if k not in ('problem', 'answer')} == recorded['row_metadata'],
                f'{path}: row metadata mismatch')
    row_hash = materializer_row_hash(all_rows)
    if expected is not None:
        require(expected.get('rows_sha256') == row_hash, f'{path}: frozen dataset identity rows hash mismatch')
        require(expected.get('rows') == len(all_rows), f'{path}: frozen dataset identity rows count mismatch')
        if expected.get('path'):
            require(Path(expected['path']).resolve() == path.resolve(), f'{path}: frozen dataset identity path mismatch')
    return {'path': str(path.resolve()), 'rows': len(all_rows), 'rows_sha256': row_hash,
            'evaluator_rows_sha256': sha(all_rows), 'verified_against_dataset_identity': expected is not None}


def level1_eval_path(domain: str) -> Path:
    """Resolve the same Level 1 confirmation source used by the evaluator."""
    directory = ROOT / 'ops/exp_scaling'
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    from materialize_modebench_harder_v2 import LEVEL1
    return Path(LEVEL1[domain]['eval']).resolve()


def verify_development_gates(recipe: dict[str, Any], domain: str) -> None:
    """Authenticate v2's forecast-only choice and both unchanged development gates."""
    selection = recipe.get('selection', {})
    require(isinstance(selection, dict)
            and selection.get('algorithm') == SELECTION_ALGORITHM
            and selection.get('weight_objective') == 'full_pool_cell_means_at_exact_controlled_allocation'
            and selection.get('selected_residual_used_for_ranking') is False
            and selection.get('outcome_independent_within_pool_cell_order') is True
            and selection.get('on_selected_gate_failure') == 'fail_without_alternate_weights_or_hash_seeds',
            f'{domain}: frozen recipe does not use forecast-only v2 selection')
    development = recipe.get('development', {})
    require(isinstance(development, dict)
            and development.get('tolerances') == TOLERANCES
            and type(development.get('selected_development_sets_scored')) is int
            and development['selected_development_sets_scored'] == 1,
            f'{domain}: frozen recipe must use unchanged tolerances and one selected development set')
    gates = development.get('gates', {})
    require(isinstance(gates, dict) and set(gates) == {'expected', 'selected'},
            f'{domain}: frozen recipe lacks both expected and selected development gates')
    for gate, metric_field, delta_field in (('expected', 'expected_metrics', 'expected_differences'),
                                           ('selected', 'selected_metrics', 'differences')):
        measured = development.get(metric_field, {})
        baselines = development.get('baseline_metrics', {})
        differences = development.get(delta_field, {})
        gate_values = gates[gate]
        require(all(isinstance(value, dict) for value in (measured, baselines, differences, gate_values)),
                f'{domain}: invalid {gate} development metrics or gates')
        for metric, tolerance in TOLERANCES.items():
            value, baseline, delta = measured.get(metric), baselines.get(metric), differences.get(metric)
            require(all(type(number) in (int, float) and math.isfinite(number)
                        for number in (value, baseline, delta))
                    and 0 <= value <= 1 and 0 <= baseline <= 1
                    and math.isclose(value - baseline, delta, rel_tol=0, abs_tol=1e-12)
                    and abs(delta) <= tolerance and gate_values.get(metric) is True,
                    f'{domain}: frozen {gate} development gate did not pass for {metric}')


def frozen_recipe(domain: str, dataset_root: Path, dataset_identity: dict) -> tuple[dict, dict]:
    """Authenticate the exact recipe bytes frozen before final row generation."""
    path = dataset_root / 'recipes' / f'{domain}.json'
    digest = file_sha(path)
    require(dataset_identity.get('recipe_sha256', {}).get(domain) == digest,
            f'{domain}: frozen recipe hash differs from dataset identity')
    bundle_path = dataset_root / 'frozen_recipes.json'
    bundle_hash = file_sha(bundle_path)
    require(dataset_identity.get('frozen_recipe_bundle_sha256') == bundle_hash,
            f'{domain}: frozen recipe bundle hash differs from dataset identity')
    bundle = json.loads(bundle_path.read_text())
    require(bundle.get('recipes_sha256', {}).get(domain) == digest,
            f'{domain}: frozen recipe hash differs from pre-generation bundle')
    require(bundle.get('confirmation_outcomes_used') is False,
            f'{domain}: recipe bundle used confirmation outcomes')
    recipe = json.loads(path.read_text())
    require(recipe.get('schema') == RECIPE_SCHEMA
            and recipe.get('domain') == domain, f'{domain}: invalid frozen recipe schema/domain')
    require(recipe.get('development_fit_pass') is True, f'{domain}: frozen development fit did not pass')
    verify_development_gates(recipe, domain)
    require(recipe.get('information_boundary', {}).get('confirmation_outcomes_used') is False,
            f'{domain}: frozen recipe used confirmation outcomes')
    seeds = recipe.get('provenance', {}).get('seeds', [])
    require(isinstance(seeds, list) and len(seeds) == 4 and len(set(seeds)) == 4
            and all(type(seed) is int and seed >= 0 for seed in seeds),
            f'{domain}: frozen recipe lacks four registered development seeds')
    return recipe, {'path': str(path.resolve()), 'sha256': digest,
                    'bundle_sha256': bundle_hash, 'development_seeds': seeds}


def verify_recipe_settings(receipt: dict, recipe: dict, domain: str, role: str,
                           expected_eval_seeds: list[int]) -> None:
    identity, provenance = receipt['identity'], recipe['provenance']
    prefix = f'{domain}/{role}'
    require(identity['interface'] == provenance.get('interface'),
            f'{prefix}: confirmation interface differs from frozen recipe')
    model_key = 'baseline_model' if role == 'baseline' else 'candidate_model'
    require(identity['model'] == provenance.get(model_key),
            f'{prefix}: confirmation model identity differs from frozen recipe')
    require(identity.get('code_sha256') == provenance.get('evaluator_and_verifier_code_sha256'),
            f'{prefix}: confirmation evaluator/verifier code differs from frozen recipe')
    require(identity['seeds'] == expected_eval_seeds,
            f'{prefix}: confirmation seeds differ from registered seeds {expected_eval_seeds}')
    require(not set(identity['seeds']) & set(provenance['seeds']),
            f'{prefix}: confirmation seeds overlap frozen development seeds')


def audit_pairs(pairs: list[dict[str, Any]], *, development: bool = False,
                dataset_root: Path | None = None, bootstrap_seed: int = 6317999,
                bootstrap_replicates: int = 10000,
                expected_eval_seeds: tuple[int, ...] | list[int] = DEFAULT_CONFIRMATION_SEEDS) -> dict[str, Any]:
    require(isinstance(pairs, list), 'pairs must be a list')
    if not development:
        require(dataset_root is not None,
                'confirmation requires --dataset-root to authenticate frozen recipes and source datasets')
        require(isinstance(expected_eval_seeds, (list, tuple)) and len(expected_eval_seeds) == 4
                and len(set(expected_eval_seeds)) == 4
                and all(type(seed) is int and seed >= 0 for seed in expected_eval_seeds),
                'exactly four distinct registered confirmation seeds are required')
    expected_eval_seeds = list(expected_eval_seeds)
    require(bootstrap_replicates >= 100, 'at least 100 bootstrap replicates are required')
    domains = [pair.get('domain') for pair in pairs]
    require(len(domains) == len(set(domains)), 'duplicate domain pairs')
    require(set(domains) <= set(DOMAINS), 'unknown domain in pairs')
    dataset_identity = None
    identity_path = dataset_root / 'identity.json' if dataset_root else None
    if identity_path:
        dataset_identity = json.loads(identity_path.read_text())
    results, errors = {}, {}
    for domain in DOMAINS:
        if domain not in domains:
            continue
        pair = pairs[domains.index(domain)]
        try:
            receipts, samples, receipt_files, datasets = {}, {}, {}, {}
            recipe, recipe_record = (frozen_recipe(domain, dataset_root, dataset_identity)
                                     if dataset_root and not development else (None, None))
            for role in ('baseline', 'candidate'):
                path = Path(pair[role])
                digest = file_sha(path)
                require(not pair.get(role + '_sha256') or pair[role + '_sha256'] == digest,
                        f'{domain}/{role}: pinned receipt hash mismatch')
                receipt = json.loads(path.read_text())
                receipts[role] = receipt
                samples[role] = validate_receipt(receipt, domain=domain, role=role, development=development)
                receipt_files[role] = {'path': str(path.resolve()), 'sha256': digest,
                                       'identity_sha256': receipt['identity_sha256']}
                dataset_path = Path(pair[role + '_dataset']) if pair.get(role + '_dataset') else None
                expected = None
                if recipe is not None:
                    verify_recipe_settings(receipt, recipe, domain, role, expected_eval_seeds)
                    pinned_source = (level1_eval_path(domain) if role == 'baseline'
                                     else (dataset_root / domain / 'eval').resolve())
                    require(receipt['identity']['source'].get('kind') == 'saved_dataset'
                            and Path(receipt['identity']['source'].get('path', '')).resolve() == pinned_source,
                            f'{domain}/{role}: confirmation source differs from frozen Level 1/final evaluation path')
                    require(dataset_path is None or dataset_path.resolve() == pinned_source,
                            f'{domain}/{role}: supplied dataset differs from frozen evaluation path')
                    dataset_path = pinned_source
                if role == 'candidate' and dataset_root:
                    expected_path = dataset_root / domain / ('dev' if development else 'eval')
                    require(dataset_path is None or dataset_path.resolve() == expected_path.resolve(),
                            f'{domain}: candidate dataset differs from final dataset root')
                    dataset_path = expected_path
                    expected = dataset_identity['domains'][domain]['dev' if development else 'eval']
                if dataset_path:
                    datasets[role] = verify_dataset(receipt, dataset_path, expected)
            left, right = receipts['baseline']['identity'], receipts['candidate']['identity']
            require(left['interface'] == right['interface'], f'{domain}: baseline/candidate interface mismatch')
            require(len(left['seeds']) == len(right['seeds']), f'{domain}: number of sampled draws differs')
            if not development:
                require(left['seeds'] == right['seeds'] == expected_eval_seeds,
                        f'{domain}: both models must use identical registered confirmation seeds {expected_eval_seeds}')
            require(left.get('code_sha256') == right.get('code_sha256'), f'{domain}: evaluator/verifier code mismatch')
            require(left['model'].get('vllm_version') == right['model'].get('vllm_version'),
                    f'{domain}: decoding engine version mismatch')
            means = {role: {metric: statistics.mean(row[metric] for row in values) for metric in METRICS}
                     for role, values in samples.items()}
            delta = {metric: means['candidate'][metric] - means['baseline'][metric] for metric in METRICS}
            checks = {metric: abs(delta[metric]) <= tolerance + 1e-12 for metric, tolerance in TOLERANCES.items()}
            domain_seed = bootstrap_seed + DOMAINS.index(domain)
            results[domain] = {'baseline_rows': len(samples['baseline']), 'candidate_rows': len(samples['candidate']),
                               'draws_per_prompt': len(left['seeds']), 'means': means,
                               'candidate_minus_baseline': delta, 'within_tolerance': checks,
                               'observed_approximate_match': all(checks.values()),
                               'bootstrap_intervals': bootstrap_differences(samples['baseline'], samples['candidate'],
                                    seed=domain_seed, replicates=bootstrap_replicates),
                               'bootstrap_seed': domain_seed, 'receipts': receipt_files,
                               'source_datasets': datasets, 'interface': left['interface'],
                               'frozen_recipe': recipe_record}
        except (KeyError, TypeError, ValueError, OSError) as error:
            errors[domain] = str(error)
    missing = [domain for domain in DOMAINS if domain not in domains]
    complete = not missing and not errors and set(results) == set(DOMAINS)
    matched = complete and all(row['observed_approximate_match'] for row in results.values())
    status = ('invalid_evidence' if errors else 'incomplete' if missing else
              'observed_approximate_match' if matched else 'outside_match_tolerance')
    return {'schema': SCHEMA, 'generated_at': datetime.now(timezone.utc).isoformat(),
            'status': status, 'phase': 'development' if development else 'confirmation',
            'all_five_domains_complete': complete, 'all_five_observed_approximate_match': matched,
            'confirmation_match_verified': matched and not development,
            'criteria': {'absolute_pass1_difference_at_most': TOLERANCES['pass1'],
                         'absolute_pass8_difference_at_most': TOLERANCES['pass8'],
                         'distinct8_is_diagnostic_only': True, 'statistical_equivalence_claimed': False,
                         'confirmation_rows_per_domain_per_model': 128, 'confirmation_draws_per_prompt': 4,
                         'samples_per_draw': 8, 'all_five_domains_required': True,
                         'expected_confirmation_seeds': expected_eval_seeds if not development else None},
            'bootstrap': {'replicates': bootstrap_replicates, 'seed': bootstrap_seed,
                          'resampling_unit': 'prompt; all sampling draws retained together',
                          'method': 'independent two-sample percentile bootstrap',
                          'intervals': [90, 95], 'intervals_are_diagnostic_not_admission_criteria': True},
            'evidence_validation': {'metrics_recomputed_from_attempts': True, 'recorded_verification_outcomes_regraded': False,
                                    'source_rows_verified_when_dataset_supplied': True,
                                    'frozen_recipe_settings_required': bool(dataset_root and not development),
                                    'baseline_source_pinned_to_level1_confirmation': bool(dataset_root and not development)},
            'dataset_identity': {'path': str(identity_path.resolve()), 'sha256': file_sha(identity_path)} if identity_path else None,
            'domains': results, 'missing_domains': missing, 'errors': errors}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs-json', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--development', action='store_true')
    parser.add_argument('--dataset-root', type=Path,
                        help='required for confirmation; pins frozen recipes and source datasets')
    parser.add_argument('--bootstrap-seed', type=int, default=6317999)
    parser.add_argument('--bootstrap-replicates', type=int, default=10000)
    parser.add_argument('--expected-eval-seeds', type=int, nargs=4, default=list(DEFAULT_CONFIRMATION_SEEDS),
                        help='four seeds registered before confirmation (default: 6319000..6319003)')
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f'fresh audit receipt required: {args.output}')
    report = audit_pairs(json.loads(args.pairs_json.read_text()), development=args.development,
                         dataset_root=args.dataset_root, bootstrap_seed=args.bootstrap_seed,
                         bootstrap_replicates=args.bootstrap_replicates,
                         expected_eval_seeds=args.expected_eval_seeds)
    report['pairs_manifest'] = {'path': str(args.pairs_json.resolve()), 'sha256': file_sha(args.pairs_json)}
    atomic_new(args.output, report)
    print(json.dumps({'status': report['status'], 'phase': report['phase'], 'output': str(args.output),
                      'missing_domains': report['missing_domains'], 'errors': report['errors']}, sort_keys=True))


if __name__ == '__main__':
    main()
