#!/usr/bin/env python3
"""Audit independent-RNG fresh Level 3/3B versus fixed Level 1/0.5B controls.

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
from evaluate_modebench_level3_independent import (DOMAINS, SCHEMA as RECEIPT_SCHEMA, atomic_new, file_sha, frozen_interface, sha, validate_seed_receipt)

SCHEMA = 'modebench-level3-independent-observed-match-audit-v2'
RECIPE_SCHEMA = 'modebench_level3_development_recipe_v2'
SELECTION_ALGORITHM = 'full_pool_cell_forecast_then_controlled_allocation_and_fixed_hash_order_v2'
TOLERANCES = {'pass1': .04, 'pass8': .08}
METRICS = ('pass1', 'pass8', 'distinct8')
DEFAULT_CONFIRMATION_SEEDS = (6329000, 6329001, 6329002, 6329003)
CAMPAIGN = ROOT / 'var/artifacts/modebench_level3_v2'
LATEST_DEVELOPMENT_SEAL = CAMPAIGN / 'python_v6/implementation_seal.json'
LATEST_DEVELOPMENT_SEAL_SHA256 = '7f3a49ae3314ef314accac14be7668c920454b8d9206ca07026e81c772a0f266'
EXECUTION_AMENDMENT_SHA256 = 'b2c57f4b711aab3381f0a8303c1be22bcf1e627a628977b94769f5501460f397'
from audit_modebench_level3_recovery_execution import validate_completion_attestation



def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def close(actual: Any, expected: float, context: str) -> None:
    require(type(actual) in (int, float) and math.isfinite(actual)
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
    validate_seed_receipt(receipt)
    require(role in ('baseline', 'candidate'), 'unknown receipt role')
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
    from evaluate_modebench_level3_independent import load_rows
    task = {'domain': domain, 'rows_jsonl' if source['kind'] == 'jsonl' else 'dataset': source['path'],
            'row_offset': offset, 'row_limit': source.get('row_limit', 0)}
    source_rows, _ = load_rows(task)
    for source_row, recorded in zip(source_rows, rows):
        spec = json.loads(source_row['answer']) if isinstance(source_row['answer'], str) else source_row['answer']
        require((sha(source_row), sha(source_row['problem']), sha(spec)) ==
                (recorded['row_sha256'], recorded['problem_sha256'], recorded['spec_sha256']),
                f'{prefix}: recorded row/spec/prompt differs from authenticated source')
        require(recorded.get('row_metadata') == {key: value for key, value in source_row.items()
                                               if key not in ('problem', 'answer')},
                f'{prefix}: row metadata differs from authenticated source')
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
            require(type(draw.get('verified_count')) is int and draw['verified_count'] == correct, f'{context}: incorrect verified_count')
            distinct = len({sha(a['canonical_key']) for a in attempts if a['verified']})
            metrics = {'pass1': correct / 8, 'pass8': float(correct > 0), 'distinct8': float(distinct)}
            for metric, value in metrics.items():
                close(draw.get(metric), value, f'{context}/draw{draw["seed"]}/{metric}')
            recalculated_draws.append(metrics)
        averages = {metric: statistics.mean(draw[metric] for draw in recalculated_draws) for metric in METRICS}
        for metric, value in averages.items():
            close(row.get(metric), value, f'{context}/{metric}')
        values.append(averages)
    require(type(receipt.get('metrics', {}).get('rows')) is int and receipt['metrics']['rows'] == len(values), f'{prefix}: summary row count mismatch')
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
    """Re-score the original fixed Level 1 control; only Level 3 is fresh."""
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


def finalizer_module():
    directory = ROOT / 'ops/exp_scaling'
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    import finalize_modebench_level3_python_v6_independent
    return finalize_modebench_level3_python_v6_independent


def required_structural_checks(split):
    common = {'row_count', 'unique_identities', 'historical_and_cross_split_disjointness',
              'exact_support_histogram', 'unchanged_verifier_contract',
              'exact_joint_support_family_histogram', 'all_final_splits_disjoint'}
    return common | ({'frozen_selected_development_rows', 'prior_train_eval_and_fixed_baseline_disjoint'}
                     if split == 'dev' else {'exact_difficulty_cell_recipe', 'exact_global_tier_quota'})


def validate_publication_metadata(identity, dataset_root):
    require(identity.get('schema') == 'modebench_level3_capability_matched_splits_independent_v2'
            and identity.get('status') == 'structural_checks_pass'
            and identity.get('decision') == 'pending_confirmation'
            and set(identity.get('domains', {})) == set(DOMAINS),
            'all-five independent structural dataset identity required')
    require(identity.get('split_sizes') == {'train': 384, 'dev': 128, 'eval': 128}
            and identity.get('generation_seed_offset') == 1_000_000, 'registered split sizes/generation seeds changed')
    boundary = identity.get('information_boundary', {})
    require(boundary.get('recipe_frozen_before_eval_generation') is True
            and boundary.get('evaluation_model_outcomes_loaded') is False
            and boundary.get('treatment_training_started') is False,
            'invalid final dataset information boundary')
    for domain, splits in identity['domains'].items():
        require(set(splits) == {'train', 'dev', 'eval'}, f'{domain}: missing or extra final splits')
        for split, count in [('train', 384), ('dev', 128), ('eval', 128)]:
            record = splits[split]
            checks = record.get('checks', {})
            require(type(record.get('rows')) is int and record['rows'] == count
                    and isinstance(checks, dict) and required_structural_checks(split) <= set(checks)
                    and all(value is True for value in checks.values()),
                    f'{domain}/{split}: required structural checks or row counts failed')
            require(valid_sha(record.get('rows_sha256'))
                    and Path(record.get('path', '')).resolve() == (dataset_root / domain / split).resolve()
                    and record.get('dataset_split') == ('train' if split == 'train' else 'multi_answer'),
                    f'{domain}/{split}: structural source identity invalid')


def protected_other_history(finalizer, domain, dataset_root):
    """Skip the publication being audited without subtracting colliding identities."""
    from datasets import load_from_disk
    blocked = finalizer.existing_ids(domain)
    data_root = ROOT / 'var/data'
    roots = set(data_root.glob('modebench_harder*')) | set(data_root.glob('modebench_level3*'))
    excluded_roots = {dataset_root.resolve(), finalizer.PRIOR_REVISION.resolve()}
    for root in sorted(roots):
        if not root.is_dir() or root.resolve() in excluded_roots:
            continue
        for split in ('train', 'dev', 'eval'):
            path = root / domain / split
            if path.is_dir() and (path / 'dataset_dict.json').is_file():
                for subset in load_from_disk(str(path)).values():
                    blocked |= finalizer.identity_set(domain, [dict(row) for row in subset])
    return blocked


def verify_all_split_sources(dataset_root, identity):
    """Bind every structural certificate to actual rows, including unscored splits."""
    from datasets import load_from_disk
    finalizer = finalizer_module()
    for domain in DOMAINS:
        rows_by_split, ids_by_split = {}, {}
        for split in ('train', 'dev', 'eval'):
            record = identity['domains'][domain][split]
            rows = [dict(row) for row in load_from_disk(str(dataset_root / domain / split))[record['dataset_split']]]
            require(len(rows) == record['rows'] and materializer_row_hash(rows) == record['rows_sha256'],
                    f'{domain}/{split}: actual rows differ from structural certificate')
            reference = finalizer.reference_rows(domain, split)
            finalizer.verify_rows(domain, rows, reference, finalizer.modes(reference), set())
            require(finalizer.cell_histogram(domain, rows) == finalizer.cell_histogram(domain, reference),
                    f'{domain}/{split}: actual joint support/family histogram differs')
            require({str(key): value for key, value in finalizer.modes(rows).items()} == record['support_histogram']
                    and finalizer.serialize_cells(finalizer.cell_histogram(domain, rows)) == record['joint_cells'],
                    f'{domain}/{split}: support certificate differs from actual rows')
            rows_by_split[split] = rows
            ids_by_split[split] = finalizer.identity_set(domain, rows)
        require(all(not ids_by_split[left] & ids_by_split[right]
                    for left, right in [('train', 'dev'), ('train', 'eval'), ('dev', 'eval')]),
                f'{domain}: actual final splits overlap')
        prior_rows, controls = finalizer.authenticated_exclusion_rows(domain)
        prior = {split: finalizer.identity_set(domain, rows) for split, rows in prior_rows.items()}
        fixed = finalizer.identity_set(domain, controls)
        candidates = set()
        pool_records = identity.get('excluded_candidate_pools', {}).get(domain)
        require(isinstance(pool_records, list) and pool_records, f'{domain}: missing candidate exclusion inventory')
        for record in pool_records:
            path = Path(record['path'])
            require(file_sha(path) == record['file_sha256'], f'{domain}: candidate exclusion source changed')
            candidates |= finalizer.identity_set(domain, finalizer.read_jsonl(path))
        protected = protected_other_history(finalizer, domain, dataset_root) | prior['train'] | prior['eval'] | fixed
        require(not ids_by_split['dev'] & protected, f'{domain}: development overlaps protected historical/control identities')
        protected |= prior['dev'] | candidates
        require(not (ids_by_split['train'] | ids_by_split['eval']) & protected,
                f'{domain}: fresh Level 3 rows overlap historical/control/candidate identities')


def verify_fixed_control_provenance(bundle, dataset_identity):
    pinned = bundle.get('prior_and_fixed_control_files_sha256', {})
    require(pinned and pinned == dataset_identity.get('prior_and_fixed_control_files_sha256'),
            'prior/fixed-control provenance differs from pre-generation freeze')
    require(pinned == finalizer_module().baseline_and_prior_pins(),
            'prior/fixed-control complete file inventory or bytes changed')
    require(bundle.get('generation_seed_offset') == dataset_identity.get('generation_seed_offset') == 1_000_000,
            'frozen generation seed offset changed')
    finalizer_module().verify_development_inputs_unchanged(bundle['development_inputs'])
    require(bundle.get('finalizer_source_sha256') == file_sha(Path(finalizer_module().__file__)),
            'independent finalizer changed after publication')
    controls = {name: str(level1_eval_path(name)) for name in DOMAINS}
    require(bundle.get('baseline_confirmation_paths') == controls
            and dataset_identity.get('baseline_confirmation_paths') == controls,
            'fixed Level 1 controls differ from prospective frozen paths')
    amendment_path = Path(bundle['prospective_amendment_path'])
    amendment_sha = file_sha(amendment_path)
    amendment = finalizer_module().validate_protocol_amendment(amendment_path)
    require(amendment['baseline_confirmation_paths'] == controls, 'amendment does not authorize frozen controls')
    require(amendment_sha == bundle.get('prospective_amendment_sha256')
            and dataset_identity.get('prospective_amendment_path') == str(amendment_path)
            and dataset_identity.get('prospective_amendment_sha256') == amendment_sha,
            'prospective fixed-control amendment changed after freeze')


def verify_recipe_split_allocation(recipe, domain, dataset_root, identity):
    from collections import Counter
    from datasets import load_from_disk
    finalizer = finalizer_module()
    require(identity['domains'][domain]['dev']['rows_sha256'] == recipe['development']['rows_sha256'],
            f'{domain}: final development differs from the sole frozen selected set')
    recorded_pools = {str(Path(record['path']).resolve()): record['file_sha256']
                      for record in identity['excluded_candidate_pools'][domain]}
    actual_pools = {str(path.resolve()): file_sha(path) for path in finalizer.candidate_pool_files(domain, recipe)}
    require(recorded_pools == actual_pools, f'{domain}: candidate exclusion file inventory is incomplete or changed')
    for split in ('train', 'eval'):
        record = identity['domains'][domain][split]
        rows = [dict(row) for row in load_from_disk(str(dataset_root / domain / split))[record['dataset_split']]]
        target = finalizer.cell_histogram(domain, finalizer.reference_rows(domain, split))
        assigned = finalizer.allocate_cells(target, recipe['weight_units'], recipe['selection']['seed'])
        for tier in range(4):
            actual = finalizer.cell_histogram(domain, [row for row in rows if row['level3_difficulty'] == tier])
            expected = Counter({cell: counts[tier] for cell, counts in assigned.items() if counts[tier]})
            require(actual == expected, f'{domain}/{split}: actual difficulty cell allocation differs from frozen recipe')
        counts = Counter(str(row['level3_difficulty']) for row in rows)
        require(counts == record['difficulty_counts'], f'{domain}/{split}: recorded difficulty counts differ')
        require(record['seed'] == finalizer.SEEDS[domain] + 1_000_000 + (100_000 if split == 'train' else 200_000),
                f'{domain}/{split}: frozen generation seed changed')


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
    directory = ROOT / 'ops/exp_scaling'
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    from finalize_modebench_level3_python_v6_independent import load_recipe
    recipe = load_recipe(path, domain)
    require(bundle.get('schema') == 'modebench_level3_frozen_recipe_bundle_independent_v2',
            'independent recipe bundle required')
    verify_fixed_control_provenance(bundle, dataset_identity)
    require(recipe.get('schema') == RECIPE_SCHEMA
            and recipe.get('domain') == domain, f'{domain}: invalid frozen recipe schema/domain')
    require(recipe.get('development_fit_pass') is True, f'{domain}: frozen development fit did not pass')
    verify_development_gates(recipe, domain)
    verify_recipe_split_allocation(recipe, domain, dataset_root, dataset_identity)
    require(recipe.get('information_boundary', {}).get('confirmation_outcomes_used') is False,
            f'{domain}: frozen recipe used confirmation outcomes')
    seeds = recipe.get('provenance', {}).get('seeds', [])
    require(seeds == [6328000, 6328001, 6328002, 6328003]
            and all(type(seed) is int for seed in seeds),
            f'{domain}: frozen recipe lacks four registered development seeds')
    return recipe, {'path': str(path.resolve()), 'sha256': digest,
                    'bundle_sha256': bundle_hash, 'development_seeds': seeds}


def verify_pinned_files_and_trees(record):
    for source, expected in record['files_sha256'].items():
        require(file_sha(Path(source)) == expected, f'prospective sealed file changed: {source}')
    for directory, expected in record['directory_files'].items():
        actual = sorted(str(path.resolve()) for path in Path(directory).rglob('*') if path.is_file())
        require(actual == expected, f'prospective sealed inventory changed: {directory}')
        require(all(source in record['files_sha256'] for source in actual),
                f'prospective sealed inventory has unpinned files: {directory}')


def authenticated_development_sources():
    """Reopen the complete inherited campaign chain, without reading outcomes."""
    from evaluate_modebench_level3_independent import load_rows, schedule_record, validate_task
    latest_path = LATEST_DEVELOPMENT_SEAL.resolve()
    require(file_sha(latest_path) == LATEST_DEVELOPMENT_SEAL_SHA256,
            'latest registered development seal changed')
    latest = json.loads(latest_path.read_text())
    verify_pinned_files_and_trees(latest)
    chains, visited = [], set()
    def visit(path, expected_hash):
        path = Path(path).resolve()
        require(file_sha(path) == expected_hash, 'inherited development seal changed')
        if path in visited:
            return
        visited.add(path)
        sealed = json.loads(path.read_text())
        require(sealed['models'] == latest['models'] and
                all(latest['files_sha256'].get(source) == digest
                    for source, digest in sealed['files_sha256'].items()),
                'latest development seal does not preserve inherited files/models')
        for key, inherited in sealed.items():
            if key.startswith('inherited_') and key.endswith('_seal'):
                inherited_path = str(Path(inherited).resolve())
                inherited_hash = sealed[key + '_sha256']
                require(latest['files_sha256'].get(inherited_path) == inherited_hash,
                        'inherited development seal is not pinned by latest seal')
                require(file_sha(Path(inherited_path)) == inherited_hash, 'inherited development seal changed')
                # Legacy v1 bytes remain pinned; their defective adjacent RNG
                # labels are not part of the independent-v2 campaign inventory.
                if key != 'inherited_v1_seal':
                    visit(inherited_path, inherited_hash)
        chains.append(sealed)
    visit(latest_path, LATEST_DEVELOPMENT_SEAL_SHA256)
    manifest, protocols, all_blocks = [], [], set()
    for sealed in chains:
        protocol_path = Path(sealed['protocol']).resolve()
        protocol_hash = sealed['protocol_sha256']
        require(latest['files_sha256'].get(str(protocol_path)) == protocol_hash
                and file_sha(protocol_path) == protocol_hash, 'development protocol is not pinned')
        protocol = json.loads(protocol_path.read_text())
        jobs = protocol['jobs']
        sources = {entry['name']: entry for entry in sealed['sources']}
        require(len(sources) == len(sealed['sources']) == len(jobs)
                and set(sources) == {job['name'] for job in jobs}, 'development source inventory differs')
        protocols.append({'path': str(protocol_path), 'sha256': protocol_hash})
        campaign_blocks = set()
        inherited_count = len(all_blocks)
        for job in jobs:
            task_path = Path(job['tasks']).resolve()
            task_hash = file_sha(task_path)
            require(latest['files_sha256'].get(str(task_path)) == task_hash,
                    'development task is not pinned')
            tasks = json.loads(task_path.read_text())
            require(isinstance(tasks, list) and len(tasks) == 1, 'development task inventory differs')
            task = tasks[0]
            validate_task(task, False)
            require(task['domain'] == job['domain'] and task['split'] == 'dev'
                    and task['seeds'] == [6328000, 6328001, 6328002, 6328003]
                    and all(type(label) is int for label in task['seeds']), 'development registered labels/domain differ')
            rows, identity = load_rows(task)
            schedule = schedule_record(job['domain'], rows, task['seeds'])
            blocks = [base for row in schedule['request_seeds'] for base in row]
            expected = sources[job['name']]
            require(identity == expected['identity'] and sha(schedule) == expected['seed_schedule_sha256'],
                    'development source or sampling schedule differs from its original seal')
            require(len(blocks) == len(set(blocks)) == expected['distinct_request_blocks']
                    and len(blocks) * 8 == expected['distinct_child_seeds'], 'development sampling coverage differs')
            require(not set(blocks) & (all_blocks | campaign_blocks),
                    'effective RNG blocks overlap another registered development source')
            campaign_blocks.update(blocks)
            manifest.append({'name': job['name'], 'domain': job['domain'], 'tasks': str(task_path),
                'tasks_sha256': task_hash, 'identity': identity, 'seeds': task['seeds'],
                'seed_schedule_sha256': sha(schedule), 'distinct_request_blocks': len(blocks),
                'distinct_child_seeds': len(blocks) * 8})
        require(len(campaign_blocks) == sealed['distinct_request_blocks'], 'development campaign block count differs')
        all_blocks.update(campaign_blocks)
        if 'inherited_distinct_request_blocks' in sealed:
            require(sealed['inherited_distinct_request_blocks'] == inherited_count
                    and sealed['combined_distinct_request_blocks'] == len(all_blocks),
                    'development combined block count differs')
    require(len(all_blocks) == latest.get('combined_distinct_request_blocks', latest['distinct_request_blocks']),
            'latest complete development RNG inventory differs')
    return {'manifest': manifest, 'protocols': protocols, 'blocks': all_blocks,
            'latest': latest, 'latest_sha256': LATEST_DEVELOPMENT_SEAL_SHA256}


def validate_recovery_execution_binding(seal, plan):
    """Keep the separate scheduler recovery proof outside scientific RNG inventory."""
    amendment_path = CAMPAIGN / 'confirmation_python_v6/prospective_execution_amendment.json'
    require(plan.get('execution') == seal.get('execution') == {'partition': 'all', 'qos': 'normal', 'preempt_mode': 'OFF'}
            and plan.get('prospective_execution_amendment_path') == str(amendment_path.resolve())
            and seal.get('prospective_execution_amendment_path') == str(amendment_path.resolve())
            and plan.get('prospective_execution_amendment_sha256') == seal.get('prospective_execution_amendment_sha256') == EXECUTION_AMENDMENT_SHA256
            and file_sha(amendment_path) == EXECUTION_AMENDMENT_SHA256
            and seal['files_sha256'].get(str(amendment_path.resolve())) == EXECUTION_AMENDMENT_SHA256,
            'prospective regular-queue confirmation amendment differs')
    amendment = json.loads(amendment_path.read_text())
    retained = {**amendment['superseded_files_sha256'], **amendment['retained_completed_evidence_sha256']}
    require(len(amendment['superseded_files_sha256']) == 1101
            and all(seal['files_sha256'].get(source) == expected and file_sha(Path(source)) == expected
                    for source, expected in retained.items()),
            'confirmation omits or changes superseded continuation or completed evidence')
    recovery = validate_completion_attestation()
    metadata = {key: value for key, value in recovery.items() if key != 'files_sha256'}
    require(seal.get('development_execution_recovery') == metadata,
            'confirmation seal recovery completion metadata differs')
    require(type(recovery['jobs']) is int and recovery['jobs'] == 11
            and type(recovery['attempts']) is int and recovery['attempts'] == 43008,
            'recovery completion coverage differs')
    require(all(seal['files_sha256'].get(source) == expected
                for source, expected in recovery['files_sha256'].items()),
            'confirmation seal omits or changes recovery execution chain files')
    return metadata


def python_v6_development_evidence():
    from audit_modebench_level3_python_v6_development import validate_completed_development_audit
    return validate_completed_development_audit(CAMPAIGN / 'python_v6/independent_completed_development_audit.json')


def validate_python_v6_development_binding(seal, plan):
    evidence = python_v6_development_evidence()
    proof = (CAMPAIGN / 'python_v6/independent_completed_development_audit.json').resolve()
    recipe = (CAMPAIGN / 'python_v6/recipe.json').resolve()
    require(evidence['path'] == str(proof) and evidence['sha256'] == file_sha(proof)
            and plan.get('required_python_v6_completed_development_audit') == str(proof)
            and evidence['files_sha256'].get(str(proof)) == evidence['sha256'],
            'canonical completed Python v6 audit binding differs')
    require(evidence['status'] == 'passed_development'
            and type(evidence['jobs']) is int and evidence['jobs'] == 4
            and type(evidence['attempts']) is int and evidence['attempts'] == 20480
            and evidence['development_gates'] == {'expected':{'pass1':True,'pass8':True},
                                                 'selected':{'pass1':True,'pass8':True}}
            and all(type(value) is bool for metrics in evidence['development_gates'].values() for value in metrics.values()),
            'all four Python v6 gates and complete audited execution are required')
    require(evidence['scientific_seal_path'] == str(LATEST_DEVELOPMENT_SEAL.resolve())
            and evidence['scientific_seal_sha256'] == LATEST_DEVELOPMENT_SEAL_SHA256
            and evidence['recipe_path'] == str(recipe) and evidence['recipe_sha256'] == file_sha(recipe)
            and evidence['files_sha256'].get(str(recipe)) == evidence['recipe_sha256'],
            'Python v6 completed audit scientific seal or recipe differs')
    metadata = {key:value for key,value in evidence.items() if key != 'files_sha256'}
    require(seal.get('development_python_v6_execution') == metadata
            and all(seal['files_sha256'].get(source) == expected for source, expected in evidence['files_sha256'].items()),
            'confirmation omits or changes Python v6 completed audit evidence')
    return metadata


def validate_confirmation_seal(dataset_root, dataset_identity, *, require_execution_claim=True):
    """Independently bind all development and confirmation schedules to the seal.

    The pre-outcome structural/frozen-recipe APIs remain usable before this seal
    exists. Final outcome auditing additionally requires the canonical claim.
    """
    from evaluate_modebench_level3_independent import load_rows, schedule_record, validate_task
    folder = CAMPAIGN / 'confirmation_python_v6'
    seal_path, plan_path = folder / 'seal.json', folder / 'plan.json'
    seal_hash = file_sha(seal_path)
    seal = json.loads(seal_path.read_text())
    plan = json.loads(plan_path.read_text())
    require(seal.get('schema') == 'modebench_level3_independent_confirmation_input_seal_v2'
            and seal.get('plan') == str(plan_path.resolve()) and seal.get('plan_sha256') == file_sha(plan_path),
            'canonical confirmation seal/plan differs')
    require(seal.get('inherited_latest_development_seal') == str(LATEST_DEVELOPMENT_SEAL.resolve())
            and seal.get('inherited_latest_development_seal_sha256') == LATEST_DEVELOPMENT_SEAL_SHA256,
            'confirmation seal lacks the latest registered development chain')
    require(seal.get('confirmation_outcomes_loaded') is False
            and seal.get('treatment_training_started') is False
            and seal.get('fresh_heldout_claim_applies_to_level3_only') is True
            and seal.get('level1_controls_are_untouched') is False, 'confirmation seal information boundary differs')
    require(Path(plan['dataset_root']).resolve() == dataset_root.resolve()
            and set(plan['domains']) == set(DOMAINS)
            and plan['confirmation_draw_labels'] == list(DEFAULT_CONFIRMATION_SEEDS)
            and seal.get('confirmation_draw_labels') == list(DEFAULT_CONFIRMATION_SEEDS),
            'confirmation plan dataset/domains/labels differ')
    verify_pinned_files_and_trees(seal)
    recovery = validate_recovery_execution_binding(seal, plan)
    v6_execution = validate_python_v6_development_binding(seal, plan)
    development = authenticated_development_sources()
    latest = development['latest']
    require(seal['models'] == latest['models']
            and all(seal['files_sha256'].get(source) == digest for source, digest in latest['files_sha256'].items())
            and seal['files_sha256'].get(str(LATEST_DEVELOPMENT_SEAL.resolve())) == LATEST_DEVELOPMENT_SEAL_SHA256,
            'confirmation seal does not preserve latest development files/models')
    require(all(seal['files_sha256'].get(source) == digest for source, digest in plan['immutable_inputs_sha256'].items())
            and seal['files_sha256'].get(str(plan_path.resolve())) == file_sha(plan_path)
            and seal['files_sha256'].get(str(Path(__file__).resolve())) == file_sha(Path(__file__)),
            'confirmation plan or auditor is not pinned by the seal')
    require(sha(seal.get('development_sources')) == sha(development['manifest'])
            and sha(seal.get('development_protocols')) == sha(development['protocols'])
            and type(seal.get('development_request_blocks_checked_disjoint')) is int
            and seal['development_request_blocks_checked_disjoint'] == len(development['blocks']),
            'confirmation seal omits or changes registered development schedules')
    bundle_path = dataset_root / 'frozen_recipes.json'
    require(file_sha(bundle_path) == dataset_identity['frozen_recipe_bundle_sha256']
            == seal.get('frozen_recipe_bundle_sha256'), 'confirmation seal frozen recipe bundle differs')
    dataset_files = sorted(str(path.resolve()) for path in dataset_root.rglob('*') if path.is_file())
    require(seal['directory_files'].get(str(dataset_root.resolve())) == dataset_files
            and all(path in seal['files_sha256'] for path in dataset_files),
            'final dataset metadata and source files are not fully sealed')
    if require_execution_claim:
        claim_path = CAMPAIGN / 'confirmation_python_v6/confirmation_execution_claim.json'
        claim = json.loads(claim_path.read_text())
        require(claim.get('seal') == str(seal_path.resolve()) and claim.get('seal_sha256') == seal_hash
                and claim.get('plan') == str(plan_path.resolve()) and claim.get('plan_sha256') == file_sha(plan_path)
                and claim.get('amendment_sha256') == seal.get('amendment_sha256')
                and claim.get('jobs') == 2 * len(DOMAINS), 'confirmation execution claim differs from prospective seal')
    jobs = plan['jobs']
    require(len(jobs) == 2 * len(DOMAINS), 'all ten prospective confirmation cells required')
    cells, blocks, expected_sources, by_name = set(), set(), [], {}
    for job in jobs:
        domain, label = job['domain'], job['model_label']
        require(domain in DOMAINS and label in ('05b', '3b') and (domain, label) not in cells,
                'duplicate or unknown prospective confirmation cell')
        cells.add((domain, label))
        task_path = Path(job['tasks']).resolve()
        require(seal['files_sha256'].get(str(task_path)) == file_sha(task_path), 'confirmation task is not pinned')
        tasks = json.loads(task_path.read_text())
        require(isinstance(tasks, list) and len(tasks) == 1, 'confirmation task inventory differs')
        task = tasks[0]
        validate_task(task, True)
        expected_path = level1_eval_path(domain) if label == '05b' else dataset_root / domain / 'eval'
        require(task['domain'] == domain and task['level'] == ('level1' if label == '05b' else 'level3')
                and task['split'] == 'eval' and task['seeds'] == list(DEFAULT_CONFIRMATION_SEEDS)
                and task['batch_size'] == 8 and task['row_limit'] == task['row_offset'] == 0
                and task.get('dataset') == str(expected_path.resolve()) and not task.get('rows_jsonl')
                and task['output'] == job['output'] and job['name'] == f'{label}_{domain}',
                'prospective confirmation task/source/settings differ')
        rows, source = load_rows(task)
        require(len(rows) == source['selected_rows'] == source['total_rows'] == 128,
                'prospective confirmation source must contain all128 rows')
        if label == '3b':
            require(materializer_row_hash(rows) == dataset_identity['domains'][domain]['eval']['rows_sha256'],
                    'prospective final Level3 source differs from publication')
        schedule = schedule_record(domain, rows, task['seeds'])
        cell_blocks = [base for row in schedule['request_seeds'] for base in row]
        require(len(cell_blocks) == len(set(cell_blocks)) == 512, 'prospective confirmation cell sampling coverage differs')
        require(not set(cell_blocks) & (development['blocks'] | blocks),
                'prospective confirmation RNG blocks overlap development or another confirmation source')
        blocks.update(cell_blocks)
        inventory = sorted(str(path.resolve()) for path in expected_path.rglob('*') if path.is_file())
        require(inventory and seal['directory_files'].get(str(expected_path.resolve())) == inventory
                and all(path in seal['files_sha256'] for path in inventory), 'confirmation source files are not fully sealed')
        record = {'name': job['name'], 'domain': domain, 'model_label': label, 'identity': source,
                  'seed_schedule_sha256': sha(schedule), 'distinct_request_blocks': len(cell_blocks),
                  'distinct_child_seeds': len(cell_blocks) * 8}
        expected_sources.append(record)
        by_name[job['name']] = {**record, 'output': str(Path(job['output']).resolve())}
    require(sha(seal.get('sources')) == sha(expected_sources)
            and seal.get('distinct_request_blocks') == len(blocks) == 5120
            and seal.get('distinct_child_seeds') == len(blocks) * 8 == 40960,
            'confirmation sealed source schedules or full coverage differ')
    verify_pinned_files_and_trees(seal)
    require(file_sha(LATEST_DEVELOPMENT_SEAL) == LATEST_DEVELOPMENT_SEAL_SHA256
            and file_sha(seal_path) == seal_hash, 'confirmation/development seal changed during validation')
    return {'path': str(seal_path.resolve()), 'sha256': seal_hash, 'plan_sha256': file_sha(plan_path),
            'latest_development_seal_sha256': development['latest_sha256'],
            'development_execution_recovery': recovery, 'development_python_v6_execution': v6_execution,
            'development_sources': len(development['manifest']), 'development_request_blocks': len(development['blocks']),
            'confirmation_sources': len(expected_sources), 'confirmation_request_blocks': len(blocks),
            'confirmation_child_seeds': len(blocks) * 8, 'sources': by_name, 'models': seal['models']}


def verify_receipt_matches_confirmation_seal(receipt, path, evidence, domain, role):
    label = '05b' if role == 'baseline' else '3b'
    expected = evidence['sources'][f'{label}_{domain}']
    require(str(Path(path).resolve()) == expected['output']
            and receipt['identity']['source'] == expected['identity']
            and receipt['identity']['seed_schedule_sha256'] == expected['seed_schedule_sha256']
            and receipt['identity']['model'] == evidence['models'][label],
            f'{domain}/{role}: receipt differs from prospective confirmation source/schedule/model')


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
    actual_blocks = {base for row in identity['seed_schedule']['request_seeds'] for base in row}
    for source in [provenance['baseline_receipt_path'],
                   *[entry['receipt_path'] for entry in provenance['pools'].values()]]:
        development_receipt = json.loads(Path(source).read_text())
        development_blocks = {base for row in development_receipt['identity']['seed_schedule']['request_seeds'] for base in row}
        require(not actual_blocks & development_blocks,
                f'{prefix}: effective RNG blocks overlap development')


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
    require(development or expected_eval_seeds == list(DEFAULT_CONFIRMATION_SEEDS),
            'v2 confirmation draw labels must equal the prospective registration')
    require(bootstrap_replicates >= 100, 'at least 100 bootstrap replicates are required')
    domains = [pair.get('domain') for pair in pairs]
    require(len(domains) == len(set(domains)), 'duplicate domain pairs')
    require(set(domains) <= set(DOMAINS), 'unknown domain in pairs')
    dataset_identity = None
    identity_path = dataset_root / 'identity.json' if dataset_root else None
    if identity_path:
        dataset_identity = json.loads(identity_path.read_text())
        validate_publication_metadata(dataset_identity, dataset_root)
        verify_all_split_sources(dataset_root, dataset_identity)
    confirmation_evidence = (validate_confirmation_seal(dataset_root, dataset_identity)
                             if dataset_root and not development else None)
    results, errors, confirmation_blocks = {}, {}, set()
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
                    verify_receipt_matches_confirmation_seal(receipt, path, confirmation_evidence, domain, role)
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
            if not development:
                left_blocks = {base for row in left['seed_schedule']['request_seeds'] for base in row}
                right_blocks = {base for row in right['seed_schedule']['request_seeds'] for base in row}
                require(not left_blocks & right_blocks, f'{domain}: baseline/candidate RNG block overlap')
                require(not (left_blocks | right_blocks) & confirmation_blocks,
                        f'{domain}: effective RNG blocks overlap another confirmation domain')
                confirmation_blocks |= left_blocks | right_blocks
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
        except (KeyError, TypeError, ValueError, OSError, RuntimeError) as error:
            errors[domain] = str(error)
    missing = [domain for domain in DOMAINS if domain not in domains]
    complete = not missing and not errors and set(results) == set(DOMAINS)
    matched = complete and all(row['observed_approximate_match'] for row in results.values())
    status = ('invalid_evidence' if errors else 'incomplete' if missing else
              'observed_approximate_match' if matched else 'outside_match_tolerance')
    return {'schema': SCHEMA, 'generated_at': datetime.now(timezone.utc).isoformat(),
            'status': status, 'phase': 'development' if development else 'confirmation',
            'all_five_domains_complete': complete, 'all_five_observed_approximate_match': matched,
            'prospective_confirmation_seal': ({key: value for key, value in confirmation_evidence.items()
                                                if key not in ('sources', 'models')} if confirmation_evidence else None),
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
                                    'independent_seed_schedules_recomputed': True,
                                    'legacy_overlapping_seed_evidence_rejected': True,
                                    'level1_controls_reused_with_corrected_sampling': True,
                                    'fresh_heldout_claim_applies_to_level3_only': True,
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
                        help='four seeds registered before confirmation (default: 6329000..6329003)')
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
