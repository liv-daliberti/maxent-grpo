#!/usr/bin/env python3
"""Freeze independent-RNG recipes and generate fresh Level 3 revision-2 splits.

This program reads development recipes and candidate rows, never evaluation
model outcomes. A materialized dataset remains pending fresh confirmation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / 'ops', ROOT / 'ops/exp_scaling', ROOT / 'src'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from datasets import Dataset, DatasetDict, load_from_disk
from fit_modebench_level3 import (
    SCHEMA as RECIPE_SCHEMA, SELECTION_ALGORITHM, TOLERANCES, cell_histogram, file_sha, generator_sources,
    hamilton, allocate_cells, read_jsonl, receipt_rows, select_rows, serialize_cells, sha,
)
from materialize_modebench_level3 import (
    DEFAULT_OUTPUT, DOMAINS, SEEDS, SPLITS, LEVEL1, REFERENCE, generator, historical_ids,
    identity_set, modes, reference_rows, row_hash, verify_rows, existing_ids,
)

from fit_modebench_level3_independent import fit_recipe as refit_independent

SCHEMA = 'modebench_level3_capability_matched_splits_independent_v2'
BASELINE_CONTROLS = {domain: Path(LEVEL1[domain]['eval']).resolve() for domain in DOMAINS}
PRIOR_REVISION = ROOT / 'var/data/modebench_level3_matched_v1'
GENERATION_SEED_OFFSET = 1_000_000
AMENDMENT_SHA256 = 'cd8168785e9d469d7d715123ff0120170e9091cd6ce7aad3d3c4ceb3aff93dd6'


def candidate_route(recipe, domain):
    """Resolve the explicit revised range; original recipes retain their route."""
    if 'candidate_revision' not in recipe:
        return None
    revision = recipe['candidate_revision']
    if not isinstance(revision, dict):
        raise ValueError('unknown or changed independent candidate revision')
    if revision.get('name') == 'graph_v7':
        import fit_modebench_level3_graph_v7_independent as revised
    elif revision.get('name') == 'python_v5':
        import fit_modebench_level3_python_v5_independent as revised
    else:
        raise ValueError('unknown or changed independent candidate revision')
    revised.validate_revision(recipe, domain)
    return revised


def recipe_generator_sources(domain, recipe):
    route = candidate_route(recipe, domain)
    return generator_sources(domain) if route is None else route.generator_sources(domain)


def recipe_generator(domain, recipe):
    route = candidate_route(recipe, domain)
    return generator(domain) if route is None else route.generator(domain)


def validate_protocol_amendment(path):
    """Authenticate the prospective fixed-control decision and its source pins.

    This specific amendment predates v2 submission. Re-authoring a new document
    with internally consistent pins cannot substitute a post-outcome decision.
    No model outcomes are loaded here.
    """
    path = Path(path).resolve()
    before = file_sha(path)
    if before != AMENDMENT_SHA256:
        raise ValueError('unregistered prospective protocol amendment')
    amendment = json.loads(path.read_text())
    if (amendment.get('schema') != 'modebench_level3_v2_fixed_control_amendment_v1'
            or amendment.get('status') != 'prospective_before_any_v2_model_outcomes'
            or amendment.get('decision') != 'reuse_original_fixed_level1_evaluation_controls_with_new_independent_draws; generate_fresh_level3_confirmation_only'):
        raise ValueError('fixed-control amendment decision or status differs')
    boundary = {
        'fresh_heldout_claim_applies_to_level3_only': True,
        'legacy_confirmation_scores_used_for_v2_recipe_ranking': False,
        'level1_controls_are_untouched': False,
        'level3_eval_generated_after_all_five_passing_recipes_frozen': True,
        'treatment_training_started': False,
        'v1_correlated_receipts_retained_as_diagnostic': True,
        'v2_model_outcomes_exist': False,
        'v2_submission_claim_exists': False,
    }
    if (set(amendment.get('information_boundary', {})) != set(boundary)
            or any(amendment['information_boundary'][key] is not value for key, value in boundary.items())):
        raise ValueError('fixed-control amendment information boundary differs')
    unchanged = amendment.get('unchanged', {})
    flags = ('calibration_tasks_and_sources', 'development_only_mixture_fitting', 'frozen_models',
             'level1_generation_law_and_data', 'sampling_interface_except_already_registered_rng_correction',
             'verifier_and_canonicalizer')
    if (any(unchanged.get(key) is not True for key in flags)
            or unchanged.get('match_tolerances') != TOLERANCES
            or unchanged.get('split_sizes') != {split: spec[0] for split, spec in SPLITS.items()}):
        raise ValueError('fixed-control amendment changes frozen scientific settings')
    for key in ('protocol', 'calibration_seal', 'capacity_failure'):
        if file_sha(amendment[f'{key}_path']) != amendment[f'{key}_sha256']:
            raise ValueError(f'fixed-control amendment {key} source changed')
    seal = json.loads(Path(amendment['calibration_seal_path']).read_text())
    prior_prefix = str(PRIOR_REVISION.resolve()) + '/'
    original_prior = {source: digest for source, digest in seal.get('files_sha256', {}).items()
                      if source.startswith(prior_prefix)}
    current_prior = {str(source.resolve()): file_sha(source)
                     for source in sorted(PRIOR_REVISION.rglob('*')) if source.is_file()}
    if not original_prior or original_prior != current_prior:
        raise ValueError('prior Level 3 publication differs from prospective calibration seal')
    expected_paths = {domain: str(directory) for domain, directory in BASELINE_CONTROLS.items()}
    if (amendment.get('baseline_confirmation_paths') != expected_paths
            or set(amendment.get('controls', {})) != set(DOMAINS)):
        raise ValueError('fixed-control amendment must bind all five original Level 1 controls')
    combined_pins = {}
    for domain, directory in BASELINE_CONTROLS.items():
        record = amendment['controls'][domain]
        current = {str(source.resolve()): file_sha(source)
                   for source in sorted(directory.rglob('*')) if source.is_file()}
        if (record.get('path') != str(directory) or record.get('rows') != 128
                or not current or current != record.get('files_sha256')):
            raise ValueError(f'{domain}: fixed-control amendment file pins differ')
        rows = [dict(row) for row in load_from_disk(str(directory))['multi_answer']]
        if (len(rows) != 128 or len(identity_set(domain, rows)) != 128
                or row_hash(rows) != record.get('rows_sha256')):
            raise ValueError(f'{domain}: fixed-control amendment rows differ')
        combined_pins.update(current)
    if amendment.get('fixed_control_files_sha256') != combined_pins:
        raise ValueError('fixed-control amendment combined file inventory differs')
    if file_sha(path) != before:
        raise ValueError('prospective protocol amendment changed during validation')
    return amendment


def baseline_and_prior_pins():
    paths = [PRIOR_REVISION / 'identity.json']
    for domain, directory in BASELINE_CONTROLS.items():
        if not (directory / 'dataset_dict.json').is_file():
            raise ValueError(f'fixed Level 1 confirmation control is missing: {domain}')
        paths.extend(path for path in directory.rglob('*') if path.is_file())
    paths.extend(path for path in PRIOR_REVISION.rglob('*') if path.is_file())
    return {str(path.resolve()): file_sha(path) for path in sorted(set(paths))}


def authenticated_exclusion_rows(domain):
    """Authenticate prior splits and load the registered fixed Level 1 controls."""
    prior = json.loads((PRIOR_REVISION / 'identity.json').read_text())
    if (prior.get('schema') != 'modebench_level3_capability_matched_splits_v1'
            or prior.get('status') != 'structural_checks_pass'):
        raise ValueError('authenticated prior Level 3 v1 publication required')
    bundle_path = PRIOR_REVISION / 'frozen_recipes.json'
    if file_sha(bundle_path) != prior.get('frozen_recipe_bundle_sha256'):
        raise ValueError('prior Level 3 frozen bundle hash mismatch')
    bundle = json.loads(bundle_path.read_text())
    if bundle.get('recipes_sha256') != prior.get('recipe_sha256'):
        raise ValueError('prior Level 3 frozen recipe bindings disagree')
    for prior_domain, digest in prior['recipe_sha256'].items():
        if file_sha(PRIOR_REVISION / 'recipes' / f'{prior_domain}.json') != digest:
            raise ValueError('prior Level 3 archived recipe changed')
    def authenticated_rows(root, split, record, expected, dataset_split):
        path = root / domain / split
        rows = [dict(row) for row in load_from_disk(str(path))[dataset_split]]
        if (len(rows) != expected or record.get('rows') != expected
                or row_hash(rows) != record.get('rows_sha256')
                or Path(record.get('path', '')).resolve() != path.resolve()
                or len(identity_set(domain, rows)) != expected):
            raise ValueError(f'{domain}/{split}: exclusion rows differ from published identity')
        return rows
    prior_rows = {}
    for split, (expected, dataset_split) in SPLITS.items():
        record = prior['domains'][domain][split]
        if not record.get('checks') or not all(value is True for value in record['checks'].values()):
            raise ValueError('prior Level 3 structural certificate did not pass')
        prior_rows[split] = authenticated_rows(PRIOR_REVISION, split, record, expected, dataset_split)
    baseline_rows = [dict(row) for row in load_from_disk(str(BASELINE_CONTROLS[domain]))['multi_answer']]
    if (len(baseline_rows) != 128 or len(identity_set(domain, baseline_rows)) != 128
            or modes(baseline_rows) != modes(reference_rows(domain, 'eval'))):
        raise ValueError('fixed Level 1 control size, uniqueness, or support histogram differs')
    return prior_rows, baseline_rows


def other_historical_ids(domain):
    """Keep every historical identity except the prior Level 3 revision."""
    blocked = existing_ids(domain)
    data_root = ROOT / 'var/data'
    roots = set(data_root.glob('modebench_harder*')) | set(data_root.glob('modebench_level3*'))
    for root in sorted(roots):
        if not root.is_dir() or root.resolve() == PRIOR_REVISION.resolve():
            continue
        for split in SPLITS:
            path = root / domain / split
            if path.is_dir() and (path / 'dataset_dict.json').exists():
                for subset in load_from_disk(str(path)).values():
                    blocked |= identity_set(domain, [dict(row) for row in subset])
    return blocked


def exclusion_sets(other_history, prior_ids, baseline_ids, pool_ids):
    """Only prior development may be reused; other occurrences stay protected."""
    protected = other_history | prior_ids['train'] | prior_ids['eval'] | baseline_ids
    return protected, protected | prior_ids['dev'] | pool_ids


def recipe_snapshot(path, domain):
    before = file_sha(path)
    recipe = load_recipe(path, domain)
    if file_sha(path) != before:
        raise ValueError('recipe bytes changed during exact development refit')
    return recipe, before


def development_input_snapshot(recipes, paths, recipe_hashes):
    """Pin authenticated receipts, pool certificates, code, and baseline sources."""
    pins, trees = {}, {}
    def pin(path, expected):
        path = Path(path).resolve()
        if file_sha(path) != expected:
            raise ValueError(f'authenticated development input changed: {path}')
        key = str(path)
        if key in pins and pins[key] != expected:
            raise ValueError('development inputs bind conflicting hashes')
        pins[key] = expected
    for domain, recipe in recipes.items():
        provenance = recipe['provenance']
        route = candidate_route(recipe, domain)
        if route is not None:
            registered = route.registration()['source_snapshot']
            for source, expected in registered['files_sha256'].items():
                pin(source, expected)
            for directory, files in registered['directory_files'].items():
                if directory in trees and trees[directory] != files:
                    raise ValueError('candidate registration binds conflicting source inventory')
                trees[directory] = files
        pin(paths[domain], recipe_hashes[domain])
        pin(provenance['baseline_receipt_path'], provenance['baseline_receipt_sha256'])
        receipt_paths = [Path(provenance['baseline_receipt_path'])]
        for pool in provenance['pools'].values():
            for path_key, hash_key in (('receipt_path', 'receipt_sha256'), ('rows_path', 'rows_file_sha256'),
                                       ('pool_identity_path', 'pool_identity_sha256')):
                pin(pool[path_key], pool[hash_key])
            receipt_paths.append(Path(pool['receipt_path']))
        for source, expected in provenance['evaluator_and_verifier_code_sha256'].items():
            pin(ROOT / source, expected)
        for source, expected in provenance['generator_sources_sha256'].items():
            pin(ROOT / source, expected)
        pin(ROOT / 'ops/exp_scaling/fit_modebench_level3.py', provenance['fitter_source_sha256'])
        pin(ROOT / 'ops/exp_scaling/fit_modebench_level3_independent.py', provenance['independent_fitter_source_sha256'])
        for receipt_path in receipt_paths:
            receipt = json.loads(receipt_path.read_text())
            source = receipt['identity']['source']
            path = Path(source['path']).resolve()
            if source['kind'] == 'saved_dataset':
                files = sorted(str(p.resolve()) for p in path.rglob('*') if p.is_file())
                trees[str(path)] = files
                for name in files:
                    pin(name, file_sha(name))
            else:
                pin(path, source['file_sha256'])
            receipt_rows(receipt)
        for split in SPLITS:
            directory = (REFERENCE / domain / split).resolve()
            files = sorted(str(p.resolve()) for p in directory.rglob('*') if p.is_file())
            if not files:
                raise ValueError('Level 2 reference dataset is missing')
            trees[str(directory)] = files
            for name in files:
                pin(name, file_sha(name))
    snapshot = {'files_sha256': pins, 'directory_files': trees}
    verify_development_inputs_unchanged(snapshot)
    return snapshot


def verify_development_inputs_unchanged(snapshot):
    for source, expected in snapshot['files_sha256'].items():
        if file_sha(source) != expected:
            raise ValueError(f'development source or code changed during generation: {source}')
    for directory, expected in snapshot['directory_files'].items():
        current = sorted(str(path.resolve()) for path in Path(directory).rglob('*') if path.is_file())
        if current != expected:
            raise ValueError(f'development source inventory changed during generation: {directory}')


def _write_json(path, payload):
    with Path(path).open('x') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write('\n')


def load_recipe(path, domain):
    recipe = json.loads(Path(path).read_text())
    if recipe.get('schema') != RECIPE_SCHEMA or recipe.get('domain') != domain:
        raise ValueError(f'wrong recipe schema/domain: {path}')
    if len(set(recipe['provenance'].get('seeds', []))) != 4:
        raise ValueError('final recipes require four calibration seeds')
    if not recipe.get('development_fit_pass'):
        raise ValueError(f'development fit did not pass: {domain}')
    selection = recipe.get('selection', {})
    if (selection.get('algorithm') != SELECTION_ALGORITHM
            or selection.get('weight_objective') != 'full_pool_cell_means_at_exact_controlled_allocation'
            or selection.get('selected_residual_used_for_ranking') is not False
            or selection.get('on_selected_gate_failure') != 'fail_without_alternate_weights_or_hash_seeds'):
        raise ValueError('final recipes require the full-pool forecast-only selection algorithm')
    development = recipe.get('development', {})
    if development.get('tolerances') != TOLERANCES or development.get('selected_development_sets_scored') != 1:
        raise ValueError('final recipes require unchanged tolerances and one scored selected set')
    gates = development.get('gates', {})
    if set(gates) != {'expected', 'selected'}:
        raise ValueError('final recipes require explicit expected and selected development gates')
    for gate, metric_field, delta_field in (('expected', 'expected_metrics', 'expected_differences'),
                                            ('selected', 'selected_metrics', 'differences')):
        for metric, tolerance in TOLERANCES.items():
            value = development.get(metric_field, {}).get(metric)
            baseline = development.get('baseline_metrics', {}).get(metric)
            delta = development.get(delta_field, {}).get(metric)
            if (not all(isinstance(x, (int, float)) and math.isfinite(x) for x in (value, baseline, delta))
                    or not math.isclose(value - baseline, delta, rel_tol=0, abs_tol=1e-12)
                    or abs(delta) > tolerance or gates[gate].get(metric) is not True):
                raise ValueError(f'{gate} development gate did not pass for {metric}')
    if recipe['information_boundary'].get('confirmation_outcomes_used', True):
        raise ValueError('a recipe selected with confirmation outcomes cannot be frozen')
    if recipe['provenance']['generator_sources_sha256'] != recipe_generator_sources(domain, recipe):
        raise ValueError(f'{domain} generator sources changed after recipe fitting')
    if recipe['provenance']['fitter_source_sha256'] != file_sha(ROOT / 'ops/exp_scaling/fit_modebench_level3.py'):
        raise ValueError('mixture allocation code changed after recipe fitting')
    # Refit only the authenticated v2 development receipts and require every
    # recorded choice to reproduce. This does not inspect confirmation results.
    route = candidate_route(recipe, domain)
    refit = refit_independent if route is None else route.fit_recipe
    authenticated = refit(
        recipe['provenance']['baseline_receipt_path'],
        [recipe['provenance']['pools'][str(tier)]['receipt_path'] for tier in range(4)],
        domain,
    )
    # JSON object keys are strings on disk; compare the exact serialized value.
    if recipe != json.loads(json.dumps(authenticated, allow_nan=False)):
        raise ValueError('independent development recipe does not reproduce exactly')
    # Validate mixture units even when a particular split has no rare cells.
    hamilton(1, recipe['weight_units'])
    return recipe


def selected_development(domain, recipe):
    pools = {}
    for difficulty in range(4):
        provenance = recipe['provenance']['pools'][str(difficulty)]
        path = Path(provenance['rows_path'])
        if file_sha(path) != provenance['rows_file_sha256']:
            raise ValueError('candidate pool bytes changed after fitting')
        rows = read_jsonl(path)
        if row_hash(rows) != provenance['rows_sha256']:
            raise ValueError('candidate pool row hash changed after fitting')
        if file_sha(provenance['pool_identity_path']) != provenance['pool_identity_sha256']:
            raise ValueError('candidate pool certificate changed after fitting')
        pools[difficulty] = rows
    reference = reference_rows(domain, 'dev')
    if row_hash(reference) != recipe['development']['reference_rows_sha256']:
        raise ValueError('development support reference changed after fitting')
    selected = select_rows(domain, pools, cell_histogram(domain, reference),
                           recipe['weight_units'], recipe['selection']['seed'])
    selected_ids = [{key: value for key, value in item.items() if key != 'row'} for item in selected]
    if selected_ids != recipe['selected_development']:
        raise ValueError('selected development rows do not reproduce from frozen weights')
    rows = [item['row'] for item in selected]
    if row_hash(rows) != recipe['development']['rows_sha256']:
        raise ValueError('selected development row hash mismatch')
    return rows


def candidate_pool_files(domain, recipe):
    paths = {Path(value['rows_path']).resolve()
             for value in recipe['provenance']['pools'].values()}
    # Exclude unsuccessful candidate pools too, not merely the four fitted pools.
    directories = {path.parent for path in paths}
    directories.add(DEFAULT_OUTPUT / 'pools' / domain)
    directories.update(root / 'pools' / domain
                       for root in (ROOT / 'var/data').glob('modebench_level3_calibration*')
                       if root.is_dir())
    for directory in directories:
        paths.update(path.resolve() for path in directory.glob('*.jsonl'))
    return sorted(paths)


def build_fresh_split(domain, split, recipe, blocked):
    reference = reference_rows(domain, split)
    expected = SPLITS[split][0]
    if len(reference) != expected:
        raise ValueError(f'{domain}/{split} Level 2 reference size changed')
    target = cell_histogram(domain, reference)
    assigned = [Counter() for _ in range(4)]
    allocation = allocate_cells(target, recipe['weight_units'], recipe['selection']['seed'])
    for key, count in target.items():
        for difficulty, required in enumerate(allocation[key]):
            if required:
                assigned[difficulty][key] = required
    rows, local_blocked = [], set(blocked)
    base_seed = SEEDS[domain] + GENERATION_SEED_OFFSET + (100_000 if split == 'train' else 200_000)
    for difficulty, cells in enumerate(assigned):
        if not cells:
            continue
        support_target = Counter()
        for key, count in cells.items():
            support_target[key[0]] += count
        extras = {'joint_target': cells} if domain == 'pantry' else {}
        generated = recipe_generator(domain, recipe)(
            domain, support_target, local_blocked, base_seed + 1000 * difficulty,
            f'level3_{split}', difficulty, multiplier=1, **extras,
        )
        if cell_histogram(domain, generated) != cells:
            raise RuntimeError(f'{domain}/{split}/{difficulty} generated support/family drift')
        local_blocked |= identity_set(domain, generated)
        rows.extend(generated)
    rows.sort(key=lambda row: sha([base_seed, 'split_order', sha(row)]))
    checks = verify_rows(domain, rows, reference, modes(reference), blocked)
    checks['exact_difficulty_cell_recipe'] = all(
        cell_histogram(domain, [row for row in rows if row['level3_difficulty'] == difficulty]) == cells
        for difficulty, cells in enumerate(assigned)
    )
    checks['exact_global_tier_quota'] = [sum(cells.values()) for cells in assigned] == hamilton(expected, recipe['weight_units'], (), recipe['selection']['seed'])
    checks['exact_joint_support_family_histogram'] = cell_histogram(domain, rows) == target
    if not all(checks.values()):
        raise RuntimeError(f'{domain}/{split} final recipe checks failed: {checks}')
    return rows, checks, base_seed



def verify_generation_inputs_unchanged(recipes, frozen, pool_records, history_records):
    """Refuse publication if another worker changed an input during generation."""
    if file_sha(Path(__file__)) != frozen['finalizer_source_sha256']:
        raise ValueError('finalizer source changed during generation')
    for domain, recipe in recipes.items():
        snapshot = frozen['generator_sources_sha256'][domain]
        if (snapshot != recipe['provenance']['generator_sources_sha256']
                or recipe_generator_sources(domain, recipe) != snapshot):
            raise ValueError(f'{domain} generator sources changed during generation')
        if file_sha(ROOT / 'ops/exp_scaling/fit_modebench_level3.py') != recipe['provenance']['fitter_source_sha256']:
            raise ValueError('fitter source changed during generation')
        current = {str(path.resolve()): file_sha(path)
                   for path in candidate_pool_files(domain, recipe)}
        recorded = {str(Path(item['path']).resolve()): item['file_sha256']
                    for item in pool_records[domain]}
        if current != recorded:
            raise ValueError(f'{domain} candidate pools changed during generation')
        if historical_ids(domain) != history_records[domain]:
            raise ValueError(f'{domain} historical identities changed during generation')


def finalize(recipes_mapping, output, protocol_amendment):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'fresh output root required: {output}')
    if set(recipes_mapping) != set(DOMAINS):
        raise ValueError('recipes mapping must cover all five domains exactly')
    initial_finalizer_sha256 = file_sha(Path(__file__))
    protocol_amendment = Path(protocol_amendment).resolve()
    validate_protocol_amendment(protocol_amendment)
    amendment_sha256 = file_sha(protocol_amendment)
    prior_and_fixed_control_pins = baseline_and_prior_pins()
    paths = {domain: Path(path).resolve() for domain, path in recipes_mapping.items()}
    snapshots = {domain: recipe_snapshot(paths[domain], domain) for domain in DOMAINS}
    recipes = {domain: value[0] for domain, value in snapshots.items()}
    recipe_hashes = {domain: value[1] for domain, value in snapshots.items()}
    development_inputs = development_input_snapshot(recipes, paths, recipe_hashes)
    if file_sha(Path(__file__)) != initial_finalizer_sha256:
        raise ValueError('finalizer changed before recipe freeze')
    # Snapshot all choices before generating any held-out prompt. This immutable
    # bundle, rather than any evaluation outcome, determines every split.
    frozen = {
        'schema': 'modebench_level3_frozen_recipe_bundle_independent_v2',
        'prior_and_fixed_control_files_sha256': prior_and_fixed_control_pins,
        'baseline_confirmation_paths': {domain: str(path) for domain, path in BASELINE_CONTROLS.items()},
        'prospective_amendment_path': str(protocol_amendment),
        'prospective_amendment_sha256': amendment_sha256,
        'generation_seed_offset': GENERATION_SEED_OFFSET,
        'development_inputs': development_inputs,
        'independent_fitter_source_sha256': file_sha(ROOT / 'ops/exp_scaling/fit_modebench_level3_independent.py'),
        'recipes_sha256': recipe_hashes,
        'generator_sources_sha256': {domain: recipe_generator_sources(domain, recipes[domain]) for domain in DOMAINS},
        'finalizer_source_sha256': file_sha(Path(__file__)),
        'confirmation_outcomes_used': False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.' + output.name + '.', dir=output.parent))
    records, pool_records, history_records = {}, {}, {}
    try:
        (staging / 'recipes').mkdir()
        for domain in DOMAINS:
            # Use bytes already authenticated above, refusing concurrent edits.
            if file_sha(paths[domain]) != recipe_hashes[domain]:
                raise ValueError('recipe changed while freezing')
            archived = staging / 'recipes' / f'{domain}.json'
            shutil.copyfile(paths[domain], archived)
            if file_sha(archived) != recipe_hashes[domain]:
                raise ValueError('recipe changed while copying frozen bytes')
        _write_json(staging / 'frozen_recipes.json', frozen)
        frozen_hash = file_sha(staging / 'frozen_recipes.json')
        for domain in DOMAINS:
            print(f'[level3] finalizing {domain}', flush=True)
            recipe = recipes[domain]
            historical = historical_ids(domain)
            history_records[domain] = set(historical)
            prior_rows, baseline_rows = authenticated_exclusion_rows(domain)
            prior_ids = {split: identity_set(domain, rows) for split, rows in prior_rows.items()}
            baseline_ids = identity_set(domain, baseline_rows)
            other_history = other_historical_ids(domain)
            pool_ids = set()
            pool_records[domain] = []
            for path in candidate_pool_files(domain, recipe):
                digest = file_sha(path)
                candidate_rows = read_jsonl(path)
                if file_sha(path) != digest:
                    raise ValueError(f'{domain} candidate pool changed while reading')
                pool_ids |= identity_set(domain, candidate_rows)
                pool_records[domain].append({'path': str(path), 'file_sha256': digest, 'rows': len(candidate_rows)})
            development = selected_development(domain, recipe)
            dev_reference = reference_rows(domain, 'dev')
            # Prior development remains development; prior train/eval and all
            # fixed Level 1 controls remain excluded from this split.
            development_blocked, blocked = exclusion_sets(other_history, prior_ids, baseline_ids, pool_ids)
            dev_checks = verify_rows(domain, development, dev_reference, modes(dev_reference), development_blocked)
            dev_checks['prior_train_eval_and_fixed_baseline_disjoint'] = not (identity_set(domain, development) & development_blocked)
            dev_checks['exact_joint_support_family_histogram'] = cell_histogram(domain, development) == cell_histogram(domain, dev_reference)
            dev_checks['frozen_selected_development_rows'] = row_hash(development) == recipe['development']['rows_sha256']
            if not all(dev_checks.values()):
                raise RuntimeError('frozen development structural checks failed')
            # Fresh train/eval exclude every candidate, including selected dev.
            built = {'dev': (development, dev_checks, None)}
            for split in ('train', 'eval'):
                rows, checks, seed = build_fresh_split(domain, split, recipe, blocked)
                blocked |= identity_set(domain, rows)
                built[split] = rows, checks, seed
            records[domain] = {}
            split_ids = {split: identity_set(domain, value[0]) for split, value in built.items()}
            if any(split_ids[a] & split_ids[b] for a, b in (('train', 'dev'), ('train', 'eval'), ('dev', 'eval'))):
                raise RuntimeError(f'{domain} final split overlap')
            for split, (expected, dataset_split) in SPLITS.items():
                rows, checks, seed = built[split]
                if len(rows) != expected:
                    raise RuntimeError('final split size mismatch')
                destination = staging / domain / split
                DatasetDict({dataset_split: Dataset.from_list(rows)}).save_to_disk(str(destination))
                records[domain][split] = {
                    'rows': len(rows), 'rows_sha256': row_hash(rows),
                    'path': str(output / domain / split), 'dataset_split': dataset_split,
                    'checks': {**checks, 'all_final_splits_disjoint': True},
                    'support_histogram': dict(sorted(modes(rows).items())),
                    'joint_cells': serialize_cells(cell_histogram(domain, rows)),
                    'difficulty_counts': dict(Counter(row['level3_difficulty'] for row in rows)),
                    'seed': seed,
                }
        identity = {
            'schema': SCHEMA, 'status': 'structural_checks_pass',
            'decision': 'pending_confirmation',
            'split_sizes': {split: spec[0] for split, spec in SPLITS.items()},
            'domains': records, 'recipe_sha256': recipe_hashes,
            'generator_sources_sha256': frozen['generator_sources_sha256'],
            'frozen_recipe_bundle_sha256': frozen_hash,
            'excluded_candidate_pools': pool_records,
            'prior_and_fixed_control_files_sha256': prior_and_fixed_control_pins,
            'baseline_confirmation_paths': {domain: str(path) for domain, path in BASELINE_CONTROLS.items()},
            'prospective_amendment_path': str(protocol_amendment),
            'prospective_amendment_sha256': amendment_sha256,
            'generation_seed_offset': GENERATION_SEED_OFFSET,
            'fairness_contract': {
                'support_histograms': 'exactly match Level 2 r5 in every split',
                'pantry_support_family_joint_histograms': 'exactly match Level 2 r5 in every split',
                'verifier_and_canonicalization': 'unchanged',
                'fresh_train_and_eval': 'disjoint from all Level 1, all Level 2, all candidate pools, prior Level 3 data, fixed Level 1 confirmation controls, and each other',
                'development': 'frozen selected candidate rows; no model-outcome-based ordering within support cells',
            },
            'information_boundary': {
                'recipe_frozen_before_eval_generation': True,
                'evaluation_model_outcomes_loaded': False,
                'treatment_training_started': False,
                'interpretation': 'Dataset construction is complete; difficulty equivalence requires fresh confirmation under the frozen interface.',
            },
        }
        verify_generation_inputs_unchanged(recipes, frozen, pool_records, history_records)
        verify_development_inputs_unchanged(development_inputs)
        for domain in DOMAINS:
            if file_sha(staging / 'recipes' / f'{domain}.json') != recipe_hashes[domain]:
                raise ValueError('archived recipe changed during generation')
        if prior_and_fixed_control_pins != baseline_and_prior_pins():
            raise ValueError('prior revision or fixed baseline control changed during generation')
        if file_sha(protocol_amendment) != amendment_sha256:
            raise ValueError('prospective protocol amendment changed during generation')
        validate_protocol_amendment(protocol_amendment)
        if frozen['independent_fitter_source_sha256'] != file_sha(ROOT / 'ops/exp_scaling/fit_modebench_level3_independent.py'):
            raise ValueError('independent fitter changed during generation')
        _write_json(staging / 'identity.json', identity)
        _write_json(staging / 'admission_fairness_report.json', identity)
        # Directory rename publishes the complete dataset only after all checks.
        if output.exists():
            raise FileExistsError(output)
        staging.rename(output)
        return identity
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipes-json', required=True, type=Path,
                        help='JSON object mapping all five domain names to fitted recipe paths')
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--protocol-amendment', required=True, type=Path,
                        help='immutable prospective amendment authorizing fixed Level 1 confirmation controls')
    args = parser.parse_args()
    mapping = json.loads(args.recipes_json.read_text())
    mapping = {domain: str((args.recipes_json.parent / path).resolve()) if not Path(path).is_absolute() else path
               for domain, path in mapping.items()}
    result = finalize(mapping, args.output_root, args.protocol_amendment)
    print(json.dumps({'decision': result['decision'], 'split_sizes': result['split_sizes'],
                      'output_root': str(args.output_root.resolve())}, sort_keys=True))


if __name__ == '__main__':
    main()
