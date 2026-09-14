#!/usr/bin/env python3
"""Freeze fitted Level 3 recipes and generate support-matched train/dev/eval splits.

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

from datasets import Dataset, DatasetDict
from fit_modebench_level3 import (
    SCHEMA as RECIPE_SCHEMA, SELECTION_ALGORITHM, TOLERANCES, cell_histogram, file_sha, generator_sources,
    hamilton, allocate_cells, read_jsonl, select_rows, serialize_cells, sha,
)
from materialize_modebench_level3 import (
    DEFAULT_OUTPUT, DOMAINS, SEEDS, SPLITS, generator, historical_ids,
    identity_set, modes, reference_rows, row_hash, verify_rows,
)

SCHEMA = 'modebench_level3_capability_matched_splits_v1'


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
    if recipe['provenance']['generator_sources_sha256'] != generator_sources(domain):
        raise ValueError(f'{domain} generator sources changed after recipe fitting')
    if recipe['provenance']['fitter_source_sha256'] != file_sha(ROOT / 'ops/exp_scaling/fit_modebench_level3.py'):
        raise ValueError('mixture allocation code changed after recipe fitting')
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
    base_seed = SEEDS[domain] + (100_000 if split == 'train' else 200_000)
    for difficulty, cells in enumerate(assigned):
        if not cells:
            continue
        support_target = Counter()
        for key, count in cells.items():
            support_target[key[0]] += count
        extras = {'joint_target': cells} if domain == 'pantry' else {}
        generated = generator(domain)(
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
                or generator_sources(domain) != snapshot):
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


def finalize(recipes_mapping, output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'fresh output root required: {output}')
    if set(recipes_mapping) != set(DOMAINS):
        raise ValueError('recipes mapping must cover all five domains exactly')
    paths = {domain: Path(path).resolve() for domain, path in recipes_mapping.items()}
    recipes = {domain: load_recipe(paths[domain], domain) for domain in DOMAINS}
    recipe_hashes = {domain: file_sha(paths[domain]) for domain in DOMAINS}
    # Snapshot all choices before generating any held-out prompt. This immutable
    # bundle, rather than any evaluation outcome, determines every split.
    frozen = {
        'schema': 'modebench_level3_frozen_recipe_bundle_v1',
        'recipes_sha256': recipe_hashes,
        'generator_sources_sha256': {domain: generator_sources(domain) for domain in DOMAINS},
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
            shutil.copyfile(paths[domain], staging / 'recipes' / f'{domain}.json')
        _write_json(staging / 'frozen_recipes.json', frozen)
        frozen_hash = file_sha(staging / 'frozen_recipes.json')
        for domain in DOMAINS:
            print(f'[level3] finalizing {domain}', flush=True)
            recipe = recipes[domain]
            historical = historical_ids(domain)
            history_records[domain] = set(historical)
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
            dev_checks = verify_rows(domain, development, dev_reference, modes(dev_reference), historical)
            dev_checks['exact_joint_support_family_histogram'] = cell_histogram(domain, development) == cell_histogram(domain, dev_reference)
            dev_checks['frozen_selected_development_rows'] = row_hash(development) == recipe['development']['rows_sha256']
            if not all(dev_checks.values()):
                raise RuntimeError('frozen development structural checks failed')
            # Fresh train/eval exclude every candidate, including selected dev.
            blocked = historical | pool_ids
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
            'fairness_contract': {
                'support_histograms': 'exactly match Level 2 r5 in every split',
                'pantry_support_family_joint_histograms': 'exactly match Level 2 r5 in every split',
                'verifier_and_canonicalization': 'unchanged',
                'fresh_train_and_eval': 'disjoint from all Level 1, all Level 2, all candidate pools, and each other',
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
    args = parser.parse_args()
    mapping = json.loads(args.recipes_json.read_text())
    mapping = {domain: str((args.recipes_json.parent / path).resolve()) if not Path(path).is_absolute() else path
               for domain, path in mapping.items()}
    result = finalize(mapping, args.output_root)
    print(json.dumps({'decision': result['decision'], 'split_sizes': result['split_sizes'],
                      'output_root': str(args.output_root.resolve())}, sort_keys=True))


if __name__ == '__main__':
    main()
