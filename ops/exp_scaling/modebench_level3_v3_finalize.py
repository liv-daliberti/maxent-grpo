#!/usr/bin/env python3
"""Publish one v3 dataset after authenticated fresh candidate development.

Countdown, MathIR and Pantry datasets/recipes are copied byte-for-byte from the
completed prior round. Only Graph v8 and Python v7 receive new train/eval
prompts; their DEV subsets are the already chosen fixed selections. No model
outcomes are evaluated here and no recipe is fitted or revised.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
for directory in ('ops', 'ops/exp_scaling', 'src'):
    sys.path.insert(0, str(ROOT / directory))
import modebench_level3_v3_common as common
import materialize_modebench_level3 as materializer
import fit_modebench_level3 as mixture

SCHEMA = 'modebench_level3_v3_adaptive_fixed_reference_splits_v1'
BUNDLE_SCHEMA = 'modebench_level3_v3_fixed_recipe_bundle_v1'
TEST_SOURCE = ROOT / 'tests/test_modebench_level3_v3_finalize.py'
EXECUTION_SEAL = common.CAMPAIGN / 'continuation/seal.json'
IDENTITY = common.DATASET / 'identity.json'
BUNDLE = common.DATASET / 'frozen_recipes.json'
require = common.require
read = common.read
digest = common.digest


def information_boundary():
    return {'adaptive_confirmation_round': 2,
        'retained_domains': list(common.RETAINED), 'fresh_candidate_domains': list(common.REVISED),
        'all_five_fresh_same_round': False, 'statistical_equivalence_claimed': False,
        'historical_level1_confirmation_used_as_fixed_reference': True,
        'new_candidate_fitting_uses_development_outcomes_only': True,
        'fresh_candidate_confirmation_outcomes_loaded': False, 'treatment_training_started': False}


def candidate_modules(domain):
    require(domain in common.REVISED, 'retained domains must never be regenerated')
    revision = {'graph_coloring': 'graph_v8', 'python_factors': 'python_v7'}[domain]
    return (importlib.import_module('modebench_level3_' + revision),
            importlib.import_module('materialize_modebench_level3_' + revision))


def authenticate_execution_seal(path, expected_sha256, registration_sha256):
    path = Path(path).resolve()
    require(path == EXECUTION_SEAL and digest(path) == expected_sha256,
            'explicit canonical additive execution seal required')
    seal = read(path)
    require(seal['schema'] == 'modebench_level3_v3_continuation_seal_v1'
            and seal['registration_path'] == str(common.REGISTRATION)
            and seal['registration_sha256'] == registration_sha256,
            'additive seal registration differs')
    registration = common.validate_registration(common.REGISTRATION, registration_sha256)
    require(all(seal['files_sha256'].get(name) == expected for name, expected in registration['files_sha256'].items())
            and seal['files_sha256'].get(str(common.REGISTRATION)) == registration_sha256,
            'additive execution seal omits scientific registration')
    for source in (Path(__file__).resolve(), TEST_SOURCE, ROOT / 'ops/audit_modebench_level3_v3.py',
                   ROOT / 'tests/test_modebench_level3_v3_audit.py',
                   ROOT / 'ops/exp_scaling/modebench_level3_v3_confirmation.py',
                   ROOT / 'tests/test_modebench_level3_v3_confirmation.py'):
        require(seal['files_sha256'].get(str(source)) == digest(source), 'future stage implementation not sealed')
    import modebench_level3_v3_development as launcher
    require(seal['scientific_seal_path'] == str(launcher.SEAL)
            and seal['files_sha256'].get(str(launcher.SEAL)) == seal['scientific_seal_sha256'],
            'additive seal omits exact new development scientific seal')
    scientific = launcher.authenticate_saved_seal(seal['scientific_seal_sha256'])
    require(seal['models'] == scientific['models']
            and all(seal['files_sha256'].get(name) == expected for name, expected in scientific['files_sha256'].items())
            and all(seal['directory_files'].get(name) == expected for name, expected in scientific['directory_files'].items()),
            'additive seal omits actual development pools, source inventory or model identities')
    common.verify_pins(seal['files_sha256'], seal['directory_files'])
    require(digest(path) == expected_sha256, 'additive execution seal changed during verification')
    return seal


def source_rows(path):
    from datasets import load_from_disk
    dataset = load_from_disk(str(path))
    require(hasattr(dataset, 'values') and len(dataset) == 1, 'one frozen dataset subset required')
    return [dict(row) for subset in dataset.values() for row in subset]


def fixed_history(domain, revision):
    """Rebuild only registered historical identities, unaffected by new output."""
    snapshot = revision['source_snapshot']
    common.verify_pins(snapshot['files_sha256'], snapshot['directory_files'])
    history = set(materializer.existing_ids(domain))
    paths = sorted({Path(name).parent for name in snapshot['files_sha256']
                    if name.endswith('/dataset_dict.json') and Path(name).parent.name in common.SPLITS
                    and Path(name).parent.parent.name == domain
                    and Path(name).parent.parent.parent.name.startswith(('modebench_harder', 'modebench_level3'))})
    for path in paths:
        history |= materializer.identity_set(domain, source_rows(path))
    require(materializer.row_hash(sorted(history, key=repr)) == snapshot['historical_identity_sha256'],
            'registered historical semantic identity inventory differs')
    blocked = set(history); pool_records = []
    new_paths = [common.POOL_ROOTS[domain] / 'pools' / domain / f'difficulty_{tier}.jsonl' for tier in range(4)]
    for path in sorted({Path(name) for name in snapshot['candidate_pool_paths']} | set(new_paths)):
        rows = mixture.read_jsonl(path)
        blocked |= materializer.identity_set(domain, rows)
        pool_records.append({'path': str(path), 'sha256': digest(path), 'rows': len(rows),
                             'rows_sha256': materializer.row_hash(rows)})
    return history, blocked, {'historical_identities': len(history), 'all_excluded_identities': len(blocked),
        'historical_identity_sha256': materializer.row_hash(sorted(history, key=repr)),
        'excluded_identity_sha256': materializer.row_hash(sorted(blocked, key=repr)),
        'candidate_pools': pool_records, 'historical_dataset_paths': list(map(str, paths))}


def selected_development(domain, recipe):
    pools = {}
    for tier in range(4):
        provenance = recipe['provenance']['pools'][str(tier)]
        require(digest(provenance['rows_path']) == provenance['rows_file_sha256']
                and digest(provenance['pool_identity_path']) == provenance['pool_identity_sha256'],
                'chosen DEV pool or certificate changed')
        rows = mixture.read_jsonl(provenance['rows_path'])
        require(materializer.row_hash(rows) == provenance['rows_sha256'], 'chosen DEV pool rows changed')
        pools[tier] = rows
    reference = materializer.reference_rows(domain, 'dev')
    selected = mixture.select_rows(domain, pools, mixture.cell_histogram(domain, reference),
                                  recipe['weight_units'], common.SELECTION_SEED)
    require([{key: value for key, value in item.items() if key != 'row'} for item in selected]
            == recipe['selected_development'], 'exact frozen development selection differs')
    rows = [item['row'] for item in selected]
    require(materializer.row_hash(rows) == recipe['development']['rows_sha256'], 'selected DEV row hash differs')
    return rows


def allocation_targets(domain, split, recipe):
    require(domain in common.REVISED and split in ('train', 'eval'), 'only fresh revised train/eval can be generated')
    target = common.reference_histograms(domain)[split]
    allocation = mixture.allocate_cells(target, recipe['weight_units'], common.SELECTION_SEED)
    return [Counter({cell: counts[tier] for cell, counts in allocation.items() if counts[tier]}) for tier in range(4)]


def build_fresh_split(domain, split, recipe, blocked, *, witnesses=True):
    generator, owner = candidate_modules(domain)
    base_seed = common.SEEDS[domain][split]
    assigned = allocation_targets(domain, split, recipe)
    current = set(blocked); rows = []
    for tier, cells in enumerate(assigned):
        if not cells:
            continue
        require(all(len(cell) == 1 for cell in cells), 'revised support cells must be marginal support only')
        target = Counter({cell[0]: count for cell, count in cells.items()})
        generated = generator.build_pool(domain, target, current, base_seed + 1000 * tier,
                                         'level3_v3_' + split, tier, multiplier=1)
        require(mixture.cell_histogram(domain, generated) == cells, 'fresh per-tier cell allocation differs')
        current |= materializer.identity_set(domain, generated); rows.extend(generated)
    rows.sort(key=lambda row: mixture.sha([base_seed, 'split_order', mixture.sha(row)]))
    reference = materializer.reference_rows(domain, split)
    checks = materializer.verify_rows(domain, rows, reference, materializer.modes(reference), blocked)
    checks['exact_joint_support_family_histogram'] = mixture.cell_histogram(domain, rows) == mixture.cell_histogram(domain, reference)
    checks['exact_difficulty_cell_recipe'] = all(mixture.cell_histogram(domain,
        [row for row in rows if row['level3_difficulty'] == tier]) == cells for tier, cells in enumerate(assigned))
    checks['exact_global_tier_quota'] = [sum(cells.values()) for cells in assigned] == mixture.hamilton(
        common.SPLITS[split], recipe['weight_units'], (), common.SELECTION_SEED)
    require(all(value is True for value in checks.values()), 'fresh structural allocation checks failed')
    count = owner.verify_witnesses(rows) if witnesses else None
    return rows, checks, {'base_seed': base_seed, 'tier_stride': 1000,
                         'tier_seeds': {str(tier): base_seed + 1000 * tier for tier in range(4)},
                         'original_structural_witnesses': count}


def byte_tree(path):
    path = Path(path)
    return {str(item.relative_to(path)): digest(item) for item in sorted(path.rglob('*')) if item.is_file()}


def copy_retained_domain(domain, staging):
    require(domain in common.RETAINED, 'only previously passed domains may be copied as retained')
    source = common.OLD_DATASET / domain
    before = byte_tree(source)
    require(before, 'retained domain has no actual split files')
    shutil.copytree(source, staging / domain)
    require(byte_tree(source) == byte_tree(staging / domain) == before, 'retained domain bytes changed during copy')
    return before


def recipe_records(development):
    records = {}
    for domain in common.DOMAINS:
        path = (common.OLD_DATASET / 'recipes' / (domain + '.json') if domain in common.RETAINED
                else common.CAMPAIGN / 'recipes' / (domain + '.json'))
        if domain in common.REVISED:
            require(development['recipes'][domain]['path'] == str(path)
                    and development['recipes'][domain]['sha256'] == digest(path), 'completed proof recipe differs')
        records[domain] = {'path': str(path), 'sha256': digest(path),
                           'role': 'retained' if domain in common.RETAINED else 'fresh_revision'}
    return records


def verify_split_structure(domain, split, rows, blocked):
    reference = materializer.reference_rows(domain, split)
    checks = materializer.verify_rows(domain, rows, reference, materializer.modes(reference), blocked)
    checks['exact_joint_support_family_histogram'] = mixture.cell_histogram(domain, rows) == mixture.cell_histogram(domain, reference)
    require(len(rows) == common.SPLITS[split] and all(value is True for value in checks.values()),
            'exact split size/support/joint histogram/identity checks required')
    return checks


def finalize(*, registration_sha256, execution_seal_path=EXECUTION_SEAL, execution_seal_sha256):
    require(not common.DATASET.exists(), 'fresh isolated v3 final dataset required')
    registration = common.validate_registration(common.REGISTRATION, registration_sha256)
    seal = authenticate_execution_seal(execution_seal_path, execution_seal_sha256, registration_sha256)
    from audit_modebench_level3_v3 import validate_completed_development_audit
    development = validate_completed_development_audit()
    require(development['registration_sha256'] == registration_sha256, 'completed DEV registration differs')
    common.fixed_references()
    recipes = recipe_records(development)
    bundle = {'schema': BUNDLE_SCHEMA, 'registration_path': str(common.REGISTRATION),
        'registration_sha256': registration_sha256, 'execution_seal_path': str(EXECUTION_SEAL),
        'execution_seal_sha256': execution_seal_sha256,
        'completed_development_audit': {'path': development['path'], 'sha256': development['sha256']},
        'recipes': recipes, 'retained_domains': list(common.RETAINED), 'fresh_domains': list(common.REVISED),
        'generation_seed_bases': common.contract()['generation_seed_bases'],
        'finalizer_source_sha256': digest(__file__), 'recipe_frozen_before_new_eval_generation': True,
        'fresh_candidate_confirmation_outcomes_used': False,
        'historical_level1_confirmation_used_as_fixed_reference': True}
    common.DATASET.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.' + common.DATASET.name + '.', dir=common.DATASET.parent))
    try:
        common.atomic_new(staging / 'frozen_recipes.json', bundle)
        (staging / 'recipes').mkdir()
        for domain, record in recipes.items():
            shutil.copyfile(record['path'], staging / 'recipes' / (domain + '.json'))
            require(digest(staging / 'recipes' / (domain + '.json')) == record['sha256'], 'archived recipe bytes differ')
        records, exclusions, retained_trees = {}, {}, {}
        from datasets import Dataset, DatasetDict
        old_identity = read(common.OLD_DATASET / 'identity.json')
        for domain in common.DOMAINS:
            records[domain] = {}
            if domain in common.RETAINED:
                retained_trees[domain] = copy_retained_domain(domain, staging)
                for split in common.SPLITS:
                    rows = source_rows(staging / domain / split)
                    checks = verify_split_structure(domain, split, rows, set())
                    old = old_identity['domains'][domain][split]
                    require(materializer.row_hash(rows) == old['rows_sha256'], 'retained source rows differ')
                    records[domain][split] = {**old, 'path': str(common.DATASET / domain / split),
                        'checks': {**checks, 'retained_split_bytes_identical': True},
                        'evidence_role': 'retained_historical_domain', 'original_path': str(common.OLD_DATASET / domain / split)}
                print(json.dumps({'event': 'retained_domain_copied', 'domain': domain}), flush=True)
                continue
            recipe = read(recipes[domain]['path'])
            history, blocked, exclusion_record = fixed_history(domain, registration['candidate_revisions'][domain])
            exclusions[domain] = exclusion_record
            dev = selected_development(domain, recipe)
            dev_checks = verify_split_structure(domain, 'dev', dev, history)
            built = {'dev': (dev, dev_checks, None)}
            for split in ('train', 'eval'):
                rows, checks, generation = build_fresh_split(domain, split, recipe, blocked)
                blocked |= materializer.identity_set(domain, rows)
                built[split] = rows, checks, generation
            split_ids = {split: materializer.identity_set(domain, value[0]) for split, value in built.items()}
            require(all(not split_ids[a] & split_ids[b] for a, b in (('train', 'dev'), ('train', 'eval'), ('dev', 'eval'))),
                    'new candidate final splits overlap')
            for split, (rows, checks, generation) in built.items():
                subset = materializer.SPLITS[split][1]
                DatasetDict({subset: Dataset.from_list(rows)}).save_to_disk(str(staging / domain / split))
                records[domain][split] = {'path': str(common.DATASET / domain / split), 'dataset_split': subset,
                    'rows': len(rows), 'rows_sha256': materializer.row_hash(rows),
                    'checks': {**checks, 'all_final_splits_disjoint': True},
                    'support_histogram': dict(sorted(materializer.modes(rows).items())),
                    'joint_cells': mixture.serialize_cells(mixture.cell_histogram(domain, rows)),
                    'difficulty_counts': dict(Counter(row['level3_difficulty'] for row in rows)),
                    'generation': generation, 'evidence_role': 'new_candidate_development' if split == 'dev' else 'fresh_candidate_' + split}
            print(json.dumps({'event': 'fresh_domain_constructed', 'domain': domain, 'rows': 640}), flush=True)
        identity = {'schema': SCHEMA, 'status': 'structural_checks_pass', 'decision': 'pending_fresh_candidate_confirmation',
            'split_sizes': dict(common.SPLITS), 'domains': records, 'recipes': recipes,
            'registration_path': str(common.REGISTRATION), 'registration_sha256': registration_sha256,
            'frozen_recipe_bundle_sha256': digest(staging / 'frozen_recipes.json'),
            'retained_domain_files_sha256': retained_trees, 'fresh_generation_exclusions': exclusions,
            'information_boundary': information_boundary()}
        common.verify_pins(development['files_sha256'], development['directory_files'])
        common.verify_pins(seal['files_sha256'], seal['directory_files'])
        require(digest(EXECUTION_SEAL) == execution_seal_sha256 and digest(__file__) == bundle['finalizer_source_sha256'],
                'generation execution/source changed')
        for domain in common.REVISED:
            _, _, current = fixed_history(domain, registration['candidate_revisions'][domain])
            require(current == exclusions[domain], 'historical/new candidate exclusions changed during generation')
        for domain, before in retained_trees.items():
            require(byte_tree(common.OLD_DATASET / domain) == byte_tree(staging / domain) == before,
                    'retained domain bytes changed before publication')
        require(read(staging / 'frozen_recipes.json') == bundle, 'frozen bundle changed before publication')
        for domain, record in recipes.items():
            require(digest(staging / 'recipes' / (domain + '.json')) == record['sha256'],
                    'archived recipe changed before publication')
            for split in common.SPLITS:
                require(materializer.row_hash(source_rows(staging / domain / split))
                        == records[domain][split]['rows_sha256'], 'written split rows differ before publication')
        common.atomic_new(staging / 'identity.json', identity)
        common.atomic_new(staging / 'admission_fairness_report.json', identity)
        require(not common.DATASET.exists(), 'fresh isolated final dataset required')
        staging.rename(common.DATASET)
        return identity
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def authenticate_dataset(*, registration_sha256):
    """Read-only exact split/provenance authentication; no model calls or writes."""
    registration = common.validate_registration(common.REGISTRATION, registration_sha256)
    before = digest(IDENTITY); identity = read(IDENTITY); bundle = read(BUNDLE)
    require(identity['schema'] == SCHEMA and identity['status'] == 'structural_checks_pass'
            and identity['decision'] == 'pending_fresh_candidate_confirmation'
            and identity['split_sizes'] == common.SPLITS and set(identity['domains']) == set(common.DOMAINS)
            and identity['registration_path'] == str(common.REGISTRATION)
            and identity['registration_sha256'] == registration_sha256
            and identity['information_boundary'] == information_boundary()
            and identity['frozen_recipe_bundle_sha256'] == digest(BUNDLE), 'canonical new dataset identity differs')
    require(read(common.DATASET / 'admission_fairness_report.json') == identity
            and bundle['schema'] == BUNDLE_SCHEMA and bundle['registration_sha256'] == registration_sha256
            and bundle['registration_path'] == str(common.REGISTRATION)
            and bundle['retained_domains'] == list(common.RETAINED) and bundle['fresh_domains'] == list(common.REVISED)
            and bundle['generation_seed_bases'] == common.contract()['generation_seed_bases']
            and bundle['historical_level1_confirmation_used_as_fixed_reference'] is True
            and bundle['finalizer_source_sha256'] == digest(__file__)
            and bundle['recipe_frozen_before_new_eval_generation'] is True
            and bundle['fresh_candidate_confirmation_outcomes_used'] is False,
            'frozen recipe/prospective generation provenance differs')
    seal = authenticate_execution_seal(bundle['execution_seal_path'], bundle['execution_seal_sha256'], registration_sha256)
    from audit_modebench_level3_v3 import validate_completed_development_audit
    development = validate_completed_development_audit()
    require(bundle['completed_development_audit'] == {'path': development['path'], 'sha256': development['sha256']}
            and bundle['recipes'] == identity['recipes'] == recipe_records(development), 'frozen recipe/proof mapping differs')
    files = common.merge_pins(seal['files_sha256'], development['files_sha256'], {str(EXECUTION_SEAL): digest(EXECUTION_SEAL)})
    trees = {**seal['directory_files'], **development['directory_files']}
    for domain in common.DOMAINS:
        archive = common.DATASET / 'recipes' / (domain + '.json')
        require(digest(archive) == identity['recipes'][domain]['sha256'], 'archived recipe bytes changed')
        actual_rows = {split: source_rows(common.DATASET / domain / split) for split in common.SPLITS}
        ids = {split: materializer.identity_set(domain, rows) for split, rows in actual_rows.items()}
        require(all(not ids[a] & ids[b] for a, b in (('train', 'dev'), ('train', 'eval'), ('dev', 'eval'))),
                'published final splits overlap')
        for split, rows in actual_rows.items():
            subset = materializer.SPLITS[split][1]
            require(set(byte_tree(common.DATASET / domain / split)) == {'dataset_dict.json',
                subset + '/state.json', subset + '/dataset_info.json', subset + '/data-00000-of-00001.arrow'},
                'exact one-shard split file inventory required')
            record = identity['domains'][domain][split]
            require(record['path'] == str(common.DATASET / domain / split)
                    and record['dataset_split'] == subset
                    and record['difficulty_counts'] == {str(key): value for key, value in Counter(
                        row['level3_difficulty'] for row in rows).items()}
                    and record['rows'] == len(rows) == common.SPLITS[split]
                    and record['rows_sha256'] == materializer.row_hash(rows)
                    and record['joint_cells'] == mixture.serialize_cells(mixture.cell_histogram(domain, rows))
                    and {str(key): value for key, value in materializer.modes(rows).items()} == record['support_histogram']
                    and all(value is True for value in record['checks'].values()), 'actual split identity/histogram/checks differ')
            verify_split_structure(domain, split, rows, set())
        if domain in common.RETAINED:
            require(byte_tree(common.DATASET / domain) == byte_tree(common.OLD_DATASET / domain)
                    == identity['retained_domain_files_sha256'][domain], 'retained domain bytes differ')
        else:
            recipe = read(archive)
            history, blocked, exclusion = fixed_history(domain, registration['candidate_revisions'][domain])
            require(exclusion == identity['fresh_generation_exclusions'][domain], 'frozen exclusion identity inventory differs')
            require(actual_rows['dev'] == selected_development(domain, recipe) and not ids['dev'] & history,
                    'actual selected DEV or historical disjointness differs')
            for split in ('train', 'eval'):
                expected, _, generation = build_fresh_split(domain, split, recipe, blocked, witnesses=False)
                require(actual_rows[split] == expected and not ids[split] & blocked,
                        'fresh split does not reproduce exactly from frozen recipe/law/seeds/exclusions')
                recorded_generation = identity['domains'][domain][split]['generation']
                require(all(recorded_generation[key] == generation[key] for key in ('base_seed', 'tier_stride', 'tier_seeds'))
                        and type(recorded_generation['original_structural_witnesses']) is int
                        and recorded_generation['original_structural_witnesses'] > 0, 'fresh generation/witness provenance differs')
                blocked |= ids[split]
    inventory = sorted(str(path.resolve()) for path in common.DATASET.rglob('*') if path.is_file())
    expected = {str(common.DATASET / name) for name in ('identity.json', 'admission_fairness_report.json', 'frozen_recipes.json')}
    expected |= {str(common.DATASET / 'recipes' / (domain + '.json')) for domain in common.DOMAINS}
    for domain in common.DOMAINS:
        for split in common.SPLITS:
            expected |= {str((common.DATASET / domain / split / name).resolve()) for name in byte_tree(common.DATASET / domain / split)}
    require(set(inventory) == expected and digest(IDENTITY) == before, 'final dataset inventory changed or has unrelated artifacts')
    for path in inventory:
        files = common.merge_pins(files, {path: digest(path)})
    trees[str(common.DATASET)] = inventory
    common.verify_pins(files, trees)
    return {'identity_metadata': {'path': str(IDENTITY), 'sha256': before, 'dataset_root': str(common.DATASET),
                                 'frozen_recipe_bundle_path': str(BUNDLE), 'frozen_recipe_bundle_sha256': digest(BUNDLE),
                                 'split_sizes': dict(common.SPLITS), 'retained_domains': list(common.RETAINED),
                                 'fresh_candidate_domains': list(common.REVISED)},
            'identity': identity, 'files_sha256': files, 'directory_files': trees}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration-sha256', required=True)
    parser.add_argument('--execution-seal', type=Path, default=EXECUTION_SEAL)
    parser.add_argument('--execution-seal-sha256', required=True)
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args(argv)
    require(args.publish, 'explicit publication flag required; use authenticate_dataset for read-only validation')
    result = finalize(registration_sha256=args.registration_sha256, execution_seal_path=args.execution_seal,
                      execution_seal_sha256=args.execution_seal_sha256)
    print(json.dumps({'status': result['status'], 'decision': result['decision'], 'rows': 3200,
                      'identity_path': str(IDENTITY), 'identity_sha256': digest(IDENTITY)}), flush=True)


if __name__ == '__main__':
    main()
