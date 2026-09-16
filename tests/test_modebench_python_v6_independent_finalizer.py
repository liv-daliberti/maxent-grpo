"""Fresh Level 3 revisions protect exposed identities and authenticated inputs."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / 'ops/exp_scaling/finalize_modebench_level3_python_v6_independent.py'
SPEC = importlib.util.spec_from_file_location('independent_finalizer_test', PATH)
finalizer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(finalizer)


def test_only_prior_development_is_reusable():
    dev_blocked, fresh_blocked = finalizer.exclusion_sets(
        {'old_level1', 'old_level2'}, {'train': {'old_train'}, 'dev': {'old_dev'}, 'eval': {'old_eval'}},
        {'fixed_control'}, {'candidate', 'old_dev'})
    assert 'old_dev' not in dev_blocked
    assert {'old_level1', 'old_level2', 'old_train', 'old_eval', 'fixed_control'} <= dev_blocked
    assert fresh_blocked == dev_blocked | {'old_dev', 'candidate'}


@pytest.mark.parametrize('other_source', ['historical', 'train', 'eval', 'control'])
def test_prior_dev_exemption_never_removes_another_protected_occurrence(other_source):
    history = {'collision'} if other_source == 'historical' else set()
    prior = {'train': set(), 'dev': {'collision'}, 'eval': set()}
    if other_source in ('train', 'eval'):
        prior[other_source].add('collision')
    control = {'collision'} if other_source == 'control' else set()
    blocked, _ = finalizer.exclusion_sets(history, prior, control, set())
    assert 'collision' in blocked


def test_recipe_snapshot_rejects_bytes_changed_during_exact_refit(tmp_path, monkeypatch):
    path = tmp_path / 'recipe.json'
    path.write_text('{}')
    def load(*args):
        path.write_text('{"changed":true}')
        return {}
    monkeypatch.setattr(finalizer, 'load_recipe', load)
    with pytest.raises(ValueError, match='changed during exact development refit'):
        finalizer.recipe_snapshot(path, 'mathir')


@pytest.mark.parametrize('kind', ['receipt', 'evaluator', 'source'])
def test_generation_checks_authenticate_all_pinned_development_inputs(tmp_path, kind):
    path = tmp_path / kind
    path.write_text('authenticated input')
    snapshot = {'files_sha256': {str(path): finalizer.file_sha(path)}, 'directory_files': {}}
    finalizer.verify_development_inputs_unchanged(snapshot)
    path.write_text('changed after exact refit')
    with pytest.raises(ValueError, match='changed during generation'):
        finalizer.verify_development_inputs_unchanged(snapshot)


def test_reference_or_source_inventory_additions_are_rejected(tmp_path):
    path = tmp_path / 'data.arrow'
    path.write_text('authenticated input')
    snapshot = {'files_sha256': {str(path): finalizer.file_sha(path)},
                'directory_files': {str(tmp_path): [str(path)]}}
    (tmp_path / 'extra.arrow').write_text('new shard')
    with pytest.raises(ValueError, match='inventory changed'):
        finalizer.verify_development_inputs_unchanged(snapshot)


@pytest.fixture
def exclusion_publication(tmp_path, monkeypatch):
    root = tmp_path / 'prior'
    (root / 'recipes').mkdir(parents=True)
    recipe = root / 'recipes/mathir.json'
    recipe.write_text('{}')
    recipe_hashes = {'mathir': finalizer.file_sha(recipe)}
    bundle = root / 'frozen_recipes.json'
    bundle.write_text(json.dumps({'recipes_sha256': recipe_hashes}))
    rows = {}
    records = {}
    for split in ('train', 'dev', 'eval'):
        rows[str(root / 'mathir' / split)] = [{'id': split, 'answer_mode_count': 5}]
        records[split] = {'rows': 1, 'path': str(root / 'mathir' / split),
                         'rows_sha256': finalizer.row_hash(rows[str(root / 'mathir' / split)]),
                         'checks': {'structural': True}}
    identity = {'schema': 'modebench_level3_capability_matched_splits_v1', 'status': 'structural_checks_pass',
                'frozen_recipe_bundle_sha256': finalizer.file_sha(bundle), 'recipe_sha256': recipe_hashes,
                'domains': {'mathir': records}}
    (root / 'identity.json').write_text(json.dumps(identity))
    baseline = tmp_path / 'original_control'
    rows[str(baseline)] = [{'id': f'baseline{index}', 'answer_mode_count': 5} for index in range(128)]
    monkeypatch.setattr(finalizer, 'PRIOR_REVISION', root)
    monkeypatch.setattr(finalizer, 'BASELINE_CONTROLS', {'mathir': baseline})
    monkeypatch.setattr(finalizer, 'SPLITS', {part: (1, 'train' if part == 'train' else 'multi_answer') for part in ('train', 'dev', 'eval')})
    monkeypatch.setattr(finalizer, 'load_from_disk', lambda path: {'train': rows[path], 'multi_answer': rows[path]})
    monkeypatch.setattr(finalizer, 'identity_set', lambda domain, selected: {r['id'] for r in selected})
    monkeypatch.setattr(finalizer, 'reference_rows', lambda *args: rows[str(baseline)])
    return root, rows


def test_exclusion_rows_authenticate_all_prior_splits_and_fixed_control(exclusion_publication):
    prior, baseline = finalizer.authenticated_exclusion_rows('mathir')
    assert set(prior) == {'train', 'dev', 'eval'}
    assert len(baseline) == 128


@pytest.mark.parametrize('split', ['train', 'dev', 'eval'])
def test_changed_prior_rows_cannot_supply_reusable_or_protected_ids(exclusion_publication, split):
    root, rows = exclusion_publication
    rows[str(root / 'mathir' / split)][0]['id'] = 'tampered'
    with pytest.raises(ValueError, match='exclusion rows differ'):
        finalizer.authenticated_exclusion_rows('mathir')


def passing_recipe():
    selection = {'algorithm': finalizer.SELECTION_ALGORITHM,
                 'weight_objective': 'full_pool_cell_means_at_exact_controlled_allocation',
                 'selected_residual_used_for_ranking': False,
                 'on_selected_gate_failure': 'fail_without_alternate_weights_or_hash_seeds'}
    metrics = {'pass1': .1, 'pass8': .3}
    development = {'tolerances': finalizer.TOLERANCES, 'selected_development_sets_scored': 1,
                   'gates': {'expected': {'pass1': True, 'pass8': True}, 'selected': {'pass1': True, 'pass8': True}},
                   'expected_metrics': metrics, 'selected_metrics': metrics, 'baseline_metrics': metrics,
                   'expected_differences': {'pass1': 0, 'pass8': 0}, 'differences': {'pass1': 0, 'pass8': 0}}
    return {'schema': finalizer.RECIPE_SCHEMA, 'domain': 'mathir', 'development_fit_pass': True,
            'selection': selection, 'development': development, 'weight_units': [5, 5, 5, 5],
            'information_boundary': {'confirmation_outcomes_used': False},
            'provenance': {'seeds': [6328000, 6328001, 6328002, 6328003], 'generator_sources_sha256': {},
                           'fitter_source_sha256': 'source', 'baseline_receipt_path': 'baseline.json',
                           'pools': {str(tier): {'receipt_path': f'tier{tier}.json'} for tier in range(4)}}}


def test_every_recipe_field_must_reproduce_in_exact_independent_refit(tmp_path, monkeypatch):
    recipe = passing_recipe()
    # The fitter produces integer tier keys; saved JSON necessarily reads strings.
    recipe['development']['actual_difficulty_counts'] = {0: 32, 1: 32, 2: 32, 3: 32}
    path = tmp_path / 'recipe.json'
    path.write_text(json.dumps(recipe))
    monkeypatch.setattr(finalizer, 'generator_sources', lambda domain: {})
    monkeypatch.setattr(finalizer, 'file_sha', lambda path: 'source')
    monkeypatch.setattr(finalizer, 'refit_independent', lambda *args: deepcopy(recipe))
    assert finalizer.load_recipe(path, 'mathir') == json.loads(json.dumps(recipe))
    changed = deepcopy(recipe)
    changed['weight_units'] = [0, 0, 0, 20]
    monkeypatch.setattr(finalizer, 'refit_independent', lambda *args: changed)
    with pytest.raises(ValueError, match='does not reproduce exactly'):
        finalizer.load_recipe(path, 'mathir')


@pytest.fixture
def protocol_amendment(tmp_path, monkeypatch):
    amendment = {
        'schema': 'modebench_level3_v2_fixed_control_amendment_v1',
        'status': 'prospective_before_any_v2_model_outcomes',
        'decision': 'reuse_original_fixed_level1_evaluation_controls_with_new_independent_draws; generate_fresh_level3_confirmation_only',
        'information_boundary': {
            'fresh_heldout_claim_applies_to_level3_only': True,
            'legacy_confirmation_scores_used_for_v2_recipe_ranking': False,
            'level1_controls_are_untouched': False,
            'level3_eval_generated_after_all_five_passing_recipes_frozen': True,
            'treatment_training_started': False,
            'v1_correlated_receipts_retained_as_diagnostic': True,
            'v2_model_outcomes_exist': False,
            'v2_submission_claim_exists': False,
        },
        'unchanged': {
            'calibration_tasks_and_sources': True,
            'development_only_mixture_fitting': True,
            'frozen_models': True,
            'level1_generation_law_and_data': True,
            'sampling_interface_except_already_registered_rng_correction': True,
            'verifier_and_canonicalizer': True,
            'match_tolerances': {'pass1': 0.04, 'pass8': 0.08},
            'split_sizes': {'train': 384, 'dev': 128, 'eval': 128},
        },
        'controls': {},
    }
    paths, rows = {}, {}
    combined = {}
    for domain in finalizer.DOMAINS:
        directory = tmp_path / domain
        directory.mkdir()
        (directory / 'dataset_dict.json').write_text('{}')
        paths[domain] = directory
        rows[str(directory)] = [{'id': f'{domain}-{index}'} for index in range(128)]
        pins = {str(directory / 'dataset_dict.json'): finalizer.file_sha(directory / 'dataset_dict.json')}
        combined.update(pins)
        amendment['controls'][domain] = {
            'path': str(directory), 'rows': 128,
            'rows_sha256': finalizer.row_hash(rows[str(directory)]), 'files_sha256': pins,
        }
    amendment['baseline_confirmation_paths'] = {domain: str(path) for domain, path in paths.items()}
    amendment['fixed_control_files_sha256'] = combined
    prior = tmp_path / 'prior'
    prior.mkdir()
    prior_identity = prior / 'identity.json'
    prior_identity.write_text('{}')
    monkeypatch.setattr(finalizer, 'PRIOR_REVISION', prior)
    for key in ('protocol', 'calibration_seal', 'capacity_failure'):
        path = tmp_path / f'{key}.json'
        content = {'files_sha256': {str(prior_identity): finalizer.file_sha(prior_identity)}} if key == 'calibration_seal' else {}
        path.write_text(json.dumps(content))
        amendment[f'{key}_path'] = str(path)
        amendment[f'{key}_sha256'] = finalizer.file_sha(path)
    monkeypatch.setattr(finalizer, 'BASELINE_CONTROLS', paths)
    monkeypatch.setattr(finalizer, 'load_from_disk', lambda path: {'multi_answer': rows[path]})
    monkeypatch.setattr(finalizer, 'identity_set', lambda domain, selected: {row['id'] for row in selected})
    path = tmp_path / 'amendment.json'
    def write():
        path.write_text(json.dumps(amendment))
        monkeypatch.setattr(finalizer, 'AMENDMENT_SHA256', finalizer.file_sha(path))
    write()
    return amendment, path, write, rows


def test_registered_amendment_authenticates_fixed_control_sources(protocol_amendment):
    amendment, path, _, _ = protocol_amendment
    assert finalizer.validate_protocol_amendment(path) == amendment


def test_new_self_consistent_amendment_cannot_replace_prospective_registration(protocol_amendment):
    amendment, path, _, _ = protocol_amendment
    amendment['created_at'] = 'after outcomes'
    path.write_text(json.dumps(amendment))
    with pytest.raises(ValueError, match='unregistered prospective'):
        finalizer.validate_protocol_amendment(path)


@pytest.mark.parametrize('field', ['level1_controls_are_untouched', 'legacy_confirmation_scores_used_for_v2_recipe_ranking',
                                   'fresh_heldout_claim_applies_to_level3_only'])
def test_amendment_rejects_changed_information_boundary(protocol_amendment, field):
    amendment, path, write, _ = protocol_amendment
    amendment['information_boundary'][field] = not amendment['information_boundary'][field]
    write()
    with pytest.raises(ValueError, match='information boundary differs'):
        finalizer.validate_protocol_amendment(path)


@pytest.mark.parametrize('kind', ['protocol', 'calibration_seal', 'capacity_failure'])
def test_amendment_rejects_changed_referenced_artifact(protocol_amendment, kind):
    amendment, path, _, _ = protocol_amendment
    Path(amendment[f'{kind}_path']).write_text('changed')
    with pytest.raises(ValueError, match=f'{kind} source changed'):
        finalizer.validate_protocol_amendment(path)


def test_amendment_rejects_unrecorded_fixed_control_shard(protocol_amendment):
    amendment, path, _, _ = protocol_amendment
    (Path(amendment['controls']['mathir']['path']) / 'extra.arrow').write_text('new')
    with pytest.raises(ValueError, match='file pins differ'):
        finalizer.validate_protocol_amendment(path)


def test_amendment_rejects_control_rows_changed_behind_file_loader(protocol_amendment):
    amendment, path, _, rows = protocol_amendment
    rows[amendment['controls']['mathir']['path']][0]['id'] = 'substituted'
    with pytest.raises(ValueError, match='rows differ'):
        finalizer.validate_protocol_amendment(path)


def test_amendment_requires_all_five_original_controls(protocol_amendment):
    amendment, path, write, _ = protocol_amendment
    amendment['controls'].pop('mathir')
    write()
    with pytest.raises(ValueError, match='all five original'):
        finalizer.validate_protocol_amendment(path)


def test_amendment_rejects_prior_publication_rewritten_before_finalization(protocol_amendment):
    _, path, _, _ = protocol_amendment
    (finalizer.PRIOR_REVISION / 'identity.json').write_text('{"coherently_rewritten":true}')
    with pytest.raises(ValueError, match='prior Level 3 publication differs'):
        finalizer.validate_protocol_amendment(path)


def test_python_v6_route_calls_the_registered_revision_validator(monkeypatch):
    import sys
    from types import SimpleNamespace
    calls = []
    route = SimpleNamespace(validate_revision=lambda recipe, domain: calls.append((recipe, domain)))
    monkeypatch.setitem(sys.modules, 'fit_modebench_level3_python_v6_independent', route)
    recipe = {'candidate_revision': {'name': 'python_v6'}}
    assert finalizer.candidate_route(recipe, 'python_factors') is route
    assert calls == [(recipe, 'python_factors')]


def test_python_v6_route_cannot_bypass_revision_authentication(monkeypatch):
    import sys
    from types import SimpleNamespace
    def changed(*args):
        raise ValueError('registered Python v6 bytes changed')
    monkeypatch.setitem(sys.modules, 'fit_modebench_level3_python_v6_independent',
                        SimpleNamespace(validate_revision=changed))
    with pytest.raises(ValueError, match='registered Python v6 bytes changed'):
        finalizer.candidate_route({'candidate_revision': {'name': 'python_v6'}}, 'python_factors')


@pytest.mark.parametrize('revision', [{'name':'python_v6_unknown'}, {'name':'python_v7'}, [], None])
def test_unregistered_python_revision_has_no_fallback(revision):
    with pytest.raises(ValueError, match='unknown or changed independent candidate revision'):
        finalizer.candidate_route({'candidate_revision': revision}, 'python_factors')


def test_original_recipe_route_is_preserved():
    assert finalizer.candidate_route({}, 'countdown') is None
