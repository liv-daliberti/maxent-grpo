from collections import Counter
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ops/exp_scaling'))
import fit_modebench_level3 as fit
import finalize_modebench_level3 as final


def make_rows(count, difficulty, prefix='pool'):
    return [{'problem': f'{prefix}-{difficulty}-{i}',
             'answer': json.dumps({'verifier': 'mathir_action_menu', 'family': prefix,
                                   'bindings': {'a': difficulty * 100000 + i}}),
             'modebench_task': 'mathir_action_menu',
             'answer_mode_count': 5, 'level3_difficulty': difficulty}
            for i in range(count)]


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True))


def write_receipt(tmp_path, name, rows, model, metric, *, split='dev'):
    source = tmp_path / (name + '.jsonl')
    source.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows))
    identity = {'source': {'kind': 'jsonl', 'path': str(source),
                           'file_sha256': fit.file_sha(source), 'rows_sha256': fit.sha(rows),
                           'selected_rows': len(rows), 'row_offset': 0, 'row_limit': 0},
                'interface': {'sample_count': 8, 'name': 'level2_qwen_r5'},
                'seeds': [77, 78, 79, 80], 'model': {'label': model},
                'code_sha256': {'verifier': 'frozen'}}
    receipt = {'status': 'complete', 'domain': 'mathir',
               'level': 'level1' if model == '05b' else 'level3',
               'split': split, 'model_label': model, 'identity': identity,
               'identity_sha256': fit.sha(identity),
               'information_boundary': {'evaluation_prompts_loaded': split == 'eval'},
               'metrics': {'pass1': metric, 'pass8': metric},
               'prompt_results': [{'row_sha256': fit.sha(row), 'pass1': metric, 'pass8': metric,
                                    'draws': [{'seed': seed, 'pass1': float(index < metric * 4),
                                               'pass8': float(index < metric * 4),
                                               'attempts': [{'verified': index < metric * 4,
                                                             'canonical_key': 'mode' if index < metric * 4 else None}
                                                            for _ in range(8)]}
                                              for index, seed in enumerate((77, 78, 79, 80))]}
                                  for row in rows]}
    path = tmp_path / (name + '.json')
    write_json(path, receipt)
    return path, source


@pytest.fixture
def fitted(tmp_path, monkeypatch):
    reference = make_rows(128, 0, 'reference')
    monkeypatch.setattr(fit, 'reference_rows', lambda domain, split: reference)
    baseline, _ = write_receipt(tmp_path, 'baseline', make_rows(128, 0, 'baseline'), '05b', 0.5)
    receipts, sources = [], []
    for difficulty, metric in enumerate((0.0, 0.25, 0.75, 1.0)):
        path, source = write_receipt(tmp_path, f'pool{difficulty}', make_rows(128, difficulty), '3b', metric)
        write_json(source.with_suffix('.identity.json'), {
            'rows_sha256': fit.row_hash(make_rows(128, difficulty)),
            'source_sha256': fit.file_sha(Path(fit.generator('mathir').__module__ and
                                             sys.modules[fit.generator('mathir').__module__].__file__)),
        })
        receipts.append(path)
        sources.append(source)
    recipe_path = tmp_path / 'recipe.json'
    recipe = fit.fit_recipe(baseline, receipts, 'mathir', recipe_path)
    return recipe, recipe_path, reference, receipts, baseline, sources


def test_hamilton_and_grid_are_exact_and_deterministic():
    grid = list(fit.weight_grid())
    assert len(grid) == 1771
    for weights in grid:
        assigned = fit.hamilton(7, weights, (5, 'family'), seed=3)
        assert sum(assigned) == 7
        assert assigned == fit.hamilton(7, weights, (5, 'family'), seed=3)
        assert all(n in (7 * w // 20, (7 * w + 19) // 20) for n, w in zip(assigned, weights))


def test_selection_keeps_pantry_joint_counts_and_ignores_scores():
    pools = {}
    for difficulty in range(4):
        rows = make_rows(8, difficulty)
        for i, row in enumerate(rows):
            row['answer_mode_family'] = 'a' if i < 4 else 'b'
        pools[difficulty] = rows
    target = Counter({(5, 'a'): 4, (5, 'b'): 4})
    selected = fit.select_rows('pantry', pools, target, (5, 5, 5, 5), 222)
    assert fit.cell_histogram('pantry', [item['row'] for item in selected]) == target
    # Selecting a frozen recipe has no score argument or dependency.
    again = fit.select_rows('pantry', pools, target, (5, 5, 5, 5), 222)
    assert selected == again


def test_full_development_fit_and_provenance(fitted):
    recipe, recipe_path, reference, receipts, baseline, sources = fitted
    assert recipe['development_fit_pass']
    assert recipe['development']['rows'] == 128
    assert recipe['development']['weight_combinations_considered'] == 1771
    assert all(abs(x) < 1e-12 for x in recipe['development']['differences'].values())
    assert recipe['provenance']['baseline_receipt_sha256'] == fit.file_sha(baseline)
    assert recipe['information_boundary']['confirmation_outcomes_used'] is False
    assert recipe_path.exists()


def test_rejects_confirmation_receipts(tmp_path):
    path, _ = write_receipt(tmp_path, 'confirmation', make_rows(2, 0), '3b', 0.5, split='eval')
    with pytest.raises(ValueError, match='development receipt'):
        fit.load_receipt(path, 'mathir', 'level3', '3b')


def test_rejects_interface_mismatch(fitted, tmp_path):
    recipe, _, _, receipts, baseline, _ = fitted
    receipt = json.loads(receipts[0].read_text())
    receipt['identity']['interface']['name'] = 'original_level1'
    receipt['identity_sha256'] = fit.sha(receipt['identity'])
    write_json(receipts[0], receipt)
    with pytest.raises(ValueError, match='identical interfaces'):
        fit.fit_recipe(baseline, receipts, 'mathir')


def test_rejects_nonpassing_development_recipe(fitted, tmp_path):
    recipe, _, _, _, _, _ = fitted
    recipe['development_fit_pass'] = False
    path = tmp_path / 'bad_recipe.json'
    write_json(path, recipe)
    with pytest.raises(ValueError, match='fit did not pass'):
        final.load_recipe(path, 'mathir')


def test_finalize_freezes_before_eval_and_uses_no_model_receipts(fitted, tmp_path, monkeypatch):
    recipe, recipe_path, reference, receipts, baseline, sources = fitted
    for path in receipts + [baseline]:
        path.unlink()  # Finalization must never reopen model-outcome files.
    output = tmp_path / 'final_data'
    monkeypatch.setattr(final, 'DOMAINS', ('mathir',))
    monkeypatch.setattr(final, 'reference_rows', lambda domain, split:
                        make_rows(384 if split == 'train' else 128, 0, 'reference'))
    monkeypatch.setattr(final, 'historical_ids', lambda domain: set())
    monkeypatch.setattr(final, 'candidate_pool_files', lambda domain, recipe: sources)
    called = []

    def fake_generator(domain, target, excluded, seed, tag, difficulty, multiplier, **extras):
        staging = next(tmp_path.glob('.final_data.*'))
        frozen = json.loads((staging / 'frozen_recipes.json').read_text())
        assert frozen['recipes_sha256']['mathir'] == fit.file_sha(recipe_path)
        assert (staging / 'recipes/mathir.json').exists()
        called.append(tag)
        return make_rows(sum(target.values()), difficulty, tag)

    monkeypatch.setattr(final, 'generator', lambda domain: fake_generator)
    identity = final.finalize({'mathir': recipe_path}, output)
    assert identity['decision'] == 'pending_confirmation'
    assert identity['information_boundary']['evaluation_model_outcomes_loaded'] is False
    assert identity['domains']['mathir']['train']['rows'] == 384
    assert identity['domains']['mathir']['dev']['rows'] == 128
    assert identity['domains']['mathir']['eval']['rows'] == 128
    assert identity['domains']['mathir']['dev']['rows_sha256'] == recipe['development']['rows_sha256']
    assert 'level3_eval' in called
    assert not list(tmp_path.glob('.final_data.*'))
    with pytest.raises(FileExistsError):
        final.finalize({'mathir': recipe_path}, output)


def test_finalize_failure_does_not_publish_partial_dataset(fitted, tmp_path, monkeypatch):
    recipe, recipe_path, reference, _, _, sources = fitted
    monkeypatch.setattr(final, 'DOMAINS', ('mathir',))
    monkeypatch.setattr(final, 'reference_rows', lambda domain, split:
                        make_rows(384 if split == 'train' else 128, 0, 'reference'))
    monkeypatch.setattr(final, 'historical_ids', lambda domain: set())
    monkeypatch.setattr(final, 'candidate_pool_files', lambda domain, recipe: sources)
    monkeypatch.setattr(final, 'generator', lambda domain: lambda *args, **kwargs: [])
    output = tmp_path / 'broken_data'
    with pytest.raises(RuntimeError, match='drift'):
        final.finalize({'mathir': recipe_path}, output)
    assert not output.exists()
    assert not list(tmp_path.glob('.broken_data.*'))


def test_fitting_refuses_single_seed_receipts(fitted):
    _, _, _, receipts, baseline, _ = fitted
    receipt = json.loads(baseline.read_text())
    receipt['identity']['seeds'] = [77]
    receipt['identity_sha256'] = fit.sha(receipt['identity'])
    write_json(baseline, receipt)
    with pytest.raises(ValueError, match='four independent'):
        fit.fit_recipe(baseline, receipts, 'mathir')


def test_local_dependency_hash_includes_imported_helpers():
    sources = fit.local_dependency_sources([ROOT / 'ops/exp_scaling/modebench_level3_graph_v2.py'])
    assert 'ops/exp_scaling/modebench_level3_discrete.py' in sources
    assert 'ops/make_modebench_data.py' in sources
    assert 'src/oat_drgrpo/python_modebench.py' in sources


def test_selection_preserves_global_tier_margins_for_rare_cells():
    target = Counter({(k,): 1 for k in (4, 5, 6, 8, 9, 12, 18)})
    pools = {difficulty: [dict(row, answer_mode_count=key[0])
                           for key, row in zip(target, make_rows(7, difficulty))]
             for difficulty in range(4)}
    weights = (0, 5, 5, 10)
    selected = fit.select_rows('graph_coloring', pools, target, weights)
    counts = Counter(item['difficulty'] for item in selected)
    assert [counts[d] for d in range(4)] == fit.hamilton(7, weights)
    assert fit.cell_histogram('graph_coloring', [item['row'] for item in selected]) == target


def test_local_dependency_hash_includes_relative_verifier_imports():
    sources = fit.local_dependency_sources([ROOT / 'src/oat_drgrpo/math_grader.py'])
    assert 'src/oat_drgrpo/pantry_plan.py' in sources
    assert 'src/oat_drgrpo/python_modebench_process.py' in sources


def test_recipe_pins_allocator_and_rejects_dependency_drift(fitted, monkeypatch):
    recipe, recipe_path, _, _, _, _ = fitted
    allocator = 'ops/exp_scaling/modebench_level3_allocation.py'
    sources = recipe['provenance']['generator_sources_sha256']
    assert sources[allocator] == fit.file_sha(ROOT / allocator)
    changed = {**sources, allocator: 'changed-after-fitting'}
    monkeypatch.setattr(final, 'generator_sources', lambda domain: changed)
    with pytest.raises(ValueError, match='sources changed'):
        final.load_recipe(recipe_path, 'mathir')


def test_recipe_records_forecast_objective_and_both_gates(fitted):
    recipe, _, _, _, _, _ = fitted
    assert recipe['schema'] == 'modebench_level3_development_recipe_v2'
    assert recipe['selection']['algorithm'] == fit.SELECTION_ALGORITHM
    assert recipe['selection']['selected_residual_used_for_ranking'] is False
    assert recipe['development']['selected_development_sets_scored'] == 1
    assert recipe['development']['expected_metrics'] == recipe['development']['selected_metrics']
    assert recipe['development']['expected_differences'] == recipe['development']['differences']
    assert recipe['development']['gates'] == {
        'expected': {'pass1': True, 'pass8': True},
        'selected': {'pass1': True, 'pass8': True},
    }


def test_forecast_uses_exact_cell_allocation_and_every_pool_row(monkeypatch):
    pools = {difficulty: make_rows(4, difficulty) for difficulty in range(4)}
    for rows in pools.values():
        rows[0]['answer_mode_count'] = 4
        for row in rows[1:]:
            row['answer_mode_count'] = 6
    target = Counter({(4,): 1, (6,): 3})
    scores = {difficulty: {fit.sha(row): {metric: difficulty / 10 + (row['answer_mode_count'] == 6) * .4
                                           for metric in fit.TOLERANCES}
                           for row in rows} for difficulty, rows in pools.items()}
    weights = (5, 5, 5, 5)
    monkeypatch.setattr(fit, 'weight_grid', lambda: iter([weights]))
    result = fit.choose_mixture('graph_coloring', pools, scores, target, {'pass1': .5, 'pass8': .5})
    allocation = fit.allocate_cells(target, weights, fit.SELECTION_SEED)
    expected = sum(allocation[key][difficulty] * (difficulty / 10 + (key == (6,)) * .4)
                   for key in target for difficulty in range(4)) / 4
    assert result['expected_metrics'] == pytest.approx({'pass1': expected, 'pass8': expected})
    assert result['selected_development_sets_scored'] == 1


def test_forecast_ranking_ignores_selected_residual_even_as_tie_break(monkeypatch):
    pools = {difficulty: make_rows(4, difficulty) for difficulty in range(4)}
    indexed = fit.ordered_cells('mathir', pools, fit.SELECTION_SEED)
    scores = {}
    for difficulty in range(4):
        # Every full cell has mean .5. The lexicographically preferred recipe
        # selects a zero-accuracy prefix; the other recipe selects .5 exactly.
        values = (.5, .5, .5, .5) if difficulty < 2 else (0., 0., 1., 1.)
        scores[difficulty] = {digest: {metric: value for metric in fit.TOLERANCES}
                              for (_, _, digest), value in zip(indexed[difficulty][(5,)], values)}
    weights = [(10, 10, 0, 0), (0, 0, 10, 10)]
    monkeypatch.setattr(fit, 'weight_grid', lambda: iter(weights))
    result = fit.choose_mixture('mathir', pools, scores, Counter({(5,): 4}), {'pass1': .5, 'pass8': .5})
    assert result['weights'] == (0, 0, 10, 10)
    assert result['expected_metrics'] == {'pass1': .5, 'pass8': .5}
    assert result['metrics'] == {'pass1': 0., 'pass8': 0.}
    assert all(result['gates']['expected'].values())
    assert not any(result['gates']['selected'].values())
    # Rearranging outcomes inside each cell changes the scored selected set,
    # while preserving every full-cell mean and therefore the chosen weights.
    for difficulty in (2, 3):
        for index, (_, _, digest) in enumerate(indexed[difficulty][(5,)]):
            scores[difficulty][digest] = {metric: float(index < 2) for metric in fit.TOLERANCES}
    rearranged = fit.choose_mixture('mathir', pools, scores, Counter({(5,): 4}), {'pass1': .5, 'pass8': .5})
    assert rearranged['weights'] == result['weights']
    assert rearranged['expected_metrics'] == result['expected_metrics']
    assert rearranged['objective'] == result['objective']
    assert rearranged['metrics'] == {'pass1': 1., 'pass8': 1.}


def test_primary_optimum_with_failing_selected_set_is_reported_failed(fitted, monkeypatch):
    _, _, _, receipts, baseline, sources = fitted
    pools = {difficulty: fit.read_jsonl(source) for difficulty, source in enumerate(sources)}
    indexed = fit.ordered_cells('mathir', pools, fit.SELECTION_SEED)
    for difficulty, receipt_path in enumerate(receipts):
        ordered = indexed[difficulty][(5,)]
        values = ([0.] * 64 + [1.] * 64) if difficulty < 2 else ([.5] * 64 + [.25] * 64)
        by_hash = {digest: value for (_, _, digest), value in zip(ordered, values)}
        payload = json.loads(receipt_path.read_text())
        for row in payload['prompt_results']:
            value = by_hash[row['row_sha256']]
            row['pass1'] = row['pass8'] = value
            for index, draw in enumerate(row['draws']):
                verified = index < value * 4
                draw['pass1'] = draw['pass8'] = float(verified)
                for attempt in draw['attempts']:
                    attempt['verified'] = verified
                    attempt['canonical_key'] = 'mode' if verified else None
        for metric in fit.TOLERANCES:
            payload['metrics'][metric] = sum(row[metric] for row in payload['prompt_results']) / 128
        write_json(receipt_path, payload)
    monkeypatch.setattr(fit, 'weight_grid', lambda: iter([(10, 10, 0, 0), (0, 0, 10, 10)]))
    result = fit.fit_recipe(baseline, receipts, 'mathir')
    assert result['weight_units'] == [10, 10, 0, 0]
    assert result['development']['expected_metrics'] == {'pass1': .5, 'pass8': .5}
    assert result['development']['selected_metrics'] == {'pass1': 0., 'pass8': 0.}
    assert result['development']['gates']['expected'] == {'pass1': True, 'pass8': True}
    assert result['development']['gates']['selected'] == {'pass1': False, 'pass8': False}
    assert result['development_fit_pass'] is False
    assert result['decision'] == 'development_fit_failed_revise_candidates'


@pytest.mark.parametrize('field', ('expected_metrics', 'expected_differences', 'gates'))
def test_finalizer_requires_new_forecast_gate_evidence(fitted, tmp_path, field):
    recipe, _, _, _, _, _ = fitted
    del recipe['development'][field]
    path = tmp_path / f'missing_{field}.json'
    write_json(path, recipe)
    with pytest.raises(ValueError, match='expected|gate'):
        final.load_recipe(path, 'mathir')


def test_finalizer_rejects_selected_only_algorithm_even_with_current_source_hash(fitted, tmp_path):
    recipe, _, _, _, _, _ = fitted
    recipe['selection']['algorithm'] = 'controlled_cell_and_global_tier_rounding_then_fixed_hash_order'
    path = tmp_path / 'old_algorithm.json'
    write_json(path, recipe)
    with pytest.raises(ValueError, match='forecast-only selection'):
        final.load_recipe(path, 'mathir')


def test_finalizer_recomputes_gate_instead_of_trusting_pass_boolean(fitted, tmp_path):
    recipe, _, _, _, _, _ = fitted
    recipe['development']['expected_metrics']['pass8'] = .7
    recipe['development']['expected_differences']['pass8'] = .2
    assert recipe['development']['gates']['expected']['pass8'] is True
    path = tmp_path / 'forged_forecast_gate.json'
    write_json(path, recipe)
    with pytest.raises(ValueError, match='expected development gate'):
        final.load_recipe(path, 'mathir')


@pytest.mark.parametrize('mutation', ['source', 'pool_bytes', 'new_pool', 'history'])
def test_finalize_refuses_inputs_changed_during_generation(fitted, tmp_path, monkeypatch, mutation):
    recipe, recipe_path, _, _, _, sources = fitted
    sources = list(sources)
    history, state = set(), {'changed': False}
    current_sources = final.generator_sources
    monkeypatch.setattr(final, 'DOMAINS', ('mathir',))
    monkeypatch.setattr(final, 'reference_rows', lambda domain, split:
                        make_rows(384 if split == 'train' else 128, 0, 'reference'))
    monkeypatch.setattr(final, 'historical_ids', lambda domain: set(history))
    monkeypatch.setattr(final, 'candidate_pool_files', lambda domain, recipe: list(sources))

    def sources_after_edit(domain):
        hashes = current_sources(domain)
        if mutation == 'source' and state['changed']:
            hashes['concurrent_generator_edit.py'] = 'changed'
        return hashes

    def fake_generator(domain, target, excluded, seed, tag, difficulty, multiplier, **extras):
        if tag == 'level3_eval' and not state['changed']:
            state['changed'] = True
            if mutation == 'pool_bytes':
                sources[0].write_text(sources[0].read_text() + '\n')
            elif mutation == 'new_pool':
                extra = tmp_path / 'concurrent_pool.jsonl'
                extra.write_text(json.dumps(make_rows(1, 0, 'concurrent')[0]) + '\n')
                sources.append(extra)
            elif mutation == 'history':
                history.add(('mathir', 'concurrent_historical_identity'))
        return make_rows(sum(target.values()), difficulty, tag)

    monkeypatch.setattr(final, 'generator_sources', sources_after_edit)
    monkeypatch.setattr(final, 'generator', lambda domain: fake_generator)
    output = tmp_path / 'changed_data'
    with pytest.raises(ValueError, match='changed during generation'):
        final.finalize({'mathir': recipe_path}, output)
    assert state['changed']
    assert not output.exists()
    assert not list(tmp_path.glob('.changed_data.*'))


def test_finalizer_refuses_source_snapshot_different_from_fitted_recipe(fitted, monkeypatch):
    recipe, _, _, _, _, _ = fitted
    snapshot = dict(recipe['provenance']['generator_sources_sha256'])
    snapshot['changed_after_recipe_validation.py'] = 'new'
    monkeypatch.setattr(final, 'generator_sources', lambda domain: snapshot)
    frozen = {'finalizer_source_sha256': fit.file_sha(Path(final.__file__)),
              'generator_sources_sha256': {'mathir': snapshot}}
    with pytest.raises(ValueError, match='generator sources changed'):
        final.verify_generation_inputs_unchanged({'mathir': recipe}, frozen, {}, {})
