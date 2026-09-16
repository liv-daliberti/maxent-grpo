from collections import Counter
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / 'ops/exp_scaling/fit_modebench_level3_fixed_reference.py'
spec = importlib.util.spec_from_file_location('fixed_reference_fitter_test', SOURCE)
fit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fit)


def fixture_data(targets=None):
    targets = targets or {'dev': Counter({(4,): 64, (5,): 64}), 'eval': Counter({(4,): 128})}
    union = fit.validate_targets(targets)
    pools, scores = {}, {}
    for tier in range(4):
        pools[tier] = [{'problem': f'{tier}/{key[0]}/{index}', 'answer_mode_count': key[0],
                       'level3_difficulty': tier}
                      for key, count in sorted(union.items()) for index in range(count)]
        metrics = {'pass1': .2, 'pass8': .5} if tier == 0 else {'pass1': .6, 'pass8': .9}
        scores[tier] = {fit.mixture.sha(row): dict(metrics) for row in pools[tier]}
    return pools, scores, targets, {'pass1': .2, 'pass8': .5}


def run(data):
    return fit.choose_fixed_reference_mixture('graph_coloring', *data)


def test_dual_forecasts_score_exactly_one_dev_set(monkeypatch):
    data = fixture_data()
    original = fit.mixture.select_rows
    calls = []
    def spy(domain, pools, target, *args):
        calls.append(target)
        return original(domain, pools, target, *args)
    monkeypatch.setattr(fit.mixture, 'select_rows', spy)
    result = run(data)
    assert calls == [data[2]['dev']]
    assert result['weights'] == (20, 0, 0, 0)
    assert result['weight_combinations_considered'] == 1771
    assert result['selected_development_sets_scored'] == 1
    assert result['selected_evaluation_sets_scored'] == 0
    assert result['development_fit_pass'] is True
    assert len(result['gates']) == 3
    assert all(len(gates) == 2 for gates in result['gates'].values())


def test_selected_failure_never_changes_weight_ranking_or_searches_again():
    data = fixture_data()
    first = run(data)
    pools, scores, targets, baseline = deepcopy(data)
    selected_hashes = {row['row_sha256'] for row in first['selected'] if row['row']['answer_mode_count'] == 4}
    for row in pools[0]:
        if row['answer_mode_count'] == 4:
            digest = fit.mixture.sha(row)
            scores[0][digest]['pass1'] = .32 if digest in selected_hashes else .08
    result = run((pools, scores, targets, baseline))
    assert result['weights'] == first['weights']
    assert result['selected'] == first['selected']
    assert result['expected_dev_metrics']['pass1'] == pytest.approx(.2)
    assert result['expected_eval_metrics']['pass1'] == pytest.approx(.2)
    assert result['metrics']['pass1'] == pytest.approx(.26)
    assert result['gates']['selected']['pass1'] is False
    assert result['development_fit_pass'] is False
    assert result['selected_development_sets_scored'] == 1


def test_eval_only_cell_changes_forecast_weight_choice():
    targets = {'dev': Counter({(4,): 128}), 'eval': Counter({(4,): 64, (5,): 64})}
    data = fixture_data(targets)
    first = run(data)
    pools, scores, targets, baseline = deepcopy(data)
    for tier in range(4):
        for row in pools[tier]:
            digest = fit.mixture.sha(row)
            if tier == 1:
                scores[tier][digest] = {'pass1': .2, 'pass8': .5}
            elif tier == 0 and row['answer_mode_count'] == 5:
                scores[tier][digest] = {'pass1': .8, 'pass8': .95}
    result = run((pools, scores, targets, baseline))
    assert result['weights'] != first['weights']
    assert result['weights'] == (0, 20, 0, 0)
    assert {row['row']['answer_mode_count'] for row in result['selected']} == {4}
    assert result['development_fit_pass'] is True


@pytest.mark.parametrize('mutation', ['missing_cell', 'extra_row', 'missing_score', 'extra_score', 'duplicate_row'])
def test_exact_union_and_full_score_coverage_required(mutation):
    pools, scores, targets, baseline = fixture_data()
    if mutation == 'missing_cell':
        pools[0] = [row for row in pools[0] if row['answer_mode_count'] != 5]
    elif mutation == 'extra_row':
        pools[0].append({'problem': 'extra', 'answer_mode_count': 4})
    elif mutation == 'missing_score':
        scores[0].pop(next(iter(scores[0])))
    elif mutation == 'extra_score':
        scores[0]['extra'] = {'pass1': .2, 'pass8': .5}
    else:
        pools[0][1] = dict(pools[0][0])
    with pytest.raises(ValueError):
        run((pools, scores, targets, baseline))


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -1, 1.01, True, '0.2'])
def test_reference_must_be_finite_probability(bad):
    pools, scores, targets, baseline = fixture_data()
    baseline['pass1'] = bad
    with pytest.raises(ValueError):
        run((pools, scores, targets, baseline))


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -1, 1.01, True, '0.2'])
def test_candidate_scores_must_be_finite_probabilities(bad):
    pools, scores, targets, baseline = fixture_data()
    scores[0][next(iter(scores[0]))]['pass1'] = bad
    with pytest.raises(ValueError):
        run((pools, scores, targets, baseline))


@pytest.mark.parametrize('split', ['dev', 'eval'])
def test_fixed_128_row_forecast_histograms(split):
    pools, scores, targets, baseline = fixture_data()
    targets[split][(4,)] -= 1
    with pytest.raises(ValueError, match='128-row'):
        run((pools, scores, targets, baseline))


def test_no_alternate_seed_or_third_domain():
    data = fixture_data()
    with pytest.raises(ValueError, match='seed is fixed'):
        fit.choose_fixed_reference_mixture('graph_coloring', *data, seed=6391702)
    with pytest.raises(ValueError, match='two prospectively'):
        fit.choose_fixed_reference_mixture('pantry', *data)


def test_no_truncated_weight_grid(monkeypatch):
    monkeypatch.setattr(fit.mixture, 'weight_grid', lambda: iter([(20, 0, 0, 0)]))
    with pytest.raises(ValueError, match='1771-weight'):
        run(fixture_data())


def wrapper_fixture(monkeypatch, tmp_path):
    import modebench_level3_v3_common as common
    import modebench_level3_discrete as discrete
    pools, scores, targets, baseline = fixture_data()
    campaign = tmp_path / 'v3'
    results = tmp_path / 'results'
    pool_root = tmp_path / 'graph_v8'
    registration_path = campaign / 'registration.json'
    digest = lambda path: 'sha:' + str(Path(path).resolve())
    monkeypatch.setattr(common, 'CAMPAIGN', campaign)
    monkeypatch.setattr(common, 'RESULTS', results)
    monkeypatch.setattr(common, 'REGISTRATION', registration_path)
    monkeypatch.setattr(common, 'POOL_ROOTS', {'graph_coloring': pool_root})
    monkeypatch.setattr(common, 'digest', digest)
    monkeypatch.setattr(discrete, 'row_identity', lambda domain, row: (domain, row['problem']))
    paths = {tier: results / f'calibration_3b_graph_v8_d{tier}.json' for tier in range(4)}
    revision = {'name': 'graph_v8', 'pool_root': str(pool_root),
                'generator_path': str(tmp_path / 'generator.py'),
                'materializer_path': str(tmp_path / 'materializer.py'),
                'development_receipts': {str(tier): str(path) for tier, path in paths.items()}}
    for kind in ('generator', 'materializer'):
        revision[kind + '_sha256'] = digest(revision[kind + '_path'])
    registration = {'candidate_revisions': {'graph_coloring': revision},
                    'files_sha256': {str(SOURCE): digest(SOURCE),
                        **{revision[kind + '_path']: revision[kind + '_sha256']
                           for kind in ('generator', 'materializer')}}}
    validated = []
    def validate(path, pin):
        assert Path(path) == registration_path and pin == 'registered'
        validated.append((path, pin))
        return deepcopy(registration)
    monkeypatch.setattr(common, 'validate_registration', validate)
    target = {'receipt': {'identity': {'interface': 'fixed', 'code_sha256': {'grader': 'frozen'}}},
              'metrics': baseline, 'provenance': {'role': 'frozen_benchmark_reference',
                  'receipt_path': 'original_eval_receipt', 'draw_labels': [6329000,6329001,6329002,6329003]}}
    monkeypatch.setattr(common, 'fixed_references', lambda: {'graph_coloring': deepcopy(target)})
    monkeypatch.setattr(common, 'authenticate_inherited', lambda: {'models': {'3b': {'frozen': '3b'}}})
    monkeypatch.setattr(common, 'reference_histograms', lambda domain: deepcopy(targets))
    monkeypatch.setattr(common, 'calibration_histogram', lambda domain: fit.validate_targets(targets))
    receipts, certificates = {}, {}
    for tier, path in paths.items():
        pool = pool_root / 'pools/graph_coloring' / f'difficulty_{tier}.jsonl'
        receipts[path] = {'identity': {'model': {'frozen': '3b'}, 'interface': 'fixed',
            'code_sha256': {'grader': 'frozen'}, 'source': {'kind': 'jsonl', 'path': str(pool),
                'file_sha256': digest(pool)}}, 'identity_sha256': 'identity:' + str(tier)}
        certificates[pool.with_suffix('.identity.json')] = {
            'candidate_revision': 'graph_v8', 'registration_path': str(registration_path),
            'registration_sha256': 'registered', 'rows_sha256': fit.mixture.row_hash(pools[tier]),
            'source_sha256': revision['generator_sha256'], 'checks': {'structural': True}}
    def load(path, domain):
        tier = next(tier for tier, expected in paths.items() if expected == path)
        return deepcopy(receipts[path]), deepcopy(pools[tier]), deepcopy(scores[tier])
    monkeypatch.setattr(common, 'validate_new_candidate_receipt', load)
    monkeypatch.setattr(common, 'read', lambda path: deepcopy(certificates[Path(path)]))
    verified = []
    def verify(pins):
        assert all(value == digest(path) for path, value in pins.items())
        verified.append(dict(pins))
    monkeypatch.setattr(common, 'verify_pins', verify)
    return locals()


def test_authenticated_wrapper_keeps_historical_reference_role(monkeypatch, tmp_path):
    env = wrapper_fixture(monkeypatch, tmp_path)
    result = fit.fit_recipe(env['registration_path'], 'registered', 'graph_coloring')
    assert result['development_fit_pass'] is True
    assert result['provenance']['reference_target']['role'] == 'frozen_benchmark_reference'
    assert result['provenance']['reference_target']['draw_labels'] == [6329000,6329001,6329002,6329003]
    assert result['provenance']['seeds'] == [6428000,6428001,6428002,6428003]
    assert result['information_boundary']['historical_level3_confirmation_used_for_mixture_fitting'] is False
    assert result['information_boundary']['new_candidate_fitting_uses_development_outcomes_only'] is True
    assert len(env['validated']) == 2 and len(env['verified']) == 1
    assert len(env['verified'][0]) == 12
    assert not env['campaign'].exists()  # In-memory verification never publishes.


@pytest.mark.parametrize('field,value', [
    ('model', {'changed':'model'}), ('interface', 'changed'), ('code_sha256', {'grader':'changed'})])
def test_wrapper_rejects_candidate_runtime_drift(monkeypatch, tmp_path, field, value):
    env = wrapper_fixture(monkeypatch, tmp_path)
    env['receipts'][env['paths'][0]]['identity'][field] = value
    with pytest.raises(ValueError, match='model, interface or evaluator'):
        fit.fit_recipe(env['registration_path'], 'registered', 'graph_coloring')


@pytest.mark.parametrize('field,value', [
    ('candidate_revision','graph_v7'), ('registration_sha256','old'),
    ('rows_sha256','wrong'), ('source_sha256','wrong'), ('checks',{'structural':1})])
def test_wrapper_rejects_unregistered_pool_certificate(monkeypatch, tmp_path, field, value):
    env = wrapper_fixture(monkeypatch, tmp_path)
    env['certificates'][next(iter(env['certificates']))][field] = value
    with pytest.raises(ValueError, match='structural certificate'):
        fit.fit_recipe(env['registration_path'], 'registered', 'graph_coloring')


def test_wrapper_rejects_old_confirmation_or_alternate_development_path(monkeypatch, tmp_path):
    env = wrapper_fixture(monkeypatch, tmp_path)
    paths = list(env['paths'].values())
    paths[0] = tmp_path / 'old_confirmation_3b_graph_coloring.json'
    with pytest.raises(ValueError, match='four registered new candidate'):
        fit.fit_recipe(env['registration_path'], 'registered', 'graph_coloring', paths)


def test_wrapper_refuses_unpinned_fitter_source(monkeypatch, tmp_path):
    env = wrapper_fixture(monkeypatch, tmp_path)
    env['registration']['files_sha256'][str(SOURCE)] = 'changed'
    with pytest.raises(ValueError, match='fitter was not'):
        fit.fit_recipe(env['registration_path'], 'registered', 'graph_coloring')


def test_wrapper_checks_evidence_again_before_return(monkeypatch, tmp_path):
    env = wrapper_fixture(monkeypatch, tmp_path)
    import modebench_level3_v3_common as common
    def changed(pins):
        raise ValueError('receipt changed during fit')
    monkeypatch.setattr(common, 'verify_pins', changed)
    with pytest.raises(ValueError, match='changed during fit'):
        fit.fit_recipe(env['registration_path'], 'registered', 'graph_coloring')


def test_cli_publishes_only_once_under_immutable_intent(monkeypatch, tmp_path):
    env = wrapper_fixture(monkeypatch, tmp_path)
    import sys
    calls = []
    def fitted(*args):
        calls.append(args)
        return {'development_fit_pass': True, 'weights': [1,0,0,0]}
    monkeypatch.setattr(fit, 'fit_recipe', fitted)
    output = env['campaign'] / 'recipes/graph_coloring.json'
    monkeypatch.setattr(sys, 'argv', ['fit', '--registration', str(env['registration_path']),
        '--registration-sha256', 'registered', '--domain', 'graph_coloring', '--output', str(output)])
    fit.main()
    before = output.read_bytes()
    assert output.with_suffix('.fit_intent.json').exists() and len(calls) == 1
    with pytest.raises(ValueError, match='ambiguous fit intent'):
        fit.main()
    assert output.read_bytes() == before and len(calls) == 1


def test_cli_failed_fit_retains_intent_and_never_retries(monkeypatch, tmp_path):
    env = wrapper_fixture(monkeypatch, tmp_path)
    import sys
    calls = []
    def failed(*args):
        calls.append(args)
        raise RuntimeError('synthetic failure during sole fit')
    monkeypatch.setattr(fit, 'fit_recipe', failed)
    output = env['campaign'] / 'recipes/graph_coloring.json'
    monkeypatch.setattr(sys, 'argv', ['fit', '--registration', str(env['registration_path']),
        '--registration-sha256', 'registered', '--domain', 'graph_coloring', '--output', str(output)])
    with pytest.raises(RuntimeError, match='sole fit'):
        fit.main()
    assert output.with_suffix('.fit_intent.json').exists() and not output.exists()
    with pytest.raises(ValueError, match='ambiguous fit intent'):
        fit.main()
    assert len(calls) == 1


def test_cli_refuses_alternate_output_before_publication(monkeypatch, tmp_path):
    env = wrapper_fixture(monkeypatch, tmp_path)
    import sys
    output = tmp_path / 'alternate.json'
    monkeypatch.setattr(sys, 'argv', ['fit', '--registration', str(env['registration_path']),
        '--registration-sha256', 'registered', '--domain', 'graph_coloring', '--output', str(output)])
    with pytest.raises(ValueError, match='canonical new recipe'):
        fit.main()
    assert not output.exists() and not env['campaign'].exists()
