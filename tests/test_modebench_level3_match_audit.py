"""Difficulty matching uses full five-domain evidence and prompt-level uncertainty."""
from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

PATH = Path(__file__).resolve().parents[1] / 'ops/audit_modebench_level3_match.py'
SPEC = spec_from_file_location('modebench_level3_match_audit_test', PATH)
audit = module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def receipt(domain='countdown', role='baseline', *, development=False, count=128,
            draw_count=4, successful=64, correct=1, distinct=False):
    label, level = ('05b', 'level1') if role == 'baseline' else ('3b', 'level3')
    split = 'dev' if development else 'eval'
    interface = audit.frozen_interface(domain)
    start = (100 if role == 'baseline' else 200) if development else 6319000
    seeds = list(range(start, start + draw_count))
    rows = [{'problem': f'{role} {domain} {index}', 'answer': json.dumps({'target': index}),
             'answer_mode_count': 4} for index in range(count)]
    prompt_results = []
    for index, row in enumerate(rows):
        successes = correct if index < successful else 0
        draws = []
        metrics = {'pass1': successes / 8, 'pass8': float(successes > 0),
                   'distinct8': float(successes if distinct else bool(successes))}
        for seed in seeds:
            attempts = [{'text': f'answer {sample}', 'verified': sample < successes,
                         'canonical_key': str(sample) if distinct and sample < successes else
                                          'valid' if sample < successes else None,
                         'token_count': 4} for sample in range(8)]
            draws.append({'seed': seed, 'verified_count': successes, 'attempts': attempts, **metrics})
        prompt_results.append({'row_index': index, 'row_sha256': audit.sha(row),
                               'spec_sha256': audit.sha(json.loads(row['answer'])),
                               'problem_sha256': audit.sha(row['problem']),
                               'row_metadata': {'answer_mode_count': 4}, 'draws': draws, **metrics})
    identity = {'schema': audit.RECEIPT_SCHEMA, 'domain': domain, 'level': level, 'split': split,
                'model': {'label': label, 'vllm_version': 'test'}, 'interface': interface,
                'interface_sha256': audit.sha(interface), 'seeds': seeds, 'code_sha256': {'test': 'frozen'},
                'source': {'total_rows': count, 'selected_rows': count, 'row_offset': 0, 'row_limit': 0,
                           'rows_sha256': audit.sha(rows), 'all_rows_sha256': audit.sha(rows)}}
    result = {'schema': audit.RECEIPT_SCHEMA, 'status': 'complete', 'domain': domain, 'level': level,
              'split': split, 'model_label': label, 'identity': identity, 'identity_sha256': audit.sha(identity),
              'sampling': {**interface, 'seeds': seeds}, 'prompt_results': prompt_results,
              'metrics': {'rows': count, **{metric: sum(row[metric] for row in prompt_results) / count
                                           for metric in audit.METRICS}},
              'information_boundary': {'evaluation_prompts_loaded': not development,
                                       'confirmation_explicitly_authorized': not development,
                                       'treatment_training_started': False}}
    return result, rows


def write_pairs(tmp_path, candidate_successful=72):
    pairs = []
    for domain in audit.DOMAINS:
        pair = {'domain': domain}
        for role in ('baseline', 'candidate'):
            payload, _ = receipt(domain, role, successful=64 if role == 'baseline' else candidate_successful)
            path = tmp_path / f'{role}_{domain}.json'
            path.write_text(json.dumps(payload))
            pair[role] = str(path)
        pairs.append(pair)
    return pairs


def test_all_five_confirmation_domains_match_with_registered_seeds(frozen_evidence):
    root, pairs = frozen_evidence
    report = audit.audit_pairs(pairs, dataset_root=root, bootstrap_replicates=100)
    assert report['status'] == 'observed_approximate_match'
    assert report['confirmation_match_verified']
    assert report['all_five_domains_complete']
    assert report['criteria']['statistical_equivalence_claimed'] is False
    for domain in report['domains'].values():
        assert domain['candidate_minus_baseline']['pass8'] == .0625
        assert domain['candidate_minus_baseline']['pass1'] == .0625 / 8
        assert set(domain['bootstrap_intervals']['pass1']) == {'90_percent', '95_percent'}
        assert len(domain['receipts']['candidate']['sha256']) == 64


def test_missing_domain_cannot_claim_five_domain_match(frozen_evidence):
    root, pairs = frozen_evidence
    report = audit.audit_pairs(pairs[:-1], dataset_root=root, bootstrap_replicates=100)
    assert report['status'] == 'incomplete'
    assert report['missing_domains'] == ['pantry']
    assert not report['all_five_observed_approximate_match']


def test_observed_match_tolerances_are_both_required(frozen_evidence):
    root, pairs = frozen_evidence
    for pair in pairs:
        rewrite_scores(pair['candidate'], successful=76)
    report = audit.audit_pairs(pairs, dataset_root=root, bootstrap_replicates=100)
    assert report['status'] == 'outside_match_tolerance'
    assert not report['confirmation_match_verified']
    assert report['domains']['countdown']['within_tolerance'] == {'pass1': True, 'pass8': False}
    rewrite_scores(pairs[0]['candidate'], correct=2)
    report = audit.audit_pairs(pairs, dataset_root=root, bootstrap_replicates=100)
    assert report['domains']['countdown']['within_tolerance'] == {'pass1': False, 'pass8': True}


def test_receipt_metric_tampering_is_rejected():
    payload, _ = receipt()
    payload['prompt_results'][0]['draws'][0]['pass1'] = .99
    with pytest.raises(ValueError, match='metric mismatch'):
        audit.validate_receipt(payload, domain='countdown', role='baseline')
    payload, _ = receipt()
    payload['prompt_results'][0]['draws'][0]['attempts'][0]['verified'] = False
    with pytest.raises(ValueError, match='verified/canonical'):
        audit.validate_receipt(payload, domain='countdown', role='baseline')
    payload, _ = receipt()
    payload['metrics']['pass8'] = .99
    with pytest.raises(ValueError, match='metric mismatch'):
        audit.validate_receipt(payload, domain='countdown', role='baseline')


def test_confirmation_requires_correct_role_full_rows_and_four_draws():
    for options, error in (({'count': 64}, '128 prompts'), ({'draw_count': 1}, 'four draws')):
        payload, _ = receipt(**options)
        with pytest.raises(ValueError, match=error):
            audit.validate_receipt(payload, domain='countdown', role='baseline')
    payload, _ = receipt()
    with pytest.raises(ValueError, match='model_label'):
        audit.validate_receipt(payload, domain='countdown', role='candidate')
    payload, _ = receipt(development=True, count=64, draw_count=1)
    assert len(audit.validate_receipt(payload, domain='countdown', role='baseline', development=True)) == 64
    with pytest.raises(ValueError, match='split'):
        audit.validate_receipt(payload, domain='countdown', role='baseline')


def test_interface_and_pinned_receipt_hash_mismatch_cannot_pass(frozen_evidence):
    root, pairs = frozen_evidence
    pair = pairs[0]
    pair['baseline_sha256'] = '0' * 64
    report = audit.audit_pairs(pairs, dataset_root=root, bootstrap_replicates=100)
    assert report['status'] == 'invalid_evidence'
    assert 'pinned receipt hash' in report['errors']['countdown']
    del pair['baseline_sha256']
    path = Path(pair['candidate'])
    candidate = json.loads(path.read_text())
    interface = audit.frozen_interface('countdown', 'original_level1')
    candidate['identity']['interface'] = interface
    candidate['identity']['interface_sha256'] = audit.sha(interface)
    candidate['sampling'] = {**interface, 'seeds': candidate['identity']['seeds']}
    candidate['identity_sha256'] = audit.sha(candidate['identity'])
    path.write_text(json.dumps(candidate))
    report = audit.audit_pairs(pairs, dataset_root=root, bootstrap_replicates=100)
    assert 'interface differs' in report['errors']['countdown']


def test_bootstrap_keeps_prompt_variation_and_is_reproducible():
    # Each entry is the average of all draws for one prompt. Resampling draws
    # independently would incorrectly turn this two-prompt population into eight.
    baseline = [{'pass1': x, 'pass8': x, 'distinct8': x} for x in (0., 1.)]
    candidate = [{'pass1': .5, 'pass8': .5, 'distinct8': .5}] * 2
    result = audit.bootstrap_differences(baseline, candidate, seed=12, replicates=1000)
    assert result == audit.bootstrap_differences(baseline, candidate, seed=12, replicates=1000)
    assert result['pass8']['95_percent'] == [-.5, .5]


def test_distinct8_is_diagnostic_not_a_matching_gate(frozen_evidence):
    root, pairs = frozen_evidence
    for pair in pairs:
        for role in ('baseline', 'candidate'):
            rewrite_scores(pair[role], successful=64, correct=4, distinct=role == 'candidate')
    report = audit.audit_pairs(pairs, dataset_root=root, bootstrap_replicates=100)
    assert report['confirmation_match_verified']
    assert report['domains']['countdown']['candidate_minus_baseline']['distinct8'] == 1.5


def test_saved_rows_are_checked_against_both_hash_conventions(tmp_path, monkeypatch):
    payload, rows = receipt(role='candidate')
    monkeypatch.setitem(sys.modules, 'datasets', SimpleNamespace(load_from_disk=lambda _: {'multi_answer': rows}))
    expected = {'rows': len(rows), 'rows_sha256': audit.materializer_row_hash(rows), 'path': str(tmp_path)}
    result = audit.verify_dataset(payload, tmp_path, expected)
    assert result['verified_against_dataset_identity']
    assert result['rows_sha256'] != result['evaluator_rows_sha256']
    with pytest.raises(ValueError, match='frozen dataset identity rows hash'):
        audit.verify_dataset(payload, tmp_path, dict(expected, rows_sha256='0' * 64))
    rows[0]['problem'] = 'Changed after evaluation'
    with pytest.raises(ValueError, match='source rows do not match'):
        audit.verify_dataset(payload, tmp_path, expected)


def test_duplicate_domain_pairs_are_rejected(frozen_evidence):
    root, pairs = frozen_evidence
    with pytest.raises(ValueError, match='duplicate domain'):
        audit.audit_pairs(pairs + [pairs[0]], dataset_root=root, bootstrap_replicates=100)


def rewrite_receipt(path, transform):
    value = json.loads(Path(path).read_text())
    transform(value)
    identity = value['identity']
    identity['interface_sha256'] = audit.sha(identity['interface'])
    value['sampling'] = {**identity['interface'], 'seeds': identity['seeds']}
    for row in value['prompt_results']:
        for draw, seed in zip(row['draws'], identity['seeds']):
            draw['seed'] = seed
    value['identity_sha256'] = audit.sha(identity)
    Path(path).write_text(json.dumps(value))


def test_confirmation_cannot_pass_without_frozen_dataset_root(tmp_path):
    with pytest.raises(ValueError, match='confirmation requires --dataset-root'):
        audit.audit_pairs(write_pairs(tmp_path), bootstrap_replicates=100)


def rewrite_scores(path, **options):
    """Change simulated outcomes while retaining authenticated source/settings."""
    value = json.loads(Path(path).read_text())
    role = 'baseline' if value['model_label'] == '05b' else 'candidate'
    replacement, _ = receipt(value['domain'], role, **options)
    for key in ('prompt_results', 'metrics'):
        value[key] = replacement[key]
    Path(path).write_text(json.dumps(value))


def recipe_document(domain):
    """Relevant fields emitted by the unchanged v2 development fitter."""
    return {
        'schema': 'modebench_level3_development_recipe_v2', 'domain': domain,
        'decision': 'development_fit_pass_pending_confirmation',
        'development_fit_pass': True,
        'selection': {
            'seed': 6391701,
            'algorithm': 'full_pool_cell_forecast_then_controlled_allocation_and_fixed_hash_order_v2',
            'weight_objective': 'full_pool_cell_means_at_exact_controlled_allocation',
            'selected_residual_used_for_ranking': False,
            'on_selected_gate_failure': 'fail_without_alternate_weights_or_hash_seeds',
            'pantry_cells_include_family': domain == 'pantry',
            'outcome_independent_within_pool_cell_order': True,
        },
        'development': {
            'baseline_metrics': {'pass1': .0625, 'pass8': .5},
            'expected_metrics': {'pass1': .0625, 'pass8': .5},
            'selected_metrics': {'pass1': .0625, 'pass8': .5},
            'expected_differences': {'pass1': 0., 'pass8': 0.},
            'differences': {'pass1': 0., 'pass8': 0.},
            'tolerances': {'pass1': .04, 'pass8': .08},
            'gates': {stage: {'pass1': True, 'pass8': True}
                      for stage in ('expected', 'selected')},
            'selected_development_sets_scored': 1,
        },
        'provenance': {
            'interface': audit.frozen_interface(domain, 'level2_qwen_r5'),
            'seeds': [6318000, 6318001, 6318002, 6318003],
            'baseline_model': {'label': '05b', 'vllm_version': 'test',
                               'path': '/frozen/baseline-checkpoint'},
            'candidate_model': {'label': '3b', 'vllm_version': 'test',
                                'path': '/frozen/candidate-checkpoint'},
            'evaluator_and_verifier_code_sha256': {'test': 'frozen'},
        },
        'information_boundary': {'development_only': True, 'confirmation_outcomes_used': False},
    }


def repin_recipe(root, domain, recipe):
    """Authenticate mutations so rejection exercises content, not stale hashes."""
    path = root / 'recipes' / f'{domain}.json'
    path.write_text(json.dumps(recipe))
    digest = audit.file_sha(path)
    bundle_path = root / 'frozen_recipes.json'
    bundle = json.loads(bundle_path.read_text())
    bundle['recipes_sha256'][domain] = digest
    bundle_path.write_text(json.dumps(bundle))
    identity_path = root / 'identity.json'
    identity = json.loads(identity_path.read_text())
    identity['recipe_sha256'][domain] = digest
    identity['frozen_recipe_bundle_sha256'] = audit.file_sha(bundle_path)
    identity_path.write_text(json.dumps(identity))
    return identity


@pytest.fixture
def frozen_evidence(tmp_path, monkeypatch):
    root = tmp_path / 'final'
    (root / 'recipes').mkdir(parents=True)
    pairs, datasets, baseline_paths = [], {}, {}
    identity = {'recipe_sha256': {}, 'domains': {}}
    for domain in audit.DOMAINS:
        pair = {'domain': domain}
        for role in ('baseline', 'candidate'):
            payload, rows = receipt(domain, role, successful=64 if role == 'baseline' else 72)
            interface = audit.frozen_interface(domain, 'level2_qwen_r5')
            payload['identity']['interface'] = interface
            payload['identity']['interface_sha256'] = audit.sha(interface)
            payload['sampling'] = {**interface, 'seeds': payload['identity']['seeds']}
            payload['identity']['model']['path'] = f'/frozen/{role}-checkpoint'
            path = ((tmp_path / 'level1' / domain / 'eval') if role == 'baseline'
                    else root / domain / 'eval').resolve()
            path.mkdir(parents=True)
            datasets[str(path)] = rows
            payload['identity']['source'].update(kind='saved_dataset', path=str(path))
            payload['identity_sha256'] = audit.sha(payload['identity'])
            receipt_path = tmp_path / f'frozen_{role}_{domain}.json'
            receipt_path.write_text(json.dumps(payload))
            pair[role] = str(receipt_path)
            if role == 'baseline':
                baseline_paths[domain] = path
            else:
                identity['domains'][domain] = {'eval': {
                    'path': str(path), 'rows': 128,
                    'rows_sha256': audit.materializer_row_hash(rows),
                }}
        recipe = recipe_document(domain)
        recipe_path = root / 'recipes' / f'{domain}.json'
        recipe_path.write_text(json.dumps(recipe))
        identity['recipe_sha256'][domain] = audit.file_sha(recipe_path)
        pairs.append(pair)
    bundle = {'recipes_sha256': identity['recipe_sha256'], 'confirmation_outcomes_used': False}
    (root / 'frozen_recipes.json').write_text(json.dumps(bundle))
    identity['frozen_recipe_bundle_sha256'] = audit.file_sha(root / 'frozen_recipes.json')
    (root / 'identity.json').write_text(json.dumps(identity))
    monkeypatch.setattr(audit, 'level1_eval_path', lambda domain: baseline_paths[domain])
    monkeypatch.setitem(sys.modules, 'datasets', SimpleNamespace(
        load_from_disk=lambda path: {'multi_answer': datasets[str(Path(path).resolve())]}))
    return root, pairs


def test_confirmation_authenticates_recipes_and_automatically_pins_baseline_data(frozen_evidence):
    root, pairs = frozen_evidence
    report = audit.audit_pairs(pairs, dataset_root=root, bootstrap_replicates=100)
    assert report['confirmation_match_verified']
    assert report['criteria']['expected_confirmation_seeds'] == list(audit.DEFAULT_CONFIRMATION_SEEDS)
    for domain, result in report['domains'].items():
        assert result['frozen_recipe']['sha256'] == audit.file_sha(root / 'recipes' / f'{domain}.json')
        assert result['source_datasets']['baseline']['rows'] == 128
        assert result['source_datasets']['candidate']['verified_against_dataset_identity']


@pytest.mark.parametrize('schema,accepted', [
    ('modebench_level3_development_recipe_v2', True),
    ('modebench_level3_development_recipe_v1', False),
])
def test_confirmation_requires_v2_recipe_even_with_consistent_frozen_hashes(
        frozen_evidence, schema, accepted):
    root, pairs = frozen_evidence
    recipe = recipe_document('countdown')
    recipe['schema'] = schema
    repin_recipe(root, 'countdown', recipe)

    report = audit.audit_pairs(pairs, dataset_root=root, bootstrap_replicates=100)
    assert report['confirmation_match_verified'] is accepted
    if accepted:
        assert report['status'] == 'observed_approximate_match'
    else:
        assert report['status'] == 'invalid_evidence'
        assert report['errors'] == {'countdown': 'countdown: invalid frozen recipe schema/domain'}


@pytest.mark.parametrize('filename,error', [('recipes/countdown.json', 'frozen recipe hash'),
                                            ('frozen_recipes.json', 'recipe bundle hash')])
def test_changed_frozen_recipe_or_bundle_bytes_cannot_pass(frozen_evidence, filename, error):
    root, pairs = frozen_evidence
    path = root / filename
    path.write_text(path.read_text() + ' ')
    report = audit.audit_pairs(pairs, dataset_root=root, bootstrap_replicates=100)
    assert not report['confirmation_match_verified']
    assert error in report['errors']['countdown']


@pytest.mark.parametrize('field,error', [('interface', 'interface differs'),
                                         ('model', 'model identity differs'),
                                         ('code_sha256', 'code differs')])
def test_matching_pair_settings_must_still_match_frozen_recipe(frozen_evidence, field, error):
    root, pairs = frozen_evidence
    for role in ('baseline', 'candidate'):
        def change(value):
            identity = value['identity']
            if field == 'interface':
                identity[field] = audit.frozen_interface('countdown', 'original_level1')
            elif field == 'model':
                identity[field]['path'] = '/different/checkpoint'
            else:
                identity[field] = {'test': 'different'}
        rewrite_receipt(pairs[0][role], change)
    report = audit.audit_pairs(pairs, dataset_root=root, bootstrap_replicates=100)
    assert error in report['errors']['countdown']


def test_confirmation_rejects_wrong_or_unpaired_registered_seeds(frozen_evidence):
    root, pairs = frozen_evidence
    rewrite_receipt(pairs[0]['candidate'], lambda value: value['identity'].__setitem__('seeds', [7, 8, 9, 10]))
    report = audit.audit_pairs(pairs, dataset_root=root, bootstrap_replicates=100)
    assert 'confirmation seeds differ' in report['errors']['countdown']
    rewrite_receipt(pairs[0]['baseline'], lambda value: value['identity'].__setitem__('seeds', [7, 8, 9, 10]))
    report = audit.audit_pairs(pairs, dataset_root=root, bootstrap_replicates=100)
    assert 'confirmation seeds differ' in report['errors']['countdown']


def test_later_registered_seed_revision_must_remain_disjoint_from_development(frozen_evidence):
    root, pairs = frozen_evidence
    development_seeds = [6318000, 6318001, 6318002, 6318003]
    for role in ('baseline', 'candidate'):
        rewrite_receipt(pairs[0][role], lambda value: value['identity'].__setitem__('seeds', development_seeds))
    report = audit.audit_pairs(pairs, dataset_root=root, bootstrap_replicates=100,
                               expected_eval_seeds=development_seeds)
    assert 'overlap frozen development seeds' in report['errors']['countdown']
    revised = [6329000, 6329001, 6329002, 6329003]
    for pair in pairs:
        for role in ('baseline', 'candidate'):
            rewrite_receipt(pair[role], lambda value: value['identity'].__setitem__('seeds', revised))
    report = audit.audit_pairs(pairs, dataset_root=root, bootstrap_replicates=100,
                               expected_eval_seeds=revised)
    assert report['confirmation_match_verified']


def test_baseline_eval_source_cannot_be_swapped_for_another_dataset(frozen_evidence):
    root, pairs = frozen_evidence
    rewrite_receipt(pairs[0]['baseline'], lambda value: value['identity']['source'].__setitem__(
        'path', str(root / 'countdown' / 'eval')))
    report = audit.audit_pairs(pairs, dataset_root=root, bootstrap_replicates=100)
    assert 'source differs from frozen Level 1/final evaluation path' in report['errors']['countdown']


def test_development_audit_still_accepts_unmatched_single_seed_draws(tmp_path):
    pairs = []
    for domain in audit.DOMAINS:
        pair = {'domain': domain}
        for role in ('baseline', 'candidate'):
            payload, _ = receipt(domain, role, development=True, count=64, draw_count=1)
            path = tmp_path / f'dev_{role}_{domain}.json'
            path.write_text(json.dumps(payload))
            pair[role] = str(path)
        pairs.append(pair)
    report = audit.audit_pairs(pairs, development=True, bootstrap_replicates=100)
    assert report['all_five_observed_approximate_match']
    assert not report['confirmation_match_verified']
    assert report['criteria']['expected_confirmation_seeds'] is None


@pytest.fixture
def frozen_recipe_case(tmp_path):
    """Small authenticated recipe fixture for guard mutations without receipts."""
    root = tmp_path / 'frozen_recipe'
    (root / 'recipes').mkdir(parents=True)
    (root / 'frozen_recipes.json').write_text(json.dumps({
        'recipes_sha256': {}, 'confirmation_outcomes_used': False,
    }))
    (root / 'identity.json').write_text(json.dumps({'recipe_sha256': {}}))
    recipe = recipe_document('countdown')
    identity = repin_recipe(root, 'countdown', recipe)
    assert audit.frozen_recipe('countdown', root, identity)[0] == recipe
    return root, recipe


@pytest.mark.parametrize('field,value', [
    ('algorithm', 'full_pool_forecast'),
    ('weight_objective', 'selected_development_residual'),
    ('selected_residual_used_for_ranking', True),
    ('selected_residual_used_for_ranking', 0),
    ('on_selected_gate_failure', 'retry_alternate_weights'),
    ('on_selected_gate_failure', 'retry_alternate_hash_seeds'),
    ('outcome_independent_within_pool_cell_order', False),
])
def test_frozen_v2_recipe_rejects_selected_outcome_search(frozen_recipe_case, field, value):
    root, recipe = frozen_recipe_case
    recipe['selection'][field] = value
    identity = repin_recipe(root, 'countdown', recipe)
    with pytest.raises(ValueError, match='forecast-only'):
        audit.frozen_recipe('countdown', root, identity)


@pytest.mark.parametrize('count', [0, 2, True, 1.0])
def test_frozen_v2_recipe_requires_exactly_one_selected_set(frozen_recipe_case, count):
    root, recipe = frozen_recipe_case
    recipe['development']['selected_development_sets_scored'] = count
    identity = repin_recipe(root, 'countdown', recipe)
    with pytest.raises(ValueError, match='one selected development set'):
        audit.frozen_recipe('countdown', root, identity)


@pytest.mark.parametrize('metric', ['pass1', 'pass8'])
def test_frozen_v2_recipe_cannot_relax_either_tolerance(frozen_recipe_case, metric):
    root, recipe = frozen_recipe_case
    assert recipe['development']['tolerances'] == {'pass1': .04, 'pass8': .08}
    recipe['development']['tolerances'][metric] += .01
    identity = repin_recipe(root, 'countdown', recipe)
    with pytest.raises(ValueError, match='unchanged tolerances'):
        audit.frozen_recipe('countdown', root, identity)


@pytest.mark.parametrize('stage', ['expected', 'selected'])
def test_frozen_v2_recipe_requires_both_gate_stages(frozen_recipe_case, stage):
    root, recipe = frozen_recipe_case
    del recipe['development']['gates'][stage]
    identity = repin_recipe(root, 'countdown', recipe)
    with pytest.raises(ValueError, match='both expected and selected'):
        audit.frozen_recipe('countdown', root, identity)


@pytest.mark.parametrize('stage', ['expected', 'selected'])
@pytest.mark.parametrize('metric', ['pass1', 'pass8'])
@pytest.mark.parametrize('mutation', [
    'inconsistent_delta', 'outside_tolerance', 'nonfinite_metric', 'nonfinite_delta',
    'out_of_range_baseline', 'missing_metric', 'missing_delta',
    'false_gate', 'numeric_gate', 'missing_gate',
])
def test_frozen_v2_recipe_recomputes_every_development_gate(
        frozen_recipe_case, stage, metric, mutation):
    root, recipe = frozen_recipe_case
    development = recipe['development']
    metric_field = stage + '_metrics'
    delta_field = 'expected_differences' if stage == 'expected' else 'differences'
    if mutation == 'inconsistent_delta':
        development[delta_field][metric] = .001
    elif mutation == 'outside_tolerance':
        delta = development['tolerances'][metric] + .001
        development[metric_field][metric] = development['baseline_metrics'][metric] + delta
        development[delta_field][metric] = delta
    elif mutation == 'nonfinite_metric':
        development[metric_field][metric] = float('nan')
    elif mutation == 'nonfinite_delta':
        development[delta_field][metric] = float('inf')
    elif mutation == 'out_of_range_baseline':
        development['baseline_metrics'][metric] = -1.
        development[metric_field][metric] = -1.
    elif mutation == 'missing_metric':
        del development[metric_field][metric]
    elif mutation == 'missing_delta':
        del development[delta_field][metric]
    elif mutation == 'false_gate':
        development['gates'][stage][metric] = False
    elif mutation == 'numeric_gate':
        development['gates'][stage][metric] = 1
    elif mutation == 'missing_gate':
        del development['gates'][stage][metric]
    identity = repin_recipe(root, 'countdown', recipe)
    with pytest.raises(ValueError, match='development gate did not pass'):
        audit.frozen_recipe('countdown', root, identity)


@pytest.mark.parametrize('stage', ['expected', 'selected'])
@pytest.mark.parametrize('metric', ['pass1', 'pass8'])
@pytest.mark.parametrize('sign', [-1, 1])
def test_frozen_v2_development_tolerance_boundaries_remain_inclusive(
        frozen_recipe_case, stage, metric, sign):
    root, recipe = frozen_recipe_case
    development = recipe['development']
    delta = sign * development['tolerances'][metric]
    development[stage + '_metrics'][metric] = development['baseline_metrics'][metric] + delta
    development['expected_differences' if stage == 'expected' else 'differences'][metric] = delta
    identity = repin_recipe(root, 'countdown', recipe)
    assert audit.frozen_recipe('countdown', root, identity)[0] == recipe
