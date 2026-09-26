"""Integrity and inference checks for the separate no-reasoning GPT curve."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from test_frontier_temperature_ablation import pair as original_pair, write_json, write_jsonl

spec = importlib.util.spec_from_file_location('gpt_curve', Path(__file__).resolve().parents[1] /
                                             'ops/analyze_gpt56_temperature_curve.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


@pytest.fixture
def curve(tmp_path):
    prototypes = original_pair.__wrapped__(tmp_path)
    reference = Path(prototypes[0]['manifest']['reference_run'])
    originals = m.read_jsonl(reference / 'requests.jsonl')
    for item in originals:
        old = item['request']
        item['request'] = {'model': m.MODEL, 'input': old['messages'], 'reasoning': {'effort': 'medium'},
                           'max_output_tokens': 8192, 'store': False}
        item['request_sha256'] = m.sha(item['request'])
    write_jsonl(reference / 'requests.jsonl', originals)
    write_json(reference / 'manifest.json', {'model': m.MODEL, 'reasoning_effort': 'medium'})
    originals = {m.identity(r): r for r in originals}
    cohorts = {}
    for temperature in m.TEMPERATURES:
        c = copy.deepcopy(prototypes[0 if temperature <= 1 else 1])
        c['manifest'].update(model=m.MODEL, reasoning_effort='none', temperature=temperature,
            experiment_condition=m.CONDITION, schema='frontier-modebench-responses-v1',
            sample_count=8, original_prompt_cohort=False, original_prompt_bytes=True,
            reference_manifest_sha256=m.file_sha(reference / 'manifest.json'),
            reference_artifact_sha256={n: m.file_sha(reference / n) for n in ('requests.jsonl', 'rows.jsonl')})
        c['directory'] = tmp_path / m.SLUGS[temperature]
        c['returned_controls'] = {}
        for key in c['requests']:
            original = originals[key]
            request = copy.deepcopy(original['request'])
            request['reasoning']['effort'] = 'none'
            request['temperature'] = temperature
            c['requests'][key] = original | {'request': request, 'request_sha256': m.sha(request),
                'reference_request_sha256': original['request_sha256'], 'temperature_condition': temperature}
            valid = c['primary'][key]['verified'] and temperature < 2
            c['raw'][key] = {'response_id': str(temperature) + original['sample_id'],
                            'response_status': 'completed' if valid else 'incomplete',
                            'incomplete_details': None if valid else {'reason': 'max_output_tokens'}}
            c['outcomes'][key]['answer_text_empty'] = not valid
            if temperature == 2:
                c['primary'][key] = c['normalized'][key] = {'verified': False, 'canonical_key': None}
        if temperature == 2:
            for section in (c['summary'], c['summary']['normalized_secondary']):
                for cell in section['cells'].values():
                    for key in cell['metrics']:
                        cell['metrics'][key]['estimate'] = 0 if key in ('pass1', 'distinct8') else None
        cohorts[temperature] = c
    return cohorts


def test_curve_authenticates_all_four_exact_conditions(curve):
    m.validate_curve(curve)


def test_curve_retains_truncations_and_does_not_connect_medium_reference(curve):
    indices = {(l, d): np.random.default_rng(l).integers(0, 8, (200, 8)) for l in m.LEVELS for d in m.DOMAINS}
    result = m.analyze_curve(curve, indices)
    assert result['reasoning_effort'] == 'none'
    analysis = result['analyses']['strict']
    high = analysis['temperatures']['2.0']
    assert high['counts']['responses'] == high['counts']['truncated_responses'] == 960
    assert high['groups']['five_domain_macro']['overall']['collision']['estimate'] is None
    assert high['groups']['five_domain_macro']['overall']['accuracy']['estimate'] == 0
    baseline = analysis['paired_contrasts_vs_t1p0']['1.0']['groups']['five_domain_macro']['overall']
    assert baseline['accuracy']['estimate'] == 0
    assert baseline['distinct8']['ci95'] == [0, 0]
    finite = analysis['temperatures']['0.5']['groups']['four_finite_support_domain_macro']['overall']
    assert finite['uniform_collision']['estimate'] == pytest.approx(.2)
    assert 'medium' not in analysis['temperatures']
    endpoint = analysis['paired_endpoint_contrast']['groups']['five_domain_macro']['overall']
    assert endpoint['accuracy']['estimate'] == -1
    assert endpoint['accuracy']['ci95'] == [-1, -1]
    assert endpoint['collision']['estimate'] is None
    assert high['groups']['five_domain_macro']['level_contrasts']['L3-L1']['accuracy']['estimate'] == 0


@pytest.mark.parametrize('change', ['reasoning', 'prompt', 'budget', 'store'])
def test_unregistered_curve_changes_are_rejected(curve, change):
    item = next(iter(curve[1.5]['requests'].values()))
    if change == 'reasoning': item['request']['reasoning']['effort'] = 'medium'
    elif change == 'prompt': item['request']['input'] = [{'role': 'user', 'content': 'different task'}]
    elif change == 'budget': item['request']['max_output_tokens'] = 16384
    else: item['request']['store'] = True
    with pytest.raises(ValueError, match='field besides registered'):
        m.validate_curve(curve)


def test_missing_temperature_is_rejected(curve):
    curve.pop(.5)
    with pytest.raises(ValueError, match='all four'):
        m.validate_curve(curve)


def test_reused_response_across_temperatures_is_rejected(curve):
    next(iter(curve[2]['raw'].values()))['response_id'] = next(iter(curve[1]['raw'].values()))['response_id']
    with pytest.raises(ValueError, match='reused between'):
        m.validate_curve(curve)


def test_changed_source_summary_is_rejected(curve):
    curve[1.5]['summary']['cells']['level2/graph_coloring']['metrics']['pass1']['estimate'] = .1
    indices = {(l, d): np.zeros((10, 8), dtype=int) for l in m.LEVELS for d in m.DOMAINS}
    with pytest.raises(ValueError, match='differs from frozen summary'):
        m.analyze_curve(curve, indices)


@pytest.mark.parametrize('temperature,effort', [(1.0, 'none'), (1.5, 'medium'), (None, 'none'), (1.5, None)])
def test_returned_controls_must_match_the_registered_curve(temperature, effort):
    raw = {'sample': {'raw_receipt': 'raw/one.json'}}
    body = {'response': {'temperature': temperature, 'reasoning': {'effort': effort}, 'top_p': .98}}
    with pytest.raises(ValueError, match='Returned temperature or reasoning differs'):
        m.returned_controls(raw, {'raw/one.json': body}, 1.5)


def test_returned_controls_preserve_exposed_top_p_without_claiming_sampler_access():
    raw = {'sample': {'raw_receipt': 'raw/one.json'}}
    body = {'response': {'temperature': 1.5, 'reasoning': {'effort': 'none'}, 'top_p': .98}}
    controls = m.returned_controls(raw, {'raw/one.json': body}, 1.5)
    assert len(controls) == 1 and list(controls.values()) == [1]
    assert json.loads(next(iter(controls)))['top_p'] == .98


def test_gate_binds_and_retains_the_registered_preflight(curve, tmp_path):
    gate = {'schema': 'frontier-gpt56-temperature-gate-v1', 'status': 'authorized_supported_controls',
            'model': m.MODEL, 'reasoning_effort': 'none', 'temperatures': list(m.TEMPERATURES),
            'total_registered_responses': 3840, 'responses_per_condition': 960, 'prompts_per_condition': 120,
            'draws_per_prompt': 8, 'domains': 5, 'levels': list(m.LEVELS), 'evidence': {}, 'conditions': []}
    for relative in ['collection_code/run_gpt56_temperature_curve.py', 'gpt56_control_probe/probe_results.json',
                     'gpt56_curve_independent_design_audit.json', 'gpt56_curve_preparation_audit.json']:
        path = tmp_path / relative
        write_json(path, {'source': 'frozen'})
        gate['evidence'][relative] = m.file_sha(path)
    for t, c in curve.items():
        run = c['directory']
        request = next(iter(c['requests'].values()))
        sample = request | {'raw_receipt': 'raw/first.json', 'response_id': 'first_' + str(t)}
        write_json(run / 'manifest.json', c['manifest'])
        write_jsonl(run / 'requests.jsonl', [request])
        write_jsonl(run / 'rows.jsonl', list(c['rows'].values()))
        write_json(run / 'preflight_result.json', {'exit_code': 0, 'terminal_samples': 1})
        write_json(run / 'sample_receipts' / (request['sample_id'] + '.json'), sample)
        write_json(run / 'raw/first.json', {'http_status': 200, 'response': {'id': sample['response_id'],
            'status': 'completed', 'temperature': t, 'reasoning': {'effort': 'none'}, 'top_p': .98}})
        gate['conditions'].append({'condition': m.SLUGS[t], 'temperature': t, 'reasoning_effort': 'none',
            'registered_responses': 960, 'manifest_sha256': m.file_sha(run / 'manifest.json'),
            'requests_sha256': m.file_sha(run / 'requests.jsonl'), 'rows_sha256': m.file_sha(run / 'rows.jsonl'),
            'preflight_result_sha256': m.file_sha(run / 'preflight_result.json'),
            'preflight_receipt': str((run / 'raw/first.json').relative_to(tmp_path)),
            'preflight_receipt_sha256': m.file_sha(run / 'raw/first.json')})
    write_json(tmp_path / 'gpt56_curve_collection_gate.json', gate)
    assert m.authenticate_gate(tmp_path)['all_four_registered_preflights_bound']
    write_json(curve[2]['directory'] / 'raw/first.json', {'replacement': 'better answer'})
    with pytest.raises(ValueError, match='Changed or unbound evidence'):
        m.authenticate_gate(tmp_path)


def test_medium_reference_stays_separate_and_checks_grader_contract(curve):
    reference = copy.deepcopy(curve[1.0])
    for sample in reference['raw'].values():
        sample['response_id'] = 'original_medium_' + sample['response_id']
    indices = {(l, d): np.zeros((20, 8), dtype=int) for l in m.LEVELS for d in m.DOMAINS}
    result = m.analyze_medium_reference(reference, curve, indices)
    assert result['connect_to_temperature_curve'] is False
    assert result['requested_temperature'] is None
    assert result['returned_temperature'] == 1.0
    assert result['reasoning_effort'] == 'medium'
    assert result['analyses']['normalized_secondary']['counts']['responses'] == 960
    assert result['analyses']['strict']['groups']['five_domain_macro']['overall']['accuracy']['estimate'] == 1
    reference['grading_audit']['frozen_grader_modules']['grader']['sha256'] = 'changed'
    with pytest.raises(ValueError, match='different executable graders'):
        m.analyze_medium_reference(reference, curve, indices)


def test_failure_categories_keep_all_draws_without_semantic_guessing(curve):
    assert m.outcome_categories(curve[.5])['totals'] == {'strict_success': 960}
    assert m.outcome_categories(curve[2])['totals'] == {'token_limit': 960}
    assert sum(m.outcome_categories(curve[1.5])['totals'].values()) == 960


def test_legacy_subset_ignores_only_unselected_cache_reconciliation(tmp_path):
    sample = {'level': 1, 'domain': 'python_factors', 'row_index': 4, 'sample_index': 0,
              'sample_id': 'selected', 'text': 'answer', 'verified': True,
              'canonical_key': 'mode', 'graded_text': 'answer'}
    key = m.identity(sample)
    for name in ('normalizer.py', 'grader.py'):
        (tmp_path / name).write_text('# frozen\n')
    digest = m.file_sha(tmp_path / 'normalizer.py')
    write_json(tmp_path / 'secondary_initial15_audit.json', {'normalizer': {'sha256': digest}})
    entry = {k: sample[k] for k in ('level', 'domain', 'row_index', 'sample_index')}
    entry.update(strict_receipt_sha256=m.sha(sample), normalization_source_sha256=digest,
        normalization={'verified': True, 'canonical_key': 'mode', 'graded_text': 'answer', 'original_text': 'answer'})
    extra = copy.deepcopy(entry)
    extra['strict_receipt_sha256'] = 'unselected historical record'
    changed = copy.deepcopy(extra)
    changed['normalization']['canonical_key'] = 'reconciled unselected mode'
    records = [entry, extra, changed]
    write_jsonl(tmp_path / 'normalized_samples.jsonl', records)
    secondary = {'cache_sha256': m.file_sha(tmp_path / 'normalized_samples.jsonl'),
        'normalization_source_path': str(tmp_path / 'normalizer.py'), 'normalization_source_sha256': digest,
        'frozen_grader_contract_path': str(tmp_path / 'grader.py'), 'frozen_grader_contract_sha256': digest,
        'initial_rule_audit_sha256': m.file_sha(tmp_path / 'secondary_initial15_audit.json')}
    assert m.normalized_medium_subset(tmp_path, {'normalized_secondary': secondary}, {key: sample}, {key: sample})[key]['verified']
    conflict = copy.deepcopy(entry)
    conflict['normalization']['canonical_key'] = 'changed selected mode'
    write_jsonl(tmp_path / 'normalized_samples.jsonl', records + [conflict])
    secondary['cache_sha256'] = m.file_sha(tmp_path / 'normalized_samples.jsonl')
    with pytest.raises(ValueError, match='Conflicting normalization cache entries'):
        m.normalized_medium_subset(tmp_path, {'normalized_secondary': secondary}, {key: sample}, {key: sample})


@pytest.fixture
def curve_with_zero(curve, tmp_path):
    zero = copy.deepcopy(curve[2.0])
    zero['manifest']['temperature'] = 0.0
    zero['directory'] = tmp_path / m.SLUGS[0.0]
    for item in zero['requests'].values():
        item['request']['temperature'] = item['temperature_condition'] = 0.0
        item['request_sha256'] = m.sha(item['request'])
    for item in zero['raw'].values():
        item['response_id'] = 'zero_' + item['response_id']
    return {0.0: zero, **curve}


def test_zero_extension_keeps_every_draw_and_updates_endpoint(curve_with_zero):
    indices = {(l, d): np.zeros((10, 8), dtype=int) for l in m.LEVELS for d in m.DOMAINS}
    result = m.analyze_curve(curve_with_zero, indices)
    assert result['temperatures'] == [0.0, .5, 1.0, 1.5, 2.0]
    assert result['total_registered_responses'] == 4800
    assert result['validation']['authenticated_native_receipt_count'] == 4800
    assert 'all_3840_native_receipts_authenticated' not in result['validation']
    for data in result['analyses'].values():
        assert data['temperatures']['0.0']['counts']['responses'] == 960
        assert data['temperatures']['0.0']['counts']['truncated_responses'] == 960
        endpoint = data['paired_endpoint_contrast']
        assert endpoint['comparison'] == 'T2.0-T0.0'
        assert endpoint['groups']['five_domain_macro']['overall']['accuracy']['estimate'] == 0
        assert endpoint['cells']['level1/countdown']['joint_eligibility']['only_t0p0_eligible'] == 0
    assert '4,800 saved responses' in m.markdown(result)
    assert 'T2.0−T0.0 accuracy' in m.markdown(result)


def test_zero_extension_rejects_reused_response(curve_with_zero):
    original = next(iter(curve_with_zero[.5]['raw'].values()))['response_id']
    next(iter(curve_with_zero[0.0]['raw'].values()))['response_id'] = original
    with pytest.raises(ValueError, match='reused between'):
        m.validate_curve(curve_with_zero)


def test_unsupported_temperature_is_not_part_of_registered_grid(curve_with_zero):
    curve_with_zero[2.5] = copy.deepcopy(curve_with_zero[2.0])
    with pytest.raises(ValueError, match='all four registered'):
        m.validate_curve(curve_with_zero)


def test_zero_has_a_separate_gate_and_retained_preflight(curve_with_zero, tmp_path):
    zero = curve_with_zero[0.0]
    run = zero['directory']
    request = next(iter(zero['requests'].values()))
    sample = request | {'raw_receipt': 'raw/first.json', 'response_id': 'zero_first'}
    write_json(run / 'manifest.json', zero['manifest'])
    write_jsonl(run / 'requests.jsonl', [request])
    write_jsonl(run / 'rows.jsonl', list(zero['rows'].values()))
    write_json(run / 'preflight_result.json', {'exit_code': 0, 'terminal_samples': 1})
    write_json(run / 'sample_receipts' / (request['sample_id'] + '.json'), sample)
    write_json(run / 'raw/first.json', {'http_status': 200, 'response': {'id': sample['response_id'],
        'status': 'completed', 'temperature': 0.0, 'reasoning': {'effort': 'none'}, 'top_p': .98}})
    gate = {'schema': 'frontier-gpt56-temperature-extension-gate-v1',
        'status': 'authorized_supported_controls', 'model': m.MODEL, 'reasoning_effort': 'none',
        'temperatures': [0.0], 'total_registered_responses': 960, 'responses_per_condition': 960,
        'prompts_per_condition': 120, 'draws_per_prompt': 8, 'domains': 5, 'levels': list(m.LEVELS),
        'evidence': {}, 'conditions': [{'condition': m.SLUGS[0.0], 'temperature': 0.0,
            'reasoning_effort': 'none', 'registered_responses': 960,
            'manifest_sha256': m.file_sha(run / 'manifest.json'),
            'requests_sha256': m.file_sha(run / 'requests.jsonl'), 'rows_sha256': m.file_sha(run / 'rows.jsonl'),
            'preflight_result_sha256': m.file_sha(run / 'preflight_result.json'),
            'preflight_receipt': str((run / 'raw/first.json').relative_to(tmp_path)),
            'preflight_receipt_sha256': m.file_sha(run / 'raw/first.json')}]}
    for relative in ['gpt56_curve_collection_gate.json', 'gpt56_requested_grid_probe/probe_results.json',
                     'gpt56_zero_control_probe/probe_results.json', 'collection_code/run_gpt56_zero_extension.py']:
        write_json(tmp_path / relative, {'source': 'frozen'})
        gate['evidence'][relative] = m.file_sha(tmp_path / relative)
    original_gate_bytes = (tmp_path / 'gpt56_curve_collection_gate.json').read_bytes()
    write_json(tmp_path / 'gpt56_zero_extension_collection_gate.json', gate)
    assert m.authenticate_extension_gate(tmp_path)['zero_registered_preflight_bound']
    assert (tmp_path / 'gpt56_curve_collection_gate.json').read_bytes() == original_gate_bytes
    gate['temperatures'].append(2.5)
    write_json(tmp_path / 'gpt56_zero_extension_collection_gate.json', gate)
    with pytest.raises(ValueError, match='Wrong GPT zero extension gate'):
        m.authenticate_extension_gate(tmp_path)
    gate['temperatures'] = [0.0]
    write_json(tmp_path / 'gpt56_zero_extension_collection_gate.json', gate)
    write_json(run / 'raw/first.json', {'replacement': 'changed preflight'})
    with pytest.raises(ValueError, match='Changed or unbound evidence'):
        m.authenticate_extension_gate(tmp_path)
