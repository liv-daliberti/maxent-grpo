"""Offline integrity guards and paired prompt-level inference checks."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('temperature_analysis', Path(__file__).resolve().parents[1] /
                                             'ops/analyze_frontier_temperature_ablation.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + '\n')


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in records))


@pytest.fixture
def pair(tmp_path):
    reference = tmp_path / 'reference'
    rows = [{'level': level, 'domain': domain, 'row_index': index,
             'metadata': {'answer_mode_count': 5}, 'problem': f'Task {level}/{domain}/{index}'}
            for level in m.LEVELS for domain in m.DOMAINS for index in range(128)]
    originals = []
    for row in rows:
        for index in range(8):
            request = {'model': 'grok-4.3', 'messages': [{'role': 'user', 'content': row['problem']}],
                       'max_tokens': 8192, 'reasoning_effort': 'medium'}
            originals.append({k: row[k] for k in ('level', 'domain', 'row_index')} |
                {'sample_index': index, 'sample_id': f"{row['level']}_{row['domain']}_{row['row_index']}_{index}",
                 'request': request, 'request_sha256': m.sha(request), 'row_sha256': m.sha(row)})
    write_jsonl(reference / 'rows.jsonl', rows)
    write_jsonl(reference / 'requests.jsonl', originals)
    write_json(reference / 'manifest.json', {'test': 'original frozen task inventory'})
    selected = m.selected_original_rows(rows)
    cohorts = []
    for label, temperature in [('t1p0', 1.0), ('t1p5', 1.5)]:
        requests, raw, grades, outcomes = {}, {}, {}, {}
        for original in originals:
            key = m.identity(original)
            if key[:3] not in selected:
                continue
            request = original['request'] | {'temperature': temperature}
            requests[key] = original | {'request': request, 'request_sha256': m.sha(request),
                                        'reference_request_sha256': original['request_sha256']}
            # T1.5 has two valid modes and one retained truncated invalid draw.
            valid = temperature == 1.0 or key[3] < 7
            canonical = ('mode_a' if temperature == 1.0 or key[3] < 4 else 'mode_b') if valid else None
            grades[key] = {'verified': valid, 'canonical_key': canonical}
            raw[key] = {'response_id': label + original['sample_id'], 'stop_reason': 'stop' if valid else 'length'}
            outcomes[key] = {'refusal': False, 'content_filtered': False}
        summary = {'normalized_secondary': {'normalization_source_sha256': 'same-normalizer',
            'initial_rule_audit_sha256': 'same-development-audit', 'frozen_grader_contract_sha256': 'same-contract',
            'cells': {}}, 'cells': {}}
        for level in m.LEVELS:
            for domain in m.DOMAINS:
                accuracy, distinct, collision = (1., 1., 1.) if temperature == 1.0 else (7/8, 2., 9/21)
                uniform = None if domain == 'countdown' else 1/5
                metrics = {'pass1': {'estimate': accuracy}, 'distinct8': {'estimate': distinct},
                    'correct_pair_collision': {'estimate': collision},
                    'uniform_correct_pair_collision': {'estimate': uniform},
                    'correct_pair_collision_excess_uniform': {'estimate': None if uniform is None else collision-uniform}}
                summary['cells'][f'level{level}/{domain}'] = {'metrics': metrics}
                summary['normalized_secondary']['cells'][f'level{level}/{domain}'] = {'metrics': copy.deepcopy(metrics)}
        cohorts.append({'directory': tmp_path / label, 'rows': copy.deepcopy(selected), 'requests': requests,
            'raw': raw, 'primary': grades, 'normalized': copy.deepcopy(grades), 'outcomes': outcomes,
            'manifest': {'model': 'grok-4.3', 'temperature': temperature, 'reference_run': str(reference),
                'reference_manifest_sha256': m.file_sha(reference / 'manifest.json'),
                'reference_artifact_sha256': {name: m.file_sha(reference / name) for name in ('rows.jsonl', 'requests.jsonl')}},
            'summary': summary, 'grading_audit': {'frozen_grader_modules': {'grader': {'sha256': 'same-source'}}},
            'sources': {}})
    return cohorts


def test_pairing_checks_original_population_and_temperature_only_payloads(pair):
    m.validate_pairing(*pair)


def test_paired_analysis_retains_invalid_truncations_and_reconstructs_cells(pair):
    indices = {(l, d): np.random.default_rng(l).integers(0, 8, (300, 8)) for l in m.LEVELS for d in m.DOMAINS}
    result = m.analyze_pair(*pair, indices)
    analysis = result['analyses']['strict']
    assert analysis['totals']['t1p5']['responses'] == 960
    assert analysis['totals']['t1p5']['truncated_responses'] == 120
    overall = analysis['groups']['five_domain_macro']['overall']
    assert overall['t1p5']['accuracy']['estimate'] == 7/8
    assert overall['t1p5_minus_t1p0']['distinct8']['estimate'] == 1
    assert overall['t1p5_minus_t1p0']['accuracy']['ci95'] == [-1/8, -1/8]
    assert overall['t1p0']['uniform_collision']['estimate'] is None
    closed = analysis['groups']['four_finite_support_domain_macro']['overall']
    assert closed['t1p0']['uniform_collision']['estimate'] == pytest.approx(.2)


def test_summary_mismatch_is_rejected(pair):
    pair[1]['summary']['cells']['level1/graph_coloring']['metrics']['pass1']['estimate'] = 1.
    indices = {(l, d): np.zeros((10, 8), dtype=int) for l in m.LEVELS for d in m.DOMAINS}
    with pytest.raises(ValueError, match='Reconstructed statistic differs'):
        m.analyze_pair(*pair, indices)


@pytest.mark.parametrize('change', ['prompt', 'reasoning', 'token_budget'])
def test_other_payload_controls_cannot_change(pair, change):
    item = next(iter(pair[1]['requests'].values()))
    if change == 'prompt':
        item['request']['messages'] = [{'role': 'user', 'content': 'different mathematical task'}]
    elif change == 'reasoning':
        item['request']['reasoning_effort'] = 'high'
    else:
        item['request']['max_tokens'] = 16384
    with pytest.raises(ValueError, match='non-temperature payload field changed'):
        m.validate_pairing(*pair)


def test_outcome_dependent_or_changed_selection_is_rejected(pair):
    row = next(iter(pair[0]['rows'].values()))
    row['problem'] = 'Changed selected public row'
    pair[1]['rows'] = copy.deepcopy(pair[0]['rows'])
    with pytest.raises(ValueError, match='outcome-blind selection'):
        m.validate_pairing(*pair)


def test_reused_native_response_id_is_rejected(pair):
    next(iter(pair[1]['raw'].values()))['response_id'] = next(iter(pair[0]['raw'].values()))['response_id']
    with pytest.raises(ValueError, match='reused across temperatures'):
        m.validate_pairing(*pair)


def test_changed_normalizer_is_rejected(pair):
    pair[1]['summary']['normalized_secondary']['normalization_source_sha256'] = 'different'
    with pytest.raises(ValueError, match='normalizer or grading contract differs'):
        m.validate_pairing(*pair)


def test_paired_bootstrap_preserves_whole_prompt_covariance():
    a = np.array([[1, 8-i, i+1, (8-i)*(7-i)//2, (8-i)*(7-i)//2, 1, 1, 1] for i in range(8)], float)
    indices = np.random.default_rng(3).integers(0, 8, (2000, 8))
    points, boots = m.paired_bootstrap(a, a.copy(), indices)
    assert np.array_equal(points['t1p5_minus_t1p0'], np.zeros(5))
    assert np.array_equal(boots['t1p5_minus_t1p0'], np.zeros((2000, 5)))


def test_pair_weighting_and_undefined_collision_are_explicit():
    a = np.array([[1, 8, 1, 28, 28, 28/5, 28, 28],
                  [1, 2, 2, 1, 0, 1/5, 1, 0]], float)
    point = m.rates(a.sum(axis=0))
    assert point[0] == 10/16
    assert point[1] == 1.5
    assert point[2] == 28/29
    empty = np.array([1, 0, 0, 0, 0, 0, 0, 0], float)
    assert np.isnan(m.rates(empty)[2])


def test_macro_does_not_drop_a_cell_without_correct_pairs():
    good = np.array([1., 1., 1., .2, .8])
    missing = np.array([0., 0., np.nan, np.nan, np.nan])
    labels = ['t1p0', 't1p5', 't1p5_minus_t1p0']
    points = {'good': {k: good for k in labels}, 'missing': {k: missing for k in labels}}
    boots = {cell: {k: np.repeat(point[k][None, :], 20, axis=0) for k in labels} for cell, point in points.items()}
    result = m.macro(points, boots, ['good', 'missing'])
    assert result['t1p0']['accuracy']['estimate'] == .5
    assert result['t1p0']['collision']['estimate'] is None
    assert result['t1p0']['collision']['defined_replicates'] == 0


def test_countdown_never_gets_a_finite_uniform_reference(pair):
    records = m.prompt_records(pair[0], 'strict')
    assert all(r['certified_support_count'] is None and r['uniform_expected_colliding_correct_pairs'] is None
               for r in records if r['domain'] == 'countdown')


def test_changed_bound_source_file_is_rejected(tmp_path):
    path = tmp_path / 'source.json'
    path.write_text('frozen')
    digest = m.file_sha(path)
    path.write_text('changed')
    with pytest.raises(ValueError, match='Changed or unbound evidence'):
        m.bound_file(path, digest)


def test_normalization_checks_all_domains_and_current_receipt_identity(tmp_path):
    raw, primary, cache = {}, {}, []
    for i, domain in enumerate(m.DOMAINS):
        key = (1, domain, i, 0)
        sample = dict(zip(('level', 'domain', 'row_index', 'sample_index'), key)) | {
            'sample_id': domain, 'text': 'answer', 'verified': True,
            'canonical_key': domain, 'graded_text': 'answer'}
        raw[key] = primary[key] = sample
        cache.append({k: sample[k] for k in ('level', 'domain', 'row_index', 'sample_index')} |
                     {'strict_receipt_sha256': m.sha(sample), 'normalization_source_sha256': '',
                      'normalization': {'verified': True, 'canonical_key': domain, 'graded_text': 'answer', 'original_text': 'answer'}})
    for name in ('normalizer.py', 'grader.py'):
        (tmp_path / name).write_text('# frozen\n')
    digest = m.file_sha(tmp_path / 'normalizer.py')
    for item in cache:
        item['normalization_source_sha256'] = digest
    write_json(tmp_path / 'secondary_initial15_audit.json', {'normalizer': {'sha256': digest}})
    write_jsonl(tmp_path / 'normalized_samples.jsonl', cache)
    secondary = {'cache_sha256': m.file_sha(tmp_path / 'normalized_samples.jsonl'),
        'normalization_source_path': str(tmp_path / 'normalizer.py'), 'normalization_source_sha256': digest,
        'frozen_grader_contract_path': str(tmp_path / 'grader.py'), 'frozen_grader_contract_sha256': digest,
        'initial_rule_audit_sha256': m.file_sha(tmp_path / 'secondary_initial15_audit.json')}
    assert set(m.normalized_grades(tmp_path, {'normalized_secondary': secondary}, raw, primary)) == set(raw)
    cache[0]['row_index'] = 99
    write_jsonl(tmp_path / 'normalized_samples.jsonl', cache)
    secondary['cache_sha256'] = m.file_sha(tmp_path / 'normalized_samples.jsonl')
    with pytest.raises(ValueError, match='cache identity differs'):
        m.normalized_grades(tmp_path, {'normalized_secondary': secondary}, raw, primary)


def test_collection_gate_retains_first_samples_and_binds_preflight(tmp_path):
    support = tmp_path / 'support.json'
    write_json(support, {'status': 'documented controls'})
    gate = {'status': 'authorized_supported_controls', 'admitted_slugs': list(m.MODEL_SLUGS),
            'temperatures': [1.0, 1.5], 'planned_terminal_responses': 3840,
            'responses_per_condition': 960, 'prompt_count_per_condition': 120,
            'preflight_responses_included_in_total': 4, 'support_review': str(support),
            'support_review_sha256': m.file_sha(support), 'source_sha256': {}, 'preflight_evidence': {}}
    for slug in m.MODEL_SLUGS:
        for suffix in ('_t1p0', '_t1p5'):
            label = slug + suffix
            run = tmp_path / label
            request = {'sample_id': 'first', 'level': 1, 'domain': 'countdown',
                       'row_index': 0, 'sample_index': 0, 'request_sha256': 'frozen'}
            sample = request | {'raw_receipt': 'raw/first.json'}
            write_jsonl(run / 'requests.jsonl', [request])
            write_json(run / 'manifest.json', {'model': m.MODEL_SLUGS[slug]})
            write_json(run / 'preflight_result.json', {'exit_code': 0, 'terminal_samples': 1, 'condition': label})
            write_json(run / 'sample_receipts/first.json', sample)
            write_json(run / 'raw/first.json', {'answer': 'saved preflight'})
            gate['preflight_evidence'][label] = {
                'manifest_sha256': m.file_sha(run / 'manifest.json'),
                'result_sha256': m.file_sha(run / 'preflight_result.json'),
                'sample_receipt_sha256': m.file_sha(run / 'sample_receipts/first.json'),
                'raw_receipt_sha256': m.file_sha(run / 'raw/first.json')}
    write_json(tmp_path / 'collection_gate.json', gate)
    assert m.authenticate_collection_gate(tmp_path)['first_registered_samples_bound']
    write_json(tmp_path / 'kimi_k3_t1p5/raw/first.json', {'answer': 'replacement for truncated preflight'})
    with pytest.raises(ValueError, match='Changed or unbound evidence'):
        m.authenticate_collection_gate(tmp_path)
