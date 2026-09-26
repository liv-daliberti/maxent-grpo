"""Offline integrity and prompt-cluster inference checks; no model calls."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('python_condition', Path(__file__).resolve().parents[1] / 'ops/analyze_frontier_python_prompt_condition.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + '\n')


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows))


def fixture_cohort(run, plain=False):
    run.mkdir()
    rows, requests, raw, audited, outcomes, cache = [], [], [], [], [], []
    for level in m.LEVELS:
        row = {'level': level, 'domain': m.DOMAIN, 'row_index': 0, 'answer': json.dumps({'cases': [6, 8]})}
        rows.append(row)
        for sample_index in range(8):
            key = f'L{level}_python_factors_000_{sample_index}'
            request = {'model': m.MODEL, 'max_tokens': 8192, 'thinking': {'type': 'adaptive'},
                       'output_config': {'effort': 'medium'}, 'messages': [{'role': 'user', 'content': m.plain_prompt([6, 8]) if plain else 'original'}]}
            if not plain:
                request['system'] = 'original'
            item = {'sample_id': key, 'level': level, 'domain': m.DOMAIN, 'row_index': 0,
                    'sample_index': sample_index, 'row_sha256': m.sha(row),
                    'request_sha256': m.sha(request), 'request': request}
            if plain:
                original_request = {**request, 'system': 'original', 'messages': [{'role': 'user', 'content': 'original'}]}
                item.update(condition=m.PLAIN_CONDITION, reference_request_sha256=m.sha(original_request))
            requests.append(item)
            refused = not plain or (level == 3 and sample_index == 7)
            text = '' if refused else '\\boxed{lambda n: 2}'
            response = {'model': m.MODEL, 'id': ('plain_' if plain else 'original_') + key,
                        'content': [] if refused else [{'type': 'text', 'text': text}],
                        'stop_reason': 'refusal' if refused else 'end_turn'}
            if refused:
                response['stop_details'] = {'type': 'refusal', 'category': 'cyber'}
            relative = 'raw_responses/' + key + '__01.json'
            receipt = {'sample_id': key, 'relative_path': relative, 'http_status': 200,
                       'request_sha256': item['request_sha256'], 'response': response}
            write_json(run / relative, receipt)
            sample = {k: v for k, v in item.items() if k != 'request'} | {
                'raw_receipt': relative, 'raw_receipt_sha256': m.sha(receipt),
                'model': m.MODEL, 'response_id': response['id'], 'text': text,
                'verified': not refused, 'canonical_key': 'python_factor:2,2' if not refused else None,
                'graded_text': text}
            raw.append(sample)
            atomic = 'sample_receipts/' + key + '.json'
            write_json(run / atomic, sample)
            audited.append(sample | {'raw_strict_' + k: sample[k] for k in m.GRADES} | {
                'raw_strict_receipt': atomic, 'raw_strict_receipt_sha256': m.file_sha(run / atomic)})
            native, _ = m.native_outcome(response)
            outcomes.append({k: sample[k] for k in ('sample_id', 'level', 'domain', 'row_index', 'sample_index',
                                                    'row_sha256', 'request_sha256', 'raw_receipt',
                                                    'raw_receipt_sha256', 'response_id')} | native | {
                'raw_receipt_file_sha256': m.file_sha(run / relative)})
    write_rows(run / 'rows.jsonl', rows)
    write_rows(run / 'requests.jsonl', requests)
    write_rows(run / 'samples.jsonl', raw)
    write_json(run / 'datasets.json', {})
    (run / 'code').mkdir()
    for name in ('grader.py', 'normalizer.py', 'audit.py', 'helper.py'):
        (run / 'code' / name).write_text('# frozen test source\n')
    manifest = {'model': m.MODEL, 'request_count': 24, 'sample_count': 8,
                'artifact_sha256': {name: m.file_sha(run / name) for name in ('rows.jsonl', 'requests.jsonl', 'datasets.json')}}
    if plain:
        original = run.parent / 'original'
        manifest.update(experiment_condition=m.PLAIN_CONDITION, reference_run=str(original),
                        reference_manifest_sha256=m.file_sha(original / 'manifest.json'))
    write_json(run / 'manifest.json', manifest)
    evidence = {str(path.relative_to(run)): m.file_sha(path) for path in run.rglob('*') if path.is_file()}
    write_json(run / 'evidence_file_sha256.json', evidence)
    write_json(run / 'completion_audit.json', {'status': 'pass', 'model': m.MODEL,
                                              'expected_responses': 24, 'saved_samples': 24, 'unique_response_ids': 24,
                                              'evidence_inventory_sha256': m.sha(evidence)})
    write_rows(run / 'audited_primary_samples.jsonl', audited)
    receipt_manifest = {s['sample_id']: {'path': s['raw_strict_receipt'], 'sha256': s['raw_strict_receipt_sha256']} for s in audited}
    grading = {'status': 'complete_for_snapshot', 'full_sampling_complete': True, 'snapshot_samples': 24,
               'expected_samples': 24, 'unresolved_groups': 0, 'python_samples': 24,
               'primary_manifest_sha256': m.file_sha(run / 'manifest.json'),
               'source_receipt_manifest': receipt_manifest, 'source_receipt_manifest_sha256': m.sha(receipt_manifest),
               'frozen_grader_modules': {'grader': {'path': str(run / 'code/grader.py'), 'sha256': m.file_sha(run / 'code/grader.py')}},
               'audit_source': 'code/audit.py', 'audit_source_sha256': m.file_sha(run / 'code/audit.py'),
               'audit_helper_source': 'code/helper.py', 'audit_helper_sha256': m.file_sha(run / 'code/helper.py'),
               'derived_primary': {'path': 'audited_primary_samples.jsonl', 'sha256': m.file_sha(run / 'audited_primary_samples.jsonl'), 'records': 24}}
    write_json(run / 'primary_python_regrade_audit.json', grading)
    norm_hash = m.file_sha(run / 'code/normalizer.py')
    write_json(run / 'secondary_initial15_audit.json', {'normalizer': {'sha256': norm_hash}})
    for sample in raw:
        cache.append({k: sample[k] for k in ('level', 'domain', 'row_index', 'sample_index')} | {
            'normalization_source_sha256': norm_hash, 'strict_receipt_sha256': m.sha(sample),
            'normalization': {k: sample[k] for k in m.GRADES} | {'original_text': sample['text']}})
    write_rows(run / 'normalized_samples.jsonl', cache)
    write_rows(run / 'provider_outcome_samples.jsonl', outcomes)
    outcome_report = {'status': 'complete', 'model': m.MODEL, 'responses': 24, 'expected_responses': 24,
                      'source_run': str(run), 'sources': {name: {'path': str(run / name), 'sha256': m.file_sha(run / name)}
                        for name in ('manifest.json', 'samples.jsonl', 'completion_audit.json', 'evidence_file_sha256.json')},
                      'audit_source_path': str(run / 'code/audit.py'), 'audit_source_sha256': m.file_sha(run / 'code/audit.py'),
                      'sample_outcomes': {'path': str(run / 'provider_outcome_samples.jsonl'),
                                          'sha256': m.file_sha(run / 'provider_outcome_samples.jsonl'), 'records': 24},
                      'totals': {'responses': 24, 'refusals': sum(o['refusal'] for o in outcomes)}, 'cells': {}}
    for level in m.LEVELS:
        subset = [o for o in outcomes if o['level'] == level]
        count = sum(o['refusal'] for o in subset)
        outcome_report['cells'][f'level{level}/{m.DOMAIN}'] = {'responses': 8, 'refusals': count,
                                                            'category_counts': {'cyber': count} if count else {}}
    write_json(run / 'provider_outcomes.json', outcome_report)
    secondary = {'cache_sha256': m.file_sha(run / 'normalized_samples.jsonl'),
                 'normalization_source_path': str(run / 'code/normalizer.py'), 'normalization_source_sha256': norm_hash,
                 'frozen_grader_contract_path': str(run / 'code/grader.py'), 'frozen_grader_contract_sha256': m.file_sha(run / 'code/grader.py'),
                 'initial_rule_audit_sha256': m.file_sha(run / 'secondary_initial15_audit.json'), 'cells': {}}
    summary = {'status': 'complete', 'received_responses': 24, 'expected_responses': 24, 'complete_prompts': 3,
               'models_returned': {m.MODEL: 24}, 'input_sha256': {name: m.file_sha(run / name) for name in
                    ('manifest.json', 'rows.jsonl', 'datasets.json', 'samples.jsonl')},
               'primary_grading_audit': {'path': str(run / 'primary_python_regrade_audit.json'),
                                         'sha256': m.file_sha(run / 'primary_python_regrade_audit.json')},
               'primary_samples_path': str(run / 'audited_primary_samples.jsonl'),
               'primary_samples_sha256': m.file_sha(run / 'audited_primary_samples.jsonl'),
               'response_records_sha256': m.sha(audited), 'normalized_secondary': secondary, 'cells': {}}
    for level in m.LEVELS:
        correct = sum(s['verified'] for s in raw if s['level'] == level)
        metrics = {'pass1': {'estimate': correct / 8}, 'distinct8': {'estimate': float(correct > 0)},
                   'correct_pair_collision': {'estimate': 1. if correct >= 2 else None}}
        summary['cells'][f'level{level}/{m.DOMAIN}'] = {'metrics': metrics}
        secondary['cells'][f'level{level}/{m.DOMAIN}'] = {'metrics': metrics}
    write_json(run / 'summary.json', summary)
    return run


@pytest.fixture
def original(tmp_path):
    return fixture_cohort(tmp_path / 'original')


@pytest.fixture
def plain(tmp_path, original):
    return fixture_cohort(tmp_path / 'plain', True)


def test_complete_evidence_and_independent_paired_rates(original, plain):
    a = m.authenticate_cohort(original, 24, prompts_per_level=1, io_workers=2)
    b = m.authenticate_cohort(plain, 24, prompts_per_level=1, io_workers=2)
    result = m.compare(a, b, replicates=500)
    assert result['levels']['3']['plain']['metrics']['native_refusal_rate'] == 1 / 8
    assert result['levels']['3']['plain']['metrics']['strict_accuracy'] == 7 / 8
    assert result['levels']['3']['original']['metrics']['strict_correct_pair_collision'] is None
    assert result['levels']['3']['plain_minus_original']['strict_accuracy']['ci95'] == [7/8, 7/8]
    assert result['levels']['3']['joint_eligibility']['strict']['only_plain_collision_eligible'] == 1
    assert result['conditions']['plain']['selected_responses'] == 24


def test_paired_bootstrap_retains_prompt_covariance():
    a = np.array([[1, 8, n, 8-n, 8-n, 1, 1, 1, 1, 1, 1] for n in range(8)], float)
    result = m.paired_contrast(a, a.copy(), 5000, np.random.default_rng(7))
    assert all(v['estimate'] == 0 and v['ci95'] == [0, 0] for v in result.values())


def test_collision_pools_pairs_and_undefined_is_not_zero():
    a = np.array([[1, 8, 0, 8, 8, 1, 1, 28, 28, 28, 28],
                  [1, 8, 0, 2, 2, 2, 2, 1, 0, 1, 0]], float)
    rates = m.rates(a.sum(axis=0))
    assert rates[1] == 10/16
    assert rates[3] == 1.5
    assert rates[5] == 28/29
    a[:, 7:] = 0
    assert m.estimates(m.rates(a.sum(axis=0)))['strict_correct_pair_collision'] is None


@pytest.mark.parametrize('relative', ['normalized_samples.jsonl', 'audited_primary_samples.jsonl',
                                      'provider_outcome_samples.jsonl', 'primary_python_regrade_audit.json'])
def test_appended_even_semantically_identical_file_rejected(original, relative):
    with (original / relative).open('a') as handle:
        handle.write('\n')
    with pytest.raises(ValueError, match='Changed or unbound'):
        m.authenticate_cohort(original, 24, prompts_per_level=1)


def test_native_body_change_rejected(original):
    path = next((original / 'raw_responses').glob('*.json'))
    body = json.loads(path.read_text()); body['response']['stop_reason'] = 'end_turn'
    write_json(path, body)
    with pytest.raises(ValueError, match='Native receipt bytes'):
        m.authenticate_cohort(original, 24, prompts_per_level=1)


def test_wrong_provider_count_rejected(original):
    path = original / 'provider_outcomes.json'
    report = json.loads(path.read_text());report['cells'][f'level1/{m.DOMAIN}']['refusals'] = 0
    write_json(path, report)
    with pytest.raises(ValueError, match='per-cell counts'):
        m.authenticate_cohort(original, 24, prompts_per_level=1)


def test_partial_primary_audit_rejected_even_when_summary_hash_updated(original):
    path = original / 'primary_python_regrade_audit.json'
    audit = json.loads(path.read_text());audit['full_sampling_complete'] = False;write_json(path, audit)
    summary = json.loads((original / 'summary.json').read_text())
    summary['primary_grading_audit']['sha256'] = m.file_sha(path);write_json(original / 'summary.json', summary)
    with pytest.raises(ValueError, match='incomplete, stale, or unresolved'):
        m.authenticate_cohort(original, 24, prompts_per_level=1)


def test_row_change_and_reused_output_are_not_a_prompt_condition(original, plain):
    a = m.authenticate_cohort(original, 24, prompts_per_level=1)
    b = m.authenticate_cohort(plain, 24, prompts_per_level=1)
    first = next(iter(b['raw']))
    b['raw'][first]['response_id'] = a['raw'][first]['response_id']
    with pytest.raises(ValueError, match='reuses original'):
        m.validate_pairing(a, b)
    b['rows'][first[:3]]['answer'] = 'changed mathematical input'
    with pytest.raises(ValueError, match='changed mathematical tasks'):
        m.validate_pairing(a, b)


def test_exact_template_changes_are_rejected(original, plain):
    a = m.authenticate_cohort(original, 24, prompts_per_level=1)
    b = m.authenticate_cohort(plain, 24, prompts_per_level=1)
    first = next(iter(b['requests']))
    b['requests'][first]['request']['messages'][0]['content'] += ' Always select the smallest divisor.'
    with pytest.raises(ValueError, match='single fixed educational template'):
        m.validate_pairing(a, b)
