"""The expanded curve must preserve whole groups, paired cells, and cohorts."""
from __future__ import annotations
import copy
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ops'))
import analyze_gpt56_temperature_curve_expanded480 as expanded
import analyze_gpt56_pass8_frontier_expanded480 as frontier


def make_population(size=32):
    rows, records = {}, []
    for level in expanded.LEVELS:
        for domain in expanded.DOMAINS:
            for row_index in range(size):
                row = {'level': level, 'domain': domain, 'row_index': row_index,
                       'metadata': {'answer_mode_count': 2}, 'prompt': f'{level}/{domain}/{row_index}'}
                rows[level, domain, row_index] = row
                correct = 8 if row_index < 8 else 0
                pairs = correct * (correct - 1) // 2
                records.append({'level': level, 'domain': domain, 'row_index': row_index,
                    'row_sha256': expanded.sha(row), 'responses': 8, 'correct_responses': correct,
                    'distinct8': 1 if correct else 0, 'correct_pairs': pairs,
                    'colliding_correct_pairs': pairs, 'certified_support_count': None if domain == 'countdown' else 2,
                    'uniform_expected_colliding_correct_pairs': None if domain == 'countdown' else pairs / 2,
                    'collision_eligible': bool(correct), 'native_refusals': 0,
                    'native_content_filtered': 0, 'truncated_responses': 0, 'empty_answers': 0})
    return rows, records


def all_temperatures(records):
    return {t: copy.deepcopy(records) for t in expanded.TEMPERATURES}


def test_combined_estimate_retains_old_and_new_zero_success_groups():
    rows, records = make_population()
    result = expanded.analyze_records(all_temperatures(records), rows, 32, replicates=200)
    overall = result['temperatures']['0.0']['groups']['five_domain_macro']['overall']
    assert overall['pass8']['estimate'] == .25
    assert overall['distinct8']['estimate'] == .25
    assert result['temperatures']['0.0']['counts']['responses'] == 3840
    assert result['temperatures']['0.0']['counts']['prompts'] == 480
    assert overall['pass8']['defined_replicates'] == 200


def test_all_temperature_contrasts_share_identical_whole_prompt_resamples():
    rows, records = make_population()
    result = expanded.analyze_records(all_temperatures(records), rows, 32, replicates=200)
    contrast = result['paired_high_temperature_contrast']
    assert contrast['comparison'] == 'T2.0-T1.5'
    for metric in ('pass8', 'distinct8', 'accuracy'):
        stats = contrast['groups']['five_domain_macro']['overall'][metric]
        assert stats['estimate'] == 0
        assert stats['ci95'] == [0., 0.]
    # Different draws are never resampled independently from their parent prompt.
    cell = result['temperatures']['0.0']['cells']['level1/countdown']
    assert np.allclose(cell['pass8']['ci95'], cell['distinct8']['ci95'])


def test_equal_cell_mean_and_all_zero_new_cohort_remain_defined():
    rows, records = make_population()
    rows = {key: row for key, row in rows.items() if key[2] >= 8}
    records = [r for r in records if r['row_index'] >= 8]
    result = expanded.analyze_records(all_temperatures(records), rows, 24, replicates=100)
    stats = result['temperatures']['2.0']['groups']['five_domain_macro']['overall']
    assert stats['pass8']['estimate'] == stats['distinct8']['estimate'] == 0
    assert stats['pass8']['ci95'] == [0., 0.]
    assert stats['collision']['estimate'] is None
    assert stats['collision']['defined_replicates'] == 0


@pytest.mark.parametrize('mutation,pattern', [
    (lambda rs: rs.pop(), 'incomplete prompt'),
    (lambda rs: rs.__setitem__(1, copy.deepcopy(rs[0])), 'Duplicate'),
    (lambda rs: rs[0].__setitem__('row_sha256', '0' * 64), 'changed row'),
    (lambda rs: rs[0].__setitem__('responses', 7), 'incomplete eight-draw'),
    (lambda rs: rs[0].__setitem__('correct_responses', 9), 'invalid correct'),
    (lambda rs: rs[0].__setitem__('distinct8', 0), 'invalid correct'),
    (lambda rs: rs[0].__setitem__('correct_pairs', 0), 'inconsistent pair'),
    (lambda rs: rs[0].__setitem__('empty_answers', 9), 'invalid retained'),
    (lambda rs: rs[0].__setitem__('certified_support_count', 2), 'support baseline'),
    (lambda rs: rs[0].__setitem__('level', True), 'invalid prompt identity'),
])
def test_incomplete_or_corrupted_prompt_groups_fail_publication(mutation, pattern):
    rows, records = make_population()
    mutation(records)
    with pytest.raises(ValueError, match=pattern):
        expanded.validate_records(records, rows, 32, 'test')


def test_missing_temperature_is_not_silently_omitted():
    rows, records = make_population()
    grid = all_temperatures(records)
    del grid[0.]
    with pytest.raises(ValueError, match='Missing or extra temperature'):
        expanded.analyze_records(grid, rows, 32, replicates=10)


def test_extra_temperature_is_not_silently_used():
    rows, records = make_population()
    grid = all_temperatures(records)
    grid[2.5] = records
    with pytest.raises(ValueError, match='Missing or extra temperature'):
        expanded.analyze_records(grid, rows, 32, replicates=10)


def test_frontier_rejects_old_cohort_presented_as_expanded():
    with pytest.raises(ValueError, match='complete expanded-480'):
        frontier.analyze({'schema': 'gpt56-none-temperature-curve-v2', 'status': 'complete'})


def test_frozen_entrypoints_find_real_repo_root(tmp_path):
    target = ROOT / 'artifacts/frontier_temperature_20260911/prompt_expansion_32_per_cell'
    spec = importlib.util.spec_from_file_location('_expanded_location_test', ROOT / 'ops/analyze_gpt56_temperature_curve_expanded480.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.ROOT == ROOT
    assert module.PLAN.parent == target


def test_native_reused_response_id_fails_before_aggregation():
    rows, records = make_population()
    row_key = next(iter(rows))
    sample_key = (*row_key, 0)
    request = {'max_output_tokens': 8192, 'store': False}
    native = {'headers': {'x-ms-served-model': 'snapshot'},
              'received_at_utc': '2026-09-12T00:00:00+00:00',
              'response': {'temperature': 0., 'reasoning': {'effort': 'none'},
                           'model': 'gpt-5.6-sol', 'max_output_tokens': 8192, 'store': False, 'top_p': .98}}
    cohort = {'raw': {sample_key: {'response_id': 'duplicate', 'raw_receipt': 'r.json'}},
              'bodies': {'r.json': native}, 'requests': {sample_key: {'request': request}},
              'summary': {'normalized_secondary': {k: 'same' for k in ('normalization_source_sha256',
                    'initial_rule_audit_sha256', 'frozen_grader_contract_sha256')}},
              'grading_audit': {'frozen_grader_modules': {}}}
    with pytest.raises(ValueError, match='response ID was reused'):
        expanded.audit_native_configuration({0.: {'old': cohort, 'new': cohort}})


def test_changed_native_reasoning_control_fails_before_aggregation():
    sample_key = (1, 'countdown', 0, 0)
    cohort = {'raw': {sample_key: {'response_id': 'r', 'raw_receipt': 'r.json'}},
              'bodies': {'r.json': {'response': {'temperature': 0., 'reasoning': {'effort': 'medium'}}}},
              'requests': {sample_key: {'request': {'max_output_tokens': 8192, 'store': False}}}}
    with pytest.raises(ValueError, match='Native controls differ'):
        expanded.audit_native_configuration({0.: {'old': cohort}})
