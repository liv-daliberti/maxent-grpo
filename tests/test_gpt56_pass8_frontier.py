"""Empirical any-correct pass@8, paired inference, and retained-cohort integrity."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('gpt_pass8', Path(__file__).resolve().parents[1] /
                                             'ops/analyze_gpt56_pass8_frontier.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def prompt_records(profile):
    records = []
    for level in m.LEVELS:
        for domain in m.DOMAINS:
            for row_index, correct in enumerate(profile):
                pairs = correct * (correct - 1) // 2
                records.append({'level': level, 'domain': domain, 'row_index': row_index,
                    'row_sha256': hashlib.sha256(f'{level}/{domain}/{row_index}'.encode()).hexdigest(),
                    'responses': 8, 'correct_responses': correct, 'distinct8': int(correct > 0),
                    'correct_pairs': pairs, 'colliding_correct_pairs': pairs,
                    'collision_eligible': correct >= 2, 'native_refusals': 0,
                    'native_content_filtered': 0, 'truncated_responses': 8 - correct,
                    'empty_answers': 0})
    return records


def metric(value):
    return {'estimate': value, 'ci95': [value, value], 'defined_replicates': m.REPLICATES}


def block(records):
    _, by_cell = m.validate_records(records, 'synthetic')
    result = m.empty_metric_block()
    for key, rs in by_cell.items():
        result['cells'][key].update(
            accuracy=metric(sum(r['correct_responses'] for r in rs) / 64),
            distinct8=metric(sum(r['distinct8'] for r in rs) / 8),
            collision=metric(1), counts=m.counts(rs), empty_answers=0)
    for group in result['groups'].values():
        for metrics in [group['overall'], *group['levels'].values(), *group['level_contrasts'].values()]:
            metrics.update(accuracy=metric(.5), distinct8=metric(1), collision=metric(1))
    result.update(prompt_records=records, counts=m.counts(records))
    return result


@pytest.fixture
def source():
    profiles = {'0.5': [0, 1, 8, 0, 0, 0, 0, 0], '1.0': [0, 1, 8, 0, 0, 0, 0, 0],
                '1.5': [0, 1, 8, 1, 0, 0, 0, 0], '2.0': [0, 0, 8, 1, 0, 0, 0, 0]}
    result = {'schema': 'gpt56-none-temperature-curve-v1', 'status': 'complete',
              'model': 'gpt-5.6-sol', 'reasoning_effort': 'none', 'temperatures': [.5, 1., 1.5, 2.],
              'bootstrap': {'seed': m.SEED, 'replicates': m.REPLICATES, 'individual_draw_pairing': False},
              'validation': {key: True for key in m.REQUIRED_VALIDATION},
              'analysis_sources': {}, 'original_metadata_sentinel': {'untouched': True}, 'analyses': {},
              'matched_medium_reference': {'model': 'gpt-5.6-sol', 'reasoning_effort': 'medium',
                  'requested_temperature': None, 'returned_temperature': 1.,
                  'connect_to_temperature_curve': False, 'analyses': {}}}
    for grading in m.GRADINGS:
        result['analyses'][grading] = {
            'temperatures': {temperature: block(prompt_records(profile)) for temperature, profile in profiles.items()},
            'paired_contrasts_vs_t1p0': {t: m.empty_metric_block() for t in m.TEMPERATURES},
            'paired_endpoint_contrast': m.empty_metric_block() | {'comparison': 'T2.0-T0.5'}}
        for group in result['analyses'][grading]['paired_endpoint_contrast']['groups'].values():
            for entry in [group['overall'], *group['levels'].values(), *group['level_contrasts'].values()]:
                entry.update(accuracy=metric(0), distinct8=metric(0))
        result['matched_medium_reference']['analyses'][grading] = block(prompt_records([8] * 8))
    return result


def test_empirical_pass8_differs_from_pass1_and_iid_shortcut(source):
    result = m.analyze(source)
    cell = result['analyses']['strict']['temperatures']['0.5']['cells']['level1/countdown']
    assert cell['accuracy']['estimate'] == 9 / 64
    assert cell['pass8']['estimate'] == 2 / 8
    assert cell['pass8']['estimate'] != pytest.approx(1 - (1 - 9 / 64) ** 8)
    assert cell['pass8']['defined_replicates'] == m.REPLICATES
    assert cell['pass8']['ci95'] == [0, .625]
    assert result['matched_medium_reference']['connect_to_temperature_curve'] is False
    assert result['matched_medium_reference']['analyses']['strict']['groups']['five_domain_macro']['overall']['pass8'] == metric(1)


def test_every_original_field_and_metric_is_preserved(source):
    before = deepcopy(source)
    result = m.analyze(source)
    assert source == before

    def preserved(old, new, path=()):
        if isinstance(old, dict):
            for key, value in old.items():
                if path == () and key == 'schema':
                    continue
                preserved(value, new[key], (*path, key))
        else:
            assert old == new, path

    preserved(source, result)
    for grading in m.GRADINGS:
        data = result['analyses'][grading]
        blocks = [*data['temperatures'].values(), *data['paired_contrasts_vs_t1p0'].values(),
                  data['paired_endpoint_contrast'], *data['pass8_pairwise_contrasts'].values(),
                  result['matched_medium_reference']['analyses'][grading]]
        for item in blocks:
            assert all('pass8' in cell for cell in item['cells'].values())
            for group in item['groups'].values():
                assert all('pass8' in metrics for metrics in
                           [group['overall'], *group['levels'].values(), *group['level_contrasts'].values()])


def test_paired_contrasts_use_same_prompt_resamples(source):
    result = m.analyze(source)
    data = result['analyses']['normalized_secondary']
    baseline = data['paired_contrasts_vs_t1p0']['1.0']
    assert baseline['groups']['five_domain_macro']['overall']['pass8'] == metric(0)
    assert data['pass8_pairwise_contrasts']['2.0-0.5']['groups'] == {
        key: {name: ({'pass8': value['pass8']} if name == 'overall' else
                    {level: {'pass8': entry['pass8']} for level, entry in value.items()})
              for name, value in group.items()}
        for key, group in data['paired_endpoint_contrast']['groups'].items()}
    rng = np.random.default_rng(m.SEED)
    # The first domain/level receives the first registered resampling draw.
    indices = rng.integers(0, 8, (m.REPLICATES, 8))
    change = np.asarray([0, -1, 0, 1, 0, 0, 0, 0], dtype=float)
    expected = m.describe(0, change[indices].mean(axis=1))
    actual = data['paired_endpoint_contrast']['cells']['level1/countdown']['pass8']
    assert actual == expected
    assert actual['ci95'][0] < 0 < actual['ci95'][1]
    assert len(data['pass8_pairwise_contrasts']) == 6


def test_zero_success_and_failed_draw_groups_remain_in_denominator(source):
    for grading in m.GRADINGS:
        source['analyses'][grading]['temperatures']['2.0'] = block(prompt_records([0] * 8))
    result = m.analyze(source)
    arm = result['analyses']['strict']['temperatures']['2.0']
    assert arm['counts']['responses'] == arm['counts']['truncated_responses'] == 960
    assert arm['counts']['prompts'] == 120
    assert arm['groups']['five_domain_macro']['overall']['pass8'] == metric(0)
    delta = result['analyses']['strict']['paired_endpoint_contrast']['groups']['five_domain_macro']['overall']
    assert delta['pass8']['estimate'] == -.25
    assert delta['pass8']['ci95'][1] < 0


def test_reordered_rows_are_aligned_by_identity(source):
    expected = m.analyze(source)
    for grading in m.GRADINGS:
        source['analyses'][grading]['temperatures']['1.5']['prompt_records'].reverse()
    result = m.analyze(source)
    for grading in m.GRADINGS:
        assert result['analyses'][grading]['pass8_pairwise_contrasts'] == expected['analyses'][grading]['pass8_pairwise_contrasts']


@pytest.mark.parametrize('change,match', [
    ('missing', '120 complete'), ('duplicate', 'duplicate prompt row identity'),
    ('duplicate_digest', 'duplicate prompt row digest'), ('draw_missing', 'all eight responses'),
    ('unpaired', 'unpaired prompt identities'), ('changed_digest', 'unpaired prompt identities'),
    ('count_mismatch', 'source counts differ'), ('cell_count_mismatch', 'source counts differ'),
    ('accuracy_mismatch', 'source accuracy differs'), ('invalid_success', 'invalid correct-response'),
    ('invalid_modes', 'invalid correct-response'), ('missing_temperature', 'incomplete temperature'),
    ('extra_cell', 'missing or extra source cells'), ('imbalanced_cell', 'eight prompts in every'),
    ('reference_unpaired', 'unpaired prompt identities')])
def test_missing_duplicate_unpaired_and_inconsistent_records_rejected(source, change, match):
    data = source['analyses']['strict']
    arm = data['temperatures']['0.5']
    records = arm['prompt_records']
    if change == 'missing': records.pop()
    elif change == 'duplicate': records[1] = deepcopy(records[0])
    elif change == 'duplicate_digest': records[1]['row_sha256'] = records[0]['row_sha256']
    elif change == 'draw_missing': records[0]['responses'] = 7
    elif change == 'unpaired': records[0]['row_index'] = 100
    elif change == 'changed_digest': records[0]['row_sha256'] = 'a' * 64
    elif change == 'count_mismatch': arm['counts']['prompts_with_correct_answer'] += 1
    elif change == 'cell_count_mismatch': arm['cells']['level1/countdown']['counts']['correct_responses'] += 1
    elif change == 'accuracy_mismatch': arm['cells']['level1/countdown']['accuracy']['estimate'] = .1
    elif change == 'invalid_success': records[0]['correct_responses'] = 9
    elif change == 'invalid_modes': records[0]['distinct8'] = 1
    elif change == 'missing_temperature': data['temperatures'].pop('1.5')
    elif change == 'extra_cell': arm['cells']['level4/countdown'] = {}
    elif change == 'imbalanced_cell':
        records[0]['level'], records[0]['row_index'] = 2, 100
    elif change == 'reference_unpaired':
        source['matched_medium_reference']['analyses']['strict']['prompt_records'][0]['row_sha256'] = 'a' * 64
    with pytest.raises(ValueError, match=match):
        m.analyze(source)


def test_source_bindings_are_checked_recursively_and_tampering_rejected(tmp_path):
    path = tmp_path / 'source.json'
    path.write_text('{"complete": true}\n')
    record = {'list': [{'deeper': {'path': str(path), 'sha256': m.file_sha(path)}}]}
    assert m.authenticate(record) == 1
    path.write_text('{"complete": false}\n')
    with pytest.raises(ValueError, match='Stale GPT temperature source binding'):
        m.authenticate(record)


def test_build_report_binds_original_and_new_analyzer(source, tmp_path):
    evidence = tmp_path / 'evidence.txt'
    evidence.write_text('authenticated evidence')
    source['analysis_sources']['evidence'] = {'path': str(evidence), 'sha256': m.file_sha(evidence)}
    path = tmp_path / 'original.json'
    path.write_text(json.dumps(source))
    result = m.build_report(path)
    assert result['analysis_sources']['original_accuracy_analysis'] == {'path': str(path), 'sha256': m.file_sha(path)}
    assert result['analysis_sources']['analyze_gpt56_pass8_frontier.py']['sha256'] == m.file_sha(m.__file__)
    assert result['validation']['pass8_computed_from_complete_prompt_groups'] is True
    rows = list(m.table_rows(result))
    assert len(rows) == 2 * 6 * 2 * 4
    assert {row['level'] for row in rows} == {'all', '1', '2', '3'}
    assert '1-(1-pass@1)^8' in m.markdown(result)
    assert 'pass8_ci95_low' in m.csv_text(result)


def test_level_and_finite_support_macros_keep_registered_domain_weights(source):
    for grading in m.GRADINGS:
        records = prompt_records([0, 1, 8, 0, 0, 0, 0, 0])
        replacements = {(r['level'], r['domain'], r['row_index']): r for r in prompt_records([8] * 8)}
        records = [replacements[m.row_identity(r)] if (r['level'], r['domain']) == (1, 'countdown') else r
                   for r in records]
        source['analyses'][grading]['temperatures']['0.5'] = block(records)
    result = m.analyze(source)
    groups = result['analyses']['strict']['temperatures']['0.5']['groups']
    assert groups['five_domain_macro']['overall']['pass8']['estimate'] == pytest.approx(.3)
    assert groups['five_domain_macro']['levels']['1']['pass8']['estimate'] == pytest.approx(.4)
    assert groups['five_domain_macro']['levels']['2']['pass8']['estimate'] == pytest.approx(.25)
    assert groups['five_domain_macro']['level_contrasts']['L2-L1']['pass8']['estimate'] == pytest.approx(-.15)
    assert groups['four_finite_support_domain_macro']['overall']['pass8']['estimate'] == pytest.approx(.25)
