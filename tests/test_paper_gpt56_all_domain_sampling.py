"""Require complete five-domain source data and preserve finite observed budgets."""
from copy import deepcopy
import json

import pytest

from ops import plot_paper_gpt56_all_domain_sampling as plot


def report_fixture():
    cells = []
    for domain in plot.DOMAINS:
        for level in plot.LEVELS:
            budget = 128 if domain == 'pantry_plan' else 64
            cells.append({'domain': domain, 'level': level, 'n_prompts': 16,
                          'max_draws': budget,
                          'support': {'mean': 2, 'range': [2, 2], 'kind': 'certified_lower_bound'},
                          'points': [{'k': k, 'distinct': {'estimate': 1, 'ci95': [1, 1]},
                                      'pass': {'estimate': 1, 'ci95': [1, 1]}}
                                     for k in plot.required_grid(budget)], 'tail': {}})
    return {'schema': 'gpt56-all-domain-discovery-v1', 'status': 'complete',
            'model': 'gpt-5.6-sol', 'grading': 'normalized_secondary', 'prompt_arm': 'original',
            'cells': cells}


def test_mixed_observed_budgets_are_preserved_without_extending_other_cells(tmp_path):
    source = tmp_path / 'analysis.json'
    source.write_text(json.dumps(report_fixture()))
    record = plot.build_record(source)
    assert len(record['cells']) == 10
    for cell in record['cells']:
        budget = 128 if cell['domain'] == 'pantry_plan' else 64
        assert cell['max_draws'] == cell['points'][-1]['k'] == budget
        assert cell['support']['kind'] == 'certified_lower_bound'
    assert record['source'] == plot.binding(source)


@pytest.mark.parametrize('mutation', ['missing_domain', 'duplicate', 'omitted_prompt',
                                      'missing_endpoint', 'nonmonotone', 'invalid_support',
                                      'impossible_distinct', 'interval'])
def test_incomplete_or_inconsistent_source_values_are_rejected(mutation):
    report = report_fixture()
    cells = report['cells']
    if mutation == 'missing_domain':
        cells.pop()
    elif mutation == 'duplicate':
        cells[-1] = deepcopy(cells[0])
    elif mutation == 'omitted_prompt':
        cells[0]['n_prompts'] = 15
    elif mutation == 'missing_endpoint':
        cells[0]['points'].pop()
    elif mutation == 'nonmonotone':
        cells[0]['points'][-1]['pass'] = {'estimate': .5, 'ci95': [.5, .5]}
    elif mutation == 'invalid_support':
        cells[0]['support']['mean'] = 3
    elif mutation == 'impossible_distinct':
        cells[0]['points'][0]['distinct'] = {'estimate': 2, 'ci95': [2, 2]}
    elif mutation == 'interval':
        cells[0]['points'][0]['distinct']['ci95'] = [float('nan'), 1]
    with pytest.raises(ValueError):
        plot.validate_cells(cells)


@pytest.mark.parametrize('key,value', [('status', 'running'), ('prompt_arm', 'neutral'),
                                      ('grading', 'strict'), ('model', 'another-model')])
def test_wrong_or_unfinished_analysis_is_rejected(tmp_path, key, value):
    report = report_fixture()
    report[key] = value
    source = tmp_path / 'analysis.json'
    source.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        plot.build_record(source)


def test_changed_source_and_output_bytes_fail_retained_figure_check(tmp_path):
    source = tmp_path / 'analysis.json'
    source.write_text(json.dumps(report_fixture()))
    output = tmp_path / 'figure'
    record = plot.build_record(source)
    record['outputs'] = {}
    for extension in ('pdf', 'png'):
        path = output.with_suffix('.' + extension)
        path.write_bytes(b'known output bytes')
        record['outputs'][extension] = plot.binding(path)
    output.with_suffix('.json').write_text(json.dumps(record))
    assert plot.check(output, source)['source'] == plot.binding(source)
    output.with_suffix('.png').write_bytes(b'changed bytes')
    with pytest.raises(ValueError, match='rendering'):
        plot.check(output, source)
    output.with_suffix('.png').write_bytes(b'known output bytes')
    source.write_text(source.read_text() + '\n')
    with pytest.raises(ValueError, match='source analysis'):
        plot.check(output, source)


def test_exact_and_lower_bound_support_labels_remain_distinct():
    cells = report_fixture()['cells'][:2]
    assert '≥' in plot.support_label(cells)
    for cell in cells:
        cell['support']['kind'] = 'exact'
    assert '≥' not in plot.support_label(cells)
    assert '2 known modes' in plot.support_label(cells)
