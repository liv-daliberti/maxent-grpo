"""Publication facts must come from the bound final analysis and its prefixes."""
import json

import pytest

from ops import build_paper_gpt56_all_domain_discovery as builder


def estimate(value):
    return {'estimate': value, 'ci95': [value, value]}


def report_fixture():
    normalized, strict = [], []
    for domain in builder.DOMAINS:
        for level in builder.LEVELS:
            support = {'mean': 6.25 if domain == 'graph_coloring' else 4.3125,
                       'range': [4, 8],
                       'kind': 'exact' if domain == 'graph_coloring' else 'certified_lower_bound'}
            for cells, prefix_successes, distinct, passed in [(normalized, 8, 2., 1.), (strict, 4, 1., .5)]:
                cells.append({'domain': domain, 'level': level, 'n_prompts': 16, 'max_draws': 128,
                              'support': support,
                              'points': [{'k': 128, 'distinct': estimate(distinct), 'pass': estimate(passed)}],
                              'tail': {'from_k': 64, 'to_k': 128, 'distinct': estimate(.2)},
                              'prompts': [{'row_index': i, 'prefix': {'64': {'pass': int(i < prefix_successes)}}}
                                          for i in range(16)]})
    return {'schema': 'gpt56-all-domain-discovery-v1', 'status': 'complete', 'model': 'gpt-5.6-sol',
            'prompt_arm': 'original', 'grading': 'normalized_secondary',
            'served_snapshot_header': 'gpt-5.6-sol-2026-07-09',
            'cells': normalized, 'strict_cells': strict, 'responses': 20480, 'prompts': 160,
            'protocol': {'reasoning': 'medium', 'max_output_tokens': 8192,
                         'temperature_and_top_p': 'omitted',
                         'bootstrap': {'unit': 'whole prompt', 'pointwise': True, 'replicates': 20000, 'seed': 20260911}},
            'sources': [], 'prior_analysis': {}, 'new_support': {}, 'support_certificate': {}, 'analyzer': {}}


def write_source(tmp_path, report=None):
    path = tmp_path / 'source.json'
    path.write_text(json.dumps(report if report is not None else report_fixture()))
    return path


def test_first64_fact_uses_ordered_prefixes_not_expanded_pool_pass_probability(tmp_path):
    record = builder.build_record(write_source(tmp_path))
    assert record['first64_prefix_successes'] == {'normalized': 80, 'strict': 40}
    text = builder.render_tex(record)
    assert 'solve 80\nof 160 problems after normalization and 40 under strict grading' in text
    assert '2.000/100.0' in text
    assert '1.000/50.0' in text
    assert '$D_N(N)-D_N(N/2)$' in text


def test_support_display_never_rounds_a_lower_bound_upward():
    assert builder.support_tex({'kind': 'certified_lower_bound', 'mean': 4.319}) == '$\\geq 4.31$'
    assert builder.support_tex({'kind': 'exact', 'mean': 6.25}) == '$6.25$'


def test_missing_domain_and_mismatched_grading_budgets_are_rejected(tmp_path):
    report = report_fixture()
    report['cells'].pop()
    with pytest.raises(ValueError, match='all ten'):
        builder.build_record(write_source(tmp_path, report))
    report = report_fixture()
    report['strict_cells'][0]['max_draws'] = 64
    with pytest.raises(ValueError, match='identical complete'):
        builder.build_record(write_source(tmp_path, report))


def test_check_detects_changed_source_bytes_and_output_tamper(tmp_path):
    source = write_source(tmp_path)
    output = tmp_path / 'published'
    builder.write(builder.build_record(source), output)
    builder.check(source, output)
    tex = output / (builder.STEM + '.tex')
    tex.write_text(tex.read_text() + '% changed\n')
    with pytest.raises(ValueError, match='Publication TeX differs'):
        builder.check(source, output)
    builder.write(builder.build_record(source), output)
    source.write_text(source.read_text() + '\n')
    with pytest.raises(ValueError, match='Publication JSON differs'):
        builder.check(source, output)
