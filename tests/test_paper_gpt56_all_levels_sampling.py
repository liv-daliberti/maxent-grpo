"""Integrity tests use synthetic pools; no synthetic publication data are retained."""
from copy import deepcopy
import itertools
import json
import math
from pathlib import Path

import pytest

from ops import plot_paper_gpt56_all_levels_sampling as plot
from ops import build_paper_gpt56_all_levels_discovery as appendix


def synthetic_report(draws=512):
    """All fifteen cells, with failures and varied multiplicities in each cell."""
    grid = tuple(2 ** power for power in range(draws.bit_length()))
    cells = []
    for domain in plot.DOMAINS:
        for level in plot.LEVELS:
            kind = 'exact' if domain == 'graph_coloring' else 'certified_lower_bound'
            # Some certified lower bounds are exceeded by observed discovery.
            support_count = 5 if kind == 'exact' else 1
            prompts, values = [], []
            for index in range(16):
                counts = ([] if index % 4 == 0 else [draws // 2] if index % 4 == 1
                          else [draws // 2 - 6, 1] if index % 4 == 2 else [draws // 3 + 30, 15, 3])
                keys = [j for j, n in enumerate(counts) for _ in range(n)]
                keys += [None] * (draws - len(keys))
                prefix = {str(k): {'distinct': float(len(set(keys[:k]) - {None})),
                                   'pass': float(any(v is not None for v in keys[:k]))}
                          for k in grid}
                prompts.append({'domain': domain, 'level': level, 'row_index': index,
                                'row_sha256': f'{index:064x}', 'responses': draws,
                                'correct_draws': sum(counts), 'mode_counts': counts,
                                'prefix': prefix, 'support': {'support_count': support_count,
                                                             'support_kind': kind}})
                values.append({k: plot.rarefaction(counts, draws, k) for k in grid})

            def summary(items):
                estimate = math.fsum(items) / len(items)
                return {'estimate': estimate, 'ci95': [min(items), max(items)]}

            points = [{'k': k, **{metric: summary([value[k][metric] for value in values])
                                  for metric in ('distinct', 'pass')}} for k in grid]
            tail = {'from_k': draws // 2, 'to_k': draws,
                    **{metric: summary([v[draws][metric] - v[draws // 2][metric] for v in values])
                       for metric in ('distinct', 'pass')}}
            cells.append({'domain': domain, 'level': level, 'n_prompts': 16, 'max_draws': draws,
                          'support': {'mean': support_count, 'range': [support_count, support_count],
                                      'kind': kind}, 'prompts': prompts, 'points': points, 'tail': tail})
    return {'schema': plot.SCHEMA, 'status': 'complete', 'model': 'gpt-5.6-sol',
            'served_snapshot_header': 'gpt-5.6-sol-2026-07-09',
            'grading': 'normalized_secondary', 'prompt_arm': 'original', 'levels': [1, 2, 3],
            'prompts': 240, 'responses': 240 * draws, 'cells': cells, 'strict_cells': deepcopy(cells),
            'protocol': {'reasoning': 'medium', 'max_output_tokens': 8192,
                         'temperature_and_top_p': 'omitted',
                         'bootstrap': {'pointwise': True, 'unit': 'whole prompt',
                                       'replicates': 20000, 'seed': 20260911}},
            'test_fixture_only': True}


@pytest.mark.parametrize('keys', [[], [0, 0], [0, 1], [0, 0, 1, None], [0, 1, 1, 2, None]])
def test_rarefaction_matches_exhaustive_uniform_subsets(keys):
    keys = keys or [None, None]
    counts = [keys.count(key) for key in set(keys) - {None}]
    for k in range(len(keys) + 1):
        subsets = list(itertools.combinations(range(len(keys)), k))
        observed = [set(keys[i] for i in indices) - {None} for indices in subsets]
        actual = plot.rarefaction(counts, len(keys), k)
        assert actual['distinct'] == pytest.approx(sum(map(len, observed)) / len(subsets))
        assert actual['pass'] == pytest.approx(sum(bool(item) for item in observed) / len(subsets))


def test_completed_all_level_pools_reconstruct_under_both_conventions():
    report = synthetic_report()
    assert plot.validate_report(report) is report
    assert all(c['max_draws'] == 512 for c in report['cells'])
    assert {c['level'] for c in report['cells']} == {1, 2, 3}
    # The synthetic non-Graph endpoint exceeds its certified lower bound.
    assert report['cells'][-1]['points'][-1]['distinct']['estimate'] > 1


@pytest.mark.parametrize('mutation', [
    'missing_level', 'duplicate_cell', 'wrong_budget', 'missing_prompt', 'duplicate_prompt',
    'missing_draw', 'count_sum', 'negative_count', 'wrong_mean', 'wrong_tail', 'wrong_ci',
    'wrong_prefix', 'missing_grid', 'support_mean', 'support_range', 'support_kind',
    'graph_exhaustiveness', 'grade_cohort', 'grade_support', 'protocol', 'total', 'incomplete',
])
def test_missing_or_tampered_scientific_evidence_is_rejected(mutation):
    report = synthetic_report()
    cell = report['cells'][0]
    prompt = cell['prompts'][3]
    if mutation == 'missing_level':
        report['cells'] = [c for c in report['cells'] if c['level'] != 1]
    elif mutation == 'duplicate_cell':
        report['cells'][-1] = deepcopy(cell)
    elif mutation == 'wrong_budget':
        cell['max_draws'] = 256
    elif mutation == 'missing_prompt':
        cell['prompts'].pop()
    elif mutation == 'duplicate_prompt':
        cell['prompts'][-1] = deepcopy(prompt)
    elif mutation == 'missing_draw':
        prompt['responses'] = 511
    elif mutation == 'count_sum':
        prompt['correct_draws'] += 1
    elif mutation == 'negative_count':
        prompt['mode_counts'] = [-1, 219]
    elif mutation == 'wrong_mean':
        cell['points'][5]['distinct']['estimate'] += .1
    elif mutation == 'wrong_tail':
        cell['tail']['distinct']['estimate'] += .1
    elif mutation == 'wrong_ci':
        cell['points'][5]['pass']['ci95'][1] = float('nan')
    elif mutation == 'wrong_prefix':
        prompt['prefix']['512']['distinct'] = 2
    elif mutation == 'missing_grid':
        cell['points'].pop(4)
    elif mutation == 'support_mean':
        cell['support']['mean'] = 4.5
    elif mutation == 'support_range':
        cell['support']['range'][0] = 4
    elif mutation == 'support_kind':
        cell['support']['kind'] = 'certified_lower_bound'
    elif mutation == 'graph_exhaustiveness':
        prompt['support']['support_count'] = 2
    elif mutation == 'grade_cohort':
        report['strict_cells'][0]['prompts'][0]['row_sha256'] = 'changed'
    elif mutation == 'grade_support':
        report['strict_cells'][3]['prompts'][0]['support']['extra'] = 'changed'
    elif mutation == 'protocol':
        report['protocol']['reasoning'] = 'none'
    elif mutation == 'total':
        report['responses'] -= 1
    elif mutation == 'incomplete':
        report['status'] = 'running'
    with pytest.raises(ValueError):
        plot.validate_report(report)


def test_known_support_labels_preserve_level_order_and_bound_direction():
    cells = [{'level': level, 'support': {'mean': mean, 'kind': 'certified_lower_bound'}}
             for level, mean in ((3, 16.8125), (1, 19.4375), (2, 17))]
    assert plot.support_label(cells) == '≥ 19.43 / 17 / 16.81\nknown modes, L1–L3'
    for cell in cells:
        cell['support']['kind'] = 'exact'
    assert plot.support_label(cells).startswith('19.44 / 17 / 16.81')


def test_y_limits_include_observed_curves_above_certified_lower_bounds():
    pytest.importorskip('matplotlib')
    import matplotlib.pyplot as plt
    report = synthetic_report()
    figure = plot.build_figure({'cells': report['cells']})
    for ax, domain in zip(figure.axes, plot.DOMAINS):
        cells = [c for c in report['cells'] if c['domain'] == domain]
        assert ax.get_ylim()[1] > max(p['distinct']['ci95'][1] for c in cells for p in c['points'])
        assert ax.get_ylim()[1] > max(c['support']['mean'] for c in cells)
    plt.close(figure)


def write_source(tmp_path):
    source = tmp_path / 'synthetic_analysis.json'
    source.write_text(json.dumps(synthetic_report()))
    return source


def test_figure_check_rejects_changed_source_and_rendering_bytes(tmp_path):
    source = write_source(tmp_path)
    record = plot.build_record(source)
    assert len(record['cells']) == 15
    assert record['analysis_provenance']['test_fixture_only'] is True
    output = tmp_path / 'figure'
    record['outputs'] = {}
    for ext in ('png', 'pdf'):
        output.with_suffix('.' + ext).write_bytes(b'test output binding only')
        record['outputs'][ext] = plot.binding(output.with_suffix('.' + ext))
    output.with_suffix('.json').write_text(json.dumps(record))
    assert plot.check(output, source)['source'] == plot.binding(source)
    output.with_suffix('.png').write_bytes(b'tampered rendering')
    with pytest.raises(ValueError, match='rendering'):
        plot.check(output, source)
    output.with_suffix('.png').write_bytes(b'test output binding only')
    source.write_text(source.read_text() + '\n')
    with pytest.raises(ValueError, match='source or renderer'):
        plot.check(output, source)


def test_appendix_keeps_all_cells_both_grades_full_curves_and_prefixes(tmp_path):
    source = write_source(tmp_path)
    record = appendix.build_record(source)
    assert len(record['rows']) == 15
    assert record['responses'] == 122880 and record['prompts'] == 240
    assert record['levels'] == [1, 2, 3]
    for row in record['rows']:
        assert len(row['prompts']) == len(row['support_counts']) == 16
        for grading in ('normalized', 'strict'):
            assert len(row['curves'][grading]) == 10
            assert set(row['prefix_successes'][grading]) == {str(k) for k in plot.K_GRID}
            assert row['tails'][grading]['from_k'] == 256
            assert row['tails'][grading]['to_k'] == 512
    assert record['prefix_successes']['normalized']['512'] == 180
    tex = appendix.render_tex(record)
    assert r'\label{app:gpt56-all-levels-discovery}' in tex
    assert '122,880' in tex and '256 to 512' in tex
    assert 'asymptotic saturation' in tex
    appendix.write(record, tmp_path)
    assert appendix.check(source, tmp_path) == record
    tex_path = tmp_path / (appendix.STEM + '.tex')
    tex_path.write_text(tex_path.read_text().replace('122,880', '122,879'))
    with pytest.raises(ValueError, match='TeX'):
        appendix.check(source, tmp_path)


def test_appendix_rejects_changed_source_or_validation_dependency(tmp_path):
    source = write_source(tmp_path)
    record = appendix.build_record(source)
    appendix.write(record, tmp_path)
    path = tmp_path / (appendix.STEM + '.json')
    changed = deepcopy(record)
    changed['validation_dependency']['sha256'] = 'tampered'
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='validation dependency'):
        appendix.check(source, tmp_path)
    appendix.write(record, tmp_path)
    source.write_text(source.read_text() + '\n')
    with pytest.raises(ValueError, match='source'):
        appendix.check(source, tmp_path)


def synthetic_interim_report():
    from ops import plot_paper_gpt56_all_levels_interim as interim
    report = synthetic_report(draws=128)
    report['schema'] = interim.SCHEMA
    return report


def test_interim_and_final_entry_points_require_their_separate_complete_budgets():
    from ops import plot_paper_gpt56_all_levels_interim as interim
    report = synthetic_interim_report()
    assert interim.validate_report(report) is report
    with pytest.raises(ValueError, match='512-draw analysis'):
        plot.validate_report(report)
    report['schema'] = plot.SCHEMA
    with pytest.raises(ValueError, match='122,880 responses'):
        plot.validate_report(report)
    with pytest.raises(ValueError, match='all 512 draws'):
        plot.validate_cells(report['cells'])
    final = synthetic_report()
    with pytest.raises(ValueError, match='128-draw interim'):
        interim.validate_report(final)
    final['schema'] = interim.SCHEMA
    with pytest.raises(ValueError, match='30,720 responses'):
        interim.validate_report(final)


@pytest.mark.parametrize('mutation', ['partial_cell', 'missing_level', 'tail', 'count', 'protocol'])
def test_interim_rejects_incomplete_or_changed_numeric_evidence(mutation):
    from ops import plot_paper_gpt56_all_levels_interim as interim
    report = synthetic_interim_report()
    if mutation == 'partial_cell':
        report['cells'][0]['max_draws'] = 64
    elif mutation == 'missing_level':
        report['cells'].pop()
    elif mutation == 'tail':
        report['cells'][0]['tail']['distinct']['estimate'] += .1
    elif mutation == 'count':
        report['cells'][0]['prompts'][1]['correct_draws'] += 1
    else:
        report['protocol']['reasoning'] = 'none'
    with pytest.raises(ValueError):
        interim.validate_report(report)


def test_interim_metadata_stays_explicitly_interim_and_binds_shared_renderer(tmp_path):
    from ops import plot_paper_gpt56_all_levels_interim as interim
    source = tmp_path / 'synthetic_interim.json'
    source.write_text(json.dumps(synthetic_interim_report()))
    record = interim.build_record(source)
    assert record['status'] == 'interim_complete'
    assert 'Interim' in record['display']['title']
    assert record['display']['x_range'] == [1, 128]
    assert record['shared_renderer'] == plot.binding(plot.__file__)
    assert record['display']['domain_backgrounds'] == plot.DOMAIN_BACKGROUNDS
    output = tmp_path / 'interim'
    record['outputs'] = {}
    for ext in ('png', 'pdf'):
        output.with_suffix('.' + ext).write_bytes(b'synthetic binding test only')
        record['outputs'][ext] = plot.binding(output.with_suffix('.' + ext))
    output.with_suffix('.json').write_text(json.dumps(record))
    assert interim.check(output, source)['status'] == 'interim_complete'
    record['shared_renderer']['sha256'] = 'changed'
    output.with_suffix('.json').write_text(json.dumps(record))
    with pytest.raises(ValueError, match='rendering helpers'):
        interim.check(output, source)


@pytest.mark.parametrize('extension', ['', '.pdf', '.png'])
def test_interim_renderer_cannot_replace_final_asset(extension):
    pytest.importorskip('matplotlib')
    from ops import plot_paper_gpt56_all_levels_interim as interim
    with pytest.raises(ValueError, match='cannot replace the final'):
        interim.render({}, Path(str(plot.OUTPUT) + extension))
