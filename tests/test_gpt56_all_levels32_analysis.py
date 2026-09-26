"""Guard32-row uncertainty, complete512 pools, and retained-cohort equivalence."""
from copy import deepcopy

import pytest
from ops import analyze_gpt56_all_levels32_discovery as analysis


def prompts(n):
    return [{'level': 1, 'domain': 'graph_coloring', 'row_index': i, 'row_sha256': str(i),
             'keys': ([None] * 512 if i < n // 2 else ['mode'] * 512),
             'support': {'support_count': i + 1, 'support_kind': 'exact'}} for i in range(n)]


def test_sixteen_problem_results_are_byte_for_byte_equivalent_to_frozen_numerics():
    rows = prompts(16)
    assert analysis.summarize_cell(rows, 1, 'graph_coloring', expected_prompts=16) == analysis.core.summarize_cell(rows, 1, 'graph_coloring')


def test_bootstrap_resamples_all32_whole_problems_with_equal_weight():
    rows = prompts(32)
    report = analysis.summarize_cell(rows, 1, 'graph_coloring')
    assert report['n_prompts'] == 32
    assert all(p['distinct']['estimate'] == .5 for p in report['points'])
    assert report['support']['mean'] == 16.5
    rng = analysis.core.np.random.default_rng(report['bootstrap']['seed'])
    indices = rng.integers(0, 32, size=(20000, 32))
    expected = analysis.core.estimate([0] * 16 + [1] * 16, indices)
    assert report['points'][-1]['distinct'] == expected
    assert expected['ci95'][0] > 0 and expected['ci95'][1] < 1
    assert report['tail']['distinct']['estimate'] == 0
    assert report['bootstrap']['replicates'] == 20000


def full_inventory():
    pools, support = {}, {}
    for level in analysis.LEVELS:
        for domain in analysis.DOMAINS:
            for i in range(32):
                key = (level, domain, i)
                pools[key] = {'level': level, 'domain': domain, 'row_index': i, 'row_sha256': str(key),
                              'strict': dict.fromkeys(range(512)), 'normalization': dict.fromkeys(range(512))}
                support[key] = {'row_sha256': str(key)}
    return pools, support


def test_complete_inventory_retains_every_failure_and_exact_budget():
    pools, support = full_inventory()
    analysis.validate_complete_pools(pools, support)
    assert sum(len(p['strict']) for p in pools.values()) == 245760


@pytest.mark.parametrize('mutation,match', [
    ('missing_cell', '480-problem'), ('missing_draw', 'all 512'),
    ('extra_draw', 'all 512'), ('support_revision', 'different problem revision'),
    ('unequal_cells', 'exactly 32'), ('identity', 'inconsistent problem identity')])
def test_partial_or_misidentified_expansion_is_rejected(mutation, match):
    pools, support = full_inventory()
    key = next(iter(pools))
    if mutation == 'missing_cell':
        pools = {k: p for k, p in pools.items() if k[:2] != key[:2]}
    elif mutation == 'missing_draw':
        pools[key]['normalization'].pop(511)
    elif mutation == 'extra_draw':
        pools[key]['strict'][512] = None
    elif mutation == 'support_revision':
        support[key]['row_sha256'] = 'changed'
    elif mutation == 'unequal_cells':
        target = (2, key[1], 1000)
        pools[target] = {**pools.pop(key), 'level': 2, 'row_index': 1000}
        support[target] = support.pop(key)
    elif mutation == 'identity':
        pools[key]['row_index'] = 1000
    with pytest.raises(ValueError, match=match):
        analysis.validate_complete_pools(pools, support)


def test_grade_execution_cache_does_not_merge_rows_texts_or_response_evidence():
    rows = {(1, 'mathir', i): {'level': 1, 'domain': 'mathir', 'row_index': i} for i in (0, 1)}
    inputs = [(0, 'same'), (0, 'same'), (1, 'same'), (0, 'same ')]
    samples = [{**rows[1, 'mathir', row], 'sample_index': i, 'sample_id': str(i),
                'text': text, 'verified': True, 'canonical_key': 'mode'} for i, (row, text) in enumerate(inputs)]
    calls = []
    def grader(level, domain, row, text):
        calls.append((row['row_index'], text))
        return {'verified': True, 'canonical_key': 'mode'}
    def normalizer(row, text, strict_grade, grader):
        strict_grade['normalization_only'] = True
        return strict_grade
    grades, unique = analysis.grade_samples(rows, samples, grader, normalizer)
    assert unique == len(calls) == 3 and len(grades) == 4
    assert len({g['raw_sample_sha256'] for g in grades}) == 4
    assert all('normalization_only' not in g['strict'] for g in grades)
    samples[1]['verified'] = False
    with pytest.raises(ValueError, match='retained strict grade'):
        analysis.grade_samples(rows, samples, grader, normalizer)


def test_full_synthetic_analysis_is_accepted_by_all15_cell_publication_validator():
    from ops import plot_paper_gpt56_all_levels32_sampling as plot
    cells = []
    for level in analysis.LEVELS:
        for domain in analysis.DOMAINS:
            rows = prompts(32)
            for row in rows:
                row.update(level=level, domain=domain, row_sha256='0' * 64)
                row['support']['support_kind'] = 'exact' if domain == 'graph_coloring' else 'certified_lower_bound'
            cells.append(analysis.summarize_cell(rows, level, domain))
    report = {'schema': 'gpt56-all-levels-discovery-v1', 'status': 'complete',
              'model': 'gpt-5.6-sol', 'grading': 'normalized_secondary', 'prompt_arm': 'original',
              'levels': [1, 2, 3], 'prompts': 480, 'responses': 245760,
              'cells': cells, 'strict_cells': deepcopy(cells),
              'protocol': {'reasoning': 'medium', 'max_output_tokens': 8192, 'temperature_and_top_p': 'omitted',
                           'bootstrap': {'replicates': 20000, 'seed': 20260911, 'unit': 'whole prompt', 'pointwise': True}}}
    assert plot.validate_report(report) is report
