"""Require the full expanded cohort before producing publication assets."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest
from ops import plot_paper_gpt56_all_levels32_sampling as plot
from ops import build_paper_gpt56_all_levels32_discovery as appendix


def expanded_fixture():
    path = Path(__file__).with_name('test_paper_gpt56_all_levels_sampling.py')
    spec = importlib.util.spec_from_file_location('_old16_publication_fixture', path)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    report = old.synthetic_report()
    report.update(prompts=480, responses=245760)
    for grading in ('cells', 'strict_cells'):
        for cell in report[grading]:
            additional = deepcopy(cell['prompts'])
            for prompt in additional:
                prompt['row_index'] += 16
                prompt['row_sha256'] = f"{prompt['row_index']:064x}"
            cell['prompts'].extend(additional)
            cell['n_prompts'] = 32
    return report


def test_full_32_problem_cohort_reconstructs_under_both_grading_conventions():
    report = expanded_fixture()
    assert plot.validate_report(report) is report
    assert sum(len(c['prompts']) for c in report['cells']) == 480


@pytest.mark.parametrize('mutation', ['old16', 'duplicate', 'missing_draw', 'wrong_weight', 'wrong_budget'])
def test_incomplete_or_misweighted_expansion_is_rejected(mutation):
    report = expanded_fixture()
    cell = report['cells'][0]
    if mutation == 'old16':
        cell['prompts'] = cell['prompts'][:16]
        cell['n_prompts'] = 16
    elif mutation == 'duplicate':
        cell['prompts'][-1] = deepcopy(cell['prompts'][0])
    elif mutation == 'missing_draw':
        cell['prompts'][-1]['responses'] = 511
    elif mutation == 'wrong_weight':
        cell['points'][-1]['distinct']['estimate'] *= 2
    else:
        cell['max_draws'] = 1024
    with pytest.raises(ValueError):
        plot.validate_report(report)


def test_expanded_publication_binds_480_problem_source_and_preserves_extension_history(tmp_path):
    path = tmp_path / 'synthetic_complete.json'
    path.write_text(json.dumps(expanded_fixture()))
    record = appendix.build_record(path)
    tex = appendix.render_tex(record)
    assert record['prompts'] == 480 and record['responses'] == 245760
    assert 'thirty-two' in tex and '245,760' in tex
    assert 'retains all 240 original problems' in tex
    assert '122,880 new responses' in tex
    assert 'eight-draw evaluations of those problems are excluded' in tex
    assert record['source'] == plot.binding(path)
