"""A third level must not let one level's mean rest on another's task mixture."""
from pathlib import Path
import importlib.util
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
for directory in ('ops', 'ops/exp_scaling'):
    if str(ROOT / directory) not in sys.path:
        sys.path.insert(0, str(ROOT / directory))
SOURCE = ROOT / 'ops/exp_scaling/build_paper_modebench_three_level_comparison.py'
SPEC = importlib.util.spec_from_file_location('modebench_three_level_comparison_test', SOURCE)
three = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = three
SPEC.loader.exec_module(three)

STEP = three.TARGET_STEP


def draws(level, domain, method, seed, value):
    return [
        dict(level=level, domain=domain, method=method, seed=seed, step=STEP,
             draw_index=draw, sample_count=8,
             evaluation_kind='fixed_seed_sampled_k_neutral',
             metrics={'any_correct_at_k': value, 'distinct_correct_modes_at_k': value * 2})
        for draw in range(4)
    ]


def cohort(values=None, dropped=()):
    """Every registered cell, with named cells left unadmitted."""
    values = values or {}
    evaluations, admission = [], []
    for level in three.LEVELS:
        for domain in three.DOMAINS:
            for method in three.METHODS:
                for seed in three.SEEDS:
                    key = (level, domain, method, seed)
                    admitted = key not in dropped
                    admission.append({'level': level, 'domain': domain, 'method': method,
                                      'seed': seed, 'admitted': admitted})
                    if admitted:
                        evaluations.extend(draws(*key, values.get(key, values.get(level, .1))))
    return evaluations, admission


def test_a_domain_partial_at_one_level_leaves_the_mean_at_every_level():
    dropped = {('level3', 'mathir', 'maxrl', 45)}
    evaluations, admission = cohort(
        values={'level1': .1, 'level2': .2, 'level3': .3,
                ('level1', 'mathir', 'drgrpo', 43): .9},
        dropped=dropped)
    result = three.build_terminal_comparison(evaluations, admission)
    assert result['partial_domains'] == ['mathir']
    assert 'mathir' not in result['complete_domains']
    # MathIR's one raised Level-1 seed cannot reach any level's mean.
    assert result['means']['level1']['drgrpo']['pass8'] == pytest.approx(.1)
    assert result['admitted_terminal_cells_by_level'] == {
        'level1': 100, 'level2': 100, 'level3': 99}


def test_domains_are_weighted_equally_and_levels_kept_apart():
    evaluations, admission = cohort(values={'level1': .1, 'level2': .2, 'level3': .3})
    result = three.build_terminal_comparison(evaluations, admission)
    assert result['complete_domains'] == list(three.DOMAINS)
    for level, value in (('level1', .1), ('level2', .2), ('level3', .3)):
        assert result['means'][level]['replay_maxrl']['pass8'] == pytest.approx(value)
        assert result['means'][level]['replay_maxrl']['distinct8'] == pytest.approx(value * 2)


def test_admission_must_enumerate_all_three_hundred_cells():
    evaluations, admission = cohort()
    with pytest.raises(RuntimeError, match='300 registered cells'):
        three.build_terminal_comparison(evaluations, admission[:-1])


def test_draws_must_match_the_declared_admission():
    evaluations, admission = cohort(dropped={('level3', 'countdown', 'maxrl', 47)})
    admission[-1]['admitted'] = not admission[-1]['admitted']
    with pytest.raises(RuntimeError, match='disagree'):
        three.build_terminal_comparison(evaluations, admission)


def training_arms(reportable=()):
    return [{'level': level, 'scale': 'qwen05b', 'domain': domain, 'method': method,
             'terminal_reportable': (level, domain) in reportable,
             'pmd_after': .2 if level == 'level1' else .4}
            for level in ('level1', 'level2') for domain in three.DOMAINS
            for method in three.METHODS]


def pmd_cells(reportable_domains, pmd=.6):
    return [{'level': 'level3', 'domain': domain, 'method': method, 'seed': seed,
             'pmd': pmd, 'reportable': domain in reportable_domains}
            for domain in three.DOMAINS for method in three.METHODS
            for seed in three.SEEDS]


def test_pmd_matching_needs_support_at_every_level():
    supported = {('level1', 'graph_coloring'), ('level2', 'graph_coloring'),
                 ('level1', 'mathir'), ('level2', 'mathir')}
    comparison = three.build_pmd_comparison(
        training_arms(supported), pmd_cells({'graph_coloring'}), 30)
    # MathIR clears the bar at Levels 1 and 2 but not at Level 3, so it leaves
    # the matched set rather than giving one level a narrower task mixture.
    assert comparison['matched_domains'] == ['graph_coloring']
    assert comparison['means']['level3']['drgrpo'] == pytest.approx(.6)
    assert comparison['means']['level1']['drgrpo'] == pytest.approx(.2)


def test_an_arm_with_no_supported_seed_is_not_reportable():
    cells = [cell for cell in pmd_cells({'graph_coloring'})
             if not (cell['domain'] == 'graph_coloring' and cell['method'] == 'maxrl')]
    cells += [{'level': 'level3', 'domain': 'graph_coloring', 'method': 'maxrl',
               'seed': seed, 'pmd': None, 'reportable': False} for seed in three.SEEDS]
    supported = {(level, 'graph_coloring') for level in ('level1', 'level2')}
    with pytest.raises(RuntimeError, match='no domain supports PCMD'):
        three.build_pmd_comparison(training_arms(supported), cells, 30)


def baseline_rows(pass8=.3, distinct8=.6, dropped=()):
    return [{'level': level, 'domain': domain, 'method': method, 'seed': seed,
             'admitted': (level, domain, method, seed) not in dropped,
             'endpoint': (None if (level, domain, method, seed) in dropped
                          else {'pass8': pass8, 'distinct8': distinct8})}
            for level in three.LEVELS for domain in three.DOMAINS
            for method in three.METHODS for seed in three.SEEDS]


def baseline_pmd(reportable_domains, pmd=.25):
    return [{'level': level, 'domain': domain, 'method': method, 'seed': seed,
             'pmd': pmd, 'reportable': (level, domain) in reportable_domains}
            for level in three.LEVELS for domain in three.DOMAINS
            for method in three.METHODS for seed in three.SEEDS]


def test_untrained_point_uses_the_domain_bases_of_the_marks_beside_it():
    supported = {(level, domain) for level in three.LEVELS
                 for domain in ('graph_coloring', 'mathir')}
    record = three.build_baseline_points(
        baseline_rows(), baseline_pmd(supported),
        ['graph_coloring', 'countdown'], ['graph_coloring', 'mathir'])
    point = record['points']['level2']
    assert point['pass8'] == pytest.approx(.3)
    assert point['pmd'] == pytest.approx(.25)
    # Only the two accuracy domains and their 4x5 cells, not all five domains.
    assert point['cells'] == 40 and point['breadth_cells'] == 40


def test_a_level_without_support_in_every_matched_domain_has_no_breadth_point():
    supported = {(level, 'graph_coloring') for level in three.LEVELS}
    supported |= {(level, 'mathir') for level in ('level1', 'level3')}
    record = three.build_baseline_points(
        baseline_rows(), baseline_pmd(supported),
        ['graph_coloring'], ['graph_coloring', 'mathir'])
    assert record['points']['level2']['pmd'] is None
    assert record['points']['level2']['breadth_domains'] == ['graph_coloring']
    # The missing breadth point never removes the accuracy point beside it.
    assert record['points']['level2']['pass8'] == pytest.approx(.3)
    assert record['points']['level1']['pmd'] == pytest.approx(.25)


def test_an_unmeasured_accuracy_domain_withdraws_that_level_s_untrained_point():
    dropped = {('level3', 'countdown', 'maxrl', seed) for seed in three.SEEDS}
    supported = {(level, 'graph_coloring') for level in three.LEVELS}
    record = three.build_baseline_points(
        baseline_rows(dropped=dropped), baseline_pmd(supported),
        ['graph_coloring', 'countdown'], ['graph_coloring'])
    # Countdown keeps its other arms, so the level still averages both domains.
    assert record['points']['level3']['pass8'] == pytest.approx(.3)
    assert record['points']['level3']['cells'] == 35


def gain_inputs():
    terminal = {'complete_domains': ['graph_coloring'],
                'means': {level: {method: {'pass8': value, 'distinct8': value * 2}
                                  for method, value in (('drgrpo', .3), ('replay_drgrpo', .5),
                                                        ('maxrl', .4), ('replay_maxrl', .45))}
                          for level in three.LEVELS}}
    pmd = {'matched_domains': ['graph_coloring'],
           'means': {level: {'drgrpo': .05, 'replay_drgrpo': .25,
                             'maxrl': .10, 'replay_maxrl': .30}
                     for level in three.LEVELS}}
    return terminal, pmd


def test_gains_difference_each_replay_arm_against_its_own_control():
    record = three.build_replay_gains(*gain_inputs())
    pair = record['levels']['level2']['replay_drgrpo']
    assert pair['control'] == 'drgrpo'
    assert pair['pass8'] == pytest.approx(.2)
    assert pair['distinct8'] == pytest.approx(.4)
    assert pair['pmd'] == pytest.approx(.2)
    # Re:Max is read against MaxRL, never against the other family's control.
    assert record['levels']['level2']['replay_maxrl']['pass8'] == pytest.approx(.05)
    assert record['levels']['level2']['replay_maxrl']['pmd'] == pytest.approx(.2)
    assert record['breadth_domains'] == ['graph_coloring']
