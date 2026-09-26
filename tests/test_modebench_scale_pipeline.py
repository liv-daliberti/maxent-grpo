from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ops/exp_scaling'))
import materialize_modebench_scale as materialize
import fit_modebench_scale as fit


def example():
    histograms = {'train': Counter({(2,): 384}), 'dev': Counter({(2,): 128}),
                  'eval': Counter({(2,): 128})}
    pools = {tier: [{'problem': f'{tier}-{index}', 'answer_mode_count': 2}
                    for index in range(128)] for tier in range(4)}
    scores = {tier: {fit.sha(row): {'pass1': tier / 10, 'pass8': tier / 5}
                    for row in rows} for tier, rows in pools.items()}
    return pools, scores, histograms


def test_pool_forecasts_fit_known_constant_rates():
    pools, scores, histograms = example()
    result = fit.choose_mixture('countdown', pools, scores, histograms, {'pass1': .1, 'pass8': .2})
    assert result['development_fit_pass']
    assert result['weight_combinations_considered'] == 1771
    assert result['selected_development_sets_scored'] == 1
    assert result['selected_evaluation_sets_scored'] == 0
    assert sum(result['weights']) == 20
    assert all(abs(x) <= .008 for x in result['differences']['dev'].values())


def test_impossible_target_is_failed_not_relaxed():
    pools, scores, histograms = example()
    result = fit.choose_mixture('countdown', pools, scores, histograms, {'pass1': .9, 'pass8': .99})
    assert not result['development_fit_pass']
    assert result['weights'] == [0, 0, 0, 20]
    assert not result['gates']['eval']['pass1']


def test_test_only_and_train_only_cells_are_in_calibration_union():
    h = {'train': Counter({(2,): 381, (7,): 3}), 'dev': Counter({(2,): 128}),
         'eval': Counter({(2,): 125, (9,): 3})}
    assert materialize.union_histogram(h) == Counter({(2,): 128, (7,): 1, (9,): 3})
    pools, scores, _ = example()
    with pytest.raises(ValueError, match='complete registered calibration cells'):
        fit.choose_mixture('countdown', pools, scores, h, {'pass1': .1, 'pass8': .2})


@pytest.mark.parametrize('mutation', ['missing', 'nan', 'pass8_below_pass1', 'extra', 'duplicate'])
def test_invalid_development_evidence_rejected(mutation):
    pools, scores, histograms = example()
    key = fit.sha(pools[0][0])
    if mutation == 'missing':
        del scores[0][key]
    elif mutation == 'nan':
        scores[0][key]['pass1'] = float('nan')
    elif mutation == 'pass8_below_pass1':
        scores[0][key]['pass1'] = .5
    elif mutation == 'extra':
        scores[0]['unknown'] = scores[0][key]
    else:
        pools[1] = deepcopy(pools[0])
    with pytest.raises(ValueError):
        fit.choose_mixture('countdown', pools, scores, histograms, {'pass1': .1, 'pass8': .2})


def test_pool_labels_distinguish_model_and_phase():
    labels = [label for level in materialize.LEVELS for phase in ('dev', 'eval')
              for label in materialize.labels(level, phase)]
    assert len(labels) == len(set(labels)) == 16
    assert not set(labels) & set(range(6329000, 6329004))
    assert materialize.generation_seed('level4', 'mathir', 'train', 0) != materialize.generation_seed('level5', 'mathir', 'train', 0)


def test_pantry_joint_histograms_roundtrip():
    joint = Counter({(3, 'vegan'): 2, (4, 'protein'): 3})
    assert materialize.deserialize_cells(fit.serialize_cells(joint)) == joint


def test_cross_level_exclusion_loads_both_pools_and_frozen_test(tmp_path):
    path = tmp_path / 'history/mathir.json'
    path.parent.mkdir()
    old = ('mathir', 'f', (('a', 1),))
    path.write_text(json.dumps({'identities': [old], 'prompt_sha256': ['old']}))
    def row(a):
        return {'problem': f'eq{a}', 'answer': json.dumps({'family': 'f', 'bindings': {'a': a}}),
                'mathir_family': 'f'}
    materialize.write_jsonl(tmp_path / 'level4/pools/mathir/difficulty_0.jsonl', [row(2)])
    materialize.write_jsonl(tmp_path / 'level5/dataset/mathir/eval.jsonl', [row(3)])
    blocked, prompts = materialize.current_exclusions(tmp_path, 'mathir')
    assert old in blocked
    assert ('mathir', 'f', (('a', 2),)) in blocked
    assert ('mathir', 'f', (('a', 3),)) in blocked
    assert fit.sha('eq2') in prompts and fit.sha('eq3') in prompts


def test_source_pin_mutation_refuses_resume(tmp_path):
    source = tmp_path / 'source.py'
    source.write_text('one')
    path = tmp_path / 'protocol.json'
    path.write_text(json.dumps({'schema': materialize.SCHEMA,
                               'files_sha256': {str(source): fit.file_sha(source)}}))
    materialize.authenticate(path)
    source.write_text('two')
    with pytest.raises(ValueError, match='registered source/input changed'):
        materialize.authenticate(path)
