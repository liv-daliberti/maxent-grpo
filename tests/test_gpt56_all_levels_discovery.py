"""Guard the new level inventory, 512-draw pools, and cross-stage identities."""
from copy import deepcopy
from pathlib import Path

import pytest
from ops import analyze_gpt56_all_levels_discovery as analysis


def stage(name, start, stop, *, message='same prompt', provider_prefix=None, level=1):
    key = (level, 'graph_coloring', 7)
    ids = {(provider_prefix or name, i) for i in range(start, stop)}
    pool = {'level': level, 'domain': key[1], 'row_index': 7,
            'row_sha256': 'same-row', 'messages_sha256': message,
            'strict': {i: 'a' for i in range(start, stop)},
            'normalization': {i: 'a' for i in range(start, stop)}}
    source = {'directory': str(Path('/tmp') / name), 'request_controls_sha256': 'same-controls',
              'served_models': ['gpt-5.6-sol'], 'served_snapshots': ['gpt-5.6-sol-2026-07-09'],
              'responses': stop-start}
    return source, {key: pool}, ids


def test_merge_preserves_all_512_slots_and_each_stage_provider_identity():
    earlier, extension = stage('earlier', 0, 64), stage('extension', 64, 512)
    _, pools = analysis.merge_authenticated([earlier, extension])
    assert set(next(iter(pools.values()))['strict']) == set(range(512))
    assert set(next(iter(earlier[1].values()))['strict']) == set(range(64))


@pytest.mark.parametrize('mutation,match', [
    ('provider', 'Repeated provider'),
    ('slots', 'Overlapping global'),
    ('prompt', 'changed problem or prompt'),
    ('controls', 'Deployment controls'),
    ('snapshot', 'model snapshots'),
    ('directory', 'Duplicate source run'),
])
def test_merge_rejects_invalid_cross_stage_evidence(mutation, match):
    earlier, extension = stage('earlier', 0, 64), stage('extension', 64, 512)
    if mutation == 'provider':
        extension[2].pop()
        extension[2].add(next(iter(earlier[2])))
    elif mutation == 'slots':
        next(iter(extension[1].values()))['strict'][0] = 'a'
    elif mutation == 'prompt':
        next(iter(extension[1].values()))['messages_sha256'] = 'changed'
    elif mutation == 'controls':
        extension[0]['request_controls_sha256'] = 'different'
    elif mutation == 'snapshot':
        extension[0]['served_snapshots'] = ['different-model-version']
    elif mutation == 'directory':
        extension[0]['directory'] = earlier[0]['directory']
    with pytest.raises(ValueError, match=match):
        analysis.merge_authenticated([earlier, extension])


def full_inventory():
    pools, support = {}, {}
    for level in (1, 2, 3):
        for domain in analysis.DOMAINS:
            for row in range(16):
                key = (level, domain, row)
                pools[key] = {'level': level, 'domain': domain, 'row_index': row,
                              'row_sha256': str(key), 'messages_sha256': 'same prompt',
                              'strict': dict.fromkeys(range(512), None),
                              'normalization': dict.fromkeys(range(512), None)}
                support[key] = {'row_sha256': str(key), 'support_count': 2}
    return pools, support


def test_complete_240_problem_inventory_includes_all_level1_failures():
    pools, support = full_inventory()
    analysis.validate_complete_pools(pools, support)
    assert sum(len(p['strict']) for p in pools.values()) == 122880


@pytest.mark.parametrize('mutation,match', [
    ('missing_level1', '240-problem'),
    ('missing_final_draw', 'all 512'),
    ('support_revision', 'different problem revision'),
    ('unequal_cell_size', 'sixteen fixed problems'),
])
def test_incomplete_all_levels_output_cannot_be_published(mutation, match):
    pools, support = full_inventory()
    key = next(iter(pools))
    if mutation == 'missing_level1':
        pools = {k:v for k,v in pools.items() if k[0] != 1}
    elif mutation == 'missing_final_draw':
        pools[key]['normalization'].pop(511)
    elif mutation == 'support_revision':
        support[key]['row_sha256'] = 'different'
    elif mutation == 'unequal_cell_size':
        replacement = (2, key[1], 1000)
        pools[replacement] = pools.pop(key)
        support[replacement] = support.pop(key)
    with pytest.raises(ValueError, match=match):
        analysis.validate_complete_pools(pools, support)


def test_512_rarefaction_retains_singletons_and_failure_denominators():
    half = analysis.core.rarefaction([511, 1], 512, 256)
    full = analysis.core.rarefaction([511, 1], 512, 512)
    assert half['distinct'] == 1.5
    assert full['distinct'] == 2
    assert full['distinct'] - half['distinct'] == .5
    failed = analysis.core.rarefaction([1], 512, 256)
    assert failed == {'distinct': .5, 'pass': .5, 'breadth': 0.0}


def test_verification_cache_is_exact_and_still_binds_each_response():
    rows={(1,'mathir',i):{'level':1,'domain':'mathir','row_index':i} for i in (0,1)}
    samples=[]
    for i,(row,text) in enumerate([(0,'same'),(0,'same'),(1,'same'),(0,'same ')]):
        samples.append({'level':1,'domain':'mathir','row_index':row,'sample_index':i,
                        'sample_id':str(i),'text':text,'verified':True,'canonical_key':'mode'})
    calls=[]
    def grader(level,domain,row,text):
        calls.append((row['row_index'],text))
        return {'verified':True,'canonical_key':'mode'}
    def normalize(row,text,strict_grade,grader):
        return dict(strict_grade)
    grades,unique=analysis.grade_samples(rows,samples,grader,normalize)
    assert unique == len(calls) == 3
    assert len(grades) == 4
    assert len({g['raw_sample_sha256'] for g in grades}) == 4
    samples[1]['verified']=False
    samples[1]['canonical_key']=None
    with pytest.raises(ValueError,match='differs from retained strict grade'):
        analysis.grade_samples(rows,samples,grader,normalize)
