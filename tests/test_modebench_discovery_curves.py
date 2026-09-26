"""Independent discovery-curve mathematics and frozen-receipt integrity tests."""
import copy
import importlib.util
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('discovery_analysis',Path(__file__).resolve().parents[1]/'ops/analyze_modebench_discovery_curves.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def row(index=0):
    return {'level':2,'domain':'python_factors','row_index':index,'problem':'same mathematical task'}


def reference(count=2,kind='certified_lower_bound'):
    return {'support_kind':kind,'support_count':count}


def samples(keys):
    assert len(keys)==64
    return [{'level':2,'domain':'python_factors','row_index':0,'sample_index':i,'verified':key is not None,
             'canonical_key':key,'text':str(key),'stop_reason':'length' if key is None else 'stop'} for i,key in enumerate(keys)]


def stats(keys,index=0,count=2,kind='certified_lower_bound'):
    return m.prompt_statistics(row(index),samples(keys),reference(count,kind))


@pytest.mark.parametrize('keys',[[None,None,None,None],['a','a','b',None],['a','b','c','d'],['a']*4])
def test_rarefaction_equals_exhaustive_uniform_subsets(keys):
    from collections import Counter
    counts=list(Counter(k for k in keys if k is not None).values());c=sum(counts)
    for k in range(5):
        subsets=list(combinations(range(4),k))
        brute_d=np.mean([len({keys[i] for i in subset if keys[i] is not None}) for subset in subsets])
        brute_p=np.mean([any(keys[i] is not None for i in subset) for subset in subsets])
        assert m.rarefied_distinct(counts,4,k)==pytest.approx(brute_d)
        assert 1-m.absence_probability(4,c,k)==pytest.approx(brute_p)


def test_all_fail_preserves_64_slots_and_undefined_conditionals():
    r=stats([None]*64)
    assert r['failed_draws']==r['truncated_draws']==64
    assert all(x['pass']==x['distinct']==x['breadth']==x['uniform_distinct']==0 for x in r['rarefaction'].values())
    assert all(x['distinct'] is None and not x['eligible'] for x in r['conditional'].values())
    a=m.paired_matrices([r],[r]);v=m.rates(a['original'].sum(axis=0))
    assert np.isnan(v[m.METRICS.index('collision_own/observed')])
    assert np.isnan(v[m.METRICS.index('conditional_joint/distinct/m1')])
    assert v[m.METRICS.index('rarefaction/distinct/k64')]==0


def test_all_one_mode_and_all_unique_boundaries():
    one=stats(['a']*64);unique=stats(list(range(64)))
    for k in m.GRID:
        assert one['rarefaction'][str(k)]['pass']==one['rarefaction'][str(k)]['distinct']==1
        assert one['rarefaction'][str(k)]['breadth']==0
        assert unique['rarefaction'][str(k)]['distinct']==k
        assert unique['conditional'][str(k)]['distinct']==k
    assert one['colliding_correct_pairs']==2016 and unique['colliding_correct_pairs']==0
    # A lower certificate of two modes does not cap observed support.
    assert unique['rarefaction']['64']['distinct']==64
    with pytest.raises(ValueError,match='exceed'):
        stats(list(range(64)),count=2,kind='exact')


def test_rarefaction_permutation_invariant_but_prefix_is_preassigned_index_order():
    first=stats(['a']*8+[None]*56)
    last=stats([None]*56+['a']*8)
    assert first['rarefaction']==last['rarefaction']
    assert first['prefix']['8']['pass']==1 and last['prefix']['8']['pass']==0
    shuffled=list(reversed(samples(['a']*8+[None]*56)))
    assert m.prompt_statistics(row(),shuffled,reference())['prefix']==first['prefix']


def test_full_pool_endpoints_and_k1_use_empirical_correctness():
    r=stats(['a']*3+['b']*2+[None]*59)
    assert r['rarefaction']['1']==pytest.approx({'pass':5/64,'distinct':5/64,'breadth':0,
                                               'uniform_distinct':5/64,'distinct_minus_uniform':0})
    assert r['rarefaction']['64']['pass']==1 and r['rarefaction']['64']['distinct']==2
    assert r['rarefaction']['8']['pass']!=pytest.approx(1-(1-5/64)**8)


def test_fixed_correct_budget_matches_brute_force_and_fails_eligibility_cleanly():
    r=stats(['a']*3+['b']+[None]*60)
    assert r['conditional']['2']['distinct']==pytest.approx(1.5)
    assert r['conditional']['4']['distinct']==2
    assert not r['conditional']['8']['eligible'] and r['conditional']['8']['distinct'] is None


@pytest.mark.parametrize('correct',range(5))
def test_correctness_matched_uniform_equals_enumerating_success_subsets(correct):
    keys=[True]*correct+[False]*(4-correct)
    for k in range(5):
        truth=np.mean([m.uniform_distinct(sum(keys[i] for i in subset),3) for subset in combinations(range(4),k)])
        assert m.correctness_matched_uniform(4,correct,k,3)==pytest.approx(truth)


def test_uniform_support_bounds_and_numerical_large_support():
    for correct in [0,1,2,8,64]:
        assert m.uniform_distinct(correct,2)<=m.uniform_distinct(correct,5)<=m.uniform_distinct(correct,10000)+1e-12
    assert m.uniform_distinct(64,10**400)==64
    assert 1/2>=1/5>=1/10000
    assert m.uniform_distinct(0,1)==0 and m.uniform_distinct(64,1)==1
    with pytest.raises(ValueError,match='Uncertified'):
        m.validate_reference(reference(kind='unknown'))


def test_own_and_joint_conditional_populations_are_distinct():
    original=[stats(['a']*8+[None]*56,0),stats(['a','b']+[None]*62,1)]
    neutral=[stats(['a']+[None]*63,0),stats(['a','b']+[None]*62,1)]
    matrices=m.paired_matrices(original,neutral)
    a=m.rates(matrices['original'].sum(axis=0));b=m.rates(matrices['neutral'].sum(axis=0))
    own=m.METRICS.index('conditional_own/distinct/m2');joint=m.METRICS.index('conditional_joint/distinct/m2')
    assert a[own]==1.5 and b[own]==2
    assert a[joint]==b[joint]==2
    counts=m.pair_counts(original,neutral)
    assert counts['original']['conditional_own_eligible']['2']==2
    assert counts['original']['conditional_joint_eligible']['2']==1


def test_correct_pair_collision_uses_pair_weights_not_prompt_means():
    rs=[stats(['a']*8+[None]*56,0,count=4),stats(['a','b']+[None]*62,1,count=8)]
    matrix=m.paired_matrices(rs,rs)['original'];point=m.rates(matrix.sum(axis=0))
    assert point[m.METRICS.index('collision_own/observed')]==28/29
    assert point[m.METRICS.index('collision_own/uniform_reference')]==pytest.approx((28/4+1/8)/29)


def test_shared_prompt_bootstrap_preserves_paired_gains_and_zero_variation():
    a=[stats(['a']*8+[None]*56,0),stats(['a','b']*16+[None]*32,1)]
    indices=np.asarray([[0,0],[0,1],[1,1]])
    points,boots=m.bootstrap_pair(m.paired_matrices(a,a),indices)
    assert np.all(boots[m.CONTRAST][np.isfinite(boots[m.CONTRAST])]==0)
    summary=m.summarize_pair(points,boots)
    assert summary[m.CONTRAST]['gain8to64/rarefaction/distinct']['ci95']==[0.,0.]
    assert summary[m.CONTRAST]['gain8to64/rarefaction/distinct']['degenerate_ci']
    assert points['original'][m.METRICS.index('gain8to64/rarefaction/distinct')]==pytest.approx(
        np.mean([r['rarefaction']['64']['distinct']-r['rarefaction']['8']['distinct'] for r in a]))


def test_seed_resampling_retains_same_prompt_replicate_and_undefined_seeds():
    points=[{a:np.asarray([x]) for a in (*m.ARMS,m.CONTRAST)} for x in [1.,3.]]
    boots=[{a:np.asarray([[x],[x+10]]) for a in (*m.ARMS,m.CONTRAST)} for x in [1.,3.]]
    p,b=m.seed_mean(points,boots,np.asarray([[0,0],[1,1]]))
    assert p['original'][0]==2 and b['original'][:,0].tolist()==[1.,13.]
    points[0]['original'][0]=np.nan
    p,b=m.seed_mean(points,boots)
    assert np.isnan(p['original'][0])


def test_missing_or_repeated_draw_cannot_enter_curve():
    with pytest.raises(ValueError,match='all 64'):
        m.prompt_statistics(row(),samples([None]*64)[:-1],reference())
    bad=samples([None]*64);bad[-1]['sample_index']=0
    with pytest.raises(ValueError,match='all 64'):
        m.prompt_statistics(row(),bad,reference())
    bad=samples([None]*64);bad[0]['verified']=True
    with pytest.raises(ValueError,match='consistent'):
        m.prompt_statistics(row(),bad,reference())


def test_fresh_seed_schedule_is_disjoint_and_checks_actual_child_slot():
    values=[m.expected_local_seed(2,'python_factors',r,i)[1] for r in range(128) for i in range(64)]
    assert len(set(values))==128*64 and min(values)>79000000
    request,child=m.expected_local_seed(3,'mathir',3,9)
    assert request==911640000+128*(10000+1000+3)+8 and child==request+1
    ck={'label':'checkpoint','training_method':'drgrpo','training_seed':43,'trained_on_level':2}
    prompt={'messages_sha256':'prompt','pair_id':'pair'}
    s={'checkpoint_label':'checkpoint','training_method':'drgrpo','training_seed':43,'trained_on_level':2,
       'draw_index':0,'token_count':3,'text':'synthetic',
       'row_sha256':m.sha(row()),'messages_sha256':'prompt','pair_id':'pair','draw_block':0,'block_draw_index':0,
       'sampling_seed':m.expected_local_seed(2,'python_factors',0,0)[0],
       'child_sampling_seed':m.expected_local_seed(2,'python_factors',0,0)[1], 'verified':False,'canonical_key':None}
    m.validate_local_draw(s,('original',2,'python_factors',0,0),row(),prompt,ck)
    bad=copy.deepcopy(s);bad['child_sampling_seed']+=1
    with pytest.raises(ValueError,match='child-seed'):
        m.validate_local_draw(bad,('original',2,'python_factors',0,0),row(),prompt,ck)


def cached_grade_fixture(tmp_path):
    import json
    directory=tmp_path;chunkdir=directory/'discovery_grading_chunks';chunkdir.mkdir()
    raw={('original',2,'python_factors',0,0):{'arm':'original','level':2,'domain':'python_factors','row_index':0,'sample_index':0,
          'text':'invalid','graded_text':'invalid','verified':False,'canonical_key':None}}
    sample=next(iter(raw.values()))
    primary={k:sample[k] for k in ('verified','canonical_key','graded_text')}
    entry={k:sample[k] for k in ('arm','level','domain','row_index','sample_index')}
    entry.update(raw_sample_sha256=m.sha(sample),strict=primary,normalization={**primary,'original_text':'invalid'})
    for name,content in [('result.json','{}'),('responses.jsonl',json.dumps(sample)+'\n'),('source.py','# frozen test source\n'),
                         ('discovery_grades.jsonl',json.dumps(entry)+'\n')]: (directory/name).write_text(content)
    design={'normalizer_sha256':'a'*64,'contract_sha256':'b'*64}
    identity={'result_sha256':m.file_sha(directory/'result.json'),'responses_sha256':m.file_sha(directory/'responses.jsonl'),
              'normalizer_sha256':design['normalizer_sha256'],'contract_sha256':design['contract_sha256'],
              'analyzer_sha256':m.file_sha(directory/'source.py')}
    chunk={'identity':identity,'start':0,'entries':[entry],'entries_sha256':m.sha([entry]),
           'python_rechecked_serially':1,'python_synthetic_warmup':[True]}
    m.write_json(chunkdir/'part_000000.json',chunk)
    audit={'status':'complete','api_calls':0,'records':1,**identity,'cache_sha256':m.file_sha(directory/'discovery_grades.jsonl'),
           'analyzer_source':m.binding(directory/'source.py'),'chunks':[m.binding(chunkdir/'part_000000.json')],
           'python_rechecked_serially':1,'python_synthetic_warmup':[True],'raw_verified':0,'strict_verified':0,
           'normalized_verified':0,'strict_changed_records':0}
    m.write_json(directory/'discovery_grading_audit.json',audit)
    return design,directory,raw,audit


def test_cached_python_grades_require_successful_warmup_and_exact_totals(tmp_path):
    design,directory,raw,audit=cached_grade_fixture(tmp_path)
    strict,normalized=m.authenticate_grades(design,directory,raw)
    assert not next(iter(strict.values()))['verified']
    bad=copy.deepcopy(audit);bad['python_synthetic_warmup']=[False];m.write_json(directory/'discovery_grading_audit.json',bad)
    with pytest.raises(ValueError,match='not warmed'):
        m.authenticate_grades(design,directory,raw)
    bad=copy.deepcopy(audit);bad['raw_verified']=1;m.write_json(directory/'discovery_grading_audit.json',bad)
    with pytest.raises(ValueError,match='total differs'):
        m.authenticate_grades(design,directory,raw)


def test_resumed_chunk_cannot_change_raw_cohort_even_if_its_hash_is_updated(tmp_path):
    import json
    design,directory,raw,audit=cached_grade_fixture(tmp_path)
    path=directory/'discovery_grading_chunks/part_000000.json';chunk=json.loads(path.read_text())
    chunk['identity']['responses_sha256']='c'*64;m.write_json(path,chunk)
    audit['chunks']=[m.binding(path)];m.write_json(directory/'discovery_grading_audit.json',audit)
    with pytest.raises(ValueError,match='chunk identity'):
        m.authenticate_grades(design,directory,raw)


def test_frozen_grading_rejects_preloaded_live_grader(monkeypatch,tmp_path):
    import sys
    monkeypatch.setitem(sys.modules,'oat_drgrpo_live',object())
    with pytest.raises(ValueError,match='isolated process'):
        m.frozen_grade_modules(tmp_path)


def test_full_report_fails_before_analysis_when_registered_draws_are_missing(monkeypatch,tmp_path):
    monkeypatch.setattr(m,'authenticate_design',lambda base:{'base':tmp_path})
    monkeypatch.setattr(m,'inventory_report',lambda *args:{'runs':[{'graded':True}],'status':'incomplete'})
    with pytest.raises(ValueError,match='must be complete'):
        m.build_report(tmp_path)
    with pytest.raises(ValueError,match='20,000'):
        m.build_report(tmp_path,replicates=100)


def test_equal_seed_collision_and_pooled_count_summaries_are_distinguished():
    a=stats(['a']*8+[None]*56);b=stats(['a','b']+[None]*62)
    count_a=m.pair_counts([a],[a]);count_b=m.pair_counts([b],[b])
    pooled=m.sum_seed_counts([count_a,count_b])
    assert pooled['original']['pooled_collision_across_checkpoints']==28/29
    assert 'separate from primary equal-seed mean' in pooled['interpretation']
    assert np.mean([1.,0.])!=pooled['original']['pooled_collision_across_checkpoints']


def test_published_curve_and_gain_descriptions_keep_bounded_reference_and_prefix_semantics():
    report={'scope':{'included_panels':[],'omitted_panels':['frontier','local']},'experiment_status':'partial_panels'}
    text=m.render_appendix(report)
    assert 'preassigned draw index' in text and 'rather than response arrival time' in text
    assert 'not exhaustive support sizes' in text and 'upper reference' in text and 'lower reference' in text
    assert 'Observed distinct counts may exceed' in text and 'partial-panel' in text
