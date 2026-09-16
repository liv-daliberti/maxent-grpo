"""Independent prompt/seed inference and failure-retention regression checks."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('prompt_ablation_analysis', Path(__file__).resolve().parents[1]/
                                             'ops/analyze_modebench_prompt_ablation.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def row(index=0,support=8):
    return {'level':2,'domain':'python_factors','row_index':index,'problem':f'Task {index}',
            'metadata':{'answer_mode_count':support}}


def samples(keys):
    return [{'sample_index':i,'verified':key is not None,'canonical_key':key,'text':str(key),
             'stop_reason':'stop' if key is not None else 'length',
             'response_status':'completed' if key is not None else 'incomplete'} for i,key in enumerate(keys)]


def test_empirical_pass8_not_plugin_estimate_and_failures_make_no_modes():
    r=m.prompt_statistics(row(),samples(['a','a','b']+[None]*5))
    assert (r['pass8'],r['distinct8'],r['b8'],r['correct_draws'])==(1,2,1,3)
    assert r['failed_draws']==r['truncated_draws']==5
    assert (r['correct_pairs'],r['colliding_correct_pairs'])==(3,1)
    assert r['uniform_expected_colliding_correct_pairs']==3/8
    assert r['uniform_expected_distinct_given_correct']==pytest.approx(8*(1-(7/8)**3))
    v=dict(zip(m.METRICS,m.rates(m.matrix([r]).sum(axis=0))))
    assert v['pass1']==3/8 and v['pass8']==1
    assert v['pass8']!=1-(1-v['pass1'])**8


def test_zero_one_success_keep_undefined_conditionals_and_all_failures():
    rs=[m.prompt_statistics(row(i),samples([None]*8 if i==0 else ['a']+[None]*7)) for i in range(2)]
    v=dict(zip(m.METRICS,m.rates(m.matrix(rs).sum(axis=0))))
    assert v['pass8']==v['distinct8']==.5 and v['b8']==0
    assert np.isnan(v['correct_pair_collision']) and np.isnan(v['uniform_correct_pair_collision'])
    assert m.counts(rs)['failed_draws']==15


def test_collision_and_uniform_reference_use_same_pairs_not_prompt_mean():
    rs=[m.prompt_statistics(row(0,4),samples(['a']*8)),
        m.prompt_statistics(row(1,8),samples(['a','b']+[None]*6))]
    v=dict(zip(m.METRICS,m.rates(m.matrix(rs).sum(axis=0))))
    assert v['correct_pair_collision']==28/29
    assert v['uniform_correct_pair_collision']==pytest.approx((28/4+1/8)/29)
    assert v['correct_pair_collision_excess_uniform']==pytest.approx((28-28/4-1/8)/29)


def test_open_support_never_gets_uniform_reference():
    r=row();r['metadata']['support_is_open']=True
    v=dict(zip(m.METRICS,m.rates(m.matrix([m.prompt_statistics(r,samples(['a']*8))]).sum(axis=0))))
    assert v['correct_pair_collision']==1
    assert np.isnan(v['uniform_correct_pair_collision']) and np.isnan(v['uniform_expected_distinct_given_correct'])


def test_equal_arms_have_zero_paired_difference_in_every_replicate():
    rs=[m.prompt_statistics(row(i),samples([str(j) for j in range(i+1)]+[None]*(7-i))) for i in range(8)]
    values=m.matrix(rs);indices=np.random.default_rng(3).integers(0,8,(2000,8))
    point,boot=m.paired_bootstrap(values,values.copy(),indices)
    np.testing.assert_array_equal(point[m.CONTRAST],np.zeros(len(m.METRICS)))
    np.testing.assert_array_equal(boot[m.CONTRAST],np.zeros((2000,len(m.METRICS))))


def test_breadth_difference_identity_holds_in_every_replicate():
    a=[m.prompt_statistics(row(i),samples(['a']*(i+1)+[None]*(7-i))) for i in range(8)]
    b=[m.prompt_statistics(row(i),samples([None]*8 if i%2 else ['a','b']+[None]*6)) for i in range(8)]
    points,boots=m.paired_bootstrap(m.matrix(a),m.matrix(b),np.random.default_rng(11).integers(0,8,(1000,8)))
    for arm in (*m.ARMS,m.CONTRAST):
        assert points[arm][3]==points[arm][2]-points[arm][1]
        np.testing.assert_array_equal(boots[arm][:,3],boots[arm][:,2]-boots[arm][:,1])


def test_macro_never_drops_a_cell_without_success():
    good=m.rates(m.matrix([m.prompt_statistics(row(),samples(['a']*8))]).sum(axis=0))
    zero=m.rates(m.matrix([m.prompt_statistics(row(),samples([None]*8))]).sum(axis=0))
    points={k:{arm:v for arm in (*m.ARMS,m.CONTRAST)} for k,v in [('good',good),('zero',zero)]}
    boots={k:{arm:np.repeat(v[None,:],100,axis=0) for arm,v in p.items()} for k,p in points.items()}
    result=m.macro(points,boots,['good','zero'])['original']
    assert result['pass8']['estimate']==result['distinct8']['estimate']==.5
    assert result['correct_pair_collision']['estimate'] is None
    assert result['correct_pair_collision']['ci95'] is None
    assert result['correct_pair_collision']['defined_bootstrap_replicates']==0


@pytest.mark.parametrize('indices',[range(7),[0,1,2,3,4,5,6,6]])
def test_incomplete_or_duplicate_draw_group_rejected(indices):
    s=samples(['a']*8)
    with pytest.raises(ValueError,match='eight unique'):
        m.prompt_statistics(row(),[{**s[0],'sample_index':i} for i in indices])


@pytest.mark.parametrize('mutation',['missing_key','false_key','not_bool','support_overflow'])
def test_bad_grades_and_impossible_support_fail_closed(mutation):
    s=samples(['a']*8);r=row(support=1)
    if mutation=='missing_key':s[0]['canonical_key']=None
    elif mutation=='false_key':s[0]['verified']=False
    elif mutation=='not_bool':s[0]['verified']=1
    else:s[0]['canonical_key']='b'
    with pytest.raises(ValueError):m.prompt_statistics(r,s)


def test_joint_eligibility_exposes_changed_conditioning_population():
    a=[m.prompt_statistics(row(i),samples(['a']*c+[None]*(8-c))) for i,c in enumerate([8,2,1,0])]
    b=[m.prompt_statistics(row(i),samples(['a']*c+[None]*(8-c))) for i,c in enumerate([2,0,2,0])]
    assert m.joint_eligibility(a,b)=={'both':1,'original_only':1,'neutral_only':1,'neither':1}
    b[0]['row_sha256']='other'
    with pytest.raises(ValueError,match='mathematical task changed'):m.joint_eligibility(a,b)


@pytest.fixture
def payload_pair(tmp_path):
    cohorts=[]
    for arm in m.ARMS:
        cohorts.append({'directory':tmp_path/arm,
            'requests':{(2,'python_factors',0,0):{'request':{'model':'model','messages':[{'role':'user','content':arm}],
                        'temperature':1,'max_tokens':8192,'reasoning_effort':'medium'},'row_sha256':'same'}},
            'raw':{0:{'response_id':arm}},'manifest':{'code_sha256':{'src/oat_drgrpo/math_grader.py':'same'}}})
    return cohorts


def test_payload_pair_allows_only_prompt_intervention(payload_pair):
    m.validate_payload_pair(*payload_pair)


@pytest.mark.parametrize('field',['temperature','max_tokens','reasoning_effort'])
def test_payload_control_change_is_rejected(payload_pair,field):
    next(iter(payload_pair[1]['requests'].values()))['request'][field]='different'
    with pytest.raises(ValueError,match='non-prompt generation control'):m.validate_payload_pair(*payload_pair)


def test_reused_native_response_is_rejected_across_arms(payload_pair):
    payload_pair[1]['raw'][0]['response_id']='original'
    with pytest.raises(ValueError,match='reused across arms'):m.validate_payload_pair(*payload_pair)


def test_changed_grader_is_rejected_across_arms(payload_pair):
    payload_pair[1]['manifest']['code_sha256']['src/oat_drgrpo/math_grader.py']='different'
    with pytest.raises(ValueError,match='graders differ'):m.validate_payload_pair(*payload_pair)


def synthetic_local_models(domain='python_factors',seeds=range(43,48)):
    result=[]
    for seed in seeds:
        records={arm:[] for arm in m.ARMS}
        for level in m.LEVELS:
            for index in range(4):
                r={**row(index),'level':level,'domain':domain}
                for arm in m.ARMS:
                    # Constant within seed, varied across seeds: prompt resampling
                    # cannot manufacture training-seed uncertainty.
                    k=(seed-43)%3+1 if arm=='neutral' else 1
                    records[arm].append(m.prompt_statistics(r,samples([str(i%k) for i in range(8)])))
        result.append({'model_id':f'm{seed}','family':'local',
                       'checkpoint':{'training_method':'drgrpo','domain':domain,'training_seed':seed},
                       'analyses':{g:{'prompts':copy.deepcopy(records)} for g in m.GRADINGS}})
    return result


def test_seed_uncertainty_not_confused_with_fixed_prompt_uncertainty():
    reps=300
    indices={(l,d):np.random.default_rng(l).integers(0,4,(reps,4)) for l in m.LEVELS for d in m.DOMAINS}
    groups=m.combine_local_seeds(synthetic_local_models(),indices,reps)
    cell=groups['drgrpo/python_factors']['analyses']['strict']['cells']['level2/python_factors']
    fixed=cell['seed_mean_fixed_seed_prompt_ci'][m.CONTRAST]['distinct8']
    seed=cell['hierarchical_seed_and_prompt_ci'][m.CONTRAST]['distinct8']
    assert fixed['estimate']==pytest.approx(.8)
    assert fixed['ci95']==[.8,.8]
    assert seed['ci95'][0]<.8<seed['ci95'][1]
    p=cell['hierarchical_seed_and_prompt_ci'][m.CONTRAST]['pass8']
    assert p['ci95']==[0,0]


def test_two_seed_pantry_has_individual_results_and_no_seed_interval():
    reps=30
    indices={(l,d):np.zeros((reps,4),int) for l in m.LEVELS for d in m.DOMAINS}
    groups=m.combine_local_seeds(synthetic_local_models('pantry_plan',[43,46]),indices,reps)
    cell=groups['drgrpo/pantry_plan']['analyses']['strict']['cells']['level3/pantry_plan']
    assert cell['hierarchical_seed_and_prompt_ci'] is None
    assert set(cell['per_seed'])=={'43','46'}
    assert 'distinct8' in cell['contrast_seed_range']


def test_missing_local_seed_cannot_be_silently_omitted():
    with pytest.raises(ValueError,match='Missing or selected local training seeds'):
        m.combine_local_seeds(synthetic_local_models(seeds=[43,44,45,46]),{},20)


def test_json_serializer_rejects_nonfinite_claims(tmp_path):
    with pytest.raises(ValueError):m.write_json(tmp_path/'report.json',{'estimate':float('nan')})


def test_degenerate_interval_and_observed_discordance_are_reported_without_equivalence_claim():
    a=[m.prompt_statistics(row(i),samples(['a']*8)) for i in range(4)]
    b=copy.deepcopy(a)
    counts=m.paired_difference_counts(a,b)
    assert counts['pass8']['discordant_prompts']==0
    assert counts['pass8']['zero_difference_prompts']==4
    assert counts['distinct8']['all_paired_differences_identical'] is True
    value=m.describe(np.zeros(len(m.METRICS)),np.zeros((20,len(m.METRICS))))
    assert value['pass8']['degenerate_ci'] is True
    assert value['pass8']['ci95']==[0,0]


def test_valid_truncated_answer_still_counts_as_correct_mode():
    s=samples(['a']*8)
    s[-1].update(stop_reason='length',response_status='incomplete')
    r=m.prompt_statistics(row(),s)
    assert r['correct_draws']==8 and r['truncated_draws']==1 and r['failed_draws']==0


@pytest.fixture
def native_cohort(tmp_path,monkeypatch):
    run=tmp_path/'original';run.mkdir();(run/'sample_receipts').mkdir()
    r=row();key=m.identity(r);prompt=[{'role':'system','content':'hint'},{'role':'user','content':r['problem']}]
    manifest={'experiment_condition':m.CONDITION,'prompt_arm':'original','ablation_manifest_sha256':'a'*64,
              'fresh_response_cohort':True,'sample_count':8,'model':'model','prepared_at_utc':'2026-09-11T00:00:00+00:00',
              'code_sha256':{'ops/frontier_modebench_normalization.py':'norm','ops/frontier_modebench_contract.py':'contract'}}
    m.write_json(run/'manifest.json',manifest)
    entry={'run_dir':str(run),'arm':'original','family':'frontier','model':'model','manifest_sha256':m.file_sha(run/'manifest.json')}
    design={'manifest_sha256':'a'*64,'rows':{key:r},'prompts':{('original',*key):{'messages':prompt}},
            'normalizer_sha256':'norm','contract_sha256':'contract'}
    requests=[{**r,'sample_index':i,'prompt_arm':'original','experiment_condition':m.CONDITION,'request':{'messages':prompt}}
              for i in range(8)]
    raw=[{**r,**s,'sample_id':f's{i}','response_id':f'response{i}','raw_receipt':f'raw{i}.json'}
         for i,s in enumerate(samples(['a']*8))]
    bodies={s['raw_receipt']:{'started_at_utc':'2026-09-11T00:01:00+00:00'} for s in raw}
    (run/'samples.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in raw))
    for s in raw:m.write_json(run/'sample_receipts'/(s['sample_id']+'.json'),s)
    inventory={'manifest':manifest,'rows':{key:r},'requests':requests}
    class FakeNative:
        @staticmethod
        def validate_native_records(inventory,records):return bodies
    monkeypatch.setattr(m,'native_inventory',lambda directory,expected:(FakeNative,inventory))
    return design,entry,inventory,raw,bodies,run


def test_native_guard_accepts_complete_fresh_requested_prompt(native_cohort):
    design,entry,*_=native_cohort
    assert len(m.validate_run_inputs(design,entry)[3])==8


def test_native_guard_rejects_missing_eighth_draw(native_cohort):
    design,entry,inventory,raw,bodies,run=native_cohort
    (run/'samples.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in raw[:-1]))
    with pytest.raises(ValueError,match='Incomplete or unexpected'):m.validate_run_inputs(design,entry)


def test_native_guard_rejects_historical_response_even_with_matching_prompt(native_cohort):
    design,entry,inventory,raw,bodies,run=native_cohort
    bodies[raw[0]['raw_receipt']]['started_at_utc']='2026-09-10T00:00:00+00:00'
    with pytest.raises(ValueError,match='Historical response'):m.validate_run_inputs(design,entry)


def test_native_guard_rejects_modified_requested_prompt(native_cohort):
    design,entry,inventory,raw,bodies,run=native_cohort
    inventory['requests'][0]['request']['messages']=[{'role':'user','content':'different'}]
    with pytest.raises(ValueError,match='Requested prompt changed'):m.validate_run_inputs(design,entry)


def test_native_guard_rejects_unregistered_original_cohort(native_cohort):
    design,entry,inventory,raw,bodies,run=native_cohort
    manifest=json.loads((run/'manifest.json').read_text());manifest['fresh_response_cohort']=False
    m.write_json(run/'manifest.json',manifest);entry['manifest_sha256']=m.file_sha(run/'manifest.json')
    with pytest.raises(ValueError,match='historical controls'):m.validate_run_inputs(design,entry)
