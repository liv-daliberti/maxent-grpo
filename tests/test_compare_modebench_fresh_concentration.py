"""Cross-study comparison preserves populations, pairing and descriptive scope."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ops'))
import compare_modebench_fresh_concentration as m
import analyze_modebench_fresh_concentration as f


class Helper:
    @staticmethod
    def selected_metrics(record):return record


def old(c):return {'collision':c,'mean8':.5,'selected_correct':2,'selected_total':11}


def summaries(keys,pid):return f.prompt_summary(keys,prompt_id=pid,row_sha256='same')


def fixture():
    left={'a':old(1),'b':old(0)};right={'a':old(0),'b':old(1)}
    a=[summaries(['x']*4,'a'),summaries(['x','y',None,None],'b')]
    b=[summaries(['x','x','y','y'],'a'),summaries(['x',None,None,None],'b')]
    new=f.seed_contrast(a,b,training_seed=43,expected_prompt_ids=['a','b'])
    previous={'paired_seed':43,'historical_primary':{'eligible_ids':['a','b'],'delta':{'collision':0}},'source_issues':[]}
    return previous,new,left,right


def test_common_population_recomputes_old_subset_instead_of_using_full_mean():
    previous,new,left,right=fixture()
    r=m.compare_seed(previous,new,left,right,Helper,expected_prompt_ids=['a','b'])
    assert r['overlap']=={'old':2,'new':1,'both':1,'old_only':1,'new_only':0,'neither':0,'jaccard':.5}
    c=r['common_population']
    assert c['prompt_ids']==['a'] and c['old_delta']==-1
    assert c['new_delta']==pytest.approx(-2/3) and c['new_minus_old_delta']==pytest.approx(1/3)
    p=c['per_prompt'][0]
    assert p['old_left_budget']==11 and p['new_left_budget']==4
    assert c['ci95'] is None


def test_missing_old_source_is_unknown_not_zero_old_eligibility():
    previous,new,left,right=fixture();previous['historical_primary']=None
    r=m.compare_seed(previous,new,None,None,Helper,expected_prompt_ids=['a','b'])
    assert r['status']=='historical_source_unavailable'
    assert r['old_joint_prompt_ids'] is None and r['overlap'] is None and r['common_population'] is None
    assert r['new_joint_prompt_ids']==['a']


def test_empty_intersection_is_retained_and_no_conditional_is_fabricated():
    previous,new,left,right=fixture();previous['historical_primary']['eligible_ids']=['b']
    r=m.compare_seed(previous,new,left,right,Helper,expected_prompt_ids=['a','b'])
    assert r['status']=='no_common_eligible_prompts'
    assert r['common_population']['n_prompts']==0 and r['common_population']['old_delta'] is None
    assert r['overlap']['old_only']==r['overlap']['new_only']==1


def test_pairing_and_full_prompt_inventory_cannot_drift():
    previous,new,left,right=fixture();new['training_seed']=44
    with pytest.raises(ValueError,match='seed pairing'):m.compare_seed(previous,new,left,right,Helper,expected_prompt_ids=['a','b'])
    new['training_seed']=43
    with pytest.raises(ValueError,match='fixed prompts'):m.compare_seed(previous,new,left,right,Helper,expected_prompt_ids=['a','b','extra'])


def test_descriptive_equal_seed_summary_retains_missing_and_adds_no_interval():
    previous,new,left,right=fixture()
    r=m.compare_seed(previous,new,left,right,Helper,expected_prompt_ids=['a','b'])
    missing={'paired_seed':44,'common_population':None}
    s=m.summarize_common([r,missing])
    assert s['registered_seeds']==[43,44] and s['undefined_seeds']==[44]
    assert s['equal_seed_means']['old_delta']==-1 and s['ci95'] is None


def test_incomplete_fresh_panel_cannot_publish_comparison(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'read_published',lambda path: {} if str(path)=='reference' else {
        'status':'complete','scope':{'panel':'registered_Level1_fresh'},
        'completeness_audit':{'status':'incomplete','expected_tasks':150,'authenticated_tasks':2}})
    with pytest.raises(ValueError,match='complete authenticated'):m.build_report('reference','fresh')


def test_published_input_manifest_detects_changed_report(tmp_path):
    path=tmp_path/'report.json';path.write_text('{}\n')
    manifest={'files':{'report.json':f.file_binding(path)['sha256']}}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest));assert m.read_published(path)=={}
    path.write_text('{"changed":true}\n')
    with pytest.raises(ValueError,match='publication content'):m.read_published(path)


def test_output_is_immutable(tmp_path):
    path=tmp_path/'exists';path.mkdir()
    with pytest.raises(ValueError,match='refusing overwrite'):m.publish({},path)


def test_complete_synthetic36block_build_preserves_all180seed_overlaps(tmp_path,monkeypatch):
    reference_path=tmp_path/'reference.json';new_path=tmp_path/'fresh.json'
    reference_path.write_text('{}');new_path.write_text('{}')
    previous,new_record,left,right=fixture()
    reference={'contrasts':[],'prompt_identity_to_row_index':{d:{'a':0,'b':1} for d in ('graph_coloring','pantry_plan')}}
    new={'status':'complete','scope':{'panel':'registered_Level1_fresh'},
         'completeness_audit':{'status':'complete','expected_tasks':150,'authenticated_tasks':150,
                               'authenticated_response_slots':1228800},'contrasts':[]}
    endpoints={};pairs=[('initial',method) for method in ('drgrpo','replay_drgrpo','maxrl','replay_maxrl')]+[
        ('drgrpo','replay_drgrpo'),('maxrl','replay_maxrl')]
    import copy
    for scale in ('qwen05b','falcon1b','qwen3b'):
        for domain in reference['prompt_identity_to_row_index']:
            for method in ('drgrpo','replay_drgrpo','maxrl','replay_maxrl'):
                for seed in range(43,48):
                    endpoints[(scale,domain,method,seed),'0']=left
                    endpoints[(scale,domain,method,seed),'3072']=right
            for a,b in pairs:
                oldrows=[];newrows=[]
                for seed in range(43,48):
                    oldrow=copy.deepcopy(previous);oldrow['paired_seed']=seed;oldrows.append(oldrow)
                    newrow=copy.deepcopy(new_record);newrow['training_seed']=seed;newrows.append(newrow)
                meta={'model_scale':scale,'domain':domain,'level':1,'wording':'original','contrast':b+'_minus_'+a,
                      'left_method':a,'right_method':b}
                reference['contrasts'].append({**meta,'registered_seeds':list(range(43,48)),'per_seed':oldrows,
                                              'primary':{'mean':0,'n':5,'ci95':[0,0]}})
                new['contrasts'].append({**meta,'seeds':newrows,
                                        'summary':f.aggregate_seed_contrasts(newrows,expected_seeds=range(43,48))})
    monkeypatch.setattr(m,'read_published',lambda path:reference if path==reference_path else new)
    monkeypatch.setattr(m,'historical_endpoints',lambda ref:(Helper,endpoints,{'synthetic_test':True}))
    result=m.build_report(reference_path,new_path)
    assert result['comparison_count']==36
    assert sum(result['direction_counts_descriptive'].values())==36
    assert sum(len(c['per_seed']) for c in result['comparisons'])==180
    assert all(c['common_population_summary']['n_defined']==5 for c in result['comparisons'])
    assert result['old_new_responses_pooled'] is False
    output=m.publish(result,tmp_path/'comparison')
    assert json.loads((output/'report.json').read_text())['comparison_count']==36
    monkeypatch.undo()
    assert m.read_published(output/'report.json')['analysis_role'].startswith('retrospective comparison sensitivity')
