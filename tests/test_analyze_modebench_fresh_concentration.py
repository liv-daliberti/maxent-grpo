"""Independent estimand, missingness, weighting, and immutable-output tests."""
import importlib.util
from itertools import combinations
import math
from pathlib import Path
import statistics
import pytest

spec=importlib.util.spec_from_file_location('fresh_analysis',Path(__file__).resolve().parents[1]/'ops/analyze_modebench_fresh_concentration.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def prompt(keys,identity='p',row_hash='same-task'):
    return m.prompt_summary(keys,prompt_id=identity,row_sha256=row_hash)


def contrast(left,right,seed=43):
    return m.seed_contrast(left,right,training_seed=seed,expected_prompt_ids=[p['prompt_id'] for p in left])


@pytest.mark.parametrize('keys,expected',[([None]*8,None),(['a']+[None]*7,None),(['a']*8,1),(list(range(8)),0),(['a','a','b',None],1/3)])
def test_collision_correct_pair_boundaries(keys,expected):
    s=prompt(keys)
    if expected is None: assert s['collision'] is None
    else: assert s['collision']==pytest.approx(expected)
    assert s['draws']==len(keys)
    assert s['correct_draws']==sum(k is not None for k in keys)


def test_rarefaction_matches_finite_population_enumeration():
    keys=['a']*4+['b']*2+['c']+[None]*3
    distinct=[len({keys[i] for i in indices if keys[i] is not None}) for indices in combinations(range(10),8)]
    s=prompt(keys)
    assert s['pass8_rarefied']==pytest.approx(statistics.mean(d>0 for d in distinct))
    assert s['distinct8_rarefied']==pytest.approx(statistics.mean(distinct))
    assert s['extra8_rarefied']==pytest.approx(statistics.mean(max(d-1,0) for d in distinct))


def test_json_key_identity_preserves_types_and_ignores_mapping_order():
    s=prompt([1,'1',{'a':1,'b':2},{'b':2,'a':1}])
    assert s['distinct_all']==3
    assert s['collision']==pytest.approx(1/6)


@pytest.mark.parametrize('field,value,message',[('prompt_id','other','identities'),('row_sha256','other','hashes'),('draws',64,'budgets')])
def test_prompt_task_budget_pairing(field,value,message):
    left,right=prompt(['a']*8),prompt(['b']*8);right[field]=value
    with pytest.raises(ValueError,match=message):m.per_prompt_contrast(left,right)


def test_pool_weighting_can_reverse_on_same_population():
    r=contrast([prompt(['a']*8,'a'),prompt(['a','b']+[None]*6,'b')],
               [prompt(['a']*4+['b']*4,'a'),prompt(['a','a']+[None]*6,'b')])
    assert r['jointly_eligible_prompts']==2
    assert r['populations']['joint_R_ge_2']['delta']['collision']==pytest.approx(3/14)
    assert r['pooled_pairs_same_joint_population']['delta']==pytest.approx(-15/29)
    assert r['pooled_minus_equal_prompt_delta']==pytest.approx(-15/29-3/14)


def test_joint_correctness_and_complete_population_are_separate():
    r=contrast([prompt(['a']*2+[None]*6,'yes'),prompt(['a']*8,'no')],
               [prompt(['a']*4+[None]*4,'yes'),prompt([None]*8,'no')])
    assert r['joint_coverage']==.5
    assert r['populations']['joint_R_ge_2']['delta']['mean_correct']==.25
    assert r['populations']['all_prompts']['delta']['mean_correct']==-.375
    assert r['populations']['all_prompts']['delta']['collision'] is None
    assert r['own_eligible']['left']['eligible_prompts']==2
    p=next(p for p in r['prompts'] if p['prompt_id']=='no')
    assert p['ineligible_reason']=='right_R_lt_2' and p['delta']['collision'] is None


def test_empty_joint_population_remains_undefined():
    r=contrast([prompt(['a']*8)],[prompt([None]*8)])
    assert not r['defined'] and r['joint_coverage']==0
    assert r['undefined_reason']=='no_joint_R_ge_2_prompts'
    assert r['populations']['joint_R_ge_2']['delta']['collision'] is None
    assert r['pooled_pairs_same_joint_population']['delta'] is None
    assert len(r['prompts'])==1


def test_missing_extra_duplicate_prompts_rejected():
    rows=[prompt(['a']*8)]
    for left,right,expected in [(rows,[],['p']),(rows,rows,['p','missing']),(rows*2,rows,['p']),(rows,rows,['p','p'])]:
        with pytest.raises(ValueError):m.seed_contrast(left,right,training_seed=43,expected_prompt_ids=expected)


def five_seeds():
    return [contrast([prompt(['a']*8)],[prompt(['a']*(8-i)+list(range(i)))],43+i) for i in range(5)]


def test_five_seed_t_interval_conditions_on_one_shared_initial_pool():
    rows=five_seeds();s=m.aggregate_seed_contrasts(rows,expected_seeds=range(43,48),shared_initial=True)
    values=[r['populations']['joint_R_ge_2']['delta']['collision'] for r in rows]
    mean=statistics.mean(values);half=2.7764451051977987*statistics.stdev(values)/math.sqrt(5)
    assert s['joint_population_effects']['collision']['ci95']==pytest.approx([mean-half,mean+half])
    assert s['independent_initial_checkpoints']==1 and s['shared_initial_output_pool']
    assert s['n_expected']==s['n_defined']==5


def test_two_seed_panels_are_descriptive():
    s=m.aggregate_seed_contrasts(five_seeds()[:2],expected_seeds=[43,44])
    assert s['n_defined']==2 and s['joint_population_effects']['collision']['ci95'] is None
    assert s['joint_population_effects']['collision']['seed_sd'] is not None


def test_undefined_fifth_seed_explicit_without_interval():
    rows=five_seeds();rows[-1]=contrast([prompt(['a']*8)],[prompt([None]*8)],47)
    s=m.aggregate_seed_contrasts(rows,expected_seeds=range(43,48))
    assert s['defined_seeds']==[43,44,45,46] and s['undefined_seeds']==[47]
    assert s['seed_effects']['47'] is None
    assert s['joint_population_effects']['collision']['ci95'] is None


def test_seed_inventory_cannot_be_selected_silently():
    rows=five_seeds()
    for current,expected in [(rows[:-1],range(43,48)),(rows,range(43,47)),(rows+[rows[0]],range(43,48)),(rows,[43,43,44,45,46,47])]:
        with pytest.raises(ValueError):m.aggregate_seed_contrasts(current,expected_seeds=expected)


def test_published_artifact_is_never_overwritten(tmp_path):
    output=tmp_path/'published';output.mkdir();marker=output/'original.json';marker.write_text('unchanged')
    with pytest.raises(ValueError,match='immutable'):m.write_artifacts({},output)
    assert marker.read_text()=='unchanged'


def test_population_reconciliation_isolates_selection_and_common_seed_population():
    records = []
    for i, seed in enumerate((43,46)):
        left = [prompt(['a']*8,'common'),prompt(['a','b']+[None]*6,'selected')]
        right = [prompt(['a']*4+['b']*4,'common'),prompt(['a']*8 if i else [None]*8,'selected')]
        records.append(contrast(left,right,seed))
    c = {'grading':'strict','wording':'original','level':2,'domain':'pantry_plan','contrast':'replay_minus_dr',
         'seeds':records,'summary':m.aggregate_seed_contrasts(records,expected_seeds=[43,46])}
    r = m.population_reconciliation(c)
    assert r['fixed_common_prompt_ids'] == ['common']
    assert r['fixed_common_population']['n_expected'] == r['fixed_common_population']['n_defined'] == 2
    assert r['fixed_common_population']['equal_prompt_delta_equal_seed'] == pytest.approx(-4/7)
    assert r['own_population_per_seed'][0]['left']['eligible_prompts'] == 2
    assert r['own_population_per_seed'][0]['right']['eligible_prompts'] == 1
    assert r['own_population_pair_pooled_then_equal_seed'] != pytest.approx(r['primary_equal_prompt_then_equal_seed'])


def test_empty_common_sensitivity_preserves_all_registered_seeds():
    records = [contrast([prompt(['a']*8)], [prompt([None]*8)], seed) for seed in (43,46)]
    c = {'grading':'strict','wording':'neutral','level':3,'domain':'pantry_plan','contrast':'replay_minus_dr',
         'seeds':records,'summary':m.aggregate_seed_contrasts(records,expected_seeds=[43,46])}
    r = m.population_reconciliation(c)
    assert r['fixed_common_prompts'] == 0
    assert r['fixed_common_population']['undefined_seeds'] == [43,46]
    assert r['fixed_common_population']['equal_prompt_delta_equal_seed'] is None


def test_common_population_preserves_initial_monte_carlo_scope():
    rows=five_seeds();summary=m.aggregate_seed_contrasts(rows,expected_seeds=range(43,48))
    summary.update(initial_weights_shared=True,initial_sampling_replicas=5,independent_initial_checkpoints=1,
                   uncertainty_scope='trained seeds and initial Monte Carlo replicas')
    c={'grading':'strict','wording':'original','level':1,'model_scale':'qwen05b','domain':'graph_coloring',
       'contrast':'drgrpo_minus_initial','seeds':rows,'summary':summary}
    fixed=m.population_reconciliation(c)['fixed_common_population']
    for field in ('initial_weights_shared','initial_sampling_replicas','independent_initial_checkpoints','uncertainty_scope'):
        assert fixed[field]==summary[field]
    assert m.population_reconciliation(c)['identity']['model_scale']=='qwen05b'
