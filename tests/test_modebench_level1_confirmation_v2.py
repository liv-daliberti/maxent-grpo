from collections import Counter
from itertools import combinations
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ops/exp_scaling'))
import materialize_modebench_level1_confirmation_v2 as reserve


def row(domain, index, support, *, prompt=None, family='breakfast_formulation'):
    if domain == 'graph_coloring':
        spec = {'verifier':'graph_coloring','n':6,'edges':[[1,2]],
                'partial_colors':[None,None,None,1,1,index+2]}
    elif domain == 'python_factors':
        spec = {'verifier':'python_factor_function','cases':[6,10,14,index+20]}
    elif domain == 'pantry':
        spec = {'verifier':'pantry_plan','family':family,'value':index,'instance_id':'ignored'}
    else:
        raise ValueError(domain)
    result = {'problem':prompt or f'problem{index}','answer':json.dumps(spec),
              'modebench_task':spec['verifier'],'answer_mode_count':support}
    if domain == 'pantry':
        result['answer_mode_family'] = family
        result['instance_fingerprint'] = reserve.semantic_identity(domain,result)[1]
    return result


def test_graph_uses_original_mask_acceptance_before_quota_filtering(monkeypatch):
    calls = []
    proposals = [row('graph_coloring',i,5 if i < 128 else 4) for i in range(160)]
    def original(limit,**kwargs):
        calls.append((limit,kwargs))
        start = (len(calls)-1)*limit
        return proposals[start:start+limit]
    monkeypatch.setattr(reserve.graph,'_synthetic_graph_rows',original)
    monkeypatch.setattr(reserve,'validate_rows',lambda *a:None)
    result = reserve.build_domain_rows('graph_coloring',Counter({(5,):128}),set(),set(),9411800)
    assert len(result) == 128 and len(calls) == 4
    for limit,kwargs in calls:
        assert limit == 32
        assert kwargs['min_completions'] == kwargs['min_solutions'] == 4
        assert kwargs['max_completions'] == 24
        assert kwargs['hidden_count'] == 3 and kwargs['max_n'] == 6 and kwargs['max_edges'] == 8
        assert kwargs['prompt_style'] == 'original' and kwargs['balance_hidden_color'] is False
    assert len(calls[-1][1]['exclude']) == 96


def test_quota_filter_keeps_semantic_and_exact_prompt_exclusions():
    historical = row('python_factors',0,16)
    repeated_prompt = row('python_factors',1,16,prompt='already observed')
    wanted = row('python_factors',2,16)
    wrong_cell = row('python_factors',3,24)
    selected = [];seen = set()
    done = reserve.keep_needed('python_factors',[historical,repeated_prompt,wrong_cell,wanted,wanted],
        Counter({(16,):1}),{reserve.semantic_identity('python_factors',historical)},
        {reserve.prompt_sha('already observed')},selected,seen)
    assert done and selected == [wanted]
    assert reserve.semantic_identity('python_factors',wrong_cell) in seen


def test_pantries_keep_original_unconditioned_family_proposals(monkeypatch,tmp_path):
    curation = tmp_path/'ingredients.json';curation.write_text('{}')
    monkeypatch.setattr(reserve,'CURATION',curation)
    calls = []
    def original(**kwargs):
        calls.append(kwargs)
        first = 16*(len(calls)-1)
        return [row('pantry',i,8) for i in range(first,first+16)]
    monkeypatch.setattr(reserve.pantry,'_build_rows',original)
    monkeypatch.setattr(reserve,'validate_rows',lambda *a:None)
    rows = reserve.build_domain_rows('pantry',Counter({(8,'breakfast_formulation'):128}),set(),set(),9412100)
    assert len(rows) == 128 and len(calls) == 8
    assert all(call['per_family'] == 4 and set(call) ==
               {'per_family','split','seed','curation','excluded_fingerprints'} for call in calls)
    assert len(calls[-1]['excluded_fingerprints']) == 112


def test_budget_exhaustion_fails_without_alternate_bounds_or_seed(monkeypatch):
    calls = []
    def original(limit,**kwargs):
        calls.append(kwargs)
        return [row('graph_coloring',i+len(calls)*100,4) for i in range(limit)]
    monkeypatch.setattr(reserve.graph,'_synthetic_graph_rows',original)
    monkeypatch.setitem(reserve.PROPOSAL_LIMITS,'graph_coloring',64)
    with pytest.raises(ValueError,match='fixed accepted-proposal budget exhausted'):
        reserve.build_domain_rows('graph_coloring',Counter({(5,):128}),set(),set(),99)
    assert len(calls) == 2
    assert [c['seed'] for c in calls] == [reserve.derived_seed(99,'accepted_chunk',i) for i in range(2)]
    assert all(c['min_completions'] == 4 and c['max_completions'] == 24 for c in calls)


def test_python_finite_law_matches_original_uniform_case_domain():
    values = reserve.python_factors._candidate_values(30)
    expected = {cases:reserve.python_factor_mode_count(cases) for cases in combinations(values,4)}
    assert dict(reserve.python_catalog(30)) == expected
    assert all(count >= 16 for count in expected.values())


def test_finite_cell_capacity_refusal_precedes_certification(monkeypatch):
    monkeypatch.setattr(reserve,'finite_cells',lambda *a:{(16,):[(6,10,14,15)]})
    def forbidden(**kwargs):
        raise AssertionError('must not certify rows after capacity failure')
    monkeypatch.setattr(reserve.python_factors,'_row',forbidden)
    with pytest.raises(ValueError,match='finite cell'):
        reserve.build_domain_rows('python_factors',Counter({(16,):128}),set(),set(),1)


@pytest.mark.parametrize('family,bindings,expected',[
    ('ax_plus_b_eq_c',{'a':2,'b':-3,'c':11},True),
    ('x_over_a_plus_b_eq_c',{'a':-9,'b':12,'c':21},True),
    ('ax_plus_b_eq_dx_plus_c',{'a':2,'b':3,'c':-6,'d':3},True),
    ('ax_plus_b_eq_c_minus_dx',{'a':2,'b':3,'c':12,'d':-1},True),
    ('ax_plus_b_eq_c',{'a':2,'b':0,'c':0},False),
    ('ax_plus_b_eq_dx_plus_c',{'a':2,'b':3,'c':9,'d':2},False),
    ('ax_plus_b_eq_c_minus_dx',{'a':2,'b':3,'c':9,'d':-2},False),
    ('ax_plus_b_eq_c',{'a':10,'b':0,'c':10},False),
])
def test_mathir_capacity_membership_preserves_original_binding_laws(family,bindings,expected):
    assert reserve.mathir_original_identity_holds(('mathir',family,tuple(sorted(bindings.items())))) is expected


def test_discovery_includes_current_prior_reserves_and_pools_but_never_model_logs(tmp_path):
    locations = ['e117_evaluation_reserve_v1/development/mathir/eval/dataset_dict.json',
                 'e117_evaluation_reserve_v1/confirmation/mathir/eval/dataset_dict.json',
                 'modebench_level3_matched_v1/mathir/eval/dataset_dict.json',
                 'modebench_level3_calibration_v7/pools/mathir/difficulty_0.jsonl',
                 'pantry_plan_modebench_v2/dev/dataset_dict.json',
                 'xdr_model_run/train_metrics.jsonl']
    for name in locations:
        path = tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('{}')
    sources = reserve.discover_sources(tmp_path)
    assert len(sources) == 5
    assert all('xdr_model_run' not in source['path'] for source in sources)
    assert sum(source['kind'] == 'jsonl' for source in sources) == 1


def test_exact_prompt_exclusion_hash_is_not_row_metadata_dependent():
    first = row('pantry',1,8)
    changed = dict(first)
    spec = json.loads(changed['answer']);spec['instance_id'] = 'another'
    changed['answer'] = json.dumps(spec)
    assert reserve.semantic_identity('pantry',changed) == reserve.semantic_identity('pantry',first)
    assert reserve.prompt_sha(changed['problem']) == reserve.prompt_sha(first['problem'])


def test_publication_refuses_existing_output_before_reading_or_sampling(tmp_path,monkeypatch):
    output = tmp_path/'existing';output.mkdir()
    def forbidden(*args):
        raise AssertionError('must not inspect or generate when destination exists')
    monkeypatch.setattr(reserve,'validate_plan',forbidden)
    with pytest.raises(ValueError,match='fresh output root required'):
        reserve.materialize(tmp_path/'missing_plan.json',output)


def test_quota_filter_rejects_duplicate_exact_prompts_across_distinct_identities():
    first = row('python_factors',1,16,prompt='same visible prompt')
    duplicate = row('python_factors',2,16,prompt='same visible prompt')
    last = row('python_factors',3,16,prompt='new visible prompt')
    selected = []
    assert reserve.keep_needed('python_factors',[first,duplicate,last],Counter({(16,):2}),
                               set(),set(),selected,set())
    assert selected == [first,last]


def test_source_snapshot_detects_candidate_file_mutation(tmp_path):
    path = tmp_path/'rows.jsonl';path.write_text(json.dumps(row('python_factors',1,16))+'\n')
    source = {'kind':'jsonl','path':str(path)}
    before = reserve.source_snapshot(source)
    path.write_text(json.dumps(row('python_factors',2,16))+'\n')
    assert reserve.source_snapshot(source) != before
