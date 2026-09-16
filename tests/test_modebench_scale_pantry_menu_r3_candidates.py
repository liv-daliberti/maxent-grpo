"""Prospective API/law regression; no new development pool or model outputs."""
import ast
import __future__
from collections import Counter
import copy
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import random
import sys
from types import CodeType

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'ops/exp_scaling'))
import modebench_scale_pantry_menu_r3_candidates as c

spec = importlib.util.spec_from_file_location('qualified_pantry_menu_scratch_for_comparison',
    c.QUALIFICATION_ROOT/'scratch_candidate.py')
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)


def raw_fixture(tier=0):
    rows = [json.loads(line) for line in (c.QUALIFICATION_ROOT/'cost'/f'tier{tier}_cost.jsonl').read_text().splitlines()]
    row = next(row for row in rows if row['answer_mode_family']=='high_fiber_snack')
    for key in ('scratch_only', 'scratch_profile', 'scale_candidate_tier'):
        row.pop(key, None)
    return row


def provider_fixture(tier=0):
    seed=int(c._canonical_sha256(['unregistered_pantry_menu_capacity_v1','cost','cost',tier,39,'high_fiber_snack'])[:14],16)
    return c._annotate(raw_fixture(tier), tier, seed, 0)


def test_exact_qualified_profile_and_bytecode():
    assert c._MENU_SIZES == q.MENU_SIZES
    assert c._AVAILABILITY == q.AVAILABILITY
    def code_signature(code):
        return (code.co_code, tuple(code_signature(x) if isinstance(x,CodeType) else x for x in code.co_consts),
            code.co_names, code.co_varnames, code.co_freevars, code.co_cellvars, code.co_flags & ~__future__.annotations.compiler_flag,
            code.co_argcount, code.co_kwonlyargcount, code.co_posonlyargcount)
    # Ignore source locations and the future-annotations bit: neither function has annotations.
    assert c._candidate.__annotations__==q._construct.__annotations__=={}
    assert code_signature(c._candidate.__code__)==code_signature(q._construct.__code__)
    for name in c._candidate.__code__.co_names:
        if name in ('_MENU_SIZES', '_AVAILABILITY'):
            assert c._candidate.__globals__[name] == q._construct.__globals__[name]
        elif name in q._construct.__globals__:
            assert c._candidate.__globals__[name] is q._construct.__globals__[name]


def test_complete_ast_is_original_with_exactly_two_law_replacements():
    before = ast.parse(inspect.getsource(c.bridge._candidate))
    changes = []
    for node in ast.walk(before):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id=='menu_size' for t in node.targets):
            node.value = ast.parse('_MENU_SIZES[tier]', mode='eval').body
            changes.append('menu')
        if isinstance(node, ast.Dict):
            for i,key in enumerate(node.keys):
                if isinstance(key,ast.Constant) and key.value=='available_g':
                    node.values[i] = ast.parse('_AVAILABILITY[tier][0] if len(_AVAILABILITY[tier]) == 1 else step * (minimum_steps + rng.choice((2, 3, 4)))', mode='eval').body
                    changes.append('availability')
    assert sorted(changes)==['availability','menu']
    assert ast.dump(before, include_attributes=False)==ast.dump(ast.parse(inspect.getsource(c._candidate)), include_attributes=False)


@pytest.mark.parametrize('tier', range(4))
def test_recorded_cost_prefix_rows_and_post_proposal_rng_equal(tier):
    # Exactly the existing first cost cell; every accepted row is already excluded.
    support,family=39,'high_fiber_snack'
    seed=int(c._canonical_sha256(['unregistered_pantry_menu_capacity_v1','cost','cost',tier,support,family])[:14],16)
    left_rng,right_rng=random.Random(seed),random.Random(seed)
    report=json.loads((c.QUALIFICATION_ROOT/'cost/report.json').read_text())
    proposals=next(x['proposals'] for x in report['cells'] if x['tier']==tier and x['family']==family)
    assert proposals in (1,2)
    for index in range(proposals):
        args=(family,support,seed,'UNREGISTERED_SCRATCH_MENU_FEASIBILITY',0,tier)
        left=c._candidate(*args,left_rng);right=q._construct(*args,right_rng)
        assert left==right and left_rng.getstate()==right_rng.getstate()
        if index<proposals-1:
            assert left is None
    assert left==raw_fixture(tier)


def test_real_complete_scratch_exclusions_and_source_closure():
    ids,prompts,pins=c._qualification()
    assert len(ids)==len(prompts)==3945
    paths=set(c.source_paths())
    assert {c.QUALIFICATION_PATH,c.EXCLUSIONS_PATH,c.BASE_SOURCE,Path(c.__file__).resolve()} <= paths
    assert {Path(path) for path in pins} <= paths
    assert all(path.is_relative_to(ROOT) for path in paths)
    assert c.identity('pantry',raw_fixture()) in ids


@pytest.mark.parametrize('which', ['base','qualification','exclusions','nested'])
def test_tampered_qualification_dependency_rejected(monkeypatch,which):
    nested=ROOT/'tests/test_modebench_scale_pantry_menu_scratch.py'
    changed={'base':c.BASE_SOURCE,'qualification':c.QUALIFICATION_PATH,'exclusions':c.EXCLUSIONS_PATH,'nested':nested}[which]
    real=c._file_sha
    monkeypatch.setattr(c,'_file_sha',lambda path:'0'*64 if Path(path)==changed else real(path))
    with pytest.raises(ValueError):c._qualification()


@pytest.mark.parametrize('kwargs', [
    {'domain':'graph_coloring'}, {'tier':True}, {'tier':-1}, {'tier':4},
    {'seed':True}, {'seed':-1}, {'tag':''}, {'multiplier':True}, {'multiplier':0},
    {'target':{7:1}}, {'target':{46:1}}, {'target':{8:-1}}, {'target':{True:1}},
    {'joint_target':None}, {'joint_target':{(8,'unknown'):1}},
    {'joint_target':{(8,'breakfast_formulation'):-1}},
    {'joint_target':{(9,'breakfast_formulation'):1}},
])
def test_invalid_native_build_contract_fails_before_generation(monkeypatch,kwargs):
    args=dict(domain='pantry',target={8:1},excluded=set(),seed=1,tag='TEST_ONLY',tier=0,
        joint_target={(8,'breakfast_formulation'):1})
    args.update(kwargs)
    monkeypatch.setattr(c,'_candidate',lambda *args:pytest.fail('candidate must not run'))
    with pytest.raises(ValueError):c.build_pool(**args)


def fake_row(fingerprint,problem,support=8,family='breakfast_formulation'):
    return dict(instance_fingerprint=fingerprint,problem=problem,answer_mode_count=support,
        answer_mode_family=family,level3_difficulty=0)


def test_whole_proposal_rejection_and_dynamic_exclusions(monkeypatch):
    scratch=('pantry','scratch');historical=('pantry','historical')
    monkeypatch.setattr(c,'_qualification',lambda:({scratch},{c._canonical_sha256('scratch prompt')},{}))
    proposals=iter([None,fake_row('scratch','a'),fake_row('historical','b'),
        fake_row('other','scratch prompt'),fake_row('accepted1','first'),
        fake_row('accepted1','different text'),fake_row('accepted2','first'),fake_row('accepted2','second')])
    calls=[]
    def candidate(*args):calls.append(args[:-1]);return next(proposals)
    monkeypatch.setattr(c,'_candidate',candidate)
    rows=c.build_pool('pantry',{8:2},{historical},7,'TEST_ONLY',0,joint_target={(8,'breakfast_formulation'):2})
    assert len(calls)==8
    assert {row['instance_fingerprint'] for row in rows}=={'accepted1','accepted2'}
    assert all(row['scale_scratch_exclusions_sha256']==c.EXCLUSIONS_SHA256 for row in rows)
    assert [call[4] for call in calls]==[0]*5+[1]*3


def test_fixed_budget_exhaustion_does_not_relax_support(monkeypatch):
    monkeypatch.setattr(c,'_qualification',lambda:(set(),set(),{}))
    monkeypatch.setattr(c,'MAX_PROPOSALS_PER_ROW',3)
    calls=[]
    def no_candidate(*args):calls.append(args);return None
    monkeypatch.setattr(c,'_candidate',no_candidate)
    with pytest.raises(RuntimeError,match='exhausted fixed proposal budget'):
        c.build_pool('pantry',{45:1},set(),7,'TEST_ONLY',3,joint_target={(45,'breakfast_formulation'):1})
    assert len(calls)==3 and all(args[1]==45 for args in calls)


def test_independent_cells_and_deterministic_display_order(monkeypatch):
    monkeypatch.setattr(c,'_qualification',lambda:(set(),set(),{}))
    def synth(family,support,seed,tag,index,tier,rng):
        value=c._canonical_sha256([family,support,rng.random()])
        return fake_row(value,'synthetic '+value,support,family)
    monkeypatch.setattr(c,'_candidate',synth)
    def generate(joint):
        marginal=Counter()
        for (support,family),n in joint.items():marginal[support]+=n
        return c.build_pool('pantry',marginal,set(),71,'TEST_ONLY',2,joint_target=joint)
    alone=generate({(8,'breakfast_formulation'):2})
    together=generate({(8,'breakfast_formulation'):2,(9,'high_fiber_snack'):1})
    assert alone==[row for row in together if row['answer_mode_count']==8]
    assert together==generate({(9,'high_fiber_snack'):1,(8,'breakfast_formulation'):2})
    assert together==sorted(together,key=lambda row:c._seed(71,2,'display',c.identity('pantry',row)))


def test_real_saved_scratch_row_refused_before_native_audit(monkeypatch):
    monkeypatch.setattr(c.original,'verify_rows',lambda *args:pytest.fail('scratch row must not be reaudited'))
    with pytest.raises(ValueError,match='reuses scratch'):
        c.verify_rows('pantry',[provider_fixture()])


@pytest.mark.parametrize('change', ['generator','profile','tier','qual_sha','exclude_sha','menu','quantity','family','attributes','tags','forbidden','limits','sodium','support',
    'task','verifier','version','support_float','seed_bool','seed_negative','index_float','index_negative','instance_id'])
def test_static_law_tampering_refused_before_native_audit(monkeypatch,change):
    monkeypatch.setattr(c,'_qualification',lambda:(set(),set(),{}))
    monkeypatch.setattr(c.original,'verify_rows',lambda *args:pytest.fail('tampered law must not reach native audit'))
    row=provider_fixture();answer=json.loads(row['answer'])
    if change=='generator':row['scale_candidate_generator']='wrong'
    if change=='profile':row['scale_candidate_profile']='{}'
    if change=='tier':row['scale_candidate_tier']=True
    if change=='qual_sha':row['scale_scratch_qualification_sha256']='0'*64
    if change=='exclude_sha':row['scale_scratch_exclusions_sha256']='0'*64
    if change=='menu':answer['ingredients'].pop()
    if change=='quantity':answer['ingredients'][0]['step_g']=20
    if change=='family':answer['family']='unknown'
    if change=='attributes':answer['ingredients'][0]['attributes_per_100g']['protein_g']='999'
    if change=='tags':answer['ingredients'][0]['tags']=['fake']
    if change=='forbidden':answer['forbidden_tags']=['fake']
    if change=='limits':answer['targets']['protein_g']['max']='20'
    if change=='sodium':answer['targets']['sodium_mg']['max']='999999'
    if change=='support':answer['certified_mode_count']=1
    if change=='task':row['modebench_task']='wrong'
    if change=='verifier':answer['verifier']='wrong'
    if change=='version':answer['pantry_version']='wrong'
    if change=='support_float':row['answer_mode_count']=39.0
    if change=='seed_bool':row['scale_generation_seed']=True
    if change=='seed_negative':row['scale_generation_seed']=-1
    if change=='index_float':row['scale_cell_index']=0.0
    if change=='index_negative':row['scale_cell_index']=-1
    if change=='instance_id':answer['instance_id']='wrong'
    row['answer']=json.dumps(answer,sort_keys=True,separators=(',',':'))
    with pytest.raises(ValueError):c.verify_rows('pantry',[row])


def test_native_audit_remains_required_after_structural_checks(monkeypatch):
    monkeypatch.setattr(c,'_qualification',lambda:(set(),set(),{}))
    marker=RuntimeError('native exact-support rejection')
    def native(*args):raise marker
    monkeypatch.setattr(c.original,'verify_rows',native)
    with pytest.raises(RuntimeError,match='native exact-support rejection'):
        c.verify_rows('pantry',[provider_fixture()])


def test_duplicate_rows_rejected_before_native_audit(monkeypatch):
    monkeypatch.setattr(c,'_qualification',lambda:(set(),set(),{}))
    monkeypatch.setattr(c.original,'verify_rows',lambda *args:pytest.fail('duplicate must not reach native audit'))
    row=provider_fixture()
    with pytest.raises(ValueError,match='reuses scratch or duplicate'):
        c.verify_rows('pantry',[row,copy.deepcopy(row)])


@pytest.mark.parametrize('change', ['support_digest','origin_difficulty','origin_legal','origin_feasible','origin_extra'])
def test_complete_native_support_and_origin_certificate_retained(monkeypatch,change):
    monkeypatch.setattr(c,'_qualification',lambda:(set(),set(),{}))
    monkeypatch.setattr(c.original,'verify_rows',lambda *args:pytest.fail('invalid certificate must not reach native audit'))
    row=provider_fixture();answer=json.loads(row['answer']);origin=json.loads(row['scale_origin_metadata'])
    if change=='support_digest':answer['certified_support_sha256']='0'*64
    if change=='origin_difficulty':origin['level3_difficulty']=1
    if change=='origin_legal':origin['level3_legal_allocation_count']+=1
    if change=='origin_feasible':origin['level3_feasible_allocation_count']+=1
    if change=='origin_extra':origin['unregistered']=True
    row['answer']=json.dumps(answer,sort_keys=True,separators=(',',':'))
    row['scale_origin_metadata']=json.dumps(origin,sort_keys=True,separators=(',',':'))
    with pytest.raises(ValueError,match='certificate or allocation metadata'):
        c.verify_rows('pantry',[row])
