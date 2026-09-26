"""Pure prospective-law checks; no native row construction or verifier calls."""
import ast
from collections import Counter
import hashlib
import importlib.util
from math import prod
from pathlib import Path
import random
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
for rel in ('ops','ops/exp_scaling','src'):
    if str(ROOT/rel) not in sys.path:sys.path.insert(0,str(ROOT/rel))
import modebench_scale_python_r4_candidates as candidate


@pytest.fixture(autouse=True)
def no_native_generation_or_verifier(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('native generation/verifier is forbidden in pure r4 tests')
    monkeypatch.setattr(candidate,'certified_python_row',forbidden)
    monkeypatch.setattr(candidate.original,'verify_rows',forbidden)


def base(tier):return tuple(sorted((candidate.semiprimes(tier)[0],50,51,52)))


def test_exact_selected_nonmonotone_four_profiles():
    assert candidate.PRIMES==(7,11,7,11)
    assert candidate.EXTRAS==((121,),(49,),(49,121),(49,121))
    assert [p['cases'] for p in candidate.PROFILES['python_factors']]==[5,5,6,6]
    assert [p['semiprime_least_prime'] for p in candidate.PROFILES['python_factors']]==[7,11,7,11]
    assert 'nonmonotone' in candidate.TIER_ORDERING
    for tier,p in enumerate(candidate.PROFILES['python_factors']):
        assert p['base_cases']==4 and p['ordinary_cases']==3 and p['base_gcd']==1
        assert p['ordinary_minimum']==48 and p['ordinary_maximum']==1000
        assert p['ordinary_maximum_smallest_factor']==5 and p['ordinary_minimum_proper_divisors']==2
        assert p['appended_prime_squares']==list(candidate.EXTRAS[tier]) and p['appended_support_multiplier']==1
        assert p['semiprime_distinct_primes'] and p['semiprime_larger_prime_ratio_maximum']==3


@pytest.mark.parametrize('tier',range(4))
def test_semiprime_catalog_balanced_distinct_and_all_factors_at_most_31(tier):
    hard=candidate.semiprimes(tier);assert hard
    for n in hard:
        d=candidate.proper_divisors(n)
        assert len(d)==2 and d[0]==candidate.PRIMES[tier] and d[0]<d[1]<=3*d[0] and n<=1000
    assert not set(hard)&set(candidate.catalog()[0])


@pytest.mark.parametrize('tier',range(4))
def test_appended_squares_have_unit_support_and_no_catalog_collision(tier):
    ordinary,_=candidate.catalog()
    assert set(candidate.EXTRAS[tier]).isdisjoint(ordinary)
    assert all(len(candidate.proper_divisors(n))==1 for n in candidate.EXTRAS[tier])
    b=base(tier);cases=tuple(sorted(b+candidate.EXTRAS[tier]))
    support=prod(len(candidate.proper_divisors(n)) for n in b)
    assert candidate.base_from_cases(cases,tier,support)==b
    assert prod(len(candidate.proper_divisors(n)) for n in cases)==support==64


@pytest.mark.parametrize('tier',range(4))
def test_exact_capacity_burns_base_projection_across_both_case_count_variants(tier):
    partner=(2,3,0,1)[tier];b=base(tier)
    other=('python_factors',tuple(sorted(b+candidate.EXTRAS[partner])))
    before=candidate.capacity(tier,64)
    assert candidate.capacity(tier,64,{other})==before-1
    assert b in candidate.blocked_projections({other})
    assert candidate._raw_capacity(tier,64)==candidate._raw_capacity(partner,64)


@pytest.mark.parametrize('tier',[-1,4,True,1.0,'1'])
def test_only_four_integer_tier_ids_are_supported(tier):
    with pytest.raises(ValueError):candidate.semiprimes(tier)


@pytest.mark.parametrize('tier',range(4))
def test_pure_raw_proposals_are_deterministic_without_native_rows(tier):
    left=random.Random(candidate._seed(7203000,tier,64,'python_factors'))
    right=random.Random(candidate._seed(7203000,tier,64,'python_factors'))
    a=[candidate._proposal(left,tier,64) for _ in range(8)]
    b=[candidate._proposal(right,tier,64) for _ in range(8)]
    assert a==b
    for proposal in a:
        assert len(set(proposal))==4
        assert prod(len(candidate.proper_divisors(n)) for n in proposal)==64


def test_original_numerical_proposal_burn_and_native_functions_are_ast_identical():
    def functions(path):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}
    before=functions(candidate.ORIGINAL_LAW_SOURCE);after=functions(Path(candidate.__file__))
    assert set(after)-set(before)=={'law_manifest'}
    for name in before:
        if name!='source_paths':assert before[name]==after[name],name
    assert hashlib.sha256(candidate.ORIGINAL_LAW_SOURCE.read_bytes()).hexdigest()==candidate.ORIGINAL_LAW_SHA


def test_prospective_manifest_pins_original_source_without_claiming_qualification():
    paths=candidate.source_paths();manifest=candidate.law_manifest()
    assert candidate.ORIGINAL_LAW_SOURCE in paths and Path(candidate.__file__).resolve() in paths
    assert manifest['files_sha256'][str(candidate.ORIGINAL_LAW_SOURCE)]==candidate.ORIGINAL_LAW_SHA
    assert manifest['profiles']==candidate.PROFILES['python_factors']
    assert not manifest['native_qualification_performed'] and not manifest['production_registration_performed']
    assert not manifest['difficulty_match_claimed']


def test_changed_original_source_pin_refuses_manifest(monkeypatch):
    monkeypatch.setattr(candidate,'ORIGINAL_LAW_SHA','a'*64)
    with pytest.raises(ValueError,match='immutable original'):candidate.law_manifest()


@pytest.mark.parametrize('kwargs',[{'domain':'other'}, {'seed':-1}, {'tier':True}, {'joint_target':Counter({1:1})}])
def test_invalid_build_request_rejected_without_native_generation(kwargs):
    args={'domain':'python_factors','target':Counter({64:1}),'excluded':set(),'seed':7203000,'tag':'SYNTHETIC_ONLY','tier':0}
    args.update(kwargs)
    with pytest.raises(ValueError):candidate.build_pool(**args)
