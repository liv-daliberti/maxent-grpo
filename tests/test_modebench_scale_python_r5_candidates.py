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
import modebench_scale_python_r5_candidates as candidate


@pytest.fixture(autouse=True)
def no_native_generation_or_verifier(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('native generation/verifier is forbidden in pure r5 tests')
    monkeypatch.setattr(candidate,'certified_python_row',forbidden)
    monkeypatch.setattr(candidate.original,'verify_rows',forbidden)


def base(tier):return tuple(sorted((candidate.semiprimes(tier)[0],50,51,52)))


def test_exact_measured_ladder_four_profiles():
    assert candidate.PRIMES==(11,11,13,17)
    assert candidate.EXTRAS==((),(25,49),(25,49),(25,49))
    assert [p['cases'] for p in candidate.PROFILES['python_factors']]==[4,6,6,6]
    assert [p['semiprime_least_prime'] for p in candidate.PROFILES['python_factors']]==[11,11,13,17]
    assert 'monotone_by_construction' not in candidate.TIER_ORDERING
    assert 'measured_order' in candidate.TIER_ORDERING
    for tier,p in enumerate(candidate.PROFILES['python_factors']):
        assert p['base_cases']==4 and p['ordinary_cases']==3 and p['base_gcd']==1
        assert p['ordinary_minimum']==48 and p['ordinary_maximum']==1000
        assert p['ordinary_maximum_smallest_factor']==5 and p['ordinary_minimum_proper_divisors']==2
        assert p['appended_prime_squares']==list(candidate.EXTRAS[tier]) and p['appended_support_multiplier']==1
        assert p['semiprime_distinct_primes'] and p['semiprime_larger_prime_ratio_maximum']==3


def test_manifest_reports_this_law_identity_not_the_predecessor():
    m=candidate.law_manifest()
    assert m['schema']=='modebench_scale_python_r5_candidate_law_manifest_v1'
    assert 'nonmonotone' not in m['status']
    assert m['candidate_module']=='modebench_scale_python_r5_candidates'


def test_structural_facts_are_computed_and_true_for_this_ladder():
    P,E=candidate.PRIMES,candidate.EXTRAS
    m=candidate.law_manifest()
    assert m['hidden_prime_non_decreasing'] is True
    assert m['appended_sets_nested_within_equal_prime_runs'] is True
    assert m['one_lever_moves_per_step'] is True
    assert all(P[i]<=P[i+1] for i in range(len(P)-1))
    for i in range(len(P)-1):
        if P[i]==P[i+1]: assert set(E[i])<=set(E[i+1]) or set(E[i+1])<=set(E[i])


def _claims_a_guarantee(manifest):
    """Scan every string a manifest emits for phrasing that advertises a guarantee."""
    banned=('monotone_by_construction','guaranteed_by_construction')
    def walk(node,key=''):
        if isinstance(node,dict):return any(walk(v,k) for k,v in node.items())
        if isinstance(node,(list,tuple)):return any(walk(v,key) for v in node)
        if isinstance(node,str) and key not in ('why_not_guaranteed','difficulty_ordering_evidence'):
            return any(b in node for b in banned)
        return False
    return walk(manifest)


def test_no_difficulty_ordering_is_ever_claimed_from_construction():
    """Revision 4 asserted an ordering it never measured. Scratch pilot 31265211 then
    measured the appended-set lever running BACKWARDS at a fixed prime, so structural
    facts must never be sold as a difficulty guarantee."""
    m=candidate.law_manifest()
    assert m['difficulty_ordering_guaranteed_by_construction'] is False
    assert 'measured' in m['difficulty_ordering_evidence']
    assert not m['difficulty_match_claimed']
    assert not any(k for k in m if k=='monotone_by_construction')
    assert not _claims_a_guarantee(m), 'a manifest string still advertises a construction guarantee'


@pytest.mark.parametrize('primes,extras',[
    ((11,11),((25,49),(9,25,49))),
    ((11,11),((),(4,9,25,49))),
    ((11,11,13,17),((),(25,49),(25,49),(25,49))),
])
def test_no_ladder_however_shaped_reports_a_construction_guarantee(monkeypatch,primes,extras):
    """The first case is the author's own pilot-2 counterexample: growing the appended
    set at a fixed prime measured EASIER (pass1 0.2422 -> 0.3216), so any predicate
    that reported a guarantee here would be unsound."""
    monkeypatch.setattr(candidate,'PRIMES',primes)
    monkeypatch.setattr(candidate,'EXTRAS',extras)
    m=candidate.law_manifest()
    assert m['difficulty_ordering_guaranteed_by_construction'] is False
    # Not tautological: this also scans every string the manifest emits, including
    # tier_ordering, which previously advertised a guarantee for every ladder shape.
    assert not _claims_a_guarantee(m)


@pytest.mark.parametrize('i',range(3))
def test_no_step_moves_two_levers_or_zero_levers(i):
    P,E=candidate.PRIMES,candidate.EXTRAS
    assert not (P[i]<P[i+1] and set(E[i+1])<set(E[i])), i
    assert not (P[i]>P[i+1]), i
    assert (P[i]==P[i+1]) != (E[i]==E[i+1]), i


@pytest.mark.parametrize('bad',[6,10,15,26,121])
def test_a_non_square_or_multi_divisor_or_high_appendix_is_rejected(monkeypatch,bad):
    monkeypatch.setattr(candidate,'PRIMES',(11,11,13,17))
    monkeypatch.setattr(candidate,'EXTRAS',((),(bad,),(bad,),(bad,)))
    assert candidate.law_manifest()['appended_primes_strictly_below_hidden_prime'] is False


@pytest.mark.parametrize('good',[4,9,25,49])
def test_valid_unit_support_squares_below_the_hidden_prime_are_accepted(monkeypatch,good):
    monkeypatch.setattr(candidate,'PRIMES',(11,11,13,17))
    monkeypatch.setattr(candidate,'EXTRAS',((),(good,),(good,),(good,)))
    assert candidate.law_manifest()['appended_primes_strictly_below_hidden_prime'] is True


@pytest.mark.parametrize('tier',range(4))
def test_every_appended_prime_stays_strictly_below_the_hidden_prime(tier):
    """The revision-4 defect: an appended square at or above the hidden prime reveals it."""
    hidden=candidate.PRIMES[tier]
    for square in candidate.EXTRAS[tier]:
        root=round(square**.5)
        assert root*root==square
        assert len(candidate.proper_divisors(square))==1
        assert root<hidden, (tier,square,root,hidden)
    assert candidate.law_manifest()['appended_primes_strictly_below_hidden_prime'] is True


@pytest.mark.parametrize('tier',range(4))
def test_required_largest_prime_is_the_hidden_prime_and_is_never_telegraphed(tier):
    b=base(tier);cases=tuple(sorted(b+candidate.EXTRAS[tier]))
    required=max(min(candidate.proper_divisors(n)) for n in cases)
    assert required==candidate.PRIMES[tier]
    unit={n for n in cases if len(candidate.proper_divisors(n))==1}
    assert all(min(candidate.proper_divisors(n))<candidate.PRIMES[tier] for n in unit)


@pytest.mark.parametrize('tier',range(4))
def test_appended_squares_carry_unit_support_and_never_collide_with_the_catalog(tier):
    ordinary,_=candidate.catalog()
    assert set(candidate.EXTRAS[tier]).isdisjoint(ordinary)
    b=base(tier);cases=tuple(sorted(b+candidate.EXTRAS[tier]))
    assert len(cases)==4+len(candidate.EXTRAS[tier])<=8
    support=prod(len(candidate.proper_divisors(n)) for n in b)
    assert candidate.base_from_cases(cases,tier,support)==b
    assert prod(len(candidate.proper_divisors(n)) for n in cases)==support==64


@pytest.mark.parametrize('tier',range(4))
def test_semiprime_catalog_balanced_distinct_and_bounded(tier):
    hard=candidate.semiprimes(tier);assert hard
    for n in hard:
        d=candidate.proper_divisors(n)
        assert len(d)==2 and d[0]==candidate.PRIMES[tier] and d[0]<d[1]<=3*d[0] and n<=1000
    assert not set(hard)&set(candidate.catalog()[0])


@pytest.mark.parametrize('tier',range(4))
def test_exact_capacity_burns_the_base_projection(tier):
    b=base(tier);other=('python_factors',tuple(sorted(b+candidate.EXTRAS[tier])))
    before=candidate.capacity(tier,64)
    assert candidate.capacity(tier,64,{other})==before-1
    assert b in candidate.blocked_projections({other})


def test_every_tier_has_usable_capacity():
    caps=[candidate._raw_capacity(tier,64) for tier in range(4)]
    assert all(c>0 for c in caps)
    assert len({candidate.semiprimes(tier) for tier in range(4)})==3


@pytest.mark.parametrize('tier',[-1,4,True,1.0,'1'])
def test_only_four_integer_tier_ids_are_supported(tier):
    with pytest.raises(ValueError):candidate.semiprimes(tier)


@pytest.mark.parametrize('tier',range(4))
def test_pure_raw_proposals_are_deterministic_without_native_rows(tier):
    left=random.Random(candidate._seed(7205000,tier,64,'python_factors'))
    right=random.Random(candidate._seed(7205000,tier,64,'python_factors'))
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


def test_revision_four_predecessor_is_pinned_immutably():
    assert hashlib.sha256(candidate.PREDECESSOR_LAW_SOURCE.read_bytes()).hexdigest()==candidate.PREDECESSOR_LAW_SHA
    manifest=candidate.law_manifest()
    assert manifest['predecessor_law_sha256']==candidate.PREDECESSOR_LAW_SHA
    assert candidate.PREDECESSOR_LAW_SOURCE in candidate.source_paths()


def test_changed_predecessor_pin_refuses_manifest(monkeypatch):
    monkeypatch.setattr(candidate,'PREDECESSOR_LAW_SHA','a'*64)
    with pytest.raises(ValueError,match='predecessor'):candidate.law_manifest()


def test_prospective_manifest_pins_original_source_without_claiming_qualification():
    paths=candidate.source_paths();manifest=candidate.law_manifest()
    assert candidate.ORIGINAL_LAW_SOURCE in paths and Path(candidate.__file__).resolve() in paths
    assert manifest['files_sha256'][str(candidate.ORIGINAL_LAW_SOURCE)]==candidate.ORIGINAL_LAW_SHA
    assert manifest['profiles']==candidate.PROFILES['python_factors']
    assert manifest['hidden_semiprime_least_primes']==[11,11,13,17]
    assert manifest['appended_prime_squares_per_tier']==[[],[25,49],[25,49],[25,49]]
    assert not manifest['native_qualification_performed'] and not manifest['production_registration_performed']
    assert not manifest['difficulty_match_claimed']


def test_changed_original_source_pin_refuses_manifest(monkeypatch):
    monkeypatch.setattr(candidate,'ORIGINAL_LAW_SHA','a'*64)
    with pytest.raises(ValueError,match='immutable original'):candidate.law_manifest()


@pytest.mark.parametrize('kwargs',[{'domain':'other'}, {'seed':-1}, {'tier':True}, {'joint_target':Counter({1:1})}])
def test_invalid_build_request_rejected_without_native_generation(kwargs):
    args={'domain':'python_factors','target':Counter({64:1}),'excluded':set(),'seed':7205000,'tag':'SYNTHETIC_ONLY','tier':0}
    args.update(kwargs)
    with pytest.raises(ValueError):candidate.build_pool(**args)
