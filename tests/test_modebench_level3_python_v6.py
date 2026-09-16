"""Fixed proposal composition, exact capacities, external support and prefixes."""
from collections import Counter
from fractions import Fraction
from itertools import combinations, islice, product
import json
from math import comb, prod
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'ops/exp_scaling'),str(ROOT/'ops'),str(ROOT/'src')]
import modebench_level3_python_v6 as candidate
from make_python_factor_mode_data import _prompt, _certified_programs
from oat_drgrpo.python_modebench import proper_divisors, python_factor_mode_count
from oat_drgrpo.python_modebench_process import validate_python_factor_function_external


def cases(row):return tuple(json.loads(row['answer'])['cases'])


@pytest.mark.parametrize('difficulty',range(4))
def test_fixed_catalog_and_composition(difficulty):
    divisors,groups=candidate.catalog(difficulty)
    lower,upper=candidate.CASE_WINDOWS[difficulty]
    band_lower,band_upper=candidate.MINIMUM_BANDS[difficulty]
    expected={n for n in range(4,1001) if lower<=n<=upper and
              len(proper_divisors(n))>=2 and proper_divisors(n)[0]<=5}
    assert set(divisors)==expected
    assert set(divisors)=={n for values in groups.values() for n in values}
    for key,values in groups.items():
        assert all(key==(len(proper_divisors(n)),int(band_lower<=n<=band_upper)) for n in values)
    for support in (16,180,420,648,1800,2520,3600):
        selected=next(candidate.case_stream(support,set(),93171,difficulty))
        assert candidate.eligible_cases(selected,difficulty)
        assert python_factor_mode_count(selected)==support
        assert band_lower<=min(selected)<=band_upper
        assert all(lower<=n<=upper for n in selected)


@pytest.mark.parametrize('difficulty',range(4))
def test_quota_multiplier_prefix_and_exclusions(difficulty):
    seed=93172
    blocked={('python_factors',next(candidate.case_stream(32,set(),seed,difficulty)))}
    small=candidate.build_pool('python_factors',Counter({32:1}),blocked,seed,'test',difficulty,1)
    large=candidate.build_pool('python_factors',Counter({16:2,32:1}),blocked,seed,'test',difficulty,3)
    cell=sorted((row for row in large if row['answer_mode_count']==32),key=lambda r:r['level3_cell_index'])
    assert small==cell[:1]
    assert list(islice(candidate.case_stream(32,blocked,seed,difficulty),3))==list(map(cases,cell))
    assert not {('python_factors',cases(row)) for row in large}&blocked


@pytest.mark.parametrize('difficulty',range(4))
def test_exact_support_original_prompt_and_external_witnesses(difficulty):
    target=Counter({16:1,180:1,420:1,2520:1,3600:1})
    rows=candidate.build_pool('python_factors',target,set(),93173,'certify',difficulty,1)
    assert Counter(row['answer_mode_count'] for row in rows)==target
    for row in rows:
        selected=cases(row);spec=json.loads(row['answer'])
        assert len(set(selected))==len(selected)==4
        assert row['problem']==_prompt(selected)
        assert python_factor_mode_count(selected)==row['answer_mode_count']==spec['num_modes']
        assert row['level3_generator']=='modebench_level3_python_candidate_v6'
        values=[validate_python_factor_function_external(program,spec) for program in _certified_programs(selected)]
        assert all(values) and len({v.canonical_key for v in values})==2
        assert validate_python_factor_function_external('lambda n:2 if n%2==0 else 3 if n%3==0 else 5',spec) is not None


@pytest.mark.parametrize('difficulty',range(4))
def test_exact_profile_capacity_and_case_exclusion(difficulty):
    _,groups=candidate.catalog(difficulty)
    for support in (16,180,420,2520,3600):
        profiles,capacities=candidate.support_profiles(difficulty,support)
        expected=sum(prod(comb(len(groups[key]),count) for key,count in profile) for profile in profiles)
        assert all(sum(key[1]*count for key,count in p)>=1 for p in profiles)
        assert candidate.available_capacity(support,difficulty,set())==expected
        selected=next(candidate.case_stream(support,set(),93174,difficulty))
        assert candidate.available_capacity(support,difficulty,{('python_factors',selected)})==expected-1


def test_uniform_integer_tickets_over_composition_profiles():
    groups={(2,0):(6,10,14),(2,1):(15,21),(4,0):(12,18)}
    profiles=((((2,0),2),((2,1),2)),(((2,1),2),((4,0),2)))
    capacities=(3,1);probabilities=Counter()
    for ticket in range(4):
        profile=profiles[0 if ticket<3 else 1]
        options=[list(combinations(groups[key],count)) for key,count in profile]
        for choices in product(*options):
            class RNG:
                def __init__(self):self.choices=iter(choices)
                def randrange(self,stop):assert stop==4;return ticket
                def sample(self,population,count):
                    picked=next(self.choices);assert len(picked)==count and set(picked)<=set(population);return picked
            selected=candidate._proposal(RNG(),groups,profiles,capacities)
            probabilities[selected]+=Fraction(1,4*prod(map(len,options)))
    assert len(probabilities)==4 and set(probabilities.values())=={Fraction(1,4)}


@pytest.mark.parametrize('difficulty',[True,-1,4,1.0])
def test_invalid_difficulty_rejected(difficulty):
    with pytest.raises(ValueError):candidate.catalog(difficulty)


def test_capacity_and_rejection_exhaustion_fail_closed(monkeypatch):
    selected=next(candidate.case_stream(16,set(),93175,0))
    monkeypatch.setattr(candidate,'available_capacity',lambda *args:0)
    with pytest.raises(RuntimeError,match='requested 1'):
        candidate.build_pool('python_factors',Counter({16:1}),set(),93175,'no',0,1)
    monkeypatch.setattr(candidate,'available_capacity',lambda *args:1)
    monkeypatch.setattr(candidate,'_proposal',lambda *args:selected)
    monkeypatch.setattr(candidate,'MAX_PROPOSALS_PER_ROW',2)
    with pytest.raises(RuntimeError,match='sampling exhausted'):
        next(candidate.case_stream(16,{('python_factors',selected)},93175,0))


def test_materializer_original_witnesses_and_prompt_tamper():
    import materialize_modebench_level3_python_v6 as materializer
    rows=candidate.build_pool('python_factors',Counter({32:1}),set(),93176,'materializer',1,1)
    assert materializer.verify_witnesses(rows)==2
    rows[0]['problem']+=' changed'
    with pytest.raises(ValueError,match='original four-case prompt'):
        materializer.verify_witnesses(rows)


def test_materializer_support_and_certificate_tamper():
    import materialize_modebench_level3_python_v6 as materializer
    rows=candidate.build_pool('python_factors',Counter({32:1}),set(),93177,'materializer',3,1)
    rows[0]['answer_mode_count']=64
    with pytest.raises(ValueError,match='canonical product support'):
        materializer.verify_witnesses(rows)
    rows[0]['answer_mode_count']=32
    spec=json.loads(rows[0]['answer']);spec['certified_mode_key_sha256']='changed'
    rows[0]['answer']=json.dumps(spec)
    with pytest.raises(ValueError,match='canonical witness certificate'):
        materializer.verify_witnesses(rows)


def test_recorded_snapshot_allows_new_unrelated_data_but_rejects_mutation(tmp_path):
    import materialize_modebench_level3_python_v6 as materializer
    source=tmp_path/'source.json';source.write_text('{}')
    snapshot={'files_sha256':{str(source):materializer.file_sha(source)},'directory_files':{}}
    (tmp_path/'legitimate_future_dataset').mkdir()
    materializer.verify_snapshot(snapshot)
    source.write_text('changed')
    with pytest.raises(ValueError,match='authenticated source changed'):
        materializer.verify_snapshot(snapshot)


def test_generation_refuses_new_exclusion_inventory(tmp_path,monkeypatch):
    import materialize_modebench_level3_python_v6 as materializer
    snapshot={'files_sha256':{},'directory_files':{},'candidate_pool_paths':[],
              'historical_identity_sha256':materializer.materializer.row_hash([])}
    monkeypatch.setattr(materializer,'pool_paths',lambda:[tmp_path/'newpool.jsonl'])
    with pytest.raises(ValueError,match='pool inventory changed'):
        materializer.verify_generation_exclusions_unchanged(snapshot)


def test_registration_precedes_construction_and_failed_attempt_is_not_retried(tmp_path,monkeypatch):
    import materialize_modebench_level3_python_v6 as materializer
    registration=tmp_path/'candidate_protocol.json';pools=tmp_path/'pools'
    monkeypatch.setattr(materializer,'REGISTRATION',registration)
    monkeypatch.setattr(materializer,'POOL_ROOT',pools)
    monkeypatch.setattr(materializer,'source_snapshot',lambda:{})
    monkeypatch.setattr(materializer,'exclusions',lambda snapshot:(set(),0))
    monkeypatch.setattr(materializer,'capacity_table',lambda blocked:{})
    monkeypatch.setattr(materializer,'registration',lambda *args:{'prospective':True})
    def fail(blocked):
        assert json.loads(registration.read_text())=={'prospective':True}
        assert not pools.exists()
        raise RuntimeError('construction did not finish')
    monkeypatch.setattr(materializer,'construct',fail)
    monkeypatch.setattr(sys,'argv',['materialize','--materialize-development'])
    with pytest.raises(RuntimeError,match='did not finish'):materializer.main()
    with pytest.raises(FileExistsError,match='fresh Python v6'):materializer.main()


def test_registration_change_during_construction_is_rejected(tmp_path,monkeypatch):
    import materialize_modebench_level3_python_v6 as materializer
    registration=tmp_path/'candidate_protocol.json'
    monkeypatch.setattr(materializer,'REGISTRATION',registration)
    monkeypatch.setattr(materializer,'POOL_ROOT',tmp_path/'pools')
    monkeypatch.setattr(materializer,'source_snapshot',lambda:{})
    monkeypatch.setattr(materializer,'exclusions',lambda snapshot:(set(),0))
    monkeypatch.setattr(materializer,'capacity_table',lambda blocked:{})
    monkeypatch.setattr(materializer,'registration',lambda *args:{'prospective':True})
    def tamper(blocked):
        registration.write_text('{"altered":true}')
        return {},{},{}
    monkeypatch.setattr(materializer,'construct',tamper)
    monkeypatch.setattr(sys,'argv',['materialize','--materialize-development'])
    with pytest.raises(ValueError,match='registration changed during construction'):
        materializer.main()
    assert not (tmp_path/'pools').exists()


def test_minimum_band_is_a_joint_profile_condition_not_case_replacement(monkeypatch):
    # Exhaust a tiny catalogue and compare profile masses against direct case sets.
    divisors={60:(2,3),62:(2,31),80:(2,4),81:(3,9),82:(2,41),84:(2,4)}
    groups={(2,1):(60,62),(2,0):(80,81,82,84)}
    monkeypatch.setattr(candidate,'catalog',lambda difficulty:(divisors,groups))
    candidate.support_profiles.cache_clear()
    try:
        profiles,capacities=candidate.support_profiles(0,16)
        actual={tuple(sorted(values)) for values in combinations(divisors,4) if min(values)<=69}
        assert sum(capacities)==len(actual)==14
        assert all(any(key[1] for key,count in profile) for profile in profiles)
        assert candidate.available_capacity(16,0,set())==len(actual)
        excluded={('python_factors',next(iter(actual)))}
        assert candidate.available_capacity(16,0,excluded)==len(actual)-1
        # Every non-excluded case set is reached exactly once before exhaustion.
        selected=list(candidate.case_stream(16,excluded,93178,0))
        assert set(selected)==actual-{next(iter(actual))}
        assert len(selected)==13
    finally:candidate.support_profiles.cache_clear()


def test_all_reference_support_cells_have_capacity():
    import materialize_modebench_level3 as materializer
    targets={split:materializer.modes(materializer.reference_rows('python_factors',split))
             for split in materializer.SPLITS}
    total=sum(targets.values(),Counter())
    assert len(total)==60
    for difficulty in range(4):
        for support,count in total.items():
            assert candidate.available_capacity(support,difficulty,set())>=4*(count+targets['dev'][support])
