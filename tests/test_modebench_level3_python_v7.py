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
import modebench_level3_python_v7 as candidate
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
        assert row['level3_generator']=='modebench_level3_python_candidate_v7'
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
    import materialize_modebench_level3_python_v7 as materializer
    rows=candidate.build_pool('python_factors',Counter({32:1}),set(),93176,'materializer',1,1)
    assert materializer.verify_witnesses(rows)==2
    rows[0]['problem']+=' changed'
    with pytest.raises(ValueError,match='original four-case prompt'):
        materializer.verify_witnesses(rows)


def test_materializer_support_and_certificate_tamper():
    import materialize_modebench_level3_python_v7 as materializer
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
    import materialize_modebench_level3_python_v7 as materializer
    source=tmp_path/'source.json';source.write_text('{}')
    snapshot={'files_sha256':{str(source):materializer.file_sha(source)},'directory_files':{}}
    (tmp_path/'legitimate_future_dataset').mkdir()
    materializer.verify_snapshot(snapshot)
    source.write_text('changed')
    with pytest.raises(ValueError,match='authenticated source changed'):
        materializer.verify_snapshot(snapshot)


def test_generation_refuses_new_exclusion_inventory(tmp_path,monkeypatch):
    import materialize_modebench_level3_python_v7 as materializer
    snapshot={'files_sha256':{},'directory_files':{},'candidate_pool_paths':[],
              'historical_identity_sha256':materializer.materializer.row_hash([])}
    monkeypatch.setattr(materializer,'pool_paths',lambda:[tmp_path/'newpool.jsonl'])
    with pytest.raises(ValueError,match='pool inventory changed'):
        materializer.verify_generation_exclusions_unchanged(snapshot)


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


def test_exact_new_laws_do_not_alias_prior_revision():
    assert candidate.CASE_WINDOWS==((60,224),(60,256),(60,224),(60,256))
    assert candidate.MINIMUM_BANDS==((60,69),(60,69),(60,79),(60,89))
    assert candidate.GENERATOR=='modebench_level3_python_candidate_v7'
    assert candidate.PROFILE=='python_minimum_case_bands_v7'
    assert candidate.MAX_SMALLEST_FACTOR==5 and candidate.MIN_DIVISORS==2


def test_calibration_support_envelope_has_every_eval_and_dev_cell():
    import materialize_modebench_level3 as base
    targets={split:base.modes(base.reference_rows('python_factors',split)) for split in ('dev','eval')}
    envelope=Counter({support:max(targets['dev'][support],targets['eval'][support]) for support in targets['dev']|targets['eval']})
    assert sum(envelope.values())==166 and len(envelope)==43
    assert all(envelope[s]>=n for hist in targets.values() for s,n in hist.items())
    for difficulty in range(4):
        rows=candidate.build_pool('python_factors',envelope,set(),8837100+1000*difficulty,'new_calibration',difficulty,1)
        assert Counter(row['answer_mode_count'] for row in rows)==envelope
        assert len(rows)==166


@pytest.mark.parametrize('field,value', [
    ('level3_generation_profile','python_minimum_case_bands_v6'),
    ('level3_python_preset','other'),('level3_case_window',[60,384]),
    ('level3_minimum_case_band',[60,79]),('level3_maximum_smallest_factor',7)])
def test_materializer_rejects_misreported_new_law_metadata(field,value):
    import materialize_modebench_level3_python_v7 as materializer
    rows=candidate.build_pool('python_factors',Counter({32:1}),set(),93191,'metadata',0,1)
    rows[0][field]=value
    with pytest.raises(ValueError,match='row metadata differs'):
        materializer.verify_witnesses(rows)


def test_descriptor_matches_new_common_seed_and_quota_contract():
    import materialize_modebench_level3_python_v7 as materializer
    record=materializer.candidate_description({})
    assert record['name']=='python_v7' and record['rows_per_tier']==166
    assert record['development_seed_base']==8837100 and record['capacity_seed_base']==9037100
    assert record['final_train_seed_base']==9237100 and record['final_eval_seed_base']==9437100
    assert sum(record['calibration_histogram'].values())==166
    assert len(record['calibration_histogram'])==43
    assert record['generator_sha256']==materializer.file_sha(candidate.__file__)
    assert record['materializer_sha256']==materializer.file_sha(materializer.__file__)
    assert set(record['development_receipts'])=={'0','1','2','3'}
    assert all(f'calibration_3b_python_v7_d{tier}.json' in path
               for tier,path in record['development_receipts'].items())


def test_materializer_requires_explicit_registration_before_model_or_generation(tmp_path,monkeypatch):
    import materialize_modebench_level3_python_v7 as materializer
    monkeypatch.setattr(materializer,'POOL_ROOT',tmp_path/'pools')
    monkeypatch.setattr(materializer,'ARTIFACTS',tmp_path/'artifacts')
    def never(*args):raise AssertionError('registration barrier was bypassed')
    monkeypatch.setattr(materializer,'source_snapshot',never)
    monkeypatch.setattr(materializer,'construct',never)
    with pytest.raises(ValueError,match='explicit registration hash'):
        materializer.main(['--materialize-development'])
    assert not (tmp_path/'pools').exists()


@pytest.mark.parametrize('mutation',['name','generator_sha256','materializer_sha256','rows_per_tier','development_receipts','source_pin','tree_pin'])
def test_registered_candidate_descriptor_and_complete_snapshot_must_match(monkeypatch,mutation):
    import copy
    import materialize_modebench_level3_python_v7 as materializer
    snapshot={'files_sha256':{'/fixed':'abc'},'directory_files':{'/tree':['/fixed']}}
    expected={'name':'python_v7','generator_sha256':'gen','materializer_sha256':'mat',
              'rows_per_tier':166,'development_receipts':{'0':'fixed-output'},'source_snapshot':snapshot}
    record={'candidate_revisions':{'python_factors':copy.deepcopy(expected)},
            'files_sha256':dict(snapshot['files_sha256']),'directory_files':dict(snapshot['directory_files'])}
    if mutation=='source_pin':record['files_sha256']['/fixed']='changed'
    elif mutation=='tree_pin':record['directory_files']['/tree']=[]
    else:record['candidate_revisions']['python_factors'][mutation]='changed'
    seen=[]
    def validate(path,pin):
        assert path==materializer.REGISTRATION and pin=='explicit-caller-pin'
        seen.append(True);return record
    monkeypatch.setattr(materializer.common,'validate_registration',validate)
    monkeypatch.setattr(materializer,'candidate_description',lambda value:expected)
    monkeypatch.setattr(materializer,'verify_generation_exclusions_unchanged',lambda value:None)
    with pytest.raises(ValueError):materializer.authenticated_registration('explicit-caller-pin')
    assert seen==[True]


def test_existing_pools_forbid_any_reconstruction(tmp_path,monkeypatch):
    import materialize_modebench_level3_python_v7 as materializer
    root=tmp_path/'pools';root.mkdir()
    monkeypatch.setattr(materializer,'POOL_ROOT',root)
    def never(*args):raise AssertionError('existing publication was reused')
    monkeypatch.setattr(materializer,'source_snapshot',never)
    with pytest.raises(ValueError,match='fresh Python v7 pool root'):
        materializer.main([])


def test_candidate_model_work_forbids_materialization(tmp_path,monkeypatch):
    import materialize_modebench_level3_python_v7 as materializer
    monkeypatch.setattr(materializer,'POOL_ROOT',tmp_path/'pools')
    monkeypatch.setattr(materializer,'ARTIFACTS',tmp_path/'artifacts')
    monkeypatch.setattr(materializer.common,'RESULTS',tmp_path)
    (tmp_path/'calibration_3b_python_v7_d2.json.batches').mkdir()
    monkeypatch.setattr(materializer,'source_snapshot',lambda:{})
    def never(*args):raise AssertionError('model-preceded generation')
    monkeypatch.setattr(materializer,'construct',never)
    with pytest.raises(ValueError,match='model work must not precede'):
        materializer.main([])


def test_changed_registration_after_construction_refuses_publication(tmp_path,monkeypatch):
    import materialize_modebench_level3_python_v7 as materializer
    monkeypatch.setattr(materializer,'POOL_ROOT',tmp_path/'pools')
    monkeypatch.setattr(materializer,'ARTIFACTS',tmp_path/'artifacts')
    monkeypatch.setattr(materializer.common,'RESULTS',tmp_path/'results')
    calls=[]
    def authenticate(pin):
        calls.append(pin)
        if len(calls)>1:raise ValueError('changed registration')
        return {},{}
    monkeypatch.setattr(materializer,'authenticated_registration',authenticate)
    monkeypatch.setattr(materializer,'exclusions',lambda value:(set(),0))
    monkeypatch.setattr(materializer,'capacity_table',lambda value:{})
    monkeypatch.setattr(materializer,'construct',lambda value:({},{},{}))
    monkeypatch.setattr(materializer,'verify_generation_exclusions_unchanged',lambda value:None)
    with pytest.raises(ValueError,match='changed registration'):
        materializer.main(['--materialize-development','--registration-sha256','pin'])
    assert calls==['pin','pin'] and not (tmp_path/'pools').exists()


def test_diagnostic_closure_pins_historical_snapshot_bytes_without_reinterpreting_superseded_pins(tmp_path,monkeypatch):
    import materialize_modebench_level3_python_v7 as materializer
    folder=tmp_path/'diagnostics';folder.mkdir()
    historical=tmp_path/'historical_snapshot.json'
    historical.write_text(json.dumps({'files_sha256':{str(tmp_path/'superseded_source.py'):'0'*64}}))
    leaf=folder/'features.json'
    leaf.write_text(json.dumps({'files_sha256':{str(historical):materializer.file_sha(historical)}}))
    diagnostic=folder/'diagnosis.json'
    diagnostic.write_text(json.dumps({'sources_sha256':{str(leaf):materializer.file_sha(leaf)}}))
    capacity=folder/'capacity.json';capacity.write_text('{}')
    monkeypatch.setattr(materializer,'DIAGNOSTIC',diagnostic)
    monkeypatch.setattr(materializer,'DIAGNOSTIC_SHA',materializer.file_sha(diagnostic))
    monkeypatch.setattr(materializer,'CAPACITY_EVIDENCE',capacity)
    monkeypatch.setattr(materializer,'CAPACITY_EVIDENCE_SHA',materializer.file_sha(capacity))
    pins=materializer.diagnostic_source_pins()
    assert set(pins)==set(map(str,(historical,leaf,diagnostic,capacity)))
    historical.write_text('changed')
    with pytest.raises(ValueError,match='diagnostic input changed'):
        materializer.diagnostic_source_pins()
