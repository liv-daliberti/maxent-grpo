#!/usr/bin/env python3
"""Register isolated Python v5 laws before publishing four development pools.

Full 384/128/128 builds for every preset are ephemeral CPU capacity evidence.
Only the four 128-row development pools are published. No fitting, model calls,
confirmation outcome reads, or scheduler actions occur in this entry point.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from itertools import islice
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[2]
for directory in ('ops/exp_scaling','ops','src'):
    sys.path.insert(0,str(ROOT/directory))
import materialize_modebench_level3 as materializer
import modebench_level3_python_v5 as candidate
from fit_modebench_level3 import file_sha,local_dependency_sources
from evaluate_modebench_level3 import atomic_new
from make_python_factor_mode_data import _certified_programs,_prompt
from oat_drgrpo.python_modebench import parse_python_factor_spec,python_factor_mode_count
from oat_drgrpo.python_modebench_process import validate_python_factor_function_external

DOMAIN='python_factors'
ARTIFACTS=ROOT/'var/artifacts/modebench_level3_v2/python_v5'
REGISTRATION=ARTIFACTS/'candidate_protocol.json'
DIAGNOSTIC=ARTIFACTS/'development_diagnostic.json'
POOL_ROOT=ROOT/'var/data/modebench_level3_calibration_python_v5'
INHERITED_SEAL=ROOT/'var/artifacts/modebench_level3_v2/graph_v7/implementation_seal.json'
INHERITED_SEAL_SHA='a66a6e172bda3b294a8efba7ed3706f834b72f241cbae063bc28dcb72ba2fb32'
BASELINE=ROOT/'var/results/modebench_level3_v2/calibration_05b_python_factors.json'
BASELINE_SHA='d72b6c76ea0d3ccf7dafb5bdb0520031e0670780437e823da31f8bf1f385ac7b'
FAILED_FIT=ROOT/'var/artifacts/modebench_level3_v2/recipes/python_factors.json'
FAILED_FIT_SHA='77f681f5efd3ca3b3d7534183111f675c9d11e3ff5a2bc116a5688a4f8374b62'
DEVELOPMENT_SEED=8337100
CAPACITY_SEED=8437100


def read(path):return json.loads(Path(path).read_text())


def pool_paths():
    roots=set((ROOT/'var/data').glob('modebench_harder*'))|set((ROOT/'var/data').glob('modebench_level3*'))
    return sorted(path.resolve() for root in roots if root.resolve()!=POOL_ROOT.resolve()
                  for path in (root/'pools'/DOMAIN).glob('*.jsonl'))


def verify_snapshot(snapshot):
    """Authenticate recorded files only; later legitimate datasets may exist."""
    for name,expected in snapshot['files_sha256'].items():
        if file_sha(name)!=expected:raise ValueError(f'authenticated source changed: {name}')
    for directory,expected in snapshot['directory_files'].items():
        actual=sorted(str(path.resolve()) for path in Path(directory).rglob('*') if path.is_file())
        if actual!=expected:raise ValueError(f'authenticated source inventory changed: {directory}')


def verify_generation_exclusions_unchanged(snapshot):
    verify_snapshot(snapshot)
    if list(map(str,pool_paths()))!=snapshot['candidate_pool_paths']:
        raise ValueError('candidate pool inventory changed during generation')
    actual=materializer.row_hash(sorted(materializer.historical_ids(DOMAIN),key=repr))
    if actual!=snapshot['historical_identity_sha256']:
        raise ValueError('historical semantic identities changed during generation')


def source_snapshot():
    if file_sha(INHERITED_SEAL)!=INHERITED_SEAL_SHA:raise ValueError('inherited Graph v7 seal changed')
    seal=read(INHERITED_SEAL)
    if len(seal['files_sha256'])!=503:raise ValueError('inherited 503-file inventory changed')
    pins=dict(seal['files_sha256']);trees=dict(seal['directory_files'])
    def pin(path,expected=None):
        name=str(Path(path).resolve());actual=file_sha(name)
        if expected is not None and actual!=expected:raise ValueError(f'fixed development evidence changed: {name}')
        if name in pins and pins[name]!=actual:raise ValueError(f'inherited source changed: {name}')
        pins[name]=actual
    pin(INHERITED_SEAL,INHERITED_SEAL_SHA);pin(BASELINE,BASELINE_SHA);pin(FAILED_FIT,FAILED_FIT_SHA);pin(DIAGNOSTIC)
    failed=read(FAILED_FIT)
    if failed['decision']!='development_fit_failed_revise_candidates' or failed['domain']!=DOMAIN:
        raise ValueError('fixed failed development recipe differs')
    for tier,record in failed['provenance']['pools'].items():
        expected=ROOT/f'var/results/modebench_level3_v2/calibration_3b_python_factors_d{tier}.json'
        if Path(record['receipt_path'])!=expected:raise ValueError('unexpected development receipt')
        pin(expected,record['receipt_sha256'])
    roots=set((ROOT/'var/data').glob('modebench_harder*'))|set((ROOT/'var/data').glob('modebench_level3*'))
    for root in roots:
        for split in materializer.SPLITS:
            directory=root/DOMAIN/split
            if (directory/'dataset_dict.json').is_file():
                trees[str(directory.resolve())]=sorted(str(path.resolve()) for path in directory.rglob('*') if path.is_file())
    for path in pool_paths():
        pin(path)
        certificate=path.with_suffix('.identity.json')
        if certificate.is_file():pin(certificate)
    for files in trees.values():
        for path in files:pin(path)
    for path,expected in local_dependency_sources([Path(candidate.__file__),Path(__file__)]).items():pin(ROOT/path,expected)
    pin(ROOT/'tests/test_modebench_level3_python_v5.py')
    snapshot={'files_sha256':pins,'directory_files':trees,'candidate_pool_paths':list(map(str,pool_paths())),
              'historical_identity_sha256':materializer.row_hash(sorted(materializer.historical_ids(DOMAIN),key=repr))}
    verify_generation_exclusions_unchanged(snapshot)
    return snapshot


def exclusions(snapshot):
    blocked=materializer.historical_ids(DOMAIN);historical=len(blocked)
    for path in snapshot['candidate_pool_paths']:
        blocked|=materializer.identity_set(DOMAIN,[json.loads(line) for line in Path(path).read_text().splitlines()])
    return blocked,historical


def verify_witnesses(rows):
    """Return two original external grader witnesses per exact four-case row."""
    total=0
    for row in rows:
        spec=json.loads(row['answer']);cases=tuple(spec['cases']);tier=row['level3_difficulty']
        if (type(tier) is not int or len(cases)!=len(set(cases)) or len(cases)!=4
                or parse_python_factor_spec(spec)!=cases or row['problem']!=_prompt(cases)
                or row.get('level3_generator')!=candidate.GENERATOR or not candidate.eligible_cases(cases,tier)):
            raise ValueError('original four-case prompt or fixed Python v5 composition differs')
        modes=python_factor_mode_count(cases)
        if modes!=row['answer_mode_count'] or modes!=spec['num_modes']:
            raise ValueError('original canonical product support differs')
        values=[validate_python_factor_function_external(program,spec) for program in _certified_programs(cases)]
        if any(value is None for value in values):raise ValueError('original external factor witness failed')
        keys={value.canonical_key for value in values}
        expected=hashlib.sha256('\n'.join(sorted(keys)).encode()).hexdigest()
        if len(keys)!=2 or spec['num_externally_certified_modes']!=2 or spec['certified_mode_key_sha256']!=expected:
            raise ValueError('original external canonical witness certificate differs')
        total+=len(keys)
    return total


def capacity_table(blocked):
    targets={split:materializer.modes(materializer.reference_rows(DOMAIN,split)) for split in materializer.SPLITS}
    total=sum(targets.values(),Counter());result={}
    # Even if every other preset consumed the same support cells, this bound
    # suffices for all four globally disjoint 640-row capacity constructions.
    for tier in candidate.PRESETS:
        result[str(tier)]={}
        for support,count in sorted(total.items()):
            available=candidate.available_capacity(support,tier,blocked)
            demand=4*count+4*targets['dev'][support]
            if available<demand:raise ValueError(f'capacity insufficient: tier{tier}/support{support}: {available}<{demand}')
            result[str(tier)][str(support)]={'raw':candidate.available_capacity(support,tier,set()),
                'remaining':available,'conservative_four_preset_demand':demand,
                'split_counts':{split:target[support] for split,target in targets.items()}}
    return result


def registration(snapshot,blocked,historical,capacity):
    for tier in candidate.PRESETS:
        output=ROOT/f'var/results/modebench_level3_v2/calibration_3b_python_v5_d{tier}.json'
        if output.exists() or Path(str(output)+'.batches').exists():raise ValueError('candidate model work precedes registration')
    return {'schema':'modebench_level3_python_candidate_protocol_v5','candidate_revision':'python_v5',
        'created_at':datetime.now(timezone.utc).isoformat(),'development_only':True,
        'generator_path':str(Path(candidate.__file__).resolve()),'generator_source_sha256':file_sha(candidate.__file__),
        'materializer_path':str(Path(__file__).resolve()),'materializer_source_sha256':file_sha(__file__),
        'pool_root':str(POOL_ROOT),'development_draw_labels':[6328000,6328001,6328002,6328003],
        'baseline_receipt_path':str(BASELINE),'baseline_receipt_sha256':BASELINE_SHA,
        'failed_development_fit_path':str(FAILED_FIT),'failed_development_fit_sha256':FAILED_FIT_SHA,
        'development_diagnostic_path':str(DIAGNOSTIC),'development_diagnostic_sha256':file_sha(DIAGNOSTIC),
        'inherited_seal_path':str(INHERITED_SEAL),'inherited_seal_sha256':INHERITED_SEAL_SHA,
        'rationale':'Corrected full-pool and selected-set pass1 gates failed. The small-window exactly-two-odd development stratum has lower pass1 with pass8 near target, including within observed support cells. Test two fixed parity windows, retain the small-factor anchor, and add one factor7 case to test the dominant repeated 2/3/5 conditional shortcut. Broad finite conditional chains remain legal. Development-derived hypotheses only.',
        'presets':candidate.PRESETS,'soft_windows':candidate.SOFT_WINDOWS,'hard_factors':candidate.HARD_FACTORS,
        'marked_case_counts':candidate.MARKED_CASE_COUNTS,'minimum_proper_divisors_per_case':2,
        'exact_case_count':4,'maximum_case_value':1000,
        'proposal_laws':{'0':'Uniform four distinct cases in48..192, each smallest factor<=5 and at least2 proper divisors.',
          '1':'Same fixed48..192 catalogue, conditioned on exactly2odd and2even cases.',
          '2':'Fixed48..384 small-factor catalogue, conditioned on exactly2odd and2even cases.',
          '3':'Exactly1 case in4..1000 with smallest proper factor7 and at least2 proper divisors, plus3 distinct cases in48..384 with smallest factor<=5 and at least2 proper divisors.'},
        'uniformity':'Integer profile capacity tickets over joint(divisor_count,composition_class), then uniform within-class samples; every eligible unordered case set has equal mass conditional on exact support.',
        'development_seed_base':DEVELOPMENT_SEED,'development_seed_rule':'base+1000*difficulty',
        'per_cell_stream_seed_rule':'SHA256(profile,seed,difficulty,support)',
        'quota_prefix_stability':'Fixed per-support RNG streams and identity exclusion only; no quota-conditioned catalog, fallback, outcome selection or seed retries.',
        'exclusions':{'historical_identities':historical,'historical_and_candidate_identities':len(blocked),
          'include_all_prior_candidates':True,'cross_new_pool_disjointness':True,
          'semantic_identity':'original python_factors tuple of four sorted cases'},
        'exact_capacity_before_new_pools':capacity,'source_snapshot':snapshot,
        'information_boundary':{'candidate_model_outcomes_exist':False,'confirmation_model_outcomes_used':False,
          'corrected_development_used_for_structural_hypothesis':True,'fitting_or_gpu_submission_performed_here':False,
          'treatment_training_started':False}}


def construct(blocked):
    pools,records={},{};current=set(blocked)
    reference=materializer.reference_rows(DOMAIN,'dev');target=materializer.modes(reference)
    for tier in candidate.PRESETS:
        seed=DEVELOPMENT_SEED+1000*tier
        rows=candidate.build_pool(DOMAIN,target,current,seed,'level3_development_pool',tier,1)
        checks=materializer.verify_rows(DOMAIN,rows,reference,target,current)
        witnesses=verify_witnesses(rows)
        checks.update(original_canonical_support_verified=True,exactly_four_distinct_cases=True)
        pools[tier]=rows;records[tier]={'schema':'modebench_level3_development_pool_v1','domain':DOMAIN,
            'difficulty':tier,'seed':seed,'rows':len(rows),'rows_sha256':materializer.row_hash(rows),
            'checks':checks,'support_histogram':dict(sorted(target.items())),
            'source_sha256':file_sha(candidate.__file__),'candidate_revision':'python_v5',
            'original_grader_witnesses':witnesses,
            'information_boundary':'development candidates only; no confirmation model outcomes used'}
        current|=materializer.identity_set(DOMAIN,rows)
        print(json.dumps({'development_difficulty':tier,'rows':len(rows),'witnesses':witnesses}),flush=True)
    after_pools=capacity_table(current);runs=[]
    for tier in candidate.PRESETS:
        for split_index,split in enumerate(materializer.SPLITS):
            reference=materializer.reference_rows(DOMAIN,split);target=materializer.modes(reference)
            seed=CAPACITY_SEED+1000*tier+10000*split_index
            rows=candidate.build_pool(DOMAIN,target,current,seed,'python_v5_capacity_'+split,tier,1)
            checks=materializer.verify_rows(DOMAIN,rows,reference,target,current)
            for support,count in target.items():
                prefix=list(islice(candidate.case_stream(support,current,seed,tier),count+1))
                cell=sorted((row for row in rows if row['answer_mode_count']==support),key=lambda row:row['level3_cell_index'])
                if [tuple(json.loads(row['answer'])['cases']) for row in cell]!=prefix[:count]:raise ValueError('quota prefix differs')
            witnesses=verify_witnesses(rows);current|=materializer.identity_set(DOMAIN,rows)
            runs.append({'difficulty':tier,'split':split,'seed':seed,'rows':len(rows),'rows_sha256':materializer.row_hash(rows),
                'support_histogram':dict(sorted(target.items())),'original_grader_witnesses':witnesses,
                'checks':{**checks,'per_cell_quota_prefix_invariance':True}})
            print(json.dumps({'capacity_difficulty':tier,'split':split,'rows':len(rows),'witnesses':witnesses}),flush=True)
    return pools,records,{'exact_capacity_after_all_four_dev_pools':after_pools,'capacity_runs':runs,
        'capacity_rows_verified':sum(run['rows'] for run in runs),
        'capacity_original_grader_witnesses':sum(run['original_grader_witnesses'] for run in runs),
        'all_capacity_rows_globally_disjoint_from_history_candidates_and_each_other':True,
        'capacity_rows_are_ephemeral_structural_witnesses_not_confirmation_data':True}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--materialize-development',action='store_true');args=parser.parse_args()
    if REGISTRATION.exists() or POOL_ROOT.exists():raise FileExistsError('fresh Python v5 registration and pool root required')
    snapshot=source_snapshot();blocked,historical=exclusions(snapshot);capacity=capacity_table(blocked)
    prospective=registration(snapshot,blocked,historical,capacity)
    registration_sha=None
    if args.materialize_development:
        atomic_new(REGISTRATION,prospective)
        registration_sha=file_sha(REGISTRATION)
    pools,records,audit=construct(blocked)
    if args.materialize_development and file_sha(REGISTRATION)!=registration_sha:
        raise ValueError('registration changed during construction')
    verify_generation_exclusions_unchanged(snapshot)
    if not args.materialize_development:
        print(json.dumps({'status':'structural_preview_pass','development_rows':512,**audit}));return
    staging=Path(tempfile.mkdtemp(prefix='.'+POOL_ROOT.name+'.',dir=POOL_ROOT.parent))
    try:
        folder=staging/'pools'/DOMAIN;folder.mkdir(parents=True)
        for tier,rows in pools.items():
            path=folder/f'difficulty_{tier}.jsonl'
            with path.open('x') as handle:
                for row in rows:handle.write(json.dumps(row,sort_keys=True)+'\n')
            records[tier].update(candidate_protocol_path=str(REGISTRATION),candidate_protocol_sha256=registration_sha)
            atomic_new(path.with_suffix('.identity.json'),records[tier])
        verify_generation_exclusions_unchanged(snapshot)
        if file_sha(REGISTRATION)!=registration_sha:raise ValueError('registration changed during generation')
        staging.rename(POOL_ROOT)
    except BaseException:
        shutil.rmtree(staging,ignore_errors=True);raise
    report={'schema':'modebench_level3_python_v5_structural_audit_v1','registration_path':str(REGISTRATION),
        'registration_sha256':registration_sha,'generator_source_sha256':file_sha(candidate.__file__),
        'pool_root':str(POOL_ROOT),'development_pools':records,**audit,'inherited_sealed_files_unchanged':503,
        'confirmation_outcomes_loaded':False,'fitting_performed':False,'gpu_submission_performed':False,
        'pool_files_sha256':{str(path):file_sha(path) for path in sorted(POOL_ROOT.rglob('*')) if path.is_file()}}
    atomic_new(ARTIFACTS/'structural_audit.json',report)
    print(json.dumps({'status':'registered_and_materialized','registration_sha256':registration_sha,
                      'development_rows':512,'capacity_rows_verified':audit['capacity_rows_verified']}),flush=True)


if __name__=='__main__':main()
