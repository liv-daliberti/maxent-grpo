#!/usr/bin/env python3
"""Build four fresh Python v7 DEV pools under explicit v3 registration.

Calibration covers the componentwise maximum DEV/EVAL support histogram
(166 rows per tier). Full 384/128/128 constructions remain ephemeral CPU
capacity witnesses. The root owns registration; this module never fits,
samples a model, launches a scheduler job, or creates confirmation outcomes.
"""
from __future__ import annotations
from collections import Counter
from datetime import datetime,timezone
from itertools import islice
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[2]
for directory in ('ops/exp_scaling','ops','src'):
    if str(ROOT/directory) not in sys.path:sys.path.insert(0,str(ROOT/directory))
import materialize_modebench_level3 as materializer
import modebench_level3_python_v7 as candidate
import modebench_level3_v3_common as common
from fit_modebench_level3 import file_sha,local_dependency_sources
from make_python_factor_mode_data import _certified_programs,_prompt
from oat_drgrpo.python_modebench import parse_python_factor_spec,python_factor_mode_count
from oat_drgrpo.python_modebench_process import validate_python_factor_function_external

DOMAIN='python_factors'
REVISION='python_v7'
ARTIFACTS=ROOT/'var/artifacts/modebench_level3_v3/python_v7'
REGISTRATION=common.REGISTRATION
POOL_ROOT=ROOT/'var/data/modebench_level3_calibration_python_v7'
DIAGNOSTIC=ROOT/'var/artifacts/modebench_level3_v2/confirmation_v6_failure_diagnostics/completed_failure_diagnosis.json'
DIAGNOSTIC_SHA='9792e20dbd7b024af6561b3f271060dcf023485f43856e0361e31024603d6edd'
CAPACITY_EVIDENCE=DIAGNOSTIC.with_name('python_v7_capacity_check.json')
CAPACITY_EVIDENCE_SHA='ef5ff4130c6803808952763c6433602605ccc9078575c2864b0e071e98f1081d'
TEST_SOURCE=ROOT/'tests/test_modebench_level3_python_v7.py'
DEVELOPMENT_SEED=8837100
CAPACITY_SEED=9037100
TRAIN_SEED=9237100
EVAL_SEED=9437100
ROWS_PER_TIER=166


def read(path):return json.loads(Path(path).read_text())


def support_histogram(cells):
    common.require(all(isinstance(cell,tuple) and len(cell)==1 and type(cell[0]) is int
                       and type(count) is int and count>=0 for cell,count in cells.items()),
                   'Python support cells must be one exact integer')
    return Counter({cell[0]:count for cell,count in cells.items()})


def reference_histograms():
    return {split:support_histogram(cells) for split,cells in common.reference_histograms(DOMAIN).items()}


def calibration_histogram():
    result=support_histogram(common.calibration_histogram(DOMAIN))
    common.require(sum(result.values())==ROWS_PER_TIER and len(result)==43,
                   'Python v7 requires166 calibration rows covering43 support cells')
    return result


def pool_paths():
    roots=set((ROOT/'var/data').glob('modebench_harder*'))|set((ROOT/'var/data').glob('modebench_level3*'))
    return sorted(path.resolve() for root in roots if root.resolve()!=POOL_ROOT.resolve()
                  for path in (root/'pools'/DOMAIN).glob('*.jsonl'))


def verify_snapshot(snapshot):
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


def diagnostic_source_pins():
    """Follow only this new diagnostic manifest family; pin raw inputs by bytes.

    Historical snapshots can contain superseded implementation hashes. Their
    current meaning is checked by authenticate_inherited(), not by recursively
    treating every JSON object as an authoritative current-file inventory.
    """
    result={};visited=set()
    def visit(path,expected):
        path=Path(path).resolve();name=str(path);actual=file_sha(path)
        common.require(actual==expected and (name not in result or result[name]==actual),
                       'completed diagnostic input changed: '+name)
        result[name]=actual
        if name in visited or path.suffix!='.json' or not path.is_relative_to(DIAGNOSTIC.parent):return
        visited.add(name);record=read(path)
        for key in ('sources_sha256','sources','files_sha256','inputs_files_sha256'):
            values=record.get(key,{}) if isinstance(record,dict) else {}
            if isinstance(values,dict):
                for child,sha in values.items():
                    if isinstance(child,str) and child.startswith('/') and isinstance(sha,str) and len(sha)==64:
                        visit(child,sha)
    visit(DIAGNOSTIC,DIAGNOSTIC_SHA)
    visit(CAPACITY_EVIDENCE,CAPACITY_EVIDENCE_SHA)
    return result


def source_snapshot():
    inherited=common.authenticate_inherited()
    pins=dict(inherited['files_sha256']);trees=dict(inherited['directory_files'])
    def pin(path,expected=None):
        name=str(Path(path).resolve());actual=file_sha(name)
        if expected is not None and actual!=expected:raise ValueError(f'fixed evidence changed: {name}')
        if name in pins and pins[name]!=actual:raise ValueError(f'inherited source changed: {name}')
        pins[name]=actual
    pin(DIAGNOSTIC,DIAGNOSTIC_SHA);pin(CAPACITY_EVIDENCE,CAPACITY_EVIDENCE_SHA)
    for path,expected in diagnostic_source_pins().items():pin(path,expected)
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
    for path,expected in local_dependency_sources([Path(candidate.__file__),Path(__file__),Path(common.__file__)]).items():
        pin(ROOT/path,expected)
    pin(TEST_SOURCE)
    snapshot={'files_sha256':pins,'directory_files':trees,'candidate_pool_paths':list(map(str,pool_paths())),
              'historical_identity_sha256':materializer.row_hash(sorted(materializer.historical_ids(DOMAIN),key=repr)),
              'inherited_confirmation':inherited['metadata']}
    verify_generation_exclusions_unchanged(snapshot)
    return snapshot


def exclusions(snapshot):
    blocked=materializer.historical_ids(DOMAIN);historical=len(blocked)
    for path in snapshot['candidate_pool_paths']:
        blocked|=materializer.identity_set(DOMAIN,[json.loads(line) for line in Path(path).read_text().splitlines()])
    return blocked,historical


def verify_witnesses(rows):
    total=0
    for row in rows:
        spec=json.loads(row['answer']);cases=tuple(spec['cases']);tier=row['level3_difficulty']
        if (type(tier) is not int or len(cases)!=len(set(cases)) or len(cases)!=4
                or parse_python_factor_spec(spec)!=cases or row['problem']!=_prompt(cases)
                or row.get('level3_generator')!=candidate.GENERATOR or not candidate.eligible_cases(cases,tier)):
            raise ValueError('original four-case prompt or fixed Python v7 minimum band differs')
        expected_metadata={'level3_generation_profile':candidate.PROFILE,
            'level3_python_preset':candidate.PRESETS[tier],
            'level3_case_window':list(candidate.CASE_WINDOWS[tier]),
            'level3_minimum_case_band':list(candidate.MINIMUM_BANDS[tier]),
            'level3_maximum_smallest_factor':candidate.MAX_SMALLEST_FACTOR,
            'level3_case_max':candidate.MAX_VALUE}
        if any(row.get(key)!=value for key,value in expected_metadata.items()):
            raise ValueError('fixed Python v7 row metadata differs')
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
    targets=reference_histograms();calibration=calibration_histogram()
    total=sum(targets.values(),Counter());result={}
    for tier in candidate.PRESETS:
        result[str(tier)]={}
        for support,count in sorted(total.items()):
            available=candidate.available_capacity(support,tier,blocked)
            demand=4*(count+calibration[support])
            if available<demand:raise ValueError(f'capacity insufficient: tier{tier}/support{support}: {available}<{demand}')
            result[str(tier)][str(support)]={'raw':candidate.available_capacity(support,tier,set()),
                'remaining':available,'conservative_four_preset_demand':demand,
                'calibration_rows_per_tier':calibration[support],
                'split_counts':{split:target[support] for split,target in targets.items()}}
    return result


def candidate_description(snapshot=None):
    policy=common.contract()
    common.require(policy['generation_seed_bases'][DOMAIN]=={'development':DEVELOPMENT_SEED,
          'capacity':CAPACITY_SEED,'train':TRAIN_SEED,'eval':EVAL_SEED}
          and policy['generation_tier_stride']==1000 and policy['pool_roots'][DOMAIN]==str(POOL_ROOT)
          and policy['calibration_rows_per_tier'][DOMAIN]==ROWS_PER_TIER,'common Python v7 policy differs')
    snapshot=source_snapshot() if snapshot is None else snapshot
    target=calibration_histogram()
    return {'name':REVISION,'domain':DOMAIN,
        'generator_path':str(Path(candidate.__file__).resolve()),'generator_sha256':file_sha(candidate.__file__),
        'materializer_path':str(Path(__file__).resolve()),'materializer_sha256':file_sha(__file__),
        'tests_path':str(TEST_SOURCE),'tests_sha256':file_sha(TEST_SOURCE),
        'pool_root':str(POOL_ROOT),'rows_per_tier':ROWS_PER_TIER,'support_cells_per_tier':43,
        'development_receipts':{str(tier):str(common.RESULTS/f'calibration_3b_python_v7_d{tier}.json') for tier in candidate.PRESETS},
        'development_seed_base':DEVELOPMENT_SEED,'capacity_seed_base':CAPACITY_SEED,
        'final_train_seed_base':TRAIN_SEED,'final_eval_seed_base':EVAL_SEED,
        'calibration_histogram':{str(support):count for support,count in sorted(target.items())},
        'presets':{str(tier):value for tier,value in candidate.PRESETS.items()},
        'case_windows':list(map(list,candidate.CASE_WINDOWS)),'minimum_case_bands':list(map(list,candidate.MINIMUM_BANDS)),
        'maximum_smallest_factor':candidate.MAX_SMALLEST_FACTOR,'minimum_proper_divisors_per_case':candidate.MIN_DIVISORS,
        'exact_case_count':4,'uniformity':'Integer profile tickets over joint(divisor_count,minimum_band_membership), requiring at least one band member; uniform within-class combinations give equal mass to every eligible distinct unordered case set conditional on support.',
        'quota_prefix_stability':'Fixed per-support RNG stream; exclusions only; no quota-dependent fallback or outcome/seed retries.',
        'completed_diagnostic_path':str(DIAGNOSTIC),'completed_diagnostic_sha256':DIAGNOSTIC_SHA,
        'independent_capacity_path':str(CAPACITY_EVIDENCE),'independent_capacity_sha256':CAPACITY_EVIDENCE_SHA,
        'source_snapshot':snapshot}


def construct(blocked):
    pools,records={},{};current=set(blocked)
    reference=materializer.reference_rows(DOMAIN,'dev');target=calibration_histogram()
    for tier in candidate.PRESETS:
        seed=DEVELOPMENT_SEED+1000*tier
        rows=candidate.build_pool(DOMAIN,target,current,seed,'level3_development_pool',tier,1)
        checks=materializer.verify_rows(DOMAIN,rows,reference,target,current)
        witnesses=verify_witnesses(rows)
        checks.update(original_canonical_support_verified=True,exactly_four_distinct_cases=True,
                      componentwise_maximum_dev_eval_calibration_histogram=True)
        pools[tier]=rows;records[tier]={'schema':'modebench_level3_development_pool_v1','domain':DOMAIN,
            'difficulty':tier,'seed':seed,'rows':len(rows),'rows_sha256':materializer.row_hash(rows),
            'checks':checks,'support_histogram':dict(sorted(target.items())),
            'source_sha256':file_sha(candidate.__file__),'candidate_revision':REVISION,
            'original_grader_witnesses':witnesses,
            'information_boundary':'Fresh v3 candidate DEV only; historical completed confirmation informed laws and fixed benchmark targets; new candidate confirmation outcomes do not exist.'}
        current|=materializer.identity_set(DOMAIN,rows)
        print(json.dumps({'development_difficulty':tier,'rows':len(rows),'witnesses':witnesses}),flush=True)
    after_pools=capacity_table(current);runs=[]
    for tier in candidate.PRESETS:
        for split_index,split in enumerate(materializer.SPLITS):
            reference=materializer.reference_rows(DOMAIN,split);target=materializer.modes(reference)
            seed=CAPACITY_SEED+1000*tier+10000*split_index
            rows=candidate.build_pool(DOMAIN,target,current,seed,'python_v7_capacity_'+split,tier,1)
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


def authenticated_registration(expected_sha):
    record=common.validate_registration(REGISTRATION,expected_sha)
    registered=record['candidate_revisions'][DOMAIN]
    expected=candidate_description(registered['source_snapshot'])
    common.require(all(registered.get(key)==value for key,value in expected.items()),
                   'registered Python v7 law/source/seed/calibration descriptor differs')
    snapshot=registered['source_snapshot']
    common.require(all(record['files_sha256'].get(path)==sha for path,sha in snapshot['files_sha256'].items()),
                   'registration omits Python source snapshot files')
    common.require(all(record['directory_files'].get(path)==files for path,files in snapshot['directory_files'].items()),
                   'registration omits Python source snapshot inventories')
    verify_generation_exclusions_unchanged(snapshot)
    return record,snapshot


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--materialize-development',action='store_true')
    parser.add_argument('--registration-sha256')
    args=parser.parse_args(argv)
    common.require(not POOL_ROOT.exists() and not (ARTIFACTS/'structural_audit.json').exists(),
                   'fresh Python v7 pool root and structural audit required')
    if args.materialize_development:
        common.require(args.registration_sha256,'explicit registration hash required before publication')
        _,snapshot=authenticated_registration(args.registration_sha256)
    else:
        common.require(args.registration_sha256 is None,'preview does not consume a registration')
        snapshot=source_snapshot()
    for tier in candidate.PRESETS:
        output=common.RESULTS/f'calibration_3b_python_v7_d{tier}.json'
        common.require(not output.exists() and not Path(str(output)+'.batches').exists(),
                       'candidate model work must not precede fresh pool materialization')
    blocked,historical=exclusions(snapshot)
    before=capacity_table(blocked)
    pools,records,audit=construct(blocked)
    verify_generation_exclusions_unchanged(snapshot)
    if not args.materialize_development:
        print(json.dumps({'status':'structural_preview_pass','candidate_revision':REVISION,
            'development_rows':4*ROWS_PER_TIER,'excluded_identities':len(blocked),
            'capacity_before_pools':before,**audit},sort_keys=True));return
    authenticated_registration(args.registration_sha256)
    staging=Path(tempfile.mkdtemp(prefix='.'+POOL_ROOT.name+'.',dir=POOL_ROOT.parent))
    try:
        folder=staging/'pools'/DOMAIN;folder.mkdir(parents=True)
        for tier,rows in pools.items():
            path=folder/f'difficulty_{tier}.jsonl'
            with path.open('x') as handle:
                for row in rows:handle.write(json.dumps(row,sort_keys=True)+'\n')
            records[tier].update(registration_path=str(REGISTRATION),registration_sha256=args.registration_sha256)
            common.atomic_new(path.with_suffix('.identity.json'),records[tier])
        verify_generation_exclusions_unchanged(snapshot)
        common.require(file_sha(REGISTRATION)==args.registration_sha256,'registration changed during generation')
        common.require(not POOL_ROOT.exists(),'fresh Python v7 pool root required')
        staging.rename(POOL_ROOT)
    except BaseException:
        shutil.rmtree(staging,ignore_errors=True);raise
    report={'schema':'modebench_level3_python_v7_structural_audit_v1','registration_path':str(REGISTRATION),
        'registration_sha256':args.registration_sha256,'generator_source_sha256':file_sha(candidate.__file__),
        'pool_root':str(POOL_ROOT),'development_pools':records,**audit,'inherited_sealed_files_unchanged':2232,
        'old_completed_confirmation_used_for_diagnostics_and_fixed_reference':True,
        'new_candidate_confirmation_outcomes_loaded':False,'fitting_performed':False,'gpu_submission_performed':False,
        'pool_files_sha256':{str(path):file_sha(path) for path in sorted(POOL_ROOT.rglob('*')) if path.is_file()}}
    common.atomic_new(ARTIFACTS/'structural_audit.json',report)
    print(json.dumps({'status':'registered_and_materialized','registration_sha256':args.registration_sha256,
        'development_rows':4*ROWS_PER_TIER,'capacity_rows_verified':audit['capacity_rows_verified']}),flush=True)

if __name__=='__main__':main()
